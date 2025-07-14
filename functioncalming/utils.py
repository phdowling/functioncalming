import types

import dataclasses

import logging

import functools
import inspect
import json
from inspect import Parameter
from pydantic_core import PydanticUndefined, to_jsonable_python
from types import MappingProxyType
from typing import Callable, Awaitable, Literal, get_origin, get_args, ForwardRef, Union

from docstring_parser import parse, Docstring
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, create_model, Field, ValidationError
from pydantic.fields import FieldInfo

from functioncalming.types import JsonCompatible, EscapedOutput


def pascal_to_snake(pascal_string):
    snake_list = [pascal_string[0].lower()]

    for char in pascal_string[1:]:
        if char.isupper():
            snake_list.extend(['_', char.lower()])
        else:
            snake_list.append(char)

    snake_case_string = ''.join(snake_list)
    return snake_case_string


class InnerValidationError(Exception):
    def __init__(self, messages, original_error):
        super().__init__(messages)
        self.original_error = original_error


class ToolCallError(ValueError):
    pass


@dataclasses.dataclass
class OpenAIFunction:
    name: str
    description: str

    callback: Callable[[...], Awaitable[BaseModel]]
    non_validating_model: type[BaseModel]
    model: type[BaseModel]
    was_defined_as_basemodel: bool

    @functools.cached_property
    def schema(self):
        # TODO use non-validating model for functions? Do the same best practices apply?
        schema = self.non_validating_model.model_json_schema()
        schema["strict"] = True  # turns on structured outputs
        schema.pop("title", None)
        schema.pop("description", None)
        return schema

    @property
    def definition(self) -> FunctionDefinition:
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=self.schema
        )

    @property
    def callback_expects_args_from_model(self) -> bool:
        return bool(self.schema["properties"])

    @functools.cached_property
    def tool_definition(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(type="function", function=self.definition)


type UnsetPlaceholder = Literal["__UNSET"]
UNSET_PLACEHOLDER: UnsetPlaceholder = "__UNSET"
_model_cache: dict[type[BaseModel], type[BaseModel]] = {}


def adjust_annotation_model_references(
        annotation,
        _forward_declarations: set[type[BaseModel]],
):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None and origin is not Literal:
        adjusted_args = []
        for arg in args:
            adjusted_args.append(
                adjust_annotation_model_references(arg, _forward_declarations)
            )
        if origin is types.UnionType:
            annotation = Union[*adjusted_args]
        else:
            annotation = origin[*adjusted_args]
    else:
        if isinstance(annotation, str):
            annotation = ForwardRef(annotation)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in _forward_declarations:
                annotation = ForwardRef(annotation.__name__)
            else:
                annotation = get_or_make_adjusted_model_for_openai(
                    annotation, _forward_declarations=_forward_declarations
                )
    return annotation


def adjust_field_info_for_openai(
        field_info: FieldInfo,
        _forward_declarations: set[type[BaseModel]],
):
    field_info = FieldInfo.merge_field_infos(field_info)  # makes a copy

    # TODO remove unsupported constraints like length, greater/less than, etc. see here
    #  https://openai.com/index/introducing-structured-outputs-in-the-api/
    #  and
    #  https://platform.openai.com/docs/guides/structured-outputs

    # make sure all references to BaseModels are the OpenAI-adjusted variant
    annotation = adjust_annotation_model_references(field_info.annotation, _forward_declarations)

    # get rid of any actual default value, but allow model to place a default placeholder token
    if field_info.default is not PydanticUndefined:
        field_info.default = PydanticUndefined
        annotation = annotation | UnsetPlaceholder
        field_info.description = "\n".join(
            [
                field_info.description or '',
                f'This field has a default value, to use it, set the field to "{UNSET_PLACEHOLDER}".'
            ]
        ).strip()

    # subsume examples
    if field_info.examples:
        field_info.description = "\n".join(
            [
                field_info.description or '',
                f"Examples:\n{"\n".join(ex for ex in field_info.examples)}"
            ]
        ).strip()
        field_info.examples = None

    return annotation, field_info


def get_or_make_adjusted_model_for_openai(
        model: type[BaseModel],
        keep_in_model_cache: bool = True,
        _forward_declarations: set[type[BaseModel]] | None = None,
):
    _forward_declarations = set() if _forward_declarations is None else _forward_declarations
    _forward_declarations.add(model)
    if model in _model_cache:
        return _model_cache[model]

    # make a clone with no custom validators so OpenAI has a better shot of instantiating the model
    non_validating_model: type[BaseModel] = create_model(
        model.__name__,
        **{
            field_name: adjust_field_info_for_openai(
                field_info=field_info,
                _forward_declarations=_forward_declarations
            )
            for field_name, field_info
            in model.model_fields.items()
        }
    )  # clone, so we don't alter the original
    non_validating_model.model_config["extra"] = "forbid"
    if keep_in_model_cache:
        _model_cache[model] = non_validating_model
    return non_validating_model


def rebuild_cached_models():
    namespace = {model.__name__: model for model in _model_cache.values()}
    for model in _model_cache.values():
        model.model_rebuild(_types_namespace=namespace)


@functools.lru_cache()
def create_openai_function(model_or_fn: BaseModel | Callable, keep_in_model_cache: bool = True) -> OpenAIFunction:
    was_defined_as_basemodel = False
    name = model_or_fn.__name__
    description = model_or_fn.__doc__

    # TODO support arbitrary type definitions like unions? We could automatically wrap the types that OpenAI does not
    #  directly support in a class like Output(result=...) and unpack it before we return it to the user

    if description is None:
        logging.warning(f"Tool {model_or_fn} does not have a docstring! Model may not know how to use it.")
        description = ""

    if isinstance(model_or_fn, type) and issubclass(model_or_fn, BaseModel):
        was_defined_as_basemodel = True
        as_model = model_or_fn

        async def callback(*args, **kwargs):  # this is just to always make the callback awaitable
            serialized = remove_unsets(kwargs)
            return model_or_fn(**serialized)

    elif inspect.isfunction(model_or_fn):
        description, param_descriptions_from_docstring = description_and_param_docs_from_docstring(description)
        as_model = basemodel_from_function(
            model_or_fn,
            name,
            param_descriptions_from_docstring
        )

        callback = create_callback_function_for_tool_use(model_or_fn, as_model)
    else:
        raise ValueError(f"Don't know how to turn {model_or_fn} into an OpenAI function")

    non_validating_model = get_or_make_adjusted_model_for_openai(as_model, keep_in_model_cache=keep_in_model_cache)

    return OpenAIFunction(
        name=name,
        description=description,
        callback=callback,
        non_validating_model=non_validating_model,
        model=as_model,
        was_defined_as_basemodel=was_defined_as_basemodel
    )


def create_abbreviated_openai_function(model_or_fn: Callable | BaseModel) -> OpenAIFunction:
    async def stub() -> None:
        pass
    stub.__name__ = model_or_fn.__name__
    stub.__doc__ = model_or_fn.__doc__
    return create_openai_function(stub, keep_in_model_cache=False)


def description_and_param_docs_from_docstring(function_docstring) -> tuple[str, dict[str, str]]:
    param_descriptions = {}
    description = function_docstring
    if function_docstring is not None:
        docstring: Docstring = parse(function_docstring)
        description = docstring.long_description or docstring.short_description or function_docstring
        param_descriptions = {p.arg_name: p.description for p in docstring.params}
    return description, param_descriptions


def remove_unsets(result: dict):
    for field_name, field_value in list(result.items()):
        if field_value == UNSET_PLACEHOLDER:
            del result[field_name]
        if isinstance(field_value, dict):
            result[field_name] = remove_unsets(field_value)
        if isinstance(field_value, list):
            result[field_name] = [remove_unsets(item) if isinstance(item, dict) else item for item in field_value]
    return result


def create_callback_function_for_tool_use(
        fn: Callable[[...], BaseModel] | Callable[[...], Awaitable[BaseModel]],
        validator: type[BaseModel]
):
    @functools.wraps(fn)
    async def callback(**kwargs):
        parsed = validator(**remove_unsets(kwargs))  # validation errors from here get raised as-is
        try:
            res = fn(**dict(parsed))
            if inspect.isawaitable(res):
                res = await res
        except ValidationError as e:
            # we do this to differentiate between a ValidationError in the function args value vs. an inner
            # validation error that has nothing to do with how the model called the tool
            raise InnerValidationError(
                "A pydantic Validation error was thrown during tool execution "
                "(but the model appears to have invoked the tool correctly), cancelling.",
                original_error=e
            ) from e
        return res

    return callback


def basemodel_from_function(model_or_fn, name, param_descriptions_from_docstring) -> type[BaseModel]:
    # TypeAdapter(model_or_fn) would be suitable too - unfortunately it doesn't care about docstrings
    params: MappingProxyType[str, Parameter] = inspect.signature(model_or_fn).parameters
    as_model = create_model(
        name,
        **{
            name: (
                param.annotation,
                Field(
                    default=(... if param.default is Parameter.empty else param.default),
                    description=param_descriptions_from_docstring.get(name)
                )
            )
            for name, param
            in params.items()
        }
    )
    return as_model


def serialize_openai_function_result(result: JsonCompatible | EscapedOutput):
    if isinstance(result, EscapedOutput):
        result = result.result_for_model

    if not isinstance(result, str):
        # if the function returns a str, just return it verbatim
        result = json.dumps(to_jsonable_python(result))

    return result
