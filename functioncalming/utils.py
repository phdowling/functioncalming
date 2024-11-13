import dataclasses

import logging

import functools
import inspect
import json
from inspect import Parameter
from types import MappingProxyType
from typing import Callable, Awaitable, TextIO

from docstring_parser import parse, Docstring
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessage
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, RootModel, create_model, Field, ValidationError
from pydantic.fields import FieldInfo

from functioncalming.types import BaseModelOrJsonCompatible, Messages


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
    definition: FunctionDefinition
    callback: Callable[[...], Awaitable[BaseModel]]
    unvalidated_model: type[BaseModel]
    model: type[BaseModel]
    callback_expects_args_from_model: bool
    was_defined_as_basemodel: bool

    @functools.cached_property
    def tool_definition(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(type="function", function=self.definition)


@functools.lru_cache()
def create_openai_function(model_or_fn: BaseModel | Callable) -> OpenAIFunction:
    # double check: do we need to iterate through the whole schema to find references to plain name and replace them?
    # could see this being the case for recursive models

    # todo is there any benefit to function-style naming?
    # name = pascal_to_snake(model_or_fn.__name__)
    was_defined_as_basemodel = False
    name = model_or_fn.__name__
    description = model_or_fn.__doc__

    if description is None:
        logging.warning(f"Tool {model_or_fn} does not have a docstring! Model may not know how to use it.")
        description = ""

    if isinstance(model_or_fn, type) and issubclass(model_or_fn, BaseModel):
        was_defined_as_basemodel = True
        as_model = model_or_fn
        annotations = model_or_fn.__annotations__
        # make a clone with no custom validators so OpenAI has a better shot of instantiating the model
        as_unvalidated_model = create_model(
            name,
            **{
                field_name: (annotations[field_name], field_info)
                for field_name, field_info
                in as_model.__fields__.items()
            }
        )  # clone, so we don't alter the original
        as_unvalidated_model.model_config["extra"] = "forbid"

        async def callback(*args, **kwargs):  # this is just to always make the callback awaitable
            return model_or_fn(*args, **kwargs)

    elif inspect.isfunction(model_or_fn):
        description, param_descriptions_from_docstring = description_and_param_docs_from_docstring(description)
        as_model = basemodel_from_function(
            model_or_fn,
            name,
            param_descriptions_from_docstring
        )
        as_unvalidated_model = as_model
        callback = create_callback_function_for_tool_use(model_or_fn, as_model)
    else:
        raise ValueError(f"Don't know how to turn {model_or_fn} into an OpenAI function")


    schema = as_model.model_json_schema()
    schema["strict"] = True  # turns on structured outputs

    expects_args_from_model = bool(schema["properties"])

    schema.pop("title", None)
    schema.pop("description", None)

    return OpenAIFunction(
        name=name,
        definition=FunctionDefinition(
            name=name,
            description=description.strip(),
            parameters=schema
        ),
        callback=callback,
        callback_expects_args_from_model=expects_args_from_model,
        unvalidated_model=as_unvalidated_model,
        model=as_model,
        was_defined_as_basemodel=was_defined_as_basemodel
    )


def create_abbreviated_openai_function(model_or_fn: Callable | BaseModel) -> OpenAIFunction:
    async def stub() -> None:
        pass
    stub.__name__ = model_or_fn.__name__
    stub.__doc__ = model_or_fn.__doc__
    return create_openai_function(stub)


def description_and_param_docs_from_docstring(function_docstring):
    param_descriptions = {}
    description = function_docstring
    if function_docstring is not None:
        docstring: Docstring = parse(function_docstring)
        description = docstring.long_description or docstring.short_description or function_docstring
        param_descriptions = {p.arg_name: p.description for p in docstring.params}
    return description, param_descriptions


def create_callback_function_for_tool_use(
        fn: Callable[[...], BaseModel] | Callable[[...], Awaitable[BaseModel]],
        validator: type[BaseModel]
):
    @functools.wraps(fn)
    async def callback(**kwargs):
        parsed = validator(**kwargs)  # validation errors from here get raised as-is
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
    # TypeAdapter(model_or_fn) might be suitable too - unfortunately it doesn't care about docstrings
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


def serialize_openai_function_result(result: BaseModelOrJsonCompatible):
    if isinstance(result, (BaseModel, RootModel)):
        return result.model_dump_json()
    return json.dumps(result)


async def invoke_callback_function(
        openai_function: OpenAIFunction,
        kwargs: dict,
) -> tuple[BaseModel, str]:
    result = await openai_function.callback(**kwargs)

    if isinstance(result, (BaseModel, RootModel)):
        serialized = result.model_dump_json()
    else:
        serialized = json.dumps(result)

    return result, serialized


class FineTuningData(BaseModel, extra="allow"):
    messages: Messages
    # todo: update this to Tool as soon as finetuning does not require legacy format anymore
    #  https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-examples
    functions: list[FunctionDefinition]



def log_finetuning_data(
        destination: str | TextIO,
        messages: Messages,
        functions: list[OpenAIFunction],
        extra_data: dict | None = None
):
    if extra_data is not None and "messages" in extra_data:
        logging.warning("'messages' key is being overwritten by extra data!")

    if extra_data is not None and "functions" in extra_data:
        logging.warning("'functions' key is being overwritten by extra data!")

    fd = FineTuningData(
        **{**dict(messages=messages, functions=[t.definition for t in functions]), **(extra_data or {})}
    )

    log_entry = f"{fd.model_dump_json()}\n"
    if isinstance(destination, str):
        with open(destination, "a") as outf:
            outf.write(log_entry)
    else:
        destination.write(log_entry)
