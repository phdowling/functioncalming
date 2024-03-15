import dataclasses

import logging

import functools
import inspect
import json
from inspect import Parameter
from types import MappingProxyType
from typing import Callable, Awaitable

from docstring_parser import parse, Docstring
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessage
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, RootModel, create_model, Field, ValidationError

type Messages = list[ChatCompletionMessageParam]


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


@dataclasses.dataclass
class OpenAIFunction:
    name: str
    definition: FunctionDefinition
    callback: Callable[[...], Awaitable[BaseModel]]
    callback_expects_user_message: bool
    callback_expects_args_from_model: bool
    is_distillery: bool

    @functools.cached_property
    def tool_definition(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(type="function", function=self.definition)


@functools.lru_cache()
def create_openai_function(model_or_fn: BaseModel | Callable) -> OpenAIFunction:
    # double check: do we need to iterate through the whole schema to find references to plain name and replace them?
    # could see this being the case for recursive models
    name = pascal_to_snake(model_or_fn.__name__)
    is_distillery = getattr(model_or_fn, "__is_functioncalming_distillery__", False)
    description = model_or_fn.__doc__
    if description is None:
        logging.warning(f"Tool {model_or_fn} does not have a docstring! Model may not know how to use it.")
        description = ""

    if isinstance(model_or_fn, type) and issubclass(model_or_fn, BaseModel):
        as_model = model_or_fn

        async def callback(*args, **kwargs):  # this is just to always make the callback awaitable
            return model_or_fn(*args, **kwargs)

    elif inspect.isfunction(model_or_fn):
        description, param_descriptions = description_and_param_docs_from_docstring(description)
        as_model = basemodel_from_function(model_or_fn, name, param_descriptions)
        callback = create_callback_function_for_tool_use(model_or_fn, as_model)
    else:
        raise ValueError(f"Don't know how to turn {model_or_fn} into an OpenAI function")

    schema = as_model.model_json_schema()

    # we want to hide this from the OpenAI model, but know whether we need to pass it back in later
    expects_user_message = schema["properties"].pop("user_message", None) is not None
    if expects_user_message and "user_message" in schema["required"]:
        schema["required"].remove("user_message")

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
        callback_expects_user_message=expects_user_message,
        callback_expects_args_from_model=expects_args_from_model,
        is_distillery=is_distillery
    )


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


def basemodel_from_function(model_or_fn, name, param_descriptions) -> type[BaseModel]:
    # TypeAdapter(model_or_fn) might be suitable too - unfortunately it doesn't care about docstrings
    params: MappingProxyType[str, Parameter] = inspect.signature(model_or_fn).parameters
    as_model = create_model(
        name,
        **{
            name: (
                param.annotation,
                Field(
                    default=(... if param.default is Parameter.empty else param.default),
                    description=param_descriptions.get(name)
                )
            )
            for name, param
            in params.items()
        }
    )
    return as_model


async def invoke_callback_function(
        openai_function: OpenAIFunction,
        kwargs: dict,
        history: Messages,
        serialize_result_for_model: bool
) -> tuple[BaseModel, str, str | None]:
    if openai_function.callback_expects_user_message:
        last_user_message = ([m["content"] for m in history if m["role"] == "user"] or [None])[-1]
        if last_user_message is None:
            raise ValueError("No user message to supply to callback function!")
        kwargs = {**kwargs, "user_message": last_user_message}

    result = await openai_function.callback(**kwargs)

    result_for_model = "{'result': 'success'}"
    serialized = None
    if serialize_result_for_model or openai_function.is_distillery:
        if isinstance(result, (BaseModel, RootModel)):
            serialized = result.model_dump_json()
        else:
            serialized = json.dumps(result)

    if serialize_result_for_model:
        result_for_model = serialized

    return result, result_for_model, serialized


class FineTuningData(BaseModel, extra="allow"):
    messages: Messages
    # todo: update this to Tool as soon as finetuning does not require legacy format anymore
    #  https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-examples
    functions: list[FunctionDefinition]
