import dataclasses
import datetime
from typing import Callable, Awaitable, Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, RootModel

type SimpleJsonCompatible = str | int | float | bool | datetime.datetime| BaseModel | RootModel | None
type JsonCompatible = SimpleJsonCompatible | dict[str, JsonCompatible] | list[JsonCompatible]

type Messages = list[ChatCompletionMessageParam]

@dataclasses.dataclass
class EscapedOutput[T: JsonCompatible]:
    """
    Helper for letting tools return arbitrary results without attempting to show them to the model verbatim.
    When returning an instance of EscapedOutput from a tool, `result_for_model` will be used as the tool call return
    value in the message history, but `CalmResponse.tool_call_results` will contain this full object.
    """
    result_for_model: T
    data: Any

type JsonCompatibleFunction[T: JsonCompatible] = Callable[..., T | EscapedOutput[T]] | Callable[..., Awaitable[T | EscapedOutput[T]]]