import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar

from openai.types.chat import ChatCompletionMessageToolCall

from functioncalming.utils import OpenAIFunction


@dataclasses.dataclass
class CalmContext:
    tool_call: ChatCompletionMessageToolCall
    openai_function: OpenAIFunction

calm_context: ContextVar[CalmContext | None] = ContextVar("calm_context", default=None)

@contextmanager
def set_calm_context(
        tool_call: ChatCompletionMessageToolCall,
        openai_function: OpenAIFunction
):
    token = calm_context.set(CalmContext(tool_call=tool_call, openai_function=openai_function))
    try:
        yield
    finally:
        calm_context.reset(token)

