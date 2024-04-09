from typing import Callable, Awaitable

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, RootModel

type SimpleJsonCompatible = str | int | float | bool
type JsonCompatible = dict[str, SimpleJsonCompatible | JsonCompatible] | list[SimpleJsonCompatible | JsonCompatible] | SimpleJsonCompatible
type BaseModelOrJsonCompatible = BaseModel | RootModel | JsonCompatible
type BaseModelFunction = Callable[..., BaseModelOrJsonCompatible] | Callable[..., Awaitable[BaseModelOrJsonCompatible]]
type Messages = list[ChatCompletionMessageParam]