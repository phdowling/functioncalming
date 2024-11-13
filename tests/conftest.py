from openai.types import CompletionUsage
from openai.types.chat.chat_completion import Choice
from typing import Literal

import logging
import dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ParsedChatCompletion

dotenv.load_dotenv()
logging.basicConfig(level=logging.DEBUG)


class _Completions:
    def __init__(self, real_client: AsyncOpenAI):
        self.real_client = real_client
        self.next_responses: list[ChatCompletion] = []

    async def create(self, *args, **kwargs) -> ChatCompletion:
        res = self.next_responses.pop(0)
        if res == "CALL_OPENAI":
            res = await self.real_client.chat.completions.create(*args, **kwargs)
        return res

    async def parse(self, *args, **kwargs) -> ParsedChatCompletion:
        res = self.next_responses.pop(0)
        if res == "CALL_OPENAI":
            res = await self.real_client.beta.chat.completions.parse(*args, **kwargs)
        return res


class _Chat:
    def __init__(self, real_client: AsyncOpenAI):
        self.completions = _Completions(real_client=real_client)


class _Beta:
    def __init__(self, chat: _Chat):
        self.chat = chat


class MockOpenAI:
    def __init__(self, real_client: AsyncOpenAI):
        self.chat = _Chat(real_client=real_client)
        self.beta = _Beta(chat=self.chat)

    def add_next_responses(
            self, *responses: ChatCompletionMessage | dict | Literal["CALL_OPENAI"], model: str = "gpt-nothing"
    ):
        msgs = [
            ChatCompletionMessage(**response) if isinstance(response, dict) else response
            for response in responses
        ]
        completions = [
            ChatCompletion(
                id="1",
                choices=[Choice(
                    index=0,
                    finish_reason="stop",
                    message=msg
                )],
                created=1,
                model=model,
                usage=CompletionUsage(
                    completion_tokens=len(msg.content) if msg.content else 1,
                    prompt_tokens=1,
                    total_tokens=len(msg.content) if msg.content else 1 + 1
                ),
                object="chat.completion"
            )
            for msg in msgs
        ]
        self.chat.completions.next_responses += completions


STRUCTURED_OUTPUTS_WERE_USED = "Using Structured Outputs without tool calling for this request."
