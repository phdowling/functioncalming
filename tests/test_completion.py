import json

import io
import os

import re

import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field, field_validator
from functioncalming.client import get_completion, get_client
from tests.conftest import MockOpenAI

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbat Symbols
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)


class Actor(BaseModel):
    """
    A person or non-human actor involved in a situation
    """
    name: str
    adjectives: list[str]


class Situation(BaseModel):
    """
    A situation or event involving a number of actors
    """
    actors: list[Actor]
    action: str


NO_EMOJIS_FOUND = "No Emojis found!"


class EmojiTranslation(BaseModel):
    """Create an Emoji-based translation of the input"""
    translation: str = Field(description="A representation of the input in the form of emojis")

    @field_validator('translation')
    @classmethod
    def emojis_only(cls, value):
        # Use the regular expression to find matches in the input string
        matches = emoji_pattern.search(value)

        # If there are matches, the string contains only emojis
        if not bool(matches):
            raise ValueError(NO_EMOJIS_FOUND)
        return value


PROMPT = """You help extract cleaned data from unstructured input text 
AND **simultaneously** (in a parallel tool call) turn the text into an Emoji-translation.
You also have a tendency to **always** make a mistake the first time you call a function 
(like using regular text instead of emojis), but then do it correctly on the next attempt.
"""

messages = [
    {'role': 'system', 'content': PROMPT},
    {'role': 'user', 'content': "The quick brown fox jumps over the lazy dog"}
]


@pytest.mark.asyncio
async def test_simple():
    mock_client = MockOpenAI(get_client())
    situation = {
        "actors": [{"name": "fox", "adjectives": ["quick", "brown"]}, {"name": "dog", "adjectives": ["lazy"]}],
        "action": "jumps over"
    }
    mock_client.add_next_responses(
        ChatCompletionMessage(**{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "Situation",
                        "arguments": json.dumps(situation)
                    }
                },
                {
                    "type": "function",
                    "id": "2",
                    "function": {
                        "name": "EmojiTranslation",
                        "arguments": json.dumps({"translation": "idk lol"})
                    }
                },
            ]
        }),
        # now comes a failure response, then:
        {"role": "assistant", "content": None, "tool_calls": [
            {
                "type": "function",
                "id": "3",
                "function": {
                    "name": "EmojiTranslation",
                    "arguments": json.dumps({"translation": "🦊↗️🐶"})
                }
            },
        ]},
    )
    messages_before = len(messages)
    with (io.StringIO() as fake_file):
        calm_response = await get_completion(
            messages=messages,
            tools=[Situation, EmojiTranslation],
            temperature=0,
            retries=1,
            rewrite_log_destination=fake_file,
            openai_client=mock_client
        )
        file_content = fake_file.getvalue()

    # check history is unaltered
    messages_after = len(messages)
    assert messages_before == messages_after

    dumped = [response.model_dump() for response in calm_response.tool_call_results]
    assert len(calm_response.tool_call_results) == 2
    assert situation in dumped

    print(
        f"Real history: {len(calm_response.messages_raw)} messages. "
        f"Rewritten history: {len(calm_response.messages)} messages."
    )
    original_history_str = json.dumps(calm_response.messages_raw)
    new_history_str = json.dumps(calm_response.messages)
    assert len(calm_response.messages_raw) != len(calm_response.messages)
    assert NO_EMOJIS_FOUND in original_history_str
    assert NO_EMOJIS_FOUND not in new_history_str

    assert "tool calls failed" in original_history_str
    assert "tool calls failed" not in new_history_str
    assert "tool calls failed" not in file_content


@pytest.mark.asyncio
async def test_no_function():
    calm_response = await get_completion(system_prompt=None, user_message="Hello")
    assert calm_response.messages[0]['content'] == "Hello"
    assert calm_response.messages[1]['role'] == 'assistant'
    assert calm_response.messages[1] == calm_response.last_message

    async def echo(text: str):
        """Echo"""
        return text

    # make sure message history is valid to continue using
    calm_response = await get_completion(
        messages=calm_response.messages,
        system_prompt=None,
        user_message="Call `echo` with the argument 'Hello'", tools=[echo]
    )
    assert calm_response.tool_call_results[0] == "Hello"
    assert calm_response.last_message['content'] == "Hello"
    assert calm_response.last_message['role'] == 'tool'


@pytest.mark.asyncio
async def test_cost_tracking():
    calm_response = await get_completion(
        system_prompt=None, user_message="Hello", model="gpt-3.5-turbo"
    )

    assert calm_response.cost > 0
    assert calm_response.usage.total_tokens > 5
    assert calm_response.model.startswith("gpt-3.5-turbo")



@pytest.mark.asyncio
async def test_unknown_costs_tracking():
    mock_client = MockOpenAI(get_client())
    mock_client.add_next_responses(
        ChatCompletionMessage(**{"role": "assistant", "content": "hello too"}), model="gpt-whatever"
    )
    calm_response = await get_completion(
        system_prompt=None, user_message="Hello", model="gpt-whatever",
        openai_client=mock_client
    )

    assert calm_response.cost == 0
    assert calm_response.unknown_costs
    assert calm_response.usage.total_tokens > 5
    assert calm_response.model.startswith("gpt-whatever")


