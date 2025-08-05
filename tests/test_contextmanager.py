from contextlib import asynccontextmanager

import pytest
from openai import NOT_GIVEN

from functioncalming import get_completion


@asynccontextmanager
async def wrap_openai_call(*, model, messages, tools, tool_choice, response_format, **kwargs):
    print('wrap_before')
    assert model is not None
    assert messages is not None
    assert tools is not None
    assert response_format is NOT_GIVEN
    yield
    print('wrap_after')

@pytest.mark.asyncio
async def test_no_function(caplog):
    calm_response = await get_completion(system_prompt=None, user_message="Hello")
    async def echo(text: str):
        """Echo"""
        return text

    # make sure message history is valid to continue using
    calm_response = await get_completion(
        messages=calm_response.messages,
        system_prompt=None,
        user_message="Call `echo` with the argument 'Hello'",
        tools=[echo],
        openai_request_context_manager=wrap_openai_call
    )
    assert 'wrap_before' not in caplog.text
    assert 'wrap_after' not in caplog.text