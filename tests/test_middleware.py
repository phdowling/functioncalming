import logging
from asyncio import CancelledError

import httpx
import pytest
from openai import NOT_GIVEN, BadRequestError

from functioncalming import get_completion
from functioncalming.client import calm_middleware

@calm_middleware
async def wrap_openai_call(*, model, messages, tools, tool_choice, response_format, **kwargs):
    logging.info('wrap_before')
    assert model is not None
    assert messages is not None
    assert tools is not None
    assert isinstance(tool_choice, dict)
    assert response_format is NOT_GIVEN
    try:
        completion = yield
        logging.info('wrap_after')
        assert completion is not None
    except BaseException as e:
        logging.info('caught')
        raise e

async def echo(text: str):
    """Echo"""
    return text

@pytest.mark.asyncio
async def test_no_function(caplog):
    # make sure message history is valid to continue using
    calm_response = await get_completion(
        system_prompt=None,
        user_message="Call `echo` with the argument 'Hello'",
        tools=[echo],
        middleware=wrap_openai_call
    )
    assert 'wrap_before' in caplog.text
    assert 'wrap_after' in caplog.text

@pytest.mark.asyncio
async def test_exceptions(caplog, httpx_mock):
    httpx_mock.add_response(
        status_code=400,
        json={"error": "Bad Request"},
    )

    # make sure message history is valid to continue using
    try:
        calm_response = await get_completion(
            system_prompt=None,
            user_message="Call `echo` with the argument 'Hello'",
            tools=[echo],
            middleware=wrap_openai_call
        )
    except BadRequestError:
        pass
    assert 'wrap_before' in caplog.text
    assert 'caught' in caplog.text
    assert 'wrap_after' not in caplog.text