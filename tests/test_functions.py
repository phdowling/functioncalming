import io

import json
import logging

import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field

from functioncalming import get_completion, get_client
from functioncalming.client import USING_STRUCTURED_OUTPUTS
from functioncalming.context import calm_context, CalmContext
from functioncalming.utils import ToolCallError
from tests.conftest import MockOpenAI


def get_weather(city: str, zip: str | None = None) -> str:
    """
    Get the weather
    :param city: city name
    :param zip: zip code
    :return: the weather
    """
    return "lots of snow"


def get_time(city: str, zip_code: str | None = None) -> str:
    context: CalmContext = calm_context.get()
    logging.info(context.tool_call.function.name)
    return "7pm"

class Something(BaseModel):
    field: str = "Hello World"

def returns_list() -> list[list[Something]]:
    return [[Something()]]

@pytest.mark.asyncio
async def test_simple_function_call():
    calm_response = await get_completion(
        user_message="What's the weather like in Berlin?",
        tools=[get_weather, get_time],
    )
    assert "snow" in json.dumps(calm_response.messages)
    assert calm_response.tool_call_results[0] == "lots of snow"


@pytest.mark.asyncio
async def test_nested_result():
    calm_response = await get_completion(
        user_message="Hello",
        tools=[returns_list],
        tool_choice='required'
    )
    res: list[list[Something]] = calm_response.tool_call_results[0]
    assert res[0][0].field == "Hello World"


@pytest.mark.asyncio
async def test_context(caplog):
    calm_response = await get_completion(
        user_message="What time is it?",
        tools=[get_weather, get_time],
    )
    assert 'get_time' in caplog.text


@pytest.mark.asyncio
async def test_shortcut():
    def noop():
        return "works"
    calm_response = await get_completion(
        user_message="Does not matter what I type here",
        tools=[noop],
    )
    assert calm_response.tool_call_results[0] == "works"
    assert calm_response.cost == 0
    assert calm_response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_wrong_function_name():
    mock_client = MockOpenAI(get_client())
    mock_client.add_next_responses(
        ChatCompletionMessage(**{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "does_not_exist",
                        "arguments": "{}"
                    }
                },
            ]
        }),
        # now comes a failure response, then:
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "3",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Munich"}'
                    }
                },
            ]
        },
    )
    with io.StringIO() as fake_file:
        history = []
        calm_response = await get_completion(
            messages=history,
            user_message="What's the weather like in Berlin?",
            tools=[get_weather, get_time],
            rewrite_log_destination=fake_file,
            retries=2,
            openai_client=mock_client
        )
        file_content = fake_file.getvalue()
    assert "does not exist" in str(calm_response.messages_raw)
    assert "lots of snow" in calm_response.tool_call_results


@pytest.mark.asyncio
async def test_retry_happens():
    raised = False

    def weird_function(text: str = "hello") -> str:
        nonlocal raised
        if not raised:
            raised = True
            raise ToolCallError("Try again!")
        return "success"

    calm_response = await get_completion(
        user_message="Hi there",
        tools=[weird_function],
        retries=1
    )
    assert calm_response.success
    assert calm_response.retries_done == 1
    assert calm_response.tool_call_results[0] == "success"

    raised = False
    calm_response = await get_completion(
        user_message="Hi there",
        tools=[weird_function],
        retries=0
    )
    assert not calm_response.success
    assert calm_response.retries_done == 0
    assert not calm_response.tool_call_results


@pytest.mark.asyncio
async def test_abbreviate(caplog):

    class GoodParam(BaseModel):
        """
        Main parameter model.
        """

        number: int = Field(
            ..., description="This is the main input parameter, please make sure to always pass it"
        )

    class BadParam(GoodParam):
        """
        This is the main param of the other function.
        If we add many tokens here, the request becomes more expensive. For that reason, it's advisable to either keep
        descriptions short, or to use abbreviated tool calling!
        This happens to have a pretty long description, most of it is irrelevant but it sure does add to the total
        token count!
        """
        another_param: dict = Field(..., description="Especially if there are more parameters!")

    class good_function(BaseModel):  # make this a basemodel so it will be called with structured outputs eventually
        """
        This is the function you need to call
        :param text: A text param
        :return:
        """
        param: GoodParam

        @property
        def result(self):
            return str(self.param.number)


    bad_tools = []
    for i in range(10):
        def bad_function(param: BadParam) -> str:
            """
            This is a function you will not call.
            :param text: A text param
            :return:
            """
            return "bad"
        bad_function.__name__ = f"bad_function_{i}"
        bad_tools.append(bad_function)

    unabbreviated_calm_response = await get_completion(
        user_message="Call the good function please, use 123 as the input number",
        tools=[good_function, *bad_tools],
        abbreviate_tools=False,
        retries=0,
        _track_raw_request_summaries=True,
    )
    assert unabbreviated_calm_response.success
    assert unabbreviated_calm_response.retries_done == 0
    assert len(unabbreviated_calm_response.raw_completions) == 1
    assert unabbreviated_calm_response.tool_call_results[0].result == "123"
    unabbreviated_prompt_len = unabbreviated_calm_response.usage.prompt_tokens
    unabbreviated_cost = unabbreviated_calm_response.cost

    abbreviated_calm_response = await get_completion(
        user_message="Call the good function please, use 123 as the input number",
        tools=[good_function, *bad_tools],
        abbreviate_tools=True,
        retries=0,
        _track_raw_request_summaries=True,
    )
    abbreviated_prompt_len = abbreviated_calm_response.usage.prompt_tokens
    abbreviated_cost = abbreviated_calm_response.cost
    assert abbreviated_calm_response.success
    assert abbreviated_calm_response.retries_done == 0
    assert len(abbreviated_calm_response.raw_completions) == 2
    assert abbreviated_calm_response.tool_call_results[0].result == "123"

    assert abbreviated_prompt_len < unabbreviated_prompt_len
    assert abbreviated_cost < unabbreviated_cost
    assert USING_STRUCTURED_OUTPUTS in caplog.text





# TODO make sure that retry information is hidden in the clean message history

# TODO simulate a correct function call that, internally, raises a validation error,
#  and make sure this does not lead to a retry!
