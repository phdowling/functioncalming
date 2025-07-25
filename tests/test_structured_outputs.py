import dataclasses
from typing import Any

import pytest
from pydantic import BaseModel, field_validator, Field, ConfigDict

from functioncalming import get_completion
from functioncalming.client import USING_STRUCTURED_OUTPUTS
from functioncalming.types import EscapedOutput


@pytest.mark.asyncio
async def test_structured_outputs_are_used(caplog):
    log_search_str = "Using Structured Outputs without tool calling for this request."

    fail_next = True

    class MyStructuredOutput(BaseModel):
        joke: str
        explanation: str

        @field_validator("joke")
        def validate_joke(cls, value):
            nonlocal fail_next
            if fail_next:
                fail_next = False
                raise ValueError("That joke was not quite funny enough, please try again.")
            return value

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Tell me a joke",
        tools=[MyStructuredOutput],
        retries=1,
        model="gpt-4o-mini"
    )

    assert isinstance(calm_response.tool_call_results[0], MyStructuredOutput)

    assert USING_STRUCTURED_OUTPUTS in caplog.text


@pytest.mark.asyncio
async def test_structured_outputs_are_not_used_in_older_models(caplog):
    fail_next = True

    class MyStructuredOutput(BaseModel):
        joke: str
        explanation: str

        @field_validator("joke")
        def validate_joke(cls, value):
            nonlocal fail_next
            if fail_next:
                fail_next = False
                raise ValueError("That joke was not quite funny enough, please try again.")
            return value

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Tell me a joke",
        tools=[MyStructuredOutput],
        retries=1,
        model="gpt-3.5-turbo"
    )

    assert isinstance(calm_response.tool_call_results[0], MyStructuredOutput)

    assert USING_STRUCTURED_OUTPUTS not in caplog.text


@pytest.mark.asyncio
async def test_field_info_gets_adjusted():
    class Joke(BaseModel):
        joke: str = Field(..., description="A joke. **Must be the same genre** of joke as the example.", examples=["Knock knock?\n> Who's there? Interrupting cow! > Interrupting cow wh-MOOOOOO"])
        another_field: str = Field(..., description="This field MUST contain the string literal 'sudo' for the response to be valid.")

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Tell me a joke",
        tools=[Joke],
        retries=1,
        model="gpt-4o-mini"
    )

    assert "knock" in calm_response.tool_call_results[0].joke
    assert calm_response.tool_call_results[0].another_field == "sudo"



@pytest.mark.asyncio
async def test_defaults_work():
    obj = object()

    class Output(BaseModel):
        model_config = ConfigDict(validate_default=False)
        leave_blank: str = Field(obj, description="leave blank.")

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Do not fill in anything",
        tools=[Output],
        retries=1,
        model="gpt-4o-mini"
    )

    assert calm_response.tool_call_results[0].leave_blank is obj

@pytest.mark.asyncio
async def test_escaping_output():
    def gen():
        i = 0
        while True:
            yield i
            i += 1


    def my_tool():
        return EscapedOutput(data=gen(), result_for_model='Okay!')

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Call my_tool",
        tools=[my_tool],
        retries=1,
        model="gpt-4o-mini"
    )
    res = calm_response.tool_call_results[0]
    assert next(res.data) == 0
    assert res.result_for_model == 'Okay!'
    assert calm_response.messages[-1]['content'] == "Okay!"

