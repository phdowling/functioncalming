import pytest
from pydantic import BaseModel, field_validator, Field

from functioncalming import get_completion
from tests.conftest import STRUCTURED_OUTPUTS_WERE_USED


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

    assert STRUCTURED_OUTPUTS_WERE_USED in caplog.text


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

    assert STRUCTURED_OUTPUTS_WERE_USED not in caplog.text