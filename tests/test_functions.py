import pytest

from functioncalming import get_completion
from functioncalming.utils import json_dump_messages


def get_weather(city: str, zip: str | None) -> str:
    """
    Get the weather
    :param city: city name
    :param zip: zip code
    :return: the weather
    """
    return "lots of snow"


def get_time(city: str, zip: str | None) -> str:
    return "7pm"


@pytest.mark.asyncio
async def test_simple_function_call():
    (weather_results, *_), message_history = await get_completion(
        user_message="What's the weather like in Berlin?",
        tools=[get_weather, get_time],
        pass_results_to_model=True
    )
    assert "snow" in json_dump_messages(message_history)
    assert weather_results == "lots of snow"

# TODO simulate the model calling a function incorrectly and make sure a retry is done

# TODO make sure that retry information is hidden in the clean message history

# TODO simulate a correct function call that, internally, raises a validation error,
#  and make sure this does not lead to a retry!
