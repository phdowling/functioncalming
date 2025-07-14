# functioncalming
Get near-guaranteed structured responses from OpenAI models using pydantic and tool calling / structured outputs.

This library provides a convenience wrapper to avoid certain repetitive patterns that I often ended up reimplementing when using the OpenAI library directly.
## Installation
`pip install functioncalming`

## Example
```python
from functioncalming import get_completion
from pydantic import BaseModel, field_validator
import datetime
import asyncio


def get_weather(city: str, postal_code: str | None = None) -> str:
    return "pretty sunny"


def get_time(city: str, zip_code: str | None = None) -> datetime.datetime:
    return datetime.datetime.now()

# tools passed in as BaseModel-derived classes will use Structured Outputs
class AddressInMunich(BaseModel):
    street: str
    city: str
    postal_code: str

    @field_validator('postal_code')
    def validate_zip(cls, value):
        if not str(value).startswith('80') or str(value).startswith('81'):
            raise ValueError("Munich postal codes start with 80 or 81")
        return value

async def main():
    calm_response = await get_completion(
        user_message="What's the weather like in Munich?",
        # tools can be python functions or Pydantic models
        tools=[get_weather, get_time, AddressInMunich],
        model="gpt-4.1",
    )
    # tools are called automatically and their results are accessible on the CalmResponse object 
    # tool_call_results is a list because the model may call multiple tools in parallel
    assert calm_response.tool_call_results[0] == "pretty sunny"
    # calm_response.messages is the rewritten message history (rewritten to hide retries)
    # the tool response message also gets generated and appended here
    assert "sunny" in calm_response.messages[-1]['content']

    calm_response_2 = await get_completion(
        user_message="Make up a random address in Munich (using the appropriate tool)",
        tools=[get_weather, get_time, AddressInMunich],
        model="gpt-4.1",
    )
    postal_code = str(calm_response_2.tool_call_results[0].postal_code)
    assert postal_code.startswith("80") or postal_code.startswith("81")
    
    print(f"Cost: ${calm_response.cost + calm_response_2.cost}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Feature overview
Main additions over OpenAI's own tool calling:
- Allows you to specify validators when using pydantic models e.g. to enforce semantics
- Automatic retries with message history cleaning
  - when the model fails to call the tool / model correctly, the error will be shown to the model for (a configurable number of) retries
  - those retries are automaticallly hidden in the "main" returned chat history, it will look like the model simply called the tool correctly the first time around
- Use plain python functions as tools
- When the model calls them, tools are actually invoked, their response messages generated and added to the result history
- Responses are wrapped in a useful CalmResponse object that neatly exposes the new message history, tool calling results, token counts, best-effort costs, etc.
- Tool abbreviation
  - If your tools generate large JSON schemas and they fill up a lot of your context window, you can "abbreviate" the tools
  - The model will then only be shown the tool without parameters, and once it tries to invoke the tool, the chat is replayed with only that tool available and its full documentation for the model to properly call it
