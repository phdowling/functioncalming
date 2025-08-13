# functioncalming

A Python library for reliable structured responses from OpenAI models using function calling and structured outputs with automatic validation, retries, and error handling.

## Why functioncalming?

Working with OpenAI's function calling can be frustrating:
- Models sometimes call functions incorrectly
- Validation errors require manual retry logic
- Message history gets cluttered with failed attempts
- Converting between functions and Pydantic models is tedious
- Cost tracking and token optimization is manual work

functioncalming solves these problems by providing:
- **Automatic retries** with intelligent error handling
- **Clean message history** that hides failed attempts
- **Seamless integration** with both functions and Pydantic models
- **Built-in validation** with helpful error messages to the model
- **Cost tracking** and token optimization features
- **Structured outputs** when available for better reliability

## Installation

```bash
pip install functioncalming
```

## Quick Start

```python
from functioncalming import get_completion
from pydantic import BaseModel, field_validator
import asyncio

def get_weather(city: str, country: str = "US") -> str:
    """Get current weather for a city"""
    return f"Sunny, 72°F in {city}, {country}"

class PersonInfo(BaseModel):
    """Extract person information from text"""
    name: str
    age: int
    occupation: str
    
    @field_validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v

async def main():
    # Function calling example
    response = await get_completion(
        user_message="What's the weather like in Paris?",
        tools=[get_weather],
        model="gpt-4o"
    )
    print(response.tool_call_results[0])  # "Sunny, 72°F in Paris, US"
    
    # Structured output example with validation
    response = await get_completion(
        user_message="Extract info: John is a 30-year-old teacher",
        tools=[PersonInfo],
        model="gpt-4o"
    )
    person = response.tool_call_results[0]
    print(f"{person.name}, age {person.age}, works as {person.occupation}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Features

### Automatic Retries with Clean History

When the model makes mistakes, functioncalming automatically retries and keeps your message history clean:

```python
class EmailAddress(BaseModel):
    email: str
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError("Email must contain @ symbol")
        return v

response = await get_completion(
    user_message="My email is john.doe.gmail.com",  # Missing @
    tools=[EmailAddress],
    retries=2  # Will automatically retry up to 2 times
)

# response.messages contains clean history as if the model got it right first time
# response.messages_raw contains the actual history with retries
print(f"Retries needed: {response.retries_done}")
print(f"Final result: {response.tool_call_results[0].email}")
```

### Mixed Function and Model Tools

Use Python functions and Pydantic models together seamlessly:

```python
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> float:
    """Calculate tip amount"""
    return bill_amount * (tip_percentage / 100)

class Receipt(BaseModel):
    """Generate a structured receipt"""
    items: list[str]
    subtotal: float
    tip: float
    total: float

response = await get_completion(
    user_message="Calculate tip for $45.50 bill and create receipt for coffee and sandwich",
    tools=[calculate_tip, Receipt],
    model="gpt-4o"
)
```

### Cost Tracking and Usage Statistics

Built-in cost tracking for all major OpenAI models:

```python
response = await get_completion(
    user_message="Hello world",
    model="gpt-4o"
)

print(f"Cost: ${response.cost:.4f}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Model: {response.model}")
print(f"Unknown costs: {response.unknown_costs}")  # True for very new models
```

### Tool Abbreviation for Large Schemas

Save tokens when using many tools with large schemas:

```python
# When you have many tools with complex schemas
response = await get_completion(
    user_message="Parse this document",
    tools=[ComplexTool1, ComplexTool2, ComplexTool3, ...],
    abbreviate_tools=True  # First shows tool names only, then full schema for chosen tool
)
```

### Multiple Tool Calls in Parallel

The model can call multiple tools in a single response:

```python
response = await get_completion(
    user_message="Get weather for NYC and LA, then create a travel comparison",
    tools=[get_weather, TravelComparison],
    model="gpt-4o"
)

# response.tool_call_results contains results from all tool calls
for i, result in enumerate(response.tool_call_results):
    print(f"Tool {i+1} result: {result}")
```

### Structured Outputs Integration

Automatically uses OpenAI's Structured Outputs when available for better reliability:

```python
class CodeGeneration(BaseModel):
    """Generate code with strict structure"""
    language: str
    code: str
    explanation: str

# Uses Structured Outputs automatically on compatible models
response = await get_completion(
    user_message="Write a Python function to reverse a string",
    tools=[CodeGeneration],
    model="gpt-4o"  # Structured Outputs supported
)
```

## Advanced Usage

### Custom OpenAI Client

```python
from openai import AsyncOpenAI
from functioncalming.client import set_openai_client

custom_client = AsyncOpenAI(api_key="your-key", timeout=30.0)

with set_openai_client(custom_client):
    response = await get_completion(
        user_message="Hello",
        tools=[SomeTool]
    )
```

### Request Middleware

Wrap OpenAI API calls with custom logic:

```python
from functioncalming.client import calm_middleware


@calm_middleware
async def log_requests(*, model, messages, tools, **kwargs):
    print(f"Making request to {model}")
    try:
        completion = yield  # OpenAI API call happens here
        print(f"Got response with {completion.usage.total_tokens} tokens")
    except Exception as e:
        print(f"Call to OpenAI failed: {e}")
        # Note: you should always re-raise here - of course, you can still wrap the call to get_completion in a try/except
        raise e
    


response = await get_completion(
    user_message="Hello",
    tools=[SomeTool],
    middleware=log_requests
)
```

### Default Values and Escaped Output

Handle default values and return non-JSON data:

```python
from functioncalming.types import EscapedOutput

class Config(BaseModel):
    name: str
    debug_mode: bool = False  # Defaults work automatically

def generate_report():
    # Return complex data that shouldn't be shown to model
    return EscapedOutput(
        result_for_model="Report generated successfully",
        data={"complex": "internal", "data": [1, 2, 3]}
    )

response = await get_completion(
    user_message="Generate config and report",
    tools=[Config, generate_report]
)

# Model sees "Report generated successfully"
# You get the full data object
report_data = response.tool_call_results[1].data
```

### Access to Tool Call Context

Get information about the current tool call from within your functions:

```python
from functioncalming.context import calm_context

def my_tool(param: str) -> str:
    context = calm_context.get()
    tool_call_id = context.tool_call.id
    function_name = context.tool_call.function.name
    return f"Called {function_name} with ID {tool_call_id}"
```

## API Reference

### `get_completion()`

Main function for getting completions with tool calling.

**Parameters:**
- `messages`: Existing message history
- `system_prompt`: System message (added to start of history)
- `user_message`: User message (added to end of history)
- `tools`: List of functions or Pydantic models to use as tools
- `tool_choice`: Control tool selection ("auto", "required", specific tool, or "none")
- `model`: OpenAI model name
- `retries`: Number of retry attempts for failed tool calls
- `abbreviate_tools`: Use tool abbreviation to save tokens
- `openai_client`: Custom OpenAI client
- `openai_request_context_manager`: Middleware for API requests

**Returns:**
`CalmResponse` object with:
- `success`: Whether all tool calls succeeded
- `tool_call_results`: List of tool call results
- `messages`: Clean message history
- `messages_raw`: Raw history including retries
- `cost`: Estimated cost in USD
- `usage`: Token usage statistics
- `model`: Model used
- `error`: Exception if any tool calls failed
- `retries_done`: Number of retries performed

### Model Registration

Register new models for cost tracking:

```python
from functioncalming.client import register_model

register_model(
    model_name="gpt-new-model",
    supports_structured_outputs=True,
    cost_per_1mm_input_tokens=2.0,
    cost_per_1mm_output_tokens=8.0
)
```

## Error Handling

functioncalming handles various error scenarios automatically:

- **Validation errors**: Shown to model for retry
- **JSON parsing errors**: Handled and reported for retry  
- **Unknown function calls**: Model is corrected and retries
- **Inner validation errors**: Raised immediately (no retry)
- **Tool call errors**: Can trigger retries based on error type

```python
from functioncalming.utils import ToolCallError

def strict_function(value: int) -> str:
    if value < 0:
        # This will trigger a retry
        raise ToolCallError("Value must be positive")
    return str(value)
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: `openai`, `pydantic`, `docstring-parser`

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_ORGANIZATION="your-org-id"  # Optional
export OPENAI_MODEL="gpt-4o"  # Default model
export OPENAI_MAX_RETRIES="2"  # Default retry count
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please check the issues page for current development priorities.