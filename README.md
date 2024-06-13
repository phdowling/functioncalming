# functioncalming
## Installation
`pip install functioncalming`

## Overview
Get (near-)guaranteed structured responses from OpenAI using pydantic and function calling (and, if you like, fine-tuning).

functioncalming uses OpenAI's function calling in combination with pydantic model validation to hide away the messy details of getting structured responses from an LLM.

functioncalming comes with support for:
- Structured responses from the LLM via pydantic models
- Structured responses from the LLM via plain python function (pydantic argument validation happens under the hood)
- Parallel function calling, as well as giving the model a choice of multiple different tools
- Automatically passing function/tool results back to the model
- Automatic message history re-writing to hide failed function calls that were re-tried
- Create fine-tuning data to make model better at calling your functions/models with near zero config
- Create fine-tuning data for distilling a complex pipeline to a simple model via a simple decorator (`@distillery`)
- Reporting the cost of your API requests (using OpenAI pricing as of April 2024)

## Who is this for?
Basically, functioncalming provides useful utilities for any case where you find yourself using function calling in OpenAI. 
However, it particularly shines in use-cases where any of the following are the case:
- LLM responses are consumed in a mostly machine-facing way (i.e. the output of the LLM is used in a workflow instead of direct conversation with a user)
- LLMs are used for data extraction, i.e. you just want to extract a possibly complex and nested structured object from an input (rather than just calling e.g. a simple `get_weather()`-style function)
- The same function(s) are called over and over again, and you want to fine-tune a cheaper model to reach the level of quality that GPT-4 offers
- A cheaper (e.g. `gpt-3.5-turbo`) model should be fine-tuned (**distilled**) to perform the task of a complex pipeline based on an expensive model (e.g. `gpt-4`) directly

## Usage
Simple example of calling two functions in parallel (may be flaky using a real model, but this is how parallel calls are done):

```python
from pydantic import BaseModel
from functioncalming.client import get_completion


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


class EmojiTranslation(BaseModel):
    translation: str


PROMPT = """You help extract cleaned data from unstructured input text 
and simultaneously (but separately) turn the text into an Emoji-translation.
You also have a tendency to always make a mistake the first time you call a function, but then do it correctly.
"""

history = [
    {'role': 'system', 'content': PROMPT},
    {'role': 'user', 'content': "The quick brown fox jumps over the lazy dog"}
]


async def main():
    calm_response = await get_completion(
        messages=history,
        tools=[Situation, EmojiTranslation],
        temperature=0,
        retries=1,
        rewrite_log_destination='finetune.jsonl', 
    )
    print(calm_response.success)
    print(calm_response.retries_done)
    print(calm_response.usage)  # total tokens used 
    print(calm_response.cost)  # estimated dollar cost of all requests that were done
    print(calm_response.tool_call_results[0].model_dump_json(
        indent=4))  # {"actors": [{"name": "fox", "adjectives": ["quick", "brown"]}, {"name": "dog", "adjectives": ["lazy"]}], "action": "jumping over"}
    print(calm_response.tool_call_results[1].model_dump_json(indent=4))  # {"translation": "🦊↗️🐶"}
    print(f"Clean, rewritten history: {len(calm_response.messages)} messages. Real history: {len(calm_response.messages_raw)} messages.")
```
## Generating fine-tuning data for distillation
functioncalming tries to make it easy to generate data for function distillation - i.e. fine-tuning a cheaper, faster "student" pipeline
to perform a complex task that can be reliably achieved using a more expensive, slower "teacher" pipeline. The ideas is to track the inputs 
and outputs of the teacher pipeline and use them to train the student pipeline to perform the task directly.

What functioncalming provides here is a simple interface to "clean up" and augment the message history of the teacher pipeline to 
have the correct format for the student fine-tuning task with no custom data cleaning scripts required.

TODO - show how to set up a distillation pipeline.

## functioncalming and instructor
Credit where it's due: functioncalming takes inspiration from https://github.com/jxnl/instructor and serves the same basic purpose.

It's an alternative (or supplement) to `instructor` that is opinionated in a different way and has (probably) slightly different priorities: 
ease of use, exposing all features of the function calling API, and providing tools for improving function calling performance and reliability.

A few differences vs instructor (as of early December 2023):
- Message history re-writing (i.e. hiding failed function call attempts from the model in subsequent calls / fine-tuning data)
  - This tends to make subsequent calls more likely to succeed if you continue sending more messages in the same conversation
  - It also makes the resulting message history more suitable for fine-tuning
- functioncalming avoids supplying/hard-coding fixed prompts (almost everywhere), while instructor has hard-coded prompts in a few places
  - This is not necessarily an advantage or disadvantage per se - in my own work I just prefer being able to customize prompts everywhere 
- Support for multiple response models (i.e. multiple tool calls) in a single completion call
- Support for multiple returned response objects (i.e. parallel tool calls, independent of whether multiple models were used)
- functioncalming handles calling functions directly and returns results
  - in instructor (from my understanding) you need to invoke the functions yourself, but it ships some helpers for doing this 
  - It also handles returning extraction/function results back to the model (not particularly difficult, but one less thing to code yourself)
- functioncalming provides its own get_completion method instead of monkey-patching OpenAI
  - not really a feature, just opinionation
  - note: it still exposes all underlying settings and config of the `openai` library via kwargs
- both libraries help with distillation, but again with different approaches/APIs (and instructor goes further with CLI utilities for triggering training runs, etc.)
- functioncalming does not ship LLM-validators for pydantic (but in principle, those from instructor should work with functioncalming)
- functioncalming does not, in its current release, support json-mode or legacy function calling as its underlying mechanisms
- Currently, instructor has much nicer docs and is probably better supported :)

It might make sense to use both libraries together. functioncalming does not handle many of the features instructor provides (e.g. LLM-based pydantic validators, fine-tuning CLI, etc.). 
If your use-case is simply to call OpenAI with multiple functions and/or to generate fine-tuning/distillation training data for a repeatable function-calling task, 
functioncalming might be a more straightforward option. 
