import dataclasses
import uuid

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal
from functools import cached_property

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, \
    ChatCompletionToolMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionAssistantMessageParam, \
    ChatCompletionToolParam, ChatCompletionNamedToolChoiceParam, ParsedChatCompletion, ParsedChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from functioncalming.context import set_calm_context
from functioncalming.utils import InnerValidationError, \
    create_openai_function, OpenAIFunction, ToolCallError, create_abbreviated_openai_function, \
    serialize_openai_function_result, rebuild_cached_models
from functioncalming.types import Messages, JsonCompatibleFunction, JsonCompatible

USING_STRUCTURED_OUTPUTS = "Using Structured Outputs without tool calling for this request."

_openai_client: ContextVar[AsyncOpenAI | None] = ContextVar('_openai_client', default=None)


def get_client():
    _client = _openai_client.get()
    if not _client:
        _client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
            max_retries=os.environ.get("OPENAI_MAX_RETRIES", 2)
        )
    return _client

@contextmanager
def set_openai_client(client: AsyncOpenAI):
    token = _openai_client.set(client)
    yield
    _openai_client.reset(token)

type DefaultBehavior = Literal['default_behavior']
DEFAULT_BEHAVIOR: DefaultBehavior = "default_behavior"

def register_model(
        model_name: str,
        supports_structured_outputs: bool,
        cost_per_1mm_input_tokens: float,
        cost_per_1mm_output_tokens: float
):
    if supports_structured_outputs:
        STRUCTURED_OUTPUTS_SUPPORTED.add(model_name)
    COSTS_BY_MODEL[model_name] = (cost_per_1mm_input_tokens, cost_per_1mm_output_tokens)

# for registering new models externally when I forget to update the library again
STRUCTURED_OUTPUTS_SUPPORTED = set()


def structured_outputs_available(model_name: str):
    if model_name == "gpt-4o-2024-05-13":
        return False
    if model_name in STRUCTURED_OUTPUTS_SUPPORTED:
        return True
    return model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('gpt-4o')


# cost per 1MM token
COSTS_BY_MODEL = {
    # o1 and 01-mini
    "o1-preview": (15.0, 60.0),
    "o1-preview-2024-09-12": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o1-mini-2024-09-12": (3.0, 12.0),
    # 4o and 4o-mini
    "gpt-4o-2024-05-13": (5.0, 15.0),  # no structured outputs yet!
    "gpt-4o-2024-08-06": (2.5, 10.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    # GPT-4 Turbo
    "gpt-4-turbo": (10., 30.),
    "gpt-4-0125-preview": (10., 30.),  # actually a Turbo model
    "gpt-4-1106-preview": (10., 30.),  # actually a Turbo model
    "gpt-4-vision-preview": (10., 30.),
    "gpt-4-1106-vision-preview": (10., 30.),
    "gpt-4-turbo-2024-04-09": (10., 30.),
    # GPT-4
    "gpt-4": (30., 60.),
    "gpt-4-0613": (30., 60.),
    "gpt-4-32k": (60., 120.),
    "gpt-4-32k-0613": (0.06, 0.12),
    # GPT 3.5
    "gpt-3.5-turbo": (0.5, 1.5),  # not sure actually
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "gpt-3.5-turbo-1106": (1., 2.),
    "gpt-3.5-turbo-instruct": (1.5, 2.0),  # not actually a chat model
    "gpt-3.5-turbo-16k-0613": (3., 4.),
    "gpt-3.5-turbo-0613": (1.5, 2.),
    "gpt-3.5-turbo-0301": (1.5, 2.),
    "gpt-3.5-turbo-16k": (3., 4.),
}

# TODO refactor these
for model_name in ('gpt-4.1', 'gpt-4.1-2025-04-14'):
    register_model(
        model_name=model_name,
        supports_structured_outputs=True,
        cost_per_1mm_input_tokens=2.0,
        cost_per_1mm_output_tokens=8.0,
    )
for model_name in ('gpt-4.1-mini', 'gpt-4.1-mini-2025-04-14'):
    register_model(
        model_name=model_name,
        supports_structured_outputs=True,
        cost_per_1mm_input_tokens=0.4,
        cost_per_1mm_output_tokens=1.6,
    )

for model_name in ('gpt-4.1-nano', 'gpt-4.1-nano-2025-04-14'):
    register_model(
        model_name=model_name,
        supports_structured_outputs=True,
        cost_per_1mm_input_tokens=0.1,
        cost_per_1mm_output_tokens=0.4,
    )

for model_name in ('gpt-4o', 'gpt-4o-2024-08-06'):
    register_model(
        model_name=model_name,
        supports_structured_outputs=True,
        cost_per_1mm_input_tokens=2.5,
        cost_per_1mm_output_tokens=10.0,
    )

for model_name in ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18'):
    register_model(
        model_name=model_name,
        supports_structured_outputs=True,
        cost_per_1mm_input_tokens=.15,
        cost_per_1mm_output_tokens=.6,
    )


class ToolCallShortcut:
    def __init__(self, message: ChatCompletionMessage):
        self.model = None
        self.choices = [Choice(finish_reason="tool_calls", index=0, message=message)]
        self.usage: CompletionUsage = CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )


@dataclasses.dataclass
class CalmResponse[T: JsonCompatible]:
    success: bool
    tool_call_results: list[T]
    messages: Messages
    error: Exception | None
    retries_done: int

    _rewritten_from: int
    _omitted_messages: Messages

    # multiple completions means there were retries
    raw_completions: list[ChatCompletion | ToolCallShortcut]
    """
    All the ChatCompletion objects returned by OpenAI during this call, ordered chronologically (last is newest).
    Multiple completions may be returned if retries were performed.
    """

    @cached_property
    def messages_raw(self):
        """The non-rewritten message history as it was actually performed against the API."""
        res = self.messages[:self._rewritten_from] + self._omitted_messages
        return res

    @property
    def last_message(self) -> ChatCompletionAssistantMessageParam:
        return self.messages[-1]

    @property
    def model(self) -> str | None:
        return self._cost_model_usage[0]

    @property
    def cost(self) -> float:
        return self._cost_model_usage[1]

    @property
    def usage(self) -> CompletionUsage:
        return self._cost_model_usage[2]

    @property
    def unknown_costs(self) -> bool:
        """
        If the model used is not in the cost lookup table, no cost can be determined and this field is True.
        This can happen with newly released models.
        """
        return self._cost_model_usage[3]

    @cached_property
    def _cost_model_usage(self) -> tuple[str | None, float, CompletionUsage, bool]:
        total_cost = 0.
        model = None
        some_cost_unknown = False
        usage = CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        for completion in self.raw_completions:
            if completion.model is None:
                continue
            # we never switch models between retries for now, not sure if that will ever change
            model = completion.model
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            usage.completion_tokens += completion_tokens
            usage.prompt_tokens += prompt_tokens
            prompt_costs_per_1mm, completion_costs_per_1mm = COSTS_BY_MODEL.get(model, (0., 0.))
            additional_cost = (
                    prompt_costs_per_1mm * prompt_tokens / 1_000_000.
                    + completion_costs_per_1mm * completion_tokens / 1_000_000.
            )

            total_cost += additional_cost
            if additional_cost == 0:
                some_cost_unknown = True
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return model, total_cost, usage, some_cost_unknown


async def get_completion[T: JsonCompatible](
        messages: Messages | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        tools: list[type[T] | JsonCompatibleFunction[T]] | None = None,
        tool_choice: DefaultBehavior | ChatCompletionToolChoiceOptionParam = DEFAULT_BEHAVIOR,
        model: Literal["gpt-3.5-turbo", "gpt-4-1106-preview"] | str = None,
        retries: int = 0,
        openai_client: AsyncOpenAI | None = None,
        abbreviate_tools: bool = False,
        abbreviation_system_prompt: str | None = "Shortcut tool calling active! If calling tools, omit arguments.",
        _track_raw_request_summaries: bool = False,
        **kwargs
) -> CalmResponse[T]:
    """
    Get a completion with validated function call responses from the chat completions API.

    :param messages: Message history. Should be None if system_prompt and/or user_message are set
    :param system_prompt: Initial system message. Will be added to the beginning of the history, typically used without
        the 'messages' param. (if set, do not supply an initial system message in 'messages')
    :param user_message: Next user message (will be appended to the message history)
    :param tools: list of available tools, given either as BaseModels or functions that return BaseModel instances
    :param tool_choice: By default, forces a tool call if only one tool is available. Otherwise, same as vanilla OpenAI
    :param abbreviate_tools: If true, tools are passed to the model without their param signature first to save tokens.
        Once the model tries to call a tool, the conversation is replayed with the full definition of that tool only.
        Setting this to True also allows one additional attempted generation (i.e. up to retries + 2 total calls to the
        API)
    :param abbreviation_system_prompt: An optional system prompt to insert in the message history before generating the
        first completion during abbreviated tool calling. Usually this should tell the model not to supply tool
        arguments to not waste tokens.
    :param retries: number of attempts to give the model to fix broken function calls (first try not counted)
    :param openai_client: optional AsyncOpenAI client to use (use set_client() to set a default client)
    :param model: Which OpenAI model to use for the completion
    :param _track_raw_request_summaries: If true, adds a _raw_request_summary field to each of the objects in
        CalmResponse.raw_completions. This can be useful for understanding the full (virtual) message history and set of
        tools that was included with each request.
    :param kwargs: Other keyword arguments to pass to the OpenAI completions API call
    :return: a CalmResponse object
    """
    if model not in COSTS_BY_MODEL:
        logging.warning(
            f"Model {model} is not (yet) known to functioncalming. "
            f"Cost tracking may be unavailable, and even if structured outputs are supported, they may be deactivated. "
            "To fix this, call register_model() with the appropriate settings for your model. "
        )

    # make a copy, we do not edit the passed-in message history
    internal_messages = messages[:] if messages is not None else []
    retries = max(0, retries)

    internal_messages = initialize_and_validate_message_history(
        messages=internal_messages,
        system_prompt=system_prompt,
        user_message=user_message
    )
    openai_client = openai_client or get_client()

    model = model or os.environ.get("OPENAI_MODEL")
    if model is None:
        raise ValueError("No model specified and OPENAI_MODEL is not set.")

    tools = tools or []
    calm_functions = process_tool_definitions(tools)

    if abbreviate_tools and len(tools) < 2:
        logging.warning("Abbreviation mode deactivated since there are not multiple tools.")
        abbreviate_tools = False

    abbreviation_mode = abbreviate_tools
    abbreviation_mode_attempted_calls: set[str] = set()

    available_function_names: set[str] = set(calm_functions.openai_functions.keys())

    # tracks successful tool call outputs
    result_instances: list[T] = []

    # for tracking what "really happened"
    total_completions_generated = 0
    total_generations_allowed = (2 if abbreviation_mode else 1) + retries
    raw_completions: list[ChatCompletion | ToolCallShortcut] = []

    # for message history rewriting
    rewrite_cutoff = len(internal_messages)
    successful_tool_calls: list[ChatCompletionMessageToolCall] = []
    successful_tool_responses: list[ChatCompletionToolMessageParam] = []

    if abbreviation_mode and abbreviation_system_prompt is not None:
        # this will be cut off once abbreviation mode is no longer active
        internal_messages.append({"role": 'system', 'content': abbreviation_system_prompt})

    had_successful_structured_output = False
    errors: list[Exception] | None = []
    while total_completions_generated < total_generations_allowed:
        openai_functions: dict[str, OpenAIFunction] = calm_functions.openai_functions

        if abbreviation_mode:
            if total_completions_generated == total_generations_allowed - 1:
                raise ToolCallError(
                    f"Ran out of retries during abbreviation phase "
                    f"({total_completions_generated} completions have been generated, 1 remains, but we are still in abbreviation mode: full tool calls can't be executed)"
                ) from ExceptionGroup("Tool calling validation errors", errors)
            openai_functions = calm_functions.abbreviated_openai_functions

        generated_completion: ChatCompletion | ParsedChatCompletion = await _generate_one_completion(
            messages=internal_messages,
            openai_functions=openai_functions,
            available_function_names=available_function_names,
            tool_choice=tool_choice,
            model=model,
            openai_client=openai_client,
            _track_raw_request_summaries=_track_raw_request_summaries,
            **kwargs
        )
        total_completions_generated += 1
        raw_completions.append(generated_completion)

        last_message: ChatCompletionMessage | ParsedChatCompletionMessage = generated_completion.choices[0].message

        exclusions = set()
        if not last_message.tool_calls:
            exclusions.add("tool_calls")  # emtpy list causes a validation error with OpenAI
        if hasattr(last_message, "parsed") and last_message.parsed is not None:
            exclusions.add("parsed")  # omit "parsed" in the data sent back to the API

        internal_messages.append(
            # make sure the history only has dict objects
            last_message.model_dump(exclude_unset=True, exclude_none=True, exclude=exclusions or None)
        )

        if last_message.tool_calls:
            # there were tool calls, let's try to execute them
            outcomes = await execute_tool_calls(
                tool_calls=last_message.tool_calls,
                openai_functions=openai_functions
            )
            if abbreviation_mode:
                # track any (valid) function that the model tried to call
                #  once we exit abbreviation mode, all of them need to be available
                abbreviation_mode_attempted_calls |= set(
                    outcome.tool_name for outcome in outcomes if outcome.tool_name is not None
                )

                # if we are still in abbreviation mode and all calls were successful:
                #   turn off abbreviation mode
                #   reset the message history to before the tool calls
                #   but only allow the tool calls that were actually made
                if not errors:
                    # end loop early, cutting all abbreviated function calls from the message history
                    internal_messages = internal_messages[:rewrite_cutoff]
                    # TODO omitted_messages is misleading when this code branch is followed
                    abbreviation_mode = False
                    # however, also restrict the set of functions to those that the model actually tried to call
                    available_function_names = abbreviation_mode_attempted_calls
                    # on the next iteration, the model will now be able to choose only from these functions,
                    #  but now with full definitions given
                    continue

            new_successful_instances = [outcome.result for outcome in outcomes if outcome.success]
            new_successful_tool_calls = [outcome.raw_tool_call for outcome in outcomes if outcome.success]
            new_successful_tool_responses = [outcome.to_response() for outcome in outcomes if outcome.success]
            # Note: this may be a mixture of successful responses and errors
            new_messages = [outcome.to_response() for outcome in outcomes]
        elif hasattr(last_message, "parsed") and last_message.parsed is not None:
            assert len(available_function_names) == 1
            oai_fun_name, = available_function_names
            oai_fun = openai_functions[oai_fun_name]
            outcome = await validate_structured_output(last_message, openai_function=oai_fun)
            had_successful_structured_output = outcome.success
            outcomes = [outcome]
            # a structured response is actually not a tool call
            new_successful_tool_calls = []
            new_successful_tool_responses = []  # no need to turn these into responses
            # we only generate a message when the structured output fails
            new_messages = [outcome.to_response()] if not outcome.success else []
            new_successful_instances = [outcome.result] if outcome.success else []
        else:
            # no tool calls: just break the loop (we're done)
            break

        # 'errors' is overwritten intentionally, we only ever care about the errors of the last tool call(s)
        errors = [outcome.result for outcome in outcomes if not outcome.success]

        if not abbreviation_mode:
            # if we are not in abbreviation mode, we just track the outputs and go on to handle errors
            result_instances += new_successful_instances
            successful_tool_calls += new_successful_tool_calls
            successful_tool_responses += new_successful_tool_responses

        internal_messages += new_messages

        if errors:
            # error handling logic actually looks the same between abbreviation mode and regular mode
            num_failed = len(errors)
            num_successful = sum(outcome.success for outcome in outcomes)

            tool_names_for_next_attempt = set(outcome.tool_name for outcome in outcomes if not outcome.success)
            if None in tool_names_for_next_attempt:
                # the model called an unknown function: we can't tell which one it needs to retry,
                # so we pass in all names again to let it retry any one of them
                tool_names_for_next_attempt = available_function_names

            logging.debug(f"Attempt {total_completions_generated}/{retries + 1}: {num_failed} errors")
            internal_messages.append({
                "role": "system",
                "content": f"{num_failed}/{num_failed + num_successful} tool calls failed. "
                           f"Please carefully recall the supplied schema definition(s) and try again."
                           f"If there were multiple calls, only repeat the failed ones!"
            })
            available_function_names = tool_names_for_next_attempt
        else:
            # defensive; if we were in abbreviation mode with no errors, the loop should have been continued above
            assert not abbreviation_mode
            # no errors: break the loop
            break

    omitted_messages = rewrite_message_history(
        messages=internal_messages,
        rewrite_cutoff=rewrite_cutoff,
        successful_tool_calls=successful_tool_calls,
        successful_tool_responses=successful_tool_responses,
        had_successful_structured_output=had_successful_structured_output
    )

    final_error = None
    if errors:
        final_error = ExceptionGroup("Tool calling validation errors", errors)

    return CalmResponse(
        success=final_error is None,
        tool_call_results=result_instances,
        messages=internal_messages,
        _rewritten_from=rewrite_cutoff,
        _omitted_messages=omitted_messages,
        error=final_error,
        retries_done=total_completions_generated - (2 if abbreviate_tools else 1),
        raw_completions=raw_completions
    )


@dataclasses.dataclass
class RawRequestSummary:
    messages: Messages
    tools: list[ChatCompletionToolParam] | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | dict | NotGiven


def _maybe_resolve_single_tool_choice(tool_choice_for_api_call, available_function_names):
    name_of_only_tool = None
    if tool_choice_for_api_call == "required" and len(available_function_names) == 1:
        name_of_only_tool, = available_function_names
    if tool_choice_for_api_call is not NOT_GIVEN and isinstance(tool_choice_for_api_call, dict):
        tool_choice_for_api_call: ChatCompletionNamedToolChoiceParam
        name_of_only_tool = tool_choice_for_api_call["function"]["name"]
    return name_of_only_tool


async def _call_openai_with_structured_outputs_if_possible(
        messages: Messages,
        model: str,
        openai_client: AsyncOpenAI,
        openai_functions: dict[str, OpenAIFunction],
        available_function_names: set[str],
        tool_choice_for_api_call: ChatCompletionToolChoiceOptionParam,
        _track_raw_request_summaries: bool,
        **kwargs
) -> ChatCompletion | ParsedChatCompletion:

    response_format: type[BaseModel] | None = None
    name_of_only_tool = _maybe_resolve_single_tool_choice(
        tool_choice_for_api_call,
        available_function_names
    )
    if name_of_only_tool is not None and openai_functions[name_of_only_tool].was_defined_as_basemodel:
        # if there is only one tool and it was defined as a BaseModel, we use response_format
        # if the tool was a function, we supply it via tools instead - OpenAI docs say that is best practice
        response_format = openai_functions[name_of_only_tool].non_validating_model

    if response_format and structured_outputs_available(model_name=model):
        # here we know that we definitely want the output to match one specific JSONSchema spec,
        # so we can use structured outputs.
        logging.debug(USING_STRUCTURED_OUTPUTS)
        generated_completion: ParsedChatCompletion[response_format] = await openai_client.beta.chat.completions.parse(
            messages=messages,
            model=model,
            response_format=response_format
        )
        tools_for_api_call = [openai_functions[name_of_only_tool].tool_definition] or NOT_GIVEN
    else:
        tools_for_api_call = [
            openai_functions[name].tool_definition
            for name
            in available_function_names
        ] or NOT_GIVEN

        generated_completion: ChatCompletion = await openai_client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools_for_api_call,
            tool_choice=tool_choice_for_api_call,
            **kwargs
        )
    if _track_raw_request_summaries:
        generated_completion._raw_request_summary = RawRequestSummary(
            messages=messages[:], tools=tools_for_api_call, tool_choice=tool_choice_for_api_call
        )
    return generated_completion


async def _generate_one_completion(
        messages: Messages,
        openai_functions: dict[str, OpenAIFunction],
        available_function_names: set[str],
        tool_choice: DefaultBehavior | ChatCompletionToolChoiceOptionParam,
        model: str,  # todo
        openai_client: AsyncOpenAI,
        _track_raw_request_summaries: bool,
        **kwargs
) -> ChatCompletion | ParsedChatCompletion:
    tool_choice_for_api_call: ChatCompletionToolChoiceOptionParam
    if tool_choice == DEFAULT_BEHAVIOR:
        tool_choice_for_api_call = get_tool_choice_for_default_behavior(available_function_names)
    else:
        tool_choice_for_api_call = tool_choice

    shortcut: ChatCompletionMessage | None = await maybe_shortcut_trivial_function_call(
        available_function_names,
        openai_functions
    )
    generated_completion: ChatCompletion | ToolCallShortcut
    if shortcut is not None:
        generated_completion = ToolCallShortcut(message=shortcut)
    else:
        generated_completion = await _call_openai_with_structured_outputs_if_possible(
            messages=messages,
            model=model,
            openai_client=openai_client,
            openai_functions=openai_functions,
            available_function_names=available_function_names,
            tool_choice_for_api_call=tool_choice_for_api_call,
            _track_raw_request_summaries=_track_raw_request_summaries,
            **kwargs
        )

    return generated_completion


@dataclasses.dataclass
class StructuredOutputOutcome:
    success: bool
    raw_content: str
    result: JsonCompatible | Exception
    tool_name: str | None

    def to_response(self) -> ChatCompletionSystemMessageParam:
        if self.success:
            raise ValueError("Shouldn't need to call to_response on a successful structured response.")
        return ChatCompletionSystemMessageParam(
            role="system",
            content=f"Error: {self.result}"
        )

@dataclasses.dataclass
class ToolCallOutcome:
    success: bool
    tool_call_id: str
    raw_tool_call: ChatCompletionMessageToolCall
    result: JsonCompatible | Exception
    tool_name: str | None

    def to_response(self) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=serialize_openai_function_result(self.result) if self.success else f"Error: {self.result}"
        )


async def validate_structured_output(
        message: ParsedChatCompletionMessage, openai_function: OpenAIFunction
) -> StructuredOutputOutcome:
    success = True
    try:
        result = await openai_function.callback(**message.parsed.model_dump())
    except Exception as e:
        result = e
        success = False
    return StructuredOutputOutcome(
        success=success,
        raw_content=message.content,
        result=result,
        tool_name=openai_function.name
    )

async def execute_tool_calls(
        tool_calls: list[ChatCompletionMessageToolCall],
        openai_functions: dict[str, OpenAIFunction]
) -> list[ToolCallOutcome]:
    outcomes = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name

        if function_name not in openai_functions:
            e = ToolCallError(f"Error: function `{function_name}` does not exist.")

            outcomes.append(
                ToolCallOutcome(
                    success=False,
                    tool_call_id=tool_call.id,
                    raw_tool_call=tool_call,
                    result=e,
                    tool_name=None
                )
            )
            continue

        openai_function = openai_functions[function_name]
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            outcomes.append(
                ToolCallOutcome(
                    success=False,
                    tool_call_id=tool_call.id,
                    raw_tool_call=tool_call,
                    result=e,
                    tool_name=function_name
                )
            )
            continue

        try:
            with set_calm_context(
                tool_call=tool_call,
                openai_function=openai_function,
            ):
                result_instance = await openai_function.callback(**arguments)
            outcomes.append(
                ToolCallOutcome(
                    success=True,
                    tool_call_id=tool_call.id,
                    raw_tool_call=tool_call,
                    result=result_instance,
                    tool_name=function_name
                )
            )
        except InnerValidationError as e:
            # There is not really a need for any custom handling vs. other exceptions here - we just raise.
            # I am just making clear that *inner* validation errors will not lead to a retry,
            # since the model did fine calling the tool here.
            # To force a retry due to a semantic error the model should correct, raise a ToolCallError instead
            raise e
        except (ValidationError, ToolCallError) as e:
            outcomes.append(
                ToolCallOutcome(
                    success=False,
                    tool_call_id=tool_call.id,
                    raw_tool_call=tool_call,
                    result=e,
                    tool_name=function_name
                )
            )
    return outcomes


async def maybe_shortcut_trivial_function_call(
        available_function_names: set[str], openai_functions: dict[str, OpenAIFunction]
) -> ChatCompletionMessage | None:
    last_message = None
    if len(available_function_names) == 1:
        only_name = list(available_function_names)[0]
        if not openai_functions[only_name].callback_expects_args_from_model:
            last_message = ChatCompletionMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=uuid.uuid4().hex,
                        type="function",
                        function=Function(
                            name=only_name,
                            arguments="{}"
                        )
                    )
                ]
            )
    return last_message


def initialize_and_validate_message_history(
        messages: Messages | None, system_prompt, user_message
) -> Messages:
    """
    Initialize and update the history in-place based on the supplied system prompt and user_message.
    After this call, we guarantee that, if a system_prompt was supplied, the first message in the history is a
    system message.
    """
    if not (messages or system_prompt or user_message):
        raise ValueError(
            "No input - supply at least 'messages', 'system_prompt' and/or 'user_message'"
        )

    if system_prompt:
        if messages:
            # TODO maybe just raise on this? Is this ever a useful way to call get_completion in practice?
            logging.warning(
                "Both 'history' and 'system_prompt' were supplied, "
                "be aware that the system_prompt will be inserted at the beginning of the history."
            )

        if messages and messages[0]["role"] == "system":
            raise ValueError(
                "First message in history was already a system message! "
                "Cowardly refusing to replace it / append another one."
            )
        messages.insert(0, ChatCompletionSystemMessageParam(role="system", content=system_prompt))

    if user_message:
        messages.append(ChatCompletionUserMessageParam(role="user", content=user_message))

    return messages


@dataclasses.dataclass
class ProcessedFunctions:
    openai_functions: dict[str, OpenAIFunction]
    abbreviated_openai_functions: dict[str, OpenAIFunction]


def process_tool_definitions(tools_raw) -> ProcessedFunctions:
    openai_functions: dict[str, OpenAIFunction] = {}
    # Note: these are indexed by their original name
    abbreviated_functions: dict[str, OpenAIFunction] = {}
    for model_or_fn in tools_raw:
        openai_function = create_openai_function(model_or_fn, keep_in_model_cache=True)
        openai_functions[openai_function.name] = openai_function

        abbreviated_function = create_abbreviated_openai_function(model_or_fn=model_or_fn)
        abbreviated_functions[abbreviated_function.name] = abbreviated_function

    rebuild_cached_models()

    return ProcessedFunctions(
        openai_functions=openai_functions, abbreviated_openai_functions=abbreviated_functions
    )


def get_tool_choice_for_default_behavior(
        openai_function_names: list[str] | set[str]
) -> ChatCompletionToolChoiceOptionParam:
    tool_choice: ChatCompletionToolChoiceOptionParam
    if not openai_function_names:
        tool_choice = NOT_GIVEN
    elif len(openai_function_names) == 1:
        tool_choice = {"type": "function", "function": {"name": list(openai_function_names)[0]}}
    else:
        tool_choice = "auto"
    return tool_choice


def rewrite_message_history(
        *,
        messages: Messages,
        rewrite_cutoff: int,
        successful_tool_calls: list[ChatCompletionMessageToolCall],
        successful_tool_responses: list[ChatCompletionToolMessageParam],
        had_successful_structured_output: bool
):
    """
    Rewrite the history by removing all messages (not just the failed calls) and replacing them with a single clean
    multi call and its responses, starting from the cutoff index.
    """
    omitted_messages = []
    stashed_final_structured_output = None
    if had_successful_structured_output:
        stashed_final_structured_output = messages[-1]

    if successful_tool_calls:
        omitted_messages = messages[rewrite_cutoff:-1 if stashed_final_structured_output else None]
        messages[rewrite_cutoff:] = [
            {
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in successful_tool_calls],
                "content": None,
            },
            *successful_tool_responses
        ]
    elif had_successful_structured_output:
        omitted_messages = messages[rewrite_cutoff:-1]
        messages[:] = messages[:rewrite_cutoff]

    if stashed_final_structured_output:
        messages.append(stashed_final_structured_output)
    return omitted_messages

