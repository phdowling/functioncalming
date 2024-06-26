import dataclasses
import functools
import uuid

import json
import logging
import os
from typing import TextIO, Literal, Dict
from functools import cached_property

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, \
    ChatCompletionToolMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, \
    ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from functioncalming.utils import invoke_callback_function, FineTuningData, InnerValidationError, \
    create_openai_function, OpenAIFunction, ToolCallError, create_abbreviated_openai_function, \
    serialize_openai_function_result, log_finetuning_data
from functioncalming.types import BaseModelOrJsonCompatible, Messages, BaseModelFunction

_client = None


def get_client():
    global _client
    if not _client:
        _client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
            max_retries=os.environ.get("OPENAI_MAX_RETRIES", 2)
        )
    return _client


def set_client(client: AsyncOpenAI):
    global _client
    _client = client


DEFAULT_BEHAVIOR = "default_behavior"


COSTS_BY_MODEL = {
    # GPT-4 Turbo
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-0125-preview": (0.01, 0.03),  # actually a Turbo model
    "gpt-4-1106-preview": (0.01, 0.03),  # actually a Turbo model
    "gpt-4-vision-preview": (0.01, 0.03),
    "gpt-4-1106-vision-preview": (0.01, 0.03),
    # GPT-4
    "gpt-4": (0.03, 0.06),
    "gpt-4-0613": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4-32k-0613": (0.06, 0.12),
    # GPT 3.5
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),
    "gpt-3.5-turbo-1106": (0.0005, 0.0015),
    "gpt-3.5-turbo-0301": (0.0005, 0.0015),  # available in playground but not in docs?
    "gpt-3.5-turbo-instruct": (0.0015, 0.0020),  # not actually a chat model
    # deprecated 3.5
    "gpt-3.5-turbo-0613": (0.0005, 0.0015),
    "gpt-3.5-turbo-16k-0613": (0.0005, 0.0015),
    "gpt-3.5-turbo-16k": (0.0005, 0.0015),
}


class ToolCallShortcut:
    def __init__(self, message: ChatCompletionMessage):
        self.model = None
        self.choices = [Choice(finish_reason="tool_calls", index=0, message=message)]
        self.usage: CompletionUsage = CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )


@dataclasses.dataclass
class CalmResponse:
    success: bool
    tool_call_results: list[BaseModelOrJsonCompatible]
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

    @cached_property
    def _cost_model_usage(self) -> tuple[str | None, float, CompletionUsage]:
        total_cost = 0.
        model = None
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
            prompt_costs_per_1k, completion_costs_per_1k = COSTS_BY_MODEL[model]
            usage.completion_tokens += completion_tokens
            usage.prompt_tokens += prompt_tokens
            total_cost += (
                    prompt_costs_per_1k * prompt_tokens / 1000.
                    + completion_costs_per_1k * completion_tokens / 1000.
            )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return model, total_cost, usage


async def get_completion(
        messages: Messages | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        tools: list[type[BaseModel] | BaseModelFunction] | None = None,
        tool_choice: Literal[DEFAULT_BEHAVIOR] | ChatCompletionToolChoiceOptionParam = DEFAULT_BEHAVIOR,
        abbreviate_tools: bool = False,
        abbreviation_system_prompt: str | None = "Shortcut tool calling active! If calling tools, omit arguments.",
        retries: int = 0,
        rewrite_log_destination: str | TextIO | None = None,
        rewrite_log_extra_data: dict | None = None,
        openai_client: AsyncOpenAI | None = None,
        model: Literal["gpt-3.5-turbo", "gpt-4-1106-preview"] | str = None,
        _track_raw_request_summaries: bool = False,
        **kwargs
) -> CalmResponse:
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
    :param rewrite_log_destination: filename or io handle to log fine-tuning data to
    :param rewrite_log_extra_data: extra data to merge into the jsonl line for this log entry
    :param openai_client: optional AsyncOpenAI client to use (use set_client() to set a default client)
    :param model: Which OpenAI model to use for the completion
    :param _track_raw_request_summaries: If true, adds a _raw_request_summary field to each of the objects in
        CalmResponse.raw_completions. This can be useful for understanding the full (virtual) message history and set of
        tools that was included with each request.
    :param kwargs: Other keyword arguments to pass to the OpenAI completions API call
    :return: a CalmResponse object
    """
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

    abbreviation_mode = abbreviate_tools
    abbreviation_mode_attempted_calls: set[str] = set()

    available_function_names: set[str] = set(calm_functions.openai_functions.keys())

    # tracks successful tool call outputs
    result_instances: list[BaseModel] = []

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

        generated_completion = await _generate_one_completion(
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

        last_message: ChatCompletionMessage = generated_completion.choices[0].message
        internal_messages.append(
            # make sure the history only has dict objects
            last_message.model_dump(exclude_unset=True, exclude_none=True)
        )

        if not last_message.tool_calls:
            # no tool calls: just break the loop (we're done)
            break

        # there were tool calls, let's try to execute them
        outcomes = await execute_tool_calls(
            tool_calls=last_message.tool_calls,
            openai_functions=openai_functions
        )

        new_successful_instances = [outcome.result for outcome in outcomes if outcome.success]
        new_successful_tool_calls = [outcome.raw_tool_call for outcome in outcomes if outcome.success]
        new_successful_tool_responses = [outcome.to_response() for outcome in outcomes if outcome.success]
        # Note: this may be a mixture of successful responses and errors
        new_messages = [outcome.to_response() for outcome in outcomes]

        # 'errors' is overwritten intentionally, we only ever care about the errors of the last tool call(s)
        errors = [outcome.result for outcome in outcomes if not outcome.success]

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
                # end loop early, cutting all of the abbreviated function calls from the message history
                internal_messages = internal_messages[:rewrite_cutoff]
                abbreviation_mode = False
                # however, also restrict the set of functions to those that the model actually tried to call
                available_function_names = abbreviation_mode_attempted_calls
                # on the next iteration, the model will now be able to choose only from these functions,
                #  but now with full definitions given
                continue
        else:
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
                           f"Please carefully recall the tool definition(s) and repeat all failed calls "
                           f"(but only repeat the failed calls!)."
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
        successful_tool_responses=successful_tool_responses
    )

    final_error = None
    if errors:
        final_error = ExceptionGroup("Tool calling validation errors", errors)
    elif rewrite_log_destination is not None:
        functions_coalesced: list[OpenAIFunction] = list(calm_functions.openai_functions.values())
        log_finetuning_data(
            destination=rewrite_log_destination,
            messages=internal_messages,
            functions=functions_coalesced,
            extra_data=rewrite_log_extra_data
        )

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


async def _generate_one_completion(
        messages: Messages,
        openai_functions: dict[str, OpenAIFunction],
        available_function_names: set[str],
        tool_choice: Literal[DEFAULT_BEHAVIOR] | ChatCompletionToolChoiceOptionParam,
        model: str,  # todo
        openai_client: AsyncOpenAI,
        _track_raw_request_summaries: bool,
        **kwargs
):
    if tool_choice == DEFAULT_BEHAVIOR:
        current_tool_choice = get_tool_choice_for_default_behavior(available_function_names)
    else:
        current_tool_choice = tool_choice

    current_tools = [openai_functions[name].tool_definition for name in available_function_names] or NOT_GIVEN

    shortcut: ChatCompletionMessage | None = await maybe_shortcut_trivial_function_call(
        available_function_names,
        openai_functions
    )
    generated_completion: ChatCompletion | ToolCallShortcut
    if shortcut is not None:
        generated_completion = ToolCallShortcut(message=shortcut)
    else:
        generated_completion = await openai_client.chat.completions.create(
            messages=messages,
            model=model,
            tools=current_tools,
            tool_choice=current_tool_choice,
            **kwargs
        )

    if _track_raw_request_summaries:
        generated_completion._raw_request_summary = RawRequestSummary(
            messages=messages[:], tools=current_tools, tool_choice=current_tool_choice
        )
    return generated_completion


@dataclasses.dataclass
class ToolCallOutcome:
    success: bool
    tool_call_id: str
    raw_tool_call: ChatCompletionMessageToolCall
    result: BaseModelOrJsonCompatible | Exception
    tool_name: str | None

    def to_response(self) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=self.tool_call_id,
            content=serialize_openai_function_result(self.result) if self.success else f"Error: {self.result}"
        )


async def execute_tool_calls(
        tool_calls: list[ChatCompletionMessageToolCall], openai_functions
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
        openai_function = create_openai_function(model_or_fn)
        openai_functions[openai_function.name] = openai_function

        abbreviated_function = create_abbreviated_openai_function(model_or_fn=model_or_fn)
        abbreviated_functions[abbreviated_function.name] = abbreviated_function

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
):
    """
    Rewrite the history by removing all messages (not just the failed calls) and replacing them with a single clean
    multi call and its responses, starting from the cutoff index.
    """
    omitted_messages = messages[rewrite_cutoff:]
    if successful_tool_calls:
        messages[rewrite_cutoff:] = [
            {
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in successful_tool_calls],
                "content": None,
            },
            *successful_tool_responses
        ]
    return omitted_messages

