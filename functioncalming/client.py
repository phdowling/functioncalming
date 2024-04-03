import dataclasses
import uuid

import json
import logging
import os
from typing import Callable, TextIO, Awaitable, Literal
from functools import cached_property

from openai._types import NOT_GIVEN
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, \
    ChatCompletionToolMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from functioncalming.utils import Messages, invoke_callback_function, FineTuningData, InnerValidationError, \
    create_openai_function, OpenAIFunction, ToolCallError

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


type SimpleJsonCompatible = str | int | float | bool
type JsonCompatible = dict[str, SimpleJsonCompatible | JsonCompatible] | list[SimpleJsonCompatible | JsonCompatible] | SimpleJsonCompatible
type BaseModelOrJsonCompatible = BaseModel | JsonCompatible
type BaseModelFunction = Callable[..., BaseModelOrJsonCompatible] | Callable[..., Awaitable[BaseModelOrJsonCompatible]]


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
        self.message: ChatCompletionMessage = message
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
    _unaltered_first_message: ChatCompletionMessageParam | None

    # multiple completions means there were retries
    raw_completions: list[ChatCompletion | ToolCallShortcut]
    """
    All the ChatCompletion objects returned by OpenAI during this call, ordered chronologically (last is newest).
    Multiple completions may be returned if retrys were performed.
    """

    @cached_property
    def messages_raw(self):
        """The non-rewritten message history as it was actually performed against the API."""
        res = self.messages[:self._rewritten_from] + self._omitted_messages
        if self._unaltered_first_message is not None:
            res[0] = self._unaltered_first_message
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
        distil_system_prompt: str | None = None,
        tools: list[type[BaseModel] | BaseModelFunction] | None = None,
        tool_choice: Literal[DEFAULT_BEHAVIOR] | ChatCompletionToolChoiceOptionParam = DEFAULT_BEHAVIOR,
        retries: int = 0,
        pass_results_to_model: bool = True,
        rewrite_log_destination: str | TextIO | None = None,
        rewrite_log_extra_data: dict | None = None,
        openai_client: AsyncOpenAI | None = None,
        model: Literal["gpt-3.5-turbo", "gpt-4-1106-preview"] | str = None,
        **kwargs
) -> CalmResponse:
    """
    Get a completion with validated function call responses from the chat completions API.

    :param messages: Message history. Should be None if system_prompt and/or user_message are set
    :param system_prompt: Initial system message (will be added to the beginning of the history - typically used without a 'history' param)
        (if set, do not supply an initial system message in 'history')
    :param user_message: Next user message (will be appended to the history)
    :param distil_system_prompt: If set, the first message of the history will be rewritten to this system message
        (useful for distillation trainng data generation)
    :param tools: list of available tools, given either as BaseModels or functions that return BaseModel instances
    :param tool_choice: By default, forces a tool call if only one tool is available. Otherwise, same as vanilla OpenAI
    :param retries: number of attempts to give the model to fix broken function calls (first try not counted)
    :param rewrite_history_in_place: If true, the history that was passed in will be modified in-place
    :param pass_results_to_model: If true, function results (or created models) are added to the message history
    :param rewrite_log_destination: filename or io handle to log fine-tuning data to
    :param rewrite_log_extra_data: extra data to merge into the jsonl line for this log entry
    :param openai_client: optional AsyncOpenAI client to use (use set_client() to set a default client)
    :param model: Which OpenAI model to use for the completion
    :param kwargs:
    :return: a tuple of (created models or function responses, rewritten message history)
    """
    # make a copy, we do not edit the passed-in message history
    if messages is not None:
        messages = messages[:]

    messages = initialize_and_validate_message_history(
        messages=messages,
        system_prompt=system_prompt,
        user_message=user_message,
        distil_system_prompt=distil_system_prompt
    )
    rewrite_cutoff = len(messages)
    openai_client = openai_client or get_client()

    model = model or os.environ.get("OPENAI_MODEL")
    if model is None:
        raise ValueError("No model specified and OPENAI_MODEL is not set.")

    tools = tools or []
    openai_functions, distillery_openai_functions = process_tool_definitions(tools)
    available_function_names: set[str] = set(openai_functions.keys())

    result_instances: list[BaseModel] = []
    tries_done = 0  # first try is not a retry
    successful_tool_calls: list[ChatCompletionMessageToolCall] = []
    successful_tool_responses: list[ChatCompletionToolMessageParam] = []
    last_message_dict = None

    errors: list[Exception] | None = []
    raw_completions: list[ChatCompletion | ToolCallShortcut] = []
    while tries_done + 1 <= 1 + retries:
        tries_done += 1
        if tool_choice == DEFAULT_BEHAVIOR:
            current_tool_choice = get_tool_choice_for_default_behavior(available_function_names)
        else:
            current_tool_choice = tool_choice

        shortcut: ChatCompletionMessage | None = await maybe_shortcut_trivial_function_call(
            available_function_names,
            openai_functions
        )

        if shortcut is not None:
            last_message = shortcut
            raw_completions.append(
                ToolCallShortcut(message=shortcut)
            )
        else:
            completion: ChatCompletion = await openai_client.chat.completions.create(
                messages=messages,
                model=model,
                tools=[openai_functions[name].tool_definition for name in available_function_names] or NOT_GIVEN,
                tool_choice=current_tool_choice,
                **kwargs
            )
            raw_completions.append(completion)
            last_message: ChatCompletionMessage = completion.choices[0].message

        messages.append(last_message.model_dump(
            exclude_unset=True, exclude_none=True
        ))  # make sure the history only has dict objects

        if not last_message.tool_calls:
            break

        num_successful = 0
        num_failed = 0
        current_errors: list[ValidationError | ToolCallError] = []
        new_available_function_names = set()
        for tool_call in last_message.tool_calls:
            function_name = tool_call.function.name

            if function_name not in openai_functions:
                e = ToolCallError(f"Error: function `{function_name}` does not exist.")
                handle_function_calling_error(
                    e=e, current_errors=current_errors, history=messages, tool_call=tool_call
                )
                new_available_function_names = available_function_names  # could be anything
                num_failed += 1
                continue

            openai_function = openai_functions[function_name]
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                new_available_function_names.add(function_name)  # available for retry
                handle_function_calling_error(
                    e=e, current_errors=current_errors, history=messages, tool_call=tool_call
                )
                num_failed += 1
                continue

            try:
                result_instance, result_for_model, maybe_serialized_result = await invoke_callback_function(
                    openai_function,
                    kwargs=arguments,
                    history=messages,
                    serialize_result_for_model=pass_results_to_model
                )
                tool_response = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=result_for_model
                )
                if openai_function.is_distillery:
                    distillery_tool = distillery_openai_functions[function_name]
                    tool_call, tool_response = adjust_distillery_call_for_clean_history(
                        new_function_name=distillery_tool.name,
                        serialized_result=maybe_serialized_result,
                        tool_call=tool_call
                    )
                messages.append(tool_response)
                result_instances.append(result_instance)
                successful_tool_calls.append(tool_call)
                successful_tool_responses.append(tool_response)
                num_successful += 1
            except InnerValidationError as e:
                # There is not really a need for any custom handling vs. other exceptions here - we just raise.
                # I am just making clear that *inner* validation errors will not lead to a retry,
                # since the model did fine calling the tool here.
                # To force a retry due to a semantic error the model should correct, raise a ToolCallError instead
                raise e
            except (ValidationError, ToolCallError) as e:
                new_available_function_names.add(function_name)  # available for retry
                handle_function_calling_error(
                    e=e, current_errors=current_errors, history=messages, tool_call=tool_call
                )
                num_failed += 1

        if not num_failed:
            errors = []
            break

        logging.debug(f"Attempt {tries_done}/{retries + 1}: {num_failed} errors")
        errors = current_errors
        messages.append({
            "role": "system",
            "content": f"{num_failed}/{num_failed + num_successful} tool calls failed. "
                       f"Please carefully recall the function definition(s) and repeat all failed calls "
                       f"(but only repeat the failed calls!)."
        })
        available_function_names = new_available_function_names

    unaltered_first_message, omitted_messages = rewrite_message_history(
        messages=messages,
        rewrite_cutoff=rewrite_cutoff,
        successful_tool_calls=successful_tool_calls,
        successful_tool_responses=successful_tool_responses,
        distil_system_prompt=distil_system_prompt,
    )

    final_error = None
    if errors:
        final_error = ExceptionGroup("Function calling validation errors", errors)
    elif rewrite_log_destination is not None:
        functions_coalesced: list[OpenAIFunction] = list({**openai_functions, **distillery_openai_functions}.values())
        log_finetuning_data(
            destination=rewrite_log_destination,
            messages=messages,
            functions=functions_coalesced,
            extra_data=rewrite_log_extra_data
        )

    return CalmResponse(
        success=final_error is None,
        tool_call_results=result_instances,
        messages=messages,
        _rewritten_from=rewrite_cutoff,
        _omitted_messages=omitted_messages,
        _unaltered_first_message=unaltered_first_message,
        error=final_error,
        retries_done=tries_done - 1,
        raw_completions=raw_completions
    )


def handle_function_calling_error(
        e: ValidationError | ToolCallError | json.JSONDecodeError,
        current_errors: list[ValidationError | ToolCallError | json.JSONDecodeError],
        history: Messages,
        tool_call
):
    tool_response = ChatCompletionToolMessageParam(
        role="tool",
        tool_call_id=tool_call.id,
        content=f"Error: {e}"
    )
    history.append(tool_response)
    current_errors.append(e)


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
        messages: Messages | None, system_prompt, user_message, distil_system_prompt
) -> Messages:
    """
    Initialize and update the history in-place based on the supplied system prompt and user_message.
    After this call, we guarantee that, if a distil_system_prompt was supplied, the first message in the history is a
    system message (which will be replaced with the distillation prompt during history rewriting).
    """
    messages = messages[:] if messages is not None else []

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

    if distil_system_prompt and messages[0]["role"] != "system":
        raise ValueError(
            "Cannot use 'distil_system_prompt' if the first message in the history is not a system message. "
            "Either use the 'system_message' param or supply a history that starts with a system message."
        )

    return messages


def process_tool_definitions(tools_raw):
    openai_functions: dict[str, OpenAIFunction] = {}
    # Note: these are indexed by their original name
    distillery_openai_functions: dict[str, OpenAIFunction] = {}
    for model_or_fn in tools_raw:
        openai_function = create_openai_function(model_or_fn)
        openai_functions[openai_function.name] = openai_function

        if openai_function.is_distillery:
            distillery_openai_function = create_openai_function(model_or_fn.__functioncalming_distil_model__)
            distillery_openai_functions[openai_function.name] = distillery_openai_function

    return openai_functions, distillery_openai_functions


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


def adjust_distillery_call_for_clean_history(
        *,
        new_function_name: str,
        serialized_result: str | None,
        tool_call: ChatCompletionMessageToolCall,
) -> tuple[ChatCompletionMessageToolCall, ChatCompletionToolMessageParam]:
    adjusted_tool_call = ChatCompletionMessageToolCall(
        id=tool_call.id,
        type="function",
        function=Function(
            name=new_function_name,
            arguments=serialized_result
        )
    )
    adjusted_tool_response = ChatCompletionToolMessageParam(
        role="tool",
        tool_call_id=tool_call.id,
        content='{"result": "success"}',
    )
    return adjusted_tool_call, adjusted_tool_response


def rewrite_message_history(
        *,
        messages: Messages,
        rewrite_cutoff: int,
        successful_tool_calls: list[ChatCompletionMessageToolCall],
        successful_tool_responses: list[ChatCompletionToolMessageParam],
        distil_system_prompt: str | None,
):
    """
    Rewrite the history by removing all messages (not just the failed calls) and replacing them with a single clean
    multi call and its responses, starting from the cutoff index.
    """
    unaltered_first_message = None
    omitted_messages = messages[rewrite_cutoff:]
    if successful_tool_calls:
        if distil_system_prompt is not None:
            unaltered_first_message = messages[0]
            assert unaltered_first_message["role"] == "system", "First message must be a system message when using 'distil_system_prompt'"
            messages[0] = {
                **messages[0],
                "content": distil_system_prompt
            }

        messages[rewrite_cutoff:] = [
            {
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in successful_tool_calls],
                "content": None,
            },
            *successful_tool_responses
        ]
    return unaltered_first_message, omitted_messages



def log_finetuning_data(
        destination: str | TextIO,
        messages: Messages,
        functions: list[OpenAIFunction],
        extra_data: dict | None = None
):
    if extra_data is not None and "messages" in extra_data:
        logging.warning("'messages' key is being overwritten by extra data!")

    if extra_data is not None and "functions" in extra_data:
        logging.warning("'functions' key is being overwritten by extra data!")

    fd = FineTuningData(
        **{**dict(messages=messages, functions=[t.definition for t in functions]), **(extra_data or {})}
    )

    log_entry = f"{fd.model_dump_json()}\n"
    if isinstance(destination, str):
        with open(destination, "a") as outf:
            outf.write(log_entry)
    else:
        destination.write(log_entry)
