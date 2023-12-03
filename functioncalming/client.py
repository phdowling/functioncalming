import json
import logging
import os
from typing import Callable, TextIO, Awaitable, Literal

from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, \
    ChatCompletionToolMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolChoiceOptionParam

from functioncalming.utils import Messages, invoke_callback_function, FineTuningData, InnerValidationError, \
    create_openai_function, OpenAIFunction

_client = None


def get_client():
    global _client
    if not _client:
        _client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION"),
        )
    return _client


def set_client(client: AsyncOpenAI):
    _client = client


DEFAULT_BEHAVIOR = "default_behavior"

# TODO technically, any json.dumps(...) compatible output is fine too
type SimpleJsonCompatible = str | int | float
type JsonCompatible = dict[str, SimpleJsonCompatible | JsonCompatible] | list[SimpleJsonCompatible | JsonCompatible]
type ModelOrJsonCompatible = BaseModel | JsonCompatible
type BaseModelFunction = Callable[..., ModelOrJsonCompatible] | Callable[..., Awaitable[ModelOrJsonCompatible]]


async def get_completion(
        messages: Messages | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        tools: list[type[BaseModel] | BaseModelFunction] | None = None,
        tool_choice: Literal[DEFAULT_BEHAVIOR] | ChatCompletionToolChoiceOptionParam = DEFAULT_BEHAVIOR,
        retries: int = 0,
        pass_results_to_model: bool = False,
        rewrite_history: bool = True,
        rewrite_system_prompt_to: str | None = None,
        rewrite_log_destination: str | TextIO | None = None,
        openai_client: AsyncOpenAI | None = None,
        model: Literal["gpt-3.5-turbo", "gpt-4-1106-preview"] | str = None,
        **kwargs
) -> tuple[list[BaseModel], Messages]:
    """
    Get a completion with validated function call responses from the chat completions API.

    :param messages: Message history. Should be None if system_prompt and/or user_message are set
    :param system_prompt: Initial system message to start off conversation
    :param rewrite_system_prompt_to:
    :param user_message: Initial user message to start off conversation. Comes after system_prompt if both are set.
    :param tools: list of available tools, given either as BaseModels or functions that return BaseModel instances
    :param tool_choice: By default, forces a tool call if only one tool is available. Otherwise, same as vanilla OpenAI
    :param retries: number of attempts to give the model to fix broken function calls (first try not counted)
    :param rewrite_history: If true, the messages list that was passed in will be modified in-place
    :param pass_results_to_model: If true, function results (or created models) are added to the message history
    :param rewrite_log_destination: filename or io handle to log fine-tuning data to
    :param openai_client: optional AsyncOpenAI client to use
    :param model: Which OpenAI model to call
    :param kwargs:
    :return: a tuple of (created models / function responses, rewritten message history)
    """
    messages = initialize_messages(
        messages=messages,
        system_prompt=system_prompt,
        user_message=user_message,
        distil_system_prompt=rewrite_system_prompt_to
    )
    rewrite_cutoff = len(messages)
    openai_client = openai_client or get_client()

    model = model or os.environ.get("OPENAI_MODEL")
    if model is None:
        raise ValueError("No model specified and OPENAI_MODEL is not set.")

    tools = tools or []
    openai_functions, distillery_openai_functions = process_tool_definitions(tools)

    if tool_choice == DEFAULT_BEHAVIOR:
        tool_choice = get_tool_choice_for_default_behavior(openai_functions)

    result_instances: list[BaseModel] = []
    retries_done = -1
    successful_tool_calls: list[ChatCompletionMessageToolCall] = []
    successful_tool_responses: list[ChatCompletionToolMessageParam] = []

    errors: list[Exception] | None = []
    while retries_done < retries:
        # TODO skip this if we'll force-call a single function that takes only user_message (common distil case)
        completion: ChatCompletion = await openai_client.chat.completions.create(
            messages=messages,
            model=model,
            tools=[openai_function.tool_definition for openai_function in openai_functions.values()],
            tool_choice=tool_choice,
            **kwargs
        )
        last_message: ChatCompletionMessage = completion.choices[0].message
        messages.append(last_message)

        if not last_message.tool_calls:
            break

        assert last_message.content is None  # TODO remove once debugged
        num_successful = 0
        num_failed = 0
        current_errors = []
        for tool_call in last_message.tool_calls:
            function_name = tool_call.function.name
            openai_function = openai_functions[function_name]
            arguments = json.loads(tool_call.function.arguments)
            try:
                result_instance, result_for_model, maybe_serialized_result = await invoke_callback_function(
                    openai_function, kwargs=arguments, user_message=user_message,
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
                # I am just making clear that *inner* validation errors should not lead to a retry,
                # since the model did fine calling the tool here.
                raise e
            except ValidationError as e:
                tool_response = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=f"{e}"
                )
                messages.append(tool_response)
                current_errors.append(e)
                num_failed += 1

        if not num_failed:
            errors = []
            break

        logging.debug(f"Attempt {retries_done + 2}/{retries + 1}: {num_failed} errors")
        errors = current_errors
        messages.append({
            "role": "system",
            "content": f"{num_failed}/{num_failed + num_successful} tool calls failed. "
                       f"Please carefully recall the function definition(s) and repeat all failed calls "
                       f"(but only repeat the failed calls!)."
        })
        retries_done += 1

    if errors:
        raise ExceptionGroup("Function calling validation errors", errors)

    messages = make_clean_history(
        messages=messages,
        rewrite_cutoff=rewrite_cutoff,
        rewrite_history=rewrite_history,
        successful_tool_calls=successful_tool_calls,
        successful_tool_responses=successful_tool_responses,
        distil_system_prompt=rewrite_system_prompt_to,
    )

    if rewrite_log_destination is not None:
        functions_coalesced: list[OpenAIFunction] = list({**openai_functions, **distillery_openai_functions}.values())
        log_finetuning_data(rewrite_log_destination, messages, functions_coalesced)

    return result_instances, messages


def initialize_messages(messages: Messages | None, system_prompt, user_message, distil_system_prompt) -> Messages:
    messages = messages if messages is not None else []
    if messages is None and not (system_prompt or user_message):
        raise ValueError(
            "No input - supply at least 'messages', 'system_prompt' and/or 'user_message'"
        )
    if messages and distil_system_prompt:
        # TODO allow this? We can just always replace/insert the first system message in the history during rewriting
        raise ValueError("For distil mode, please use 'system_prompt' and/or 'user_message' instead of 'messages'")
    if messages and (system_prompt or user_message):
        # TODO see above/below - this is not necessarily the best approach
        raise ValueError("Supply (only 'messages') or ('system_prompt' and/or 'user_message'), but not both.")
    if not messages:
        # TODO maybe just always call this? Why not allow combining messages with system_prompt and/or user_message?
        messages = make_initial_messages(messages=messages, system_prompt=system_prompt, user_message=user_message)
    return messages


def make_initial_messages(messages: Messages, system_prompt: str, user_message: str) -> Messages:
    if system_prompt:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
    if user_message:
        messages.append(ChatCompletionUserMessageParam(role="user", content=user_message))
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


# def maybe_convert_tool_call_for_distillation(model_or_fn) -> ChatCompletionToolParam | None:
#     new_tool: ChatCompletionToolParam | None = None
#     if getattr(model_or_fn, "__is_functioncalming_distillery__", False):
#         new_tool: ChatCompletionToolParam = to_tool(model_or_fn.__functioncalming_distil_model__)
#     return new_tool


def get_tool_choice_for_default_behavior(
        openai_functions: dict[str, OpenAIFunction]
) -> ChatCompletionToolChoiceOptionParam:
    tool_choice: ChatCompletionToolChoiceOptionParam
    if len(openai_functions) == 1:
        tool_choice = {"type": "function", "function": {"name": list(openai_functions.keys())[0]}}
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


def make_clean_history(
        *,
        messages: Messages,
        rewrite_cutoff: int,
        rewrite_history: bool,
        successful_tool_calls: list[ChatCompletionMessageToolCall],
        successful_tool_responses: list[ChatCompletionToolMessageParam],
        distil_system_prompt: str | None,
):
    if successful_tool_calls:
        # rewrite the history by removing all messages (not just the failed calls)
        # and replacing them with a single clean multi call and its responses
        if not rewrite_history:
            messages = [message.copy() for message in messages]

        if distil_system_prompt is not None:
            messages[0]["content"] = distil_system_prompt

        messages[rewrite_cutoff:] = [
            ChatCompletionMessage(
                role="assistant",
                tool_calls=successful_tool_calls,
                content=None,
            ),
            *successful_tool_responses
        ]
    return messages


def log_finetuning_data(
        log_finetuning_to: str | TextIO,
        messages: Messages,
        functions: list[OpenAIFunction]
):
    fd = FineTuningData(messages=messages, functions=[t.definition for t in functions])
    log_entry = f"{fd.model_dump_json()}\n"
    if isinstance(log_finetuning_to, str):
        with open(log_finetuning_to, "a") as outf:
            outf.write(log_entry)
    else:
        log_finetuning_to.write(log_entry)
