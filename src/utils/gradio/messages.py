"""Tools for integrating with the Gradio chatbot UI."""

from typing import TYPE_CHECKING

from agents import StreamEvent, stream_events
from agents.items import MessageOutputItem, RunItem, ToolCallItem, ToolCallOutputItem
from gradio.components.chatbot import ChatMessage
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from ..pretty_printing import pretty_print


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


def gradio_messages_to_oai_chat(
    messages: list[ChatMessage | dict],
) -> list["ChatCompletionMessageParam"]:
    """Translate Gradio chat message history to OpenAI format."""
    output: list["ChatCompletionMessageParam"] = []
    for message in messages:
        if isinstance(message, dict):
            output.append(message)  # type: ignore[arg-type]
            continue

        message_content = message.content
        assert isinstance(message_content, str), message_content
        output.append({"role": message.role, "content": message_content})  # type: ignore[arg-type,misc]

    return output


def _oai_response_output_item_to_gradio(item: RunItem) -> list[ChatMessage] | None:
    """Map OAI SDK new RunItem (response.new_items) to gr messages.

    Returns None if message is of unknown/unsupported type.
    """
    print(type(item))
    pretty_print(item)

    if isinstance(item, ToolCallItem):
        raw_item = item.raw_item

        if isinstance(raw_item, ResponseFunctionToolCall):
            return [
                ChatMessage(
                    role="assistant",
                    content=f"```\n{raw_item.arguments}\n```\n`{raw_item.call_id}`",
                    metadata={
                        "title": f"Used tool `{raw_item.name}`",
                    },
                )
            ]

    if isinstance(item, ToolCallOutputItem):
        function_output = item.raw_item["output"]
        call_id = item.raw_item.get("call_id", None)

        if isinstance(function_output, str):
            return [
                ChatMessage(
                    role="assistant",
                    content=f"> {function_output}\n\n`{call_id}`",
                    metadata={
                        "title": "Tool response",
                    },
                )
            ]

    if isinstance(item, MessageOutputItem):
        message_content = item.raw_item

        output_texts: list[str] = []
        for response_text in message_content.content:
            if isinstance(response_text, ResponseOutputText):
                output_texts.append(response_text.text)

        return [ChatMessage(role="assistant", content=_text) for _text in output_texts]

    return None


def oai_agent_items_to_gradio_messages(
    new_items: list[RunItem],
) -> list[ChatMessage]:
    """Parse agent sdk "new items" into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.
    """
    output: list[ChatMessage] = []
    for item in new_items:
        maybe_messages = _oai_response_output_item_to_gradio(item)
        if maybe_messages is not None:
            output.extend(maybe_messages)

    return output


def oai_agent_stream_to_gradio_messages(
    stream_event: StreamEvent,
) -> list[ChatMessage]:
    """Parse agent sdk "stream event" into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.
    """
    output: list[ChatMessage] = []
    if isinstance(stream_event, stream_events.RawResponsesStreamEvent):
        data = stream_event.data
        if isinstance(data, ResponseCompletedEvent):
            for message in data.response.output:
                if isinstance(message, ResponseOutputMessage):
                    for _item in message.content:
                        if isinstance(_item, ResponseOutputText):
                            output.append(
                                ChatMessage(role="assistant", content=_item.text)
                            )

                elif isinstance(message, ResponseFunctionToolCall):
                    output.append(
                        ChatMessage(
                            role="assistant",
                            content=f"```\n{message.arguments}\n```",
                            metadata={
                                "title": f"Used tool `{message.name}`",
                            },
                        )
                    )
    elif isinstance(stream_event, stream_events.RunItemStreamEvent):
        name = stream_event.name
        item = stream_event.item
        if name == "tool_output" and isinstance(item, ToolCallOutputItem):
            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{item.output}\n```",
                    metadata={
                        "title": "*Tool call output*",
                    },
                )
            )

    return output
