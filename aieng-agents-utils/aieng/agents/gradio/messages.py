"""Tools for integrating with the Gradio chatbot UI."""

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING

import gradio as gr
from gradio.components.chatbot import ChatMessage, MetadataDict
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputText
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_message import ResponseOutputMessage
from PIL import Image

from agents import StreamEvent, stream_events
from agents.items import MessageOutputItem, RunItem, ToolCallItem, ToolCallOutputItem


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


__all__ = [
    "gradio_messages_to_oai_chat",
    "oai_agent_items_to_gradio_messages",
    "oai_agent_stream_to_gradio_messages",
]


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


def _process_tool_output_for_images(output_str: str) -> tuple[str, list[Image.Image]]:
    """Extract images from tool output JSON if present.

    Returns tuple of (text_content, list_of_images).
    """
    images = []
    try:
        # Try to parse as JSON to extract images
        output_data = json.loads(output_str)

        # Check if results contain PNG data
        if isinstance(output_data, dict) and "results" in output_data:
            for result in output_data["results"]:
                if isinstance(result, dict) and "png" in result:
                    # Decode base64 PNG
                    png_data = result["png"]
                    img_bytes = base64.b64decode(png_data)
                    img = Image.open(BytesIO(img_bytes))
                    images.append(img)

        return output_str, images
    except (json.JSONDecodeError, Exception):
        # If not JSON or error parsing, return as-is
        return output_str, images


def _oai_response_output_item_to_gradio(
    item: RunItem, is_final_output: bool
) -> list[ChatMessage] | None:
    """Map OAI SDK new RunItem (response.new_items) to gr messages.

    Returns None if message is of unknown/unsupported type.
    """
    if isinstance(item, ToolCallItem):
        raw_item = item.raw_item

        if isinstance(raw_item, ResponseFunctionToolCall):
            return [
                ChatMessage(
                    role="assistant",
                    content=f"```\n{raw_item.arguments}\n```",
                    metadata={
                        "title": f"🛠️ Used tool `{raw_item.name}`",
                    },
                )
            ]

    if isinstance(item, ToolCallOutputItem):
        function_output = item.raw_item["output"]
        call_id = item.raw_item.get("call_id", None)

        if isinstance(function_output, str):
            text_content, images = _process_tool_output_for_images(function_output)
            messages = [
                ChatMessage(
                    role="assistant",
                    content=f"> {text_content}\n\n`{call_id}`",
                    metadata={
                        "title": "*Tool call output*",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            ]

            # Add images as separate messages
            for img in images:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=gr.Image(
                            img,
                            format="png",
                            container=False,
                            interactive=True,
                            buttons=["download"],
                        ),
                    )
                )
            return messages

    if isinstance(item, MessageOutputItem):
        message_content = item.raw_item

        output_texts: list[str] = []
        for response_text in message_content.content:
            if isinstance(response_text, ResponseOutputText):
                output_texts.append(response_text.text)

        return [
            ChatMessage(
                role="assistant",
                content=_text,
                metadata={
                    "title": "Intermediate Step",
                    "status": "done",  # This makes it collapsed by default
                }
                if not is_final_output
                else MetadataDict(),
            )
            for _text in output_texts
        ]

    return None


def oai_agent_items_to_gradio_messages(
    new_items: list[RunItem], is_final_output: bool = True
) -> list[ChatMessage]:
    """Parse agent sdk "new items" into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.
    """
    output: list[ChatMessage] = []
    for item in new_items:
        maybe_messages = _oai_response_output_item_to_gradio(item, is_final_output)
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
            # The completed event may contain multiple output messages,
            # including tool calls and final outputs.
            # If there is at least one tool call, we mark the response as a thought.
            is_thought = len(data.response.output) > 1 and any(
                isinstance(message, ResponseFunctionToolCall)
                for message in data.response.output
            )

            for message in data.response.output:
                if isinstance(message, ResponseOutputMessage):
                    for _item in message.content:
                        if isinstance(_item, ResponseOutputText):
                            output.append(
                                ChatMessage(
                                    role="assistant",
                                    content=_item.text,
                                    metadata={
                                        "title": "🧠 Thought",
                                        "id": data.sequence_number,
                                    }
                                    if is_thought
                                    else MetadataDict(),
                                )
                            )
                elif isinstance(message, ResponseFunctionToolCall):
                    output.append(
                        ChatMessage(
                            role="assistant",
                            content=f"```\n{message.arguments}\n```",
                            metadata={
                                "title": f"🛠️ Used tool `{message.name}`",
                            },
                        )
                    )

    elif isinstance(stream_event, stream_events.RunItemStreamEvent):
        name = stream_event.name
        item = stream_event.item

        if name == "tool_output" and isinstance(item, ToolCallOutputItem):
            text_content, images = _process_tool_output_for_images(item.output)

            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{text_content}\n```",
                    metadata={
                        "title": "*Tool call output*",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            )

            # Add images as separate messages
            for img in images:
                output.append(
                    ChatMessage(
                        role="assistant",
                        content=gr.Image(
                            img,
                            format="png",
                            container=False,
                            interactive=True,
                            buttons=["download"],
                        ),
                    )
                )

    return output
