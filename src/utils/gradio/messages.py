"""Tools for integrating with the Gradio chatbot UI."""

from typing import TYPE_CHECKING

from gradio.components.chatbot import ChatMessage


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


def gradio_messages_to_oai_chat(
    messages: list[ChatMessage],
) -> list["ChatCompletionMessageParam"]:
    """Translate Gradio chat message history to OpenAI format."""
    output: list["ChatCompletionMessageParam"] = []
    for message in messages:
        message_content = message.content
        if isinstance(message_content, str):
            output.append({"role": message.role, "content": message_content})  # type: ignore[arg-type,misc]

    return output
