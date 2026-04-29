"""Utilities for managing Gradio interface."""

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    from aieng.agents._optional_extras import EXTRA_GRADIO, raise_missing_optional

    raise_missing_optional(
        EXTRA_GRADIO, missing=getattr(exc, "name", None), from_exc=exc
    )


def get_common_gradio_config() -> dict[str, gr.Component]:
    """Get common Gradio components for agent demos.

    This includes a chatbot for displaying messages, a textbox for user input,
    and a hidden state component for maintaining session state across turns.

    Returns
    -------
    dict[str, gr.Component]
        A dictionary containing the common Gradio components.
    """
    return {
        "chatbot": gr.Chatbot(height=600),
        "textbox": gr.Textbox(lines=1, placeholder="Enter your prompt"),
        # Additional input to maintain session state across multiple turns
        # NOTE: Examples must be a list of lists when additional inputs are provided
        "additional_inputs": gr.State(value={}, render=False),
    }
