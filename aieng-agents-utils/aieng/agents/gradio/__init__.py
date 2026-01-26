"""Utilities for managing Gradio interface."""

import gradio as gr


COMMON_GRADIO_CONFIG = {
    "chatbot": gr.Chatbot(height=600),
    "textbox": gr.Textbox(lines=1, placeholder="Enter your prompt"),
    # Additional input to maintain session state across multiple turns
    # NOTE: Examples must be a list of lists when additional inputs are provided
    "additional_inputs": gr.State(value={}, render=False),
}
