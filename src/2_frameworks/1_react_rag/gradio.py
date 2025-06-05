"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

import logging

import agents
import gradio as gr
from elasticsearch import AsyncElasticsearch
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    AsyncESKnowledgeBase,
    Configs,
    oai_agent_items_to_gradio_messages,
    pretty_print,
)


SYSTEM_MESSAGE = """\
Answer the question using the search tool. \
You must explain your reasons for invoking the tool. \
Be sure to mention the sources. \
If the search did not return intended results, try again. \
Do not make up information."""


async def _main(question: str, gr_messages: list[ChatMessage]):
    configs = Configs.from_env_var()
    async_es_client = AsyncElasticsearch(configs.es_host, api_key=configs.es_api_key)
    async_openai_client = AsyncOpenAI()
    async_knowledgebase = AsyncESKnowledgeBase(
        async_es_client,
        es_collection_name="enwiki-20250501",
    )

    main_agent = agents.Agent(
        name="Wikipedia Agent",
        instructions=SYSTEM_MESSAGE,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model="gpt-4o-mini", openai_client=async_openai_client
        ),
    )
    gr_messages.append(ChatMessage(role="user", content=question))
    yield gr_messages

    responses = await agents.Runner.run(main_agent, input=question)
    gr_messages += oai_agent_items_to_gradio_messages(responses.new_items)
    pretty_print(gr_messages)
    yield gr_messages

    await async_es_client.close()


if __name__ == "__main__":
    Configs.from_env_var()
    logging.basicConfig(level=logging.INFO)

    with gr.Blocks(title="OAI Agent SDK ReAct") as app:
        chatbot = gr.Chatbot(type="messages", label="Agent")
        chat_message = gr.Textbox(lines=1, label="Ask a question")
        chat_message.submit(_main, [chat_message, chatbot], [chatbot])

    app.launch(server_name="0.0.0.0")
