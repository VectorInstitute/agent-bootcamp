"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""

import asyncio
import contextlib
import logging
import signal
import sys
import os
import json
import openai
import numpy as np

import agents
from agents import function_tool
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

from src.prompts import REACT_INSTRUCTIONS, KB_SEARCH_INSTRUCTIONS, QA_SEARCH_INSTRUCTIONS, EVALUATOR_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)


configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


@function_tool
async def qa_search_tool(user_query:str) -> list:
    """Search the QA dataset for a question that is semantically similar to the user query."""
    qa_dataset = { 
        1 : {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "context": "Paris is the capital city of France, known for its art, fashion, and culture."
        },
        2: {
            "question": "What software using the Blinnk layout engine is Fluff Busting Purity available for?",
            "answer": "Opera",
            "context": """{"title": ["Vivaldi (web browser)","Internet Explorer shell","GtkHTML","Firefox","Web browser engine","Opera (web browser)","Presto (layout engine)","Fluff Busting Purity","Tasman (layout engine)","KaXUL"], "sentences": [["Vivaldi is a freeware, cross-platform web browser developed by Vivaldi Technologies, a company founded by Opera Software co-founder and former CEO Jon Stephenson von Tetzchner and Tatsuki Tomita."," The browser was officially launched on April 12, 2016."," The browser is aimed at staunch technologists, heavy Internet users, and previous Opera web browser users disgruntled by Opera's transition from the Presto layout engine to the Blink layout engine, which removed many popular features."," Vivaldi aims to revive the old, popular features of Opera 12."," The browser has gained popularity since the launch of its first technical preview."," The browser has 1 million users as of January 2017."],["An Internet Explorer shell is any computer software that uses the Trident rendering engine of the Internet Explorer web browser."," Although the term \"Trident shell\" is probably more accurate for describing these applications (including Internet Explorer itself), the term \"Internet Explorer shell\", or \"IE shell\", is in common parlance."," This means that these software products are not actually full-fledged web browsers in their own right but are simply an alternate interface for Internet Explorer; they share the same limitations of the Trident engine, typically contain the same bugs as IE browsers based on the same version of Trident, and any security vulnerabilities found in IE will generally apply to these browsers as well."," Strictly speaking, programs that use Tasman (layout engine), used in Internet Explorer 5 for Apple Mac, are also IE shells, but, because Internet Explorer for Mac was discontinued in 2003, and Tasman was further developed independent of IE, it tends to be thought of as a separate layout engine."],["GtkHTML is a layout engine written in C using the GTK+ widget toolkit."," It is primarily used by Novell Evolution and other GTK+ applications."," The Balsa email client used GtkHTML as its layout engine for displaying emails until recently."," In the long run, GtkHTML is planned to be phased out in favor of WebKit in GNOME."],["Mozilla Firefox (or simply Firefox) is a free and open-source web browser developed by the Mozilla Foundation and its subsidiary the Mozilla Corporation."," Firefox is available for Windows, macOS and Linux operating systems, with its Firefox for Android available for Android (formerly Firefox for mobile, it also ran on the discontinued Firefox OS), and uses the Gecko layout engine to render web pages, which implements current and anticipated web standards."," An additional version, Firefox for iOS, was released in late 2015, but this version does not use Gecko due to Apple's restrictions limiting third-party web browsers to the WebKit-based layout engine built into iOS."],["A web browser engine (sometimes called web layout engine or web rendering engine) is a computer program that renders marked up content (such as HTML, XML, image files, etc.) and formatting information (such as CSS, XSL, etc.)."," A layout engine is a typical component of web browsers, email clients, e-book readers, on-line help systems, or other applications that require the displaying (and editing) of web pages."],["Opera is a web browser for Windows, macOS, and Linux operating systems developed by Opera Software."," It uses the Blink layout engine."," An earlier version using the Presto layout engine is still available, and runs on FreeBSD systems."," According to Opera, the browser had more than 350 million users worldwide in the 4th quarter of 2014."," Total Opera mobile users reached 291 million in June 2015."," According to SlashGeek, Opera has originated features later adopted by other web browsers, including Speed Dial, pop-up blocking, browser sessions, private browsing, and tabbed browsing."],["Presto was the layout engine of the Opera web browser for a decade."," It was released on 28 January 2003 in Opera 7, and later used to power the Opera Mini and Opera Mobile browsers."," As of Opera 15, the desktop browser uses a Chromium backend, replacing Presto with the Blink layout engine."],["Fluff Busting Purity, or FB Purity for short (previously known as Facebook Purity) is a web browser extension designed to customise the Facebook website's user interface and add extra functionality."," Developed by Steve Fernandez, a UK-based programmer, it was first released in 2009 as a Greasemonkey script, as donationware."," It is available for Firefox , Google Chrome , Microsoft Edge , Safari, Opera and the Maxthon Cloud Browser ."],["Tasman is a discontinued layout engine developed by Microsoft for inclusion in the Macintosh version of Internet Explorer 5."," Tasman was an attempt to improve support for web standards, as defined by the World Wide Web Consortium."," At the time of its release, Tasman was seen as the layout engine with the best support for web standards such as HTML and CSS."," Internet Explorer for Mac is no longer supported, but newer versions of Tasman are incorporated in some other Microsoft products."],["KaXUL (\"KDE Advanced XUL\") is a reimplemetation of Mozilla's own XUL framework for KDE."," Written by George Staikos, it allows for XUL applications - both client- and server-side - to be read by native Qt widgets."," uXUL (\"UI XUL\"), also made by Staikos, takes a XUL application, uses KaXUL to convert it, and then run it as a native KDE plugin."," Used together, one can access XUL applications using Konqueror or any other Web browser using the KHTML layout engine."," Previously, XUL applications were only used by browsers using the Gecko layout engine, which is used, most famously, by Mozilla Firefox for the generation of its extensions."]]}"""
        },
    }

    # _embed_client = openai.OpenAI(
    #     api_key=os.getenv("EMBEDDING_API_KEY"),
    #     base_url=os.getenv("EMBEDDING_BASE_URL"),
    #     max_retries=5)
    
    # #embed user query
    # user_query_embedding = _embed_client.embeddings.create(input=user_query, model=os.getenv('EMBEDDING_MODEL_NAME'))
    # user_query_embedding = np.array(user_query_embedding.data[0].embedding)
    # user_query_embedding = user_query_embedding.reshape(1, -1)

    # cosi_list = []
    # qa_embedding_list = _embed_client.embeddings.create(input=qa_dataset, model=os.getenv('EMBEDDING_MODEL_NAME'))
    # for i, qa_embedding in enumerate(qa_embedding_list.data):
    #         qa_embedding = np.array(qa_embedding.embedding)
    #         qa_embedding = qa_embedding.reshape(1,-1)
    #         similarity_score = cosine_similarity(user_query_embedding, qa_embedding)[0][0]
    #         cosi_list.append({"faq":faq_list[i], "sim":similarity_score})

    # sorted_qa = sorted(cosi_list, key=lambda d: d["sim"], reverse=True)
    # sorted_faqs_list = [i["faq"] for i in sorted_qa]
    # 
    # return "\n".join(f" {i}\n"for i in sorted_faqs_list)

    return json.dumps(qa_dataset)

qa_search_agent = agents.Agent(
    name="QASearchAgent",
    instructions=QA_SEARCH_INSTRUCTIONS,
    tools=[qa_search_tool],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

kb_search_agent = agents.Agent(
    name="KBSearchAgent",
    instructions=KB_SEARCH_INSTRUCTIONS,
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],

    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

evaluator_agent = agents.Agent(
    name="EvaluatorAgent",
    instructions=EVALUATOR_INSTRUCTIONS,
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-flash", openai_client=async_openai_client
    )
)

main_agent = agents.Agent(
    name="MainAgent",
    instructions=REACT_INSTRUCTIONS,

    tools=[
        qa_search_agent.as_tool(
            tool_name="qa_search_Agent",
            tool_description = "Perform a search of the QA database and retrieve question/answer/context tuples related to input query."
       ),
        kb_search_agent.as_tool(
            tool_name="kb_search_agent",
            tool_description="Perform a search of a knowledge base and synthesize the search results to answer input question.",
        ),
        evaluator_agent.as_tool(
            tool_name="evaluator_agent",
            tool_description="Evaluate the output of the knowledge base search agent.",
        )
    ],

    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro", openai_client=async_openai_client
    ),
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Calen-Multi-Agent-V1.0") as span:
        score_is_answer_correct = []
        score_explanation = []

        span.update(input=question)
        result_stream = agents.Runner.run_streamed(main_agent, input=question)

        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)

            if len(gr_messages) > 0:
                yield gr_messages

            try:
                # Assume `event` is your RunItemStreamEvent
                if _item.name == "tool_output" and _item.item.type == "tool_call_output_item":
                        tool_output = json.loads(_item.item.output)

                        explanation = tool_output.get("explanation")
                        is_correct = tool_output.get("is_answer_correct")

                        score_is_answer_correct.append(is_correct)
                        score_explanation.append(explanation)

                        print("✅ is_answer_correct:", is_correct)
                        print("🧠 explanation:", explanation)

            except: 
                continue

        span.update(output=result_stream.final_output)

        if len(score_is_answer_correct) > 0:
            langfuse_client.create_score(
                    name="is_answer_correct",
                    value=score_is_answer_correct[0],
                    comment=score_explanation[0],
                    trace_id=langfuse_client.get_current_trace_id()
                )

demo = gr.ChatInterface(
    _main,
    title="Hitachi Multi-Agent Knowledge Retrieval System",
    type="messages",
    examples=[
        "Where should I go in France? ",
        "Where is the government of France located? ",
        "Check expected answers for  'web browser' topic?"
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(server_name="0.0.0.0")
    finally:
        asyncio.run(_cleanup_clients())
