"""Flight Search Agent Demo using Gradio."""

import gradio as gr
from dotenv import load_dotenv
import os
import asyncio
import json

from flight_api import RapidApiFlightSearchClient,RapidApiAirportSearchClient 
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI

DESCRIPTION = """\
Enter a flight-related query below. The agent will use RapidAPI's Google Flights API to find relevant flight information. \
Try queries like: "Find flights from Montreal to Vancouver" or "Show me flight prices from Toronto to New York."
"""

load_dotenv()

# Initialize RapidAPI flight search client and agent
async def agent_flight_search(query: str):
    flight_search_client = RapidApiFlightSearchClient(
        api_key=os.environ.get("RAPIDAPI_KEY"),
        host="google-flights2.p.rapidapi.com"
    )

    airport_search_client = RapidApiAirportSearchClient(
        api_key=os.environ.get("RAPIDAPI_KEY"),
        host="google-flights4.p.rapidapi.com"
    )

    # Wrap the airport search method as a tool
    flight_search_tool = function_tool(flight_search_client.search_flight)

    # Wrap the airport search method as a tool
    airport_search_tool = function_tool(airport_search_client.search_airport)

    async_openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    model = OpenAIChatCompletionsModel(
        model=os.environ.get("AGENT_LLM_NAME"),
        openai_client=async_openai_client,
    )

    flight_agent = Agent(
        name="Flight Search Agent",
        instructions="""
            You are a travel assistant that helps users find flights information using RapidAPI's Google Flights API.
            If the user provides city names instead of airport codes, first use airport_search_tool to map the origin and 
            destination's cities to airports' IATA codes and initialize departure_id and arrival_id with extracted code
            and then use the flight search tool to get flight information based on extracetd departure_id and arrival_id.
            You do not need to ask for currency or language codes unless the user specifies them—default values are used.
            Extract the travel date from the user query and format it as YYYY-MM-DD and initialize outbound_date with it.
            if date is not specified, use the current date to initialize outbound_date and call the api with current date withought asking the user.
            before calling the flight search tool.
            Example : 
            User: Find flights from Toronto to Montreal
            Agent: Resolves Toronto → YYZ, Montreal → YUL
            Agent: Calls flight API with origin=YYZ, destination=YUL """,
        tools=[flight_search_tool,airport_search_tool],
        model=model,
    )

    response = await Runner.run(
        flight_agent,
        input=query,
        run_config=RunConfig(
            model=model,
            tracing_disabled=True
        )
    )

    results = []
    for item in response.new_items:
        if isinstance(item.raw_item, (dict, list)):
            results.append(item.raw_item)
        else:
            results.append(str(item.raw_item))
    results.append({"final_output": str(response.final_output)})

    await async_openai_client.close()
    return json.dumps(results, indent=2)

def sync_agent_flight_search(query: str) -> str:
    """Sync wrapper for Gradio."""
    return asyncio.run(agent_flight_search(query))

json_codeblock = gr.Code(language="json", wrap_lines=True)

demo = gr.Interface(
    fn=sync_agent_flight_search,
    inputs=["text"],
    outputs=[json_codeblock],
    title="Flight Search Agent Demo",
    description=DESCRIPTION,
    examples=[
        "Find flights from Montreal to Vancouver",
        "Show me flight prices from Toronto to New York",
        "Search for flights from Paris to Rome",
        "Get flight info from Los Angeles to Tokyo",
    ],
)

demo.launch(server_name="0.0.0.0", share=True)
