# 2.2 Multi-agent Planner-Researcher Setup via OpenAI Agents SDK

This folder introduces a multi-agent architecture, featuring a planner agent, a research agent and a writer agent.
The planner agents take a user query and breaks it down into search queries for the knowledge base. The research
each performs a search for each query by calling a search tool. The writer agent then sythesizes the search results into
a summary that is presented to the user.

## Run

```bash
uv run -m src.2_frameworks.2_multi_agent.gradio
```
