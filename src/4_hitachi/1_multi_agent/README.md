# 4.1 Multi-agent Orchestrator-QA Search-Knowledge Base Search via OpenAI Agents SDK

This folder introduces a multi-agent architecture, featuring an orchestrator agent and two search agents, one with access to QA dataset and the other with access to Knowledge Base dataset.

The orchestrator agents take a user query and breaks it down into search queries for the QA dataset. It then takes the returned QA pairs and breaks down searches for the Knowledge Base. The Knowledge Base search agent calls the search tool and synthesizes the results into an answer for each question. The orchestrator agent then receives the resulting answers and evaluates them based on the ground truth answers retrieved from the QA search.

## Run

```bash
uv run --env-file .env \
-m src.4_hitachi.1_multi_agent.orchestrator_gradio
```
