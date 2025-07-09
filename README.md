# Agent Bootcamp

----------------------------------------------------------------------------------------

This is a collection of reference implementations for Vector Institute's **Agent Bootcamp**, taking place between June and September 2025. The repository demonstrates modern agentic workflows for retrieval-augmented generation (RAG), evaluation, and orchestration using the latest Python tools and frameworks.

## Reference Implementations

This repository includes several modules, each showcasing a different aspect of agent-based RAG systems:

**1. Basics: Reason-and-Act RAG**
A minimal Reason-and-Act (ReAct) agent for knowledge retrieval, implemented without any agent framework.

- **[1.0 Search Demo](src/1_basics/0_search_demo/README.md)**
  A simple demo showing the capabilities (and limitations) of a knowledgebase search.


- **[1.1 ReAct Agent for RAG](src/1_basics/1_react_rag/README.md)**
  Basic ReAct agent for step-by-step retrieval and answer generation.

**2. Frameworks: OpenAI Agents SDK**
  Showcases the use of the OpenAI agents SDK to reduce boilerplate and improve readability.

- **[2.1 ReAct Agent for RAG - OpenAI SDK](src/2_frameworks/1_react_rag/README.md)**
  Implements the same Reason-and-Act agent using the high-level abstractions provided by the OpenAI Agents SDK. This approach reduces boilerplate and improves readability.
  The use of langfuse for making the agent less of a black-box is also introduced in this module.

- **[2.2 Multi-agent Setup for Deep Research](src/2_frameworks/2_multi_agent/README.md)**
  Demo of a multi-agent architecture with planner, researcher, and writer agents collaborating on complex queries.

**3. Evals: Automated Evaluation Pipelines**
  Contains scripts and utilities for evaluating agent performance using LLM-as-a-judge and synthetic data generation. Includes tools for uploading datasets, running evaluations, and integrating with [Langfuse](https://langfuse.com/) for traceability.

- **[3.1 LLM-as-a-Judge](src/3_evals/1_llm_judge/README.md)**
  Automated evaluation pipelines using LLM-as-a-judge with Langfuse integration.

- **[3.2 Evaluation on Synthetic Dataset](src/3_evals/2_synthetic_data/README.md)**
  Showcases the generation of synthetic evaluation data for testing agents.


## Getting Started

Set your API keys in `.env`. Use `.env.example` as a template.

```bash
cp -v .env.example .env
```

Run integration tests to validate that your API keys are set up correctly.

```bash
PYTHONPATH="." uv run pytest -sv tests/tool_tests/test_integration.py
```

## Reference Implementations

### 1. Basics

Interactive knowledge base demo. Access the gradio interface in your browser (see forwarded ports.)

```bash
uv run --env-file .env -m src.1_basics.0_search_demo.gradio
```

Basic Reason-and-Act Agent- command line version. To exit, press `Control-\`.

```bash
uv run --env-file .env -m src.1_basics.1_react_rag.main
```

Interactive web version of the Gradio Reason-and-Act Agent.

```bash
uv run --env-file .env -m src.1_basics.1_react_rag.gradio
```


### 2. Frameworks

Reason-and-Act Agent without the boilerplate- using the OpenAI Agent SDK.

```bash
uv run --env-file .env -m src.2_frameworks.1_react_rag.basic
uv run --env-file .env -m src.2_frameworks.1_react_rag.gradio
uv run --env-file .env -m src.2_frameworks.1_react_rag.langfuse_gradio
```

Multi-agent examples, also via the OpenAI Agent SDK.

```bash
uv run --env-file .env -m src.2_frameworks.2_multi_agent.gradio
```

### 3. Evals

Synthetic data.

```bash
uv run -m src.3_evals.2_synthetic_data.synthesize_data \
--source_dataset hf://vector-institute/hotpotqa@d997ecf:train \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--limit 18
```

Run LLM-as-a-judge Evaluation on synthetic data

```bash
uv run \
--env-file .env \
-m src.3_evals.1_llm_judge.run_eval \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name enwiki_weaviate \
--limit 18
```

## Requirements

- Python 3.12+

### Tidbit

If you're curious about what "uv" stands for, it appears to have been more or
less chosen [randomly](https://github.com/astral-sh/uv/issues/1349#issuecomment-1986451785).
