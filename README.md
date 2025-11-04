# Agent Bootcamp

----------------------------------------------------------------------------------------

This is a collection of reference implementations for Vector Institute's **Agent Bootcamp**, taking place between June and September 2025. The repository demonstrates modern agentic workflows for retrieval-augmented generation (RAG), evaluation, and orchestration using the latest Python tools and frameworks.

## Reference Implementations

This repository includes several modules, each showcasing a different aspect of agent-based RAG systems:

**2. Frameworks: OpenAI Agents SDK**
  Showcases the use of the OpenAI agents SDK to reduce boilerplate and improve readability.

- **[2.1 ReAct Agent for RAG - OpenAI SDK](src/2_frameworks/1_react_rag/README.md)**
  Implements the same Reason-and-Act agent using the high-level abstractions provided by the OpenAI Agents SDK. This approach reduces boilerplate and improves readability.
  The use of langfuse for making the agent less of a black-box is also introduced in this module.

- **[2.2 Multi-agent Setup for Deep Research](src/2_frameworks/2_multi_agent/README.md)**
  Demo of a multi-agent architecture to improve efficiency on long-context inputs, reduce latency, and reduce LLM costs. Two versions are available- "efficient" and "verbose". For the build days, you should start from the "efficient" version as that provides greater flexibility and is easier to follow.

**3. Evals: Automated Evaluation Pipelines**
  Contains scripts and utilities for evaluating agent performance using LLM-as-a-judge and synthetic data generation. Includes tools for uploading datasets, running evaluations, and integrating with [Langfuse](https://langfuse.com/) for traceability.

- **[3.1 LLM-as-a-Judge](src/3_evals/1_llm_judge/README.md)**
  Automated evaluation pipelines using LLM-as-a-judge with Langfuse integration.

- **[3.2 Evaluation on Synthetic Dataset](src/3_evals/2_synthetic_data/README.md)**
  Showcases the generation of synthetic evaluation data for testing agents.

We also provide "basic" no-framework implementations. These are meant to showcase how agents work behind the scene and are excessively verbose in the implementation. You should not use these as the basis for real projects.

**1. Basics: Reason-and-Act RAG**
A minimal Reason-and-Act (ReAct) agent for knowledge retrieval, implemented without any agent framework.

- **[1.0 Search Demo](src/1_basics/0_search_demo/README.md)**
  A simple demo showing the capabilities (and limitations) of a knowledgebase search.

- **[1.1 ReAct Agent for RAG](src/1_basics/1_react_rag/README.md)**
  Basic ReAct agent for step-by-step retrieval and answer generation.


## Getting Started

Set your API keys in `.env`. Use `.env.example` as a template.

```bash
cp -v .env.example .env
```

Run integration tests to validate that your API keys are set up correctly.

```bash
PYTHONPATH="." uv run --env-file .env pytest -sv tests/tool_tests/test_integration.py
```

## Reference Implementations

For "Gradio App" reference implementations, running the script would print out a "public URL" ending in `gradio.live` (might take a few seconds to appear.) To access the gradio app with the full streaming capabilities, copy and paste this `gradio.live` URL into a new browser tab.

For all reference implementations, to exit, press "Ctrl/Control-C" and wait up to ten seconds. If you are a Mac user, you should use "Control-C" and not "Command-C". Please note that by default, the gradio web app reloads automatically as you edit the Python script. There is no need to manually stop and restart the program each time you make some code changes.

You might see warning messages like the following:

```json
ERROR:openai.agents:[non-fatal] Tracing client error 401: {
  "error": {
    "message": "Incorrect API key provided. You can find your API key at https://platform.openai.com/account/api-keys.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

These warnings can be safely ignored, as they are the result of a bug in the upstream libraries. Your agent traces will be uploaded to LangFuse as configured.

### 1. Basics

Interactive knowledge base demo. Access the gradio interface in your browser to see if your knowledge base meets your expectations.

```bash
PYTHONPATH="." uv run --env-file .env gradio src/1_basics/0_search_demo/app.py
```

Basic Reason-and-Act Agent- for demo purposes only.

As noted above, these are unnecessarily verbose for real applications.

```bash
# PYTHONPATH="." uv run --env-file .env src/1_basics/1_react_rag/cli.py
# PYTHONPATH="." uv run --env-file .env gradio src/1_basics/1_react_rag/app.py
```


### 2. Frameworks

Reason-and-Act Agent without the boilerplate- using the OpenAI Agent SDK.

```bash
PYTHONPATH="." uv run --env-file .env src/2_frameworks/1_react_rag/cli.py
PYTHONPATH="." uv run --env-file .env gradio src/2_frameworks/1_react_rag/langfuse_gradio.py
```

Multi-agent examples, also via the OpenAI Agent SDK.

```bash
PYTHONPATH="." uv run --env-file .env gradio src/2_frameworks/2_multi_agent/efficient.py
# Verbose option- greater control over the agent flow, but less flexible.
# PYTHONPATH="." uv run --env-file .env gradio src/2_frameworks/2_multi_agent/verbose.py
```

Python Code Interpreter demo- using the OpenAI Agent SDK, E2B for secure code sandbox, and LangFuse for observability. Refer to [src/2_frameworks/3_code_interpreter/README.md](src/2_frameworks/3_code_interpreter/README.md) for details.


### 3. Evals

Synthetic data.

```bash
uv run --env-file .env \
-m src.3_evals.2_synthetic_data.synthesize_data \
--source_dataset hf://vector-institute/hotpotqa@d997ecf:train \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--limit 18
```

Quantify embedding diversity of synthetic data

```bash
# Baseline: "Real" dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset \
--run_name cosine_similarity_bge_m3

# Synthetic dataset
uv run \
--env-file .env \
-m src.3_evals.2_synthetic_data.annotate_diversity \
--langfuse_dataset_name search-dataset-synthetic-20250609 \
--run_name cosine_similarity_bge_m3
```

Visualize embedding diversity of synthetic data

```bash
uv run \
--env-file .env \
gradio src/3_evals/2_synthetic_data/gradio_visualize_diversity.py
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
- API keys as configured in `.env`.

### Tidbit

If you're curious about what "uv" stands for, it appears to have been more or
less chosen [randomly](https://github.com/astral-sh/uv/issues/1349#issuecomment-1986451785).
