# Agent Bootcamp

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/agent-bootcamp/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/agent-bootcamp/actions/workflows/code_checks.yml)
[![docs](https://github.com/VectorInstitute/agent-bootcamp/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/agent-bootcamp/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/agent-bootcamp/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/agent-bootcamp)
[![license](https://img.shields.io/github/license/VectorInstitute/agent-bootcamp.svg)](https://github.com/VectorInstitute/agent-bootcamp/blob/main/LICENSE)

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



## Requirements

- Python 3.12+

## Getting Started

Clone the repository:

```bash
git clone https://github.com/VectorInstitute/agent-bootcamp
cd agent-bootcamp
```

### Setup Instructions

1. **Install [uv](https://github.com/astral-sh/uv):**

    ```bash
    pip install uv
    ```

2. **Create and activate a virtual environment:**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    uv sync --dev
    ```

4. **Configure environment variables:**

    ```bash
    cp .env.example .env
    # Edit .env and add all required environment variables
    ```

You are now ready to explore the Agent Bootcamp reference implementations!

---
For more details on each module, see the respective README files linked above.
