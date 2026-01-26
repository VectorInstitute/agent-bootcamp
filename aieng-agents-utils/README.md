# aieng-agents-utils

A utility library for building AI agent applications with support for knowledge bases, code interpreter, web search, and observability. Built for the Vector Institute Agents Bootcamp
by the AI Engineering team.

## Features

### 🤖 Agent Tools

- **Code Interpreter** - Execute Python code in isolated E2B sandboxes with file upload support
- **Gemini Grounding with Google Search** - Web search capabilities with citation tracking
- **Weaviate Knowledge Base** - Vector database integration for RAG applications
- **News Events** - Fetch structured current events from Wikipedia

### 📊 Data Processing

- **PDF to Dataset** - Convert PDF documents to HuggingFace datasets using multimodal OCR
- **Dataset Chunking** - Token-aware text chunking for embedding models
- **Dataset Loading** - Unified interface for loading datasets from multiple sources

### 🔧 Utilities

- **Async Client Manager** - Lifecycle management for async clients (OpenAI, Weaviate)
- **Progress Tracking** - Rich progress bars for async operations with rate limiting
- **Gradio Integration** - Message format conversion between Gradio and OpenAI SDK
- **Langfuse Integration** - OpenTelemetry-based observability and tracing
- **Environment Configuration** - Type-safe environment variable management with Pydantic
- **Session Management** - Persistent conversation sessions with SQLite backend

## Installation

### Using uv (recommended)

```bash
uv pip install aieng-agents-utils
```

### Using pip

```bash
pip install aieng-agents-utils
```

## Quick Start

### Environment Setup

Create a `.env` file with your API keys:

```env
# Required for most features
OPENAI_API_KEY=your_openai_key
# or
GEMINI_API_KEY=your_gemini_key

# For Weaviate knowledge base
WEAVIATE_API_KEY=your_weaviate_key
WEAVIATE_HTTP_HOST=your_instance.weaviate.cloud
WEAVIATE_GRPC_HOST=grpc-your_instance.weaviate.cloud

# For code interpreter (optional)
E2B_API_KEY=your_e2b_key

# For Langfuse observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx

# For embedding models (optional)
EMBEDDING_API_KEY=your_embedding_key
EMBEDDING_BASE_URL=https://your-embedding-service
```

### Basic Usage Examples

#### Using Tools with OpenAI Agents SDK

```python
from aieng.agents.tools import (
    CodeInterpreter,
    AsyncWeaviateKnowledgeBase,
    get_weaviate_async_client,
)
from aieng.agents import AsyncClientManager
import agents

# Initialize client manager
manager = AsyncClientManager()

# Create an agent with tools
agent = agents.Agent(
    name="Research Assistant",
    instructions="Help users with code and research questions.",
    tools=[
        agents.function_tool(manager.knowledgebase.search_knowledgebase),
        agents.function_tool(CodeInterpreter().run_code),
    ],
    model=agents.OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=manager.openai_client,
    ),
)

# Run the agent
response = await agents.Runner.run(
    agent,
    input="Search for information about transformers and create a visualization."
)

# Clean up
await manager.close()
```

#### Using the Code Interpreter

```python
from aieng.agents.tools import CodeInterpreter

interpreter = CodeInterpreter(
    template="<your template ID",
    timeout=300,
)

result = await interpreter.run_code(
    code="""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.savefig("sine_wave.png")
""",
    files=[]
)

print(result.stdout)
print(result.results)  # Contains base64 PNG data
```

#### Fetching News Events

```python
from aieng.agents.tools import get_news_events

news_events = await get_news_events()

# Access events by category
for category, events in news_events.root.items():
    print(f"\n{category}:")
    for event in events:
        print(f"  - {event.description}")
```

#### Using Gemini Grounding with Google Search

```python
from aieng.agents.tools import GeminiGroundingWithGoogleSearch

search_tool = GeminiGroundingWithGoogleSearch(
    base_url="https://your-search-proxy",
    api_key="your_api_key"
)

response = await search_tool.search(
    query="Latest developments in transformer architecture"
)

print(response.text_with_citations)
print(f"Citations: {response.citations}")
```

#### Knowledge Base Search

```python
from aieng.agents import AsyncClientManager

manager = AsyncClientManager()

results = await manager.knowledgebase.search_knowledgebase(
    keyword="machine learning"
)

for result in results:
    print(f"Title: {result.source.title}")
    print(f"Section: {result.source.section}")
    print(f"Snippet: {result.highlight.text[0][:200]}...")
    print()

await manager.close()
```

#### Langfuse Tracing

```python
from aieng.agents import setup_langfuse_tracer, set_up_logging
from dotenv import load_dotenv

load_dotenv()
set_up_logging()

# Setup tracing
tracer = setup_langfuse_tracer(service_name="my_agent_app")

# Your agent code here - traces will automatically be sent to Langfuse
```

#### Async Operations with Progress

```python
from aieng.agents import gather_with_progress, rate_limited
import asyncio

async def fetch_data(url):
    # Your async operation
    await asyncio.sleep(1)
    return f"Data from {url}"

# Run with progress bar
urls = ["url1", "url2", "url3"]
semaphore = asyncio.Semaphore(2)  # Max 2 concurrent

tasks = [
    rate_limited(lambda u=url: fetch_data(u), semaphore=semaphore)
    for url in urls
]

results = await gather_with_progress(
    tasks,
    description="Fetching data..."
)
```

## Command-Line Tools

The package includes console scripts for data processing:

### Convert PDFs to HuggingFace Dataset

```bash
pdf_to_hf_dataset \
    --input-path ./documents \
    --output-dir ./dataset \
    --recursive \
    --model gemini-2.5-flash \
    --chunk-size 512
```

Key options:

- `--input-path`: PDF file or directory
- `--output-dir`: Where to save the dataset
- `--recursive`: Scan directories recursively
- `--model`: OCR model to use
- `--chunk-size`: Max tokens per chunk
- `--structured-ocr`: Use structured JSON output
- `--skip-toc-detection`: Skip table of contents pages

### Chunk Existing Dataset

```bash
chunk_hf_dataset \
    --hf_dataset_path_or_name my-org/my-dataset \
    --chunk_size 512 \
    --chunk_overlap 128 \
    --save_to_hub \
    --hub_repo_id my-org/chunked-dataset
```

## Advanced Usage

### Custom Client Configuration

```python
from aieng.agents import Configs, AsyncClientManager

# Load custom configuration
configs = Configs(
    default_planner_model="gpt-4o",
    default_worker_model="gpt-4o-mini",
    weaviate_collection_name="my_collection",
)

manager = AsyncClientManager(configs=configs)
```

### Gradio Integration

```python
from aieng.agents import (
    gradio_messages_to_oai_chat,
    oai_agent_stream_to_gradio_messages,
)
import gradio as gr
import agents

def chat_fn(message, history):
    # Convert Gradio messages to OpenAI format
    oai_messages = gradio_messages_to_oai_chat(history)

    # Run agent and stream response
    async for gradio_msg in oai_agent_stream_to_gradio_messages(
        agent, message, session
    ):
        yield gradio_msg

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    chatbot.chat(chat_fn)
```

### Session Persistence

```python
from aieng.agents import get_or_create_agent_session

# In your Gradio handler
session = get_or_create_agent_session(history, session_state)

# Use the session with agents
response = await agents.Runner.run(agent, input=message, session=session)
```

## Development

### Running Tests

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run --env-file .env pytest tests/
```

### Project Layout

This package is part of the [Vector Institute AI Engineering Agents Bootcamp](https://github.com/VectorInstitute/agent-bootcamp). It provides reusable utilities for the bootcamp's reference implementations.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) file for details.

## Support

- **Documentation**: See the [Agents Bootcamp docs](../docs/)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/VectorInstitute/agent-bootcamp/issues)
- **Questions**: Open a discussion on GitHub
