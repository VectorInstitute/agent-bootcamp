"""
Tool for finding relevant stock symbols from the Weaviate financial news knowledge base.

This tool queries the Weaviate financial news collection to retrieve unique
stock tickers and uses an LLM to filter them based on user queries.
"""

from typing import List
from pathlib import Path
import os
import json
import asyncio

import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

# Import client manager from the utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.client_manager import AsyncClientManager

# Cache for symbols and company mapping
_cached_symbols: List[str] | None = None
_cached_companies: dict[str, str] | None = None  # ticker -> company name
_client_manager = None

# Weaviate collection name (from WEAVIATE_COLLECTION_NAME env var)
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION_NAME", "Hitachi_finance_news")


def get_client_manager() -> AsyncClientManager:
    """Get or create the client manager."""
    global _client_manager

    if _client_manager is None:
        _client_manager = AsyncClientManager()

    return _client_manager


def _get_weaviate_sync_client():
    """Create a synchronous Weaviate client from environment variables."""
    http_host = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
    api_key = os.getenv("WEAVIATE_API_KEY", "")

    # Weaviate Cloud uses connect_to_weaviate_cloud (single host, port 443)
    if http_host.endswith(".weaviate.cloud"):
        cluster_url = f"https://{http_host}"
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=AuthApiKey(api_key),
        )

    # Otherwise use custom connection for self-hosted instances
    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
        http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
        grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "localhost"),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
        auth_credentials=AuthApiKey(api_key),
    )


def get_all_symbols() -> List[str]:
    """
    Get all unique stock tickers from the Weaviate knowledge base.

    Iterates through the Weaviate collection and collects
    unique ticker symbols and their corresponding company names.

    Returns:
        Sorted list of unique stock tickers
    """
    global _cached_symbols, _cached_companies

    if _cached_symbols is not None:
        return _cached_symbols

    client = _get_weaviate_sync_client()
    try:
        col = client.collections.get(WEAVIATE_COLLECTION)

        tickers = set()
        companies: dict[str, str] = {}

        for obj in col.iterator(
            include_vector=False,
            return_properties=["ticker", "company"],
        ):
            ticker = obj.properties.get("ticker")
            company = obj.properties.get("company")
            if ticker:
                tickers.add(ticker)
                if company and ticker not in companies:
                    companies[ticker] = company

        _cached_symbols = sorted(tickers)
        _cached_companies = companies
        return _cached_symbols

    except Exception as e:
        raise RuntimeError(f"Failed to load tickers from Weaviate: {e}")
    finally:
        client.close()


def get_company_mapping() -> dict[str, str]:
    """
    Get ticker -> company name mapping from the Weaviate knowledge base.

    Returns:
        Dictionary mapping ticker symbols to company names
    """
    if _cached_companies is None:
        get_all_symbols()  # populates both caches
    return _cached_companies or {}


async def filter_symbols_with_llm_async(query: str, symbols: List[str]) -> List[str]:
    """
    Use an LLM to filter symbols based on the query (async version).

    Args:
        query: Natural-language query describing what to filter for
        symbols: List of all available symbols

    Returns:
        Filtered list of relevant symbols
    """
    client_manager = get_client_manager()

    # Build a readable list with company names
    company_map = get_company_mapping()
    symbol_list = ", ".join(
        f"{s} ({company_map[s]})" if s in company_map else s for s in symbols
    )

    prompt = f"""Given this list of stock symbols from our financial knowledge base, identify and return ALL symbols that match this query: "{query}"

Available symbols:
{symbol_list}

Instructions:
- Return ALL matching stock symbols as a JSON array
- Focus on the query requirements (sector, industry, characteristics)
- IGNORE any numeric limits like "top N" - return ALL relevant matches
- Use your knowledge of which companies operate in which sectors
- Return ONLY a JSON object with a "symbols" array, nothing else

Example: For "tech stocks", return ALL technology company symbols, not just the top few.
Response format: {{"symbols": ["AAPL", "GOOGL", "MSFT", "NVDA", ...]}}"""

    try:
        response = await client_manager.openai_client.chat.completions.create(
            model=client_manager.configs.default_worker_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        filtered_symbols = result.get("symbols", [])

        # Validate that returned symbols are in the original list
        valid_symbols = [s for s in filtered_symbols if s in symbols]

        return valid_symbols

    except Exception as e:
        # Fallback: return all symbols if filtering fails
        print(f"Warning: LLM filtering failed ({e}), returning all symbols")
        return symbols


def filter_symbols_with_llm(query: str, symbols: List[str]) -> List[str]:
    """
    Use an LLM to filter symbols based on the query (sync wrapper).

    Args:
        query: Natural-language query describing what to filter for
        symbols: List of all available symbols

    Returns:
        Filtered list of relevant symbols
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're already inside an event loop (e.g. Jupyter / Gradio)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run, filter_symbols_with_llm_async(query, symbols)
            ).result()
    else:
        return asyncio.run(filter_symbols_with_llm_async(query, symbols))


def find_relevant_symbols(query: str, use_llm_filter: bool = True) -> List[str]:
    """
    Find stock symbols relevant to the query from the Weaviate knowledge base.
    Uses an LLM internally to filter symbols based on the query.

    Args:
        query: Natural-language query describing the type of companies or time period,
               e.g. 'List all the top automotive stocks of 2012'
        use_llm_filter: Whether to use LLM for filtering (default: True)

    Returns:
        Sorted list of filtered stock symbols relevant to the query.
    """
    all_symbols = get_all_symbols()

    if not use_llm_filter:
        return all_symbols

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: No LLM API key set, returning all symbols without filtering")
        return all_symbols

    # Use LLM to filter symbols based on query
    filtered = filter_symbols_with_llm(query, all_symbols)

    return sorted(filtered)


# Keep backward-compatible alias
find_relevant_sp500_symbols = find_relevant_symbols


# OpenAI tool schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_relevant_symbols",
        "description": (
            "Find relevant stock symbols from the Weaviate financial news knowledge base "
            "The tool uses an LLM internally to filter "
            "symbols based on the query, returning only symbols that match the specified "
            "criteria (sector, industry, time period, ranking, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query describing the type of companies or "
                        "time period, e.g. 'List all the top automotive stocks of 2012' "
                        "or 'top 3 tech stocks of 2010'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


# Tool implementation mapping for OpenAI agent integration
TOOL_IMPLEMENTATIONS = {
    "find_relevant_symbols": find_relevant_symbols,
    "find_relevant_sp500_symbols": find_relevant_symbols,  # backward compat
}


def run_agent_with_tool(user_query: str, client) -> str:
    """
    Run an OpenAI agent that can use the symbol-filtering tool.

    Args:
        user_query: User's natural-language query
        client: OpenAI client instance

    Returns:
        Final response from the agent
    """
    import json

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_query}],
        tools=[TOOL_SCHEMA],
    )

    choice = response.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)

    if tool_calls:
        # Process the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments  # JSON string

        args = json.loads(tool_args)
        result = TOOL_IMPLEMENTATIONS[tool_name](**args)

        # Send tool result back to the model for final answer
        followup = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_query},
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(result),
                },
            ],
        )
        return followup.choices[0].message.content

    # If no tool call, just return the model's direct answer
    return choice.message.content
