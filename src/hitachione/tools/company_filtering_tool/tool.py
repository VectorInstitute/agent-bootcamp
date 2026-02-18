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
import re
import difflib
import csv
import io
import logging
from urllib import request

import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

# Import client manager from the utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.client_manager import AsyncClientManager


logger = logging.getLogger(__name__)

# Cache for symbols and company mapping
_cached_symbols: List[str] | None = None
_cached_companies: dict[str, str] | None = None  # ticker -> company name
_client_manager = None

# Weaviate collection name (from WEAVIATE_COLLECTION_NAME env var)
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION_NAME", "Hitachi_finance_news")
SP500_CONSTITUENTS_URL = os.getenv(
    "SP500_CONSTITUENTS_URL",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
)
SP500_UNIVERSE_ENABLED = os.getenv("SP500_UNIVERSE_ENABLED", "true").lower() == "true"

_COMMON_TYPO_FIXES = {
    "enery": "energy",
    "techonology": "technology",
    "teh": "the",
}

_SECTOR_TERMS = [
    "energy",
    "technology",
    "tech",
    "automotive",
    "healthcare",
    "financial",
    "finance",
    "retail",
]

_SECTOR_TICKER_MAP = {
    "energy": {"TSLA"},
    "technology": {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"},
    "tech": {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"},
    "automotive": {"TSLA"},
    "financial": {"JPM", "V"},
    "finance": {"JPM", "V"},
    "retail": {"AMZN", "WMT"},
}


def _normalize_query(query: str) -> str:
    """Normalize obvious typos in user queries (e.g., 'enery' -> 'energy')."""
    normalized = query

    for wrong, right in _COMMON_TYPO_FIXES.items():
        normalized = re.sub(rf"\b{re.escape(wrong)}\b", right, normalized, flags=re.IGNORECASE)

    words = normalized.split()
    fixed_words: list[str] = []
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if len(clean) >= 5 and clean not in _SECTOR_TERMS:
            matches = difflib.get_close_matches(clean, _SECTOR_TERMS, n=1, cutoff=0.86)
            if matches:
                fixed_words.append(word.lower().replace(clean, matches[0]))
                continue
        fixed_words.append(word)

    return " ".join(fixed_words)


def _deterministic_sector_filter(query: str, symbols: List[str]) -> List[str]:
    """Apply deterministic sector-based filtering when sector terms are present."""
    q = query.lower()
    symbol_set = set(symbols)
    matched: set[str] = set()

    for sector, sector_symbols in _SECTOR_TICKER_MAP.items():
        if re.search(rf"\b{re.escape(sector)}\b", q):
            matched |= (sector_symbols & symbol_set)

    return sorted(matched)


def _has_explicit_sector_term(query: str) -> bool:
    """Whether the query explicitly mentions a known sector term."""
    q = query.lower()
    return any(re.search(rf"\b{re.escape(sector)}\b", q) for sector in _SECTOR_TICKER_MAP)


def _load_sp500_constituents() -> tuple[list[str], dict[str, str]]:
    """Load S&P 500 constituents (ticker + company) from a CSV source."""
    try:
        with request.urlopen(SP500_CONSTITUENTS_URL, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        reader = csv.DictReader(io.StringIO(raw))
        symbols: set[str] = set()
        companies: dict[str, str] = {}

        for row in reader:
            symbol = (row.get("Symbol") or "").strip().upper()
            name = (row.get("Name") or "").strip()
            if not symbol:
                continue
            symbols.add(symbol)
            if name:
                companies[symbol] = name

        return sorted(symbols), companies
    except Exception as exc:
        logger.warning("Failed to load S&P500 constituents from %s: %s", SP500_CONSTITUENTS_URL, exc)
        return [], {}


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
                ticker = str(ticker).upper()
                tickers.add(ticker)
                if company and ticker not in companies:
                    companies[ticker] = str(company)

        # Merge in full S&P500 constituents universe so retrieval is not limited
        # to currently indexed Weaviate objects.
        if SP500_UNIVERSE_ENABLED:
            sp500_symbols, sp500_companies = _load_sp500_constituents()
            tickers.update(sp500_symbols)
            for symbol, name in sp500_companies.items():
                companies.setdefault(symbol, name)

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
    normalized_query = _normalize_query(query)

    deterministic = _deterministic_sector_filter(normalized_query, all_symbols)

    if not use_llm_filter:
        # For non-LLM mode, prefer deterministic matches when available.
        return deterministic or all_symbols

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: No LLM API key set, returning all symbols without filtering")
        return deterministic or all_symbols

    # Use LLM to filter symbols based on query
    filtered = filter_symbols_with_llm(normalized_query, all_symbols)

    explicit_sector = _has_explicit_sector_term(normalized_query)

    # If the user explicitly asked for a known sector, constrain to that sector
    # to avoid model drift (e.g. TSLA leaking into "tech").
    # Only hard-constrain to deterministic matches when that rule has enough
    # coverage. A single-ticker deterministic hit (e.g., energy -> TSLA) is too
    # narrow for broad sector queries and should not suppress LLM discoveries.
    if explicit_sector and deterministic and len(deterministic) >= 2:
        constrained = sorted(set(filtered) & set(deterministic))
        if constrained:
            return constrained
        return deterministic

    # Otherwise combine LLM output with deterministic hints so obvious matches
    # are not dropped by the model.
    merged = sorted(set(filtered) | set(deterministic))

    # Prefer merged candidate set; fall back gracefully if model returned nothing.
    if merged:
        return merged
    if deterministic:
        return deterministic

    return all_symbols


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
