"""
Test harness for the Sentiment Analysis Tool (Weaviate-backed).

Run:
    cd src/hitachione/tools/sentiment_analysis_tool
    python3 -W ignore::ResourceWarning test.py all          # full suite
    python3 -W ignore::ResourceWarning test.py data         # Weaviate queries only (no LLM)
    python3 -W ignore::ResourceWarning test.py sentiment    # LLM sentiment analysis tests
    python3 -W ignore::ResourceWarning test.py schema       # show tool schemas
    python3 -W ignore::ResourceWarning test.py interactive  # interactive prompt
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from tool import (
    TOOL_SCHEMAS,
    TOOL_IMPLEMENTATIONS,
    TOOL_SCHEMA_TICKER,
    TOOL_SCHEMA_YEAR,
    TOOL_SCHEMA_NEWS,
    TOOL_SCHEMA_TEXT,
    query_weaviate_by_ticker,
    query_weaviate_by_year,
    query_weaviate_by_topic,
    analyze_sentiment_sync,
    analyze_ticker_sentiment_sync,
    analyze_year_sentiment_sync,
    analyze_news_sentiment_sync,
    EXAMPLE_TEXT,
)

# ── Known data in the Weaviate collection ──
KNOWN_TICKERS = ["AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "V", "WMT"]
UNKNOWN_TICKER = "ZZZZZ"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def _pass(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _has_llm_key() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


# ──────────────────────────────────────────────────────────────────────────
# Schema tests
# ──────────────────────────────────────────────────────────────────────────

def test_tool_schemas() -> None:
    """Validate all tool schema structures."""
    _section("Tool Schemas")

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "required" in fn["parameters"]
        _pass(f"Schema '{fn['name']}' is valid")

    # Verify implementation mapping
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert name in TOOL_IMPLEMENTATIONS, f"No implementation for '{name}'"
        assert callable(TOOL_IMPLEMENTATIONS[name])
    _pass(f"All {len(TOOL_SCHEMAS)} schemas have implementations")

    print("\nSchemas:")
    for schema in TOOL_SCHEMAS:
        print(json.dumps(schema, indent=2))


# ──────────────────────────────────────────────────────────────────────────
# Weaviate data-retrieval tests (no LLM required)
# ──────────────────────────────────────────────────────────────────────────

def test_query_by_ticker_known() -> None:
    """Retrieve data for known tickers and verify results."""
    _section("Query by Ticker — Known Tickers")

    for ticker in ["AAPL", "TSLA", "JPM"]:
        t0 = time.time()
        records = query_weaviate_by_ticker(ticker, limit=20)
        elapsed = time.time() - t0

        assert isinstance(records, list), f"Expected list, got {type(records)}"
        assert len(records) > 0, f"No records returned for {ticker}"

        # Each record should have text
        for rec in records:
            assert "text" in rec or "title" in rec, f"Record missing text/title for {ticker}"

        _pass(f"{ticker}: {len(records)} records ({elapsed:.2f}s)")


def test_query_by_ticker_unknown() -> None:
    """Verify unknown ticker returns empty list."""
    _section("Query by Ticker — Unknown Ticker")

    records = query_weaviate_by_ticker(UNKNOWN_TICKER)
    assert isinstance(records, list)
    assert len(records) == 0, f"Expected 0 records for {UNKNOWN_TICKER}, got {len(records)}"
    _pass(f"{UNKNOWN_TICKER}: 0 records as expected")


def test_query_by_ticker_case_insensitive() -> None:
    """Verify ticker lookup is case-insensitive."""
    _section("Query by Ticker — Case Insensitivity")

    upper = query_weaviate_by_ticker("AAPL", limit=10)
    lower = query_weaviate_by_ticker("aapl", limit=10)

    assert len(upper) == len(lower), (
        f"Count differs: 'AAPL' returned {len(upper)}, 'aapl' returned {len(lower)}"
    )
    _pass("'AAPL' and 'aapl' return same number of results")


def test_query_by_year() -> None:
    """Retrieve data for a year range."""
    _section("Query by Year")

    # Try several years — the dataset may span different years
    for year in [2020, 2023, 2024]:
        records = query_weaviate_by_year(year, limit=10)
        assert isinstance(records, list)
        if records:
            for rec in records:
                date_str = str(rec.get("date", ""))
                assert str(year) in date_str[:4], (
                    f"Record date '{date_str}' doesn't match year {year}"
                )
            _pass(f"Year {year}: {len(records)} records")
        else:
            print(f"  - Year {year}: no records (may not be in dataset)")


def test_query_by_topic() -> None:
    """Retrieve articles matching a topic via BM25 search."""
    _section("Query by Topic (BM25)")

    for topic in ["earnings", "inflation", "technology"]:
        t0 = time.time()
        records = query_weaviate_by_topic(topic, limit=5)
        elapsed = time.time() - t0

        assert isinstance(records, list)
        # BM25 should find something for broad financial topics
        _pass(f"'{topic}': {len(records)} records ({elapsed:.2f}s)")

        if records:
            # Spot-check structure
            for rec in records:
                assert isinstance(rec, dict)


def test_query_record_properties() -> None:
    """Verify returned records contain expected properties."""
    _section("Record Properties")

    records = query_weaviate_by_ticker("MSFT", limit=5)
    assert len(records) > 0, "No records for MSFT"

    expected = {"text", "title", "date", "dataset_source"}
    for i, rec in enumerate(records[:3]):
        present = set(rec.keys()) & expected
        missing = expected - present
        if not missing:
            _pass(f"Record {i}: has all expected properties")
        else:
            # Some properties may be null for certain dataset sources
            print(f"  - Record {i}: missing {missing} (may be null)")


# ──────────────────────────────────────────────────────────────────────────
# LLM sentiment analysis tests
# ──────────────────────────────────────────────────────────────────────────

def test_analyze_sentiment_free_text() -> None:
    """Analyze sentiment of free-form text."""
    _section("Free Text Sentiment (LLM)")

    if not _has_llm_key():
        print("  ⚠️  No LLM API key — skipping")
        return

    t0 = time.time()
    result = analyze_sentiment_sync(EXAMPLE_TEXT)
    elapsed = time.time() - t0

    assert isinstance(result, dict)
    assert "label" in result
    assert "rating" in result
    assert "rationale" in result
    assert result["label"] in ("Negative", "Neutral", "Positive", "unknown")
    if result["rating"] is not None:
        assert 1 <= result["rating"] <= 10
        # Verify label matches rating
        if result["rating"] <= 4:
            assert result["label"] == "Negative"
        elif result["rating"] == 5:
            assert result["label"] == "Neutral"
        else:
            assert result["label"] == "Positive"

    _pass(
        f"rating={result['rating']}, label={result['label']}, "
        f"{elapsed:.1f}s"
    )
    print(f"    Rationale: {result['rationale'][:120]}")


def test_analyze_ticker_sentiment_known() -> None:
    """Analyze ticker sentiment for known tickers."""
    _section("Ticker Sentiment — Known (LLM)")

    if not _has_llm_key():
        print("  ⚠️  No LLM API key — skipping")
        return

    for ticker in ["AAPL", "NVDA"]:
        t0 = time.time()
        result = analyze_ticker_sentiment_sync(ticker)
        elapsed = time.time() - t0

        assert isinstance(result, dict)
        assert result["ticker"] == ticker
        assert result["rating"] is not None, f"Rating is None for {ticker}"
        assert isinstance(result["rating"], int)
        assert 1 <= result["rating"] <= 10
        # Verify label matches rating scale
        if result["rating"] <= 4:
            assert result["label"] == "Negative", f"Expected Negative for rating {result['rating']}"
        elif result["rating"] == 5:
            assert result["label"] == "Neutral", f"Expected Neutral for rating {result['rating']}"
        else:
            assert result["label"] == "Positive", f"Expected Positive for rating {result['rating']}"
        assert len(result.get("rationale", "")) > 10
        assert isinstance(result.get("references", []), list)

        _pass(
            f"{ticker}: rating={result['rating']}, label={result['label']}, "
            f"{elapsed:.1f}s"
        )
        print(f"    Rationale: {result['rationale'][:120]}")


def test_analyze_ticker_sentiment_unknown() -> None:
    """Analyze ticker sentiment for unknown ticker returns graceful result."""
    _section("Ticker Sentiment — Unknown")

    result = analyze_ticker_sentiment_sync(UNKNOWN_TICKER)
    assert isinstance(result, dict)
    assert result["ticker"] == UNKNOWN_TICKER
    assert result["rating"] is None
    assert result["label"] == "unknown"
    assert "no data" in result["rationale"].lower() or "not found" in result["rationale"].lower() or len(result["rationale"]) > 0
    _pass(f"{UNKNOWN_TICKER}: rating=None, label=unknown (no LLM call needed)")


def test_analyze_year_sentiment() -> None:
    """Analyze year sentiment (requires data for that year in Weaviate)."""
    _section("Year Sentiment (LLM)")

    if not _has_llm_key():
        print("  ⚠️  No LLM API key — skipping")
        return

    # First find a year that has data
    for year in [2024, 2023, 2020]:
        records = query_weaviate_by_year(year, limit=5)
        if records:
            break
    else:
        print("  ⚠️  No year data found in KB — skipping")
        return

    t0 = time.time()
    result = analyze_year_sentiment_sync(year)
    elapsed = time.time() - t0

    assert isinstance(result, dict)
    assert result["year"] == year
    assert "label" in result
    assert "rating" in result
    assert "rationale" in result
    if result["rating"] is not None:
        assert 1 <= result["rating"] <= 10
        if result["rating"] <= 4:
            assert result["label"] == "Negative"
        elif result["rating"] == 5:
            assert result["label"] == "Neutral"
        else:
            assert result["label"] == "Positive"

    _pass(
        f"Year {year}: rating={result['rating']}, label={result['label']}, "
        f"{elapsed:.1f}s"
    )
    print(f"    Rationale: {result['rationale'][:120]}")


def test_analyze_news_sentiment() -> None:
    """Analyze news sentiment by topic query."""
    _section("News Sentiment by Topic (LLM)")

    if not _has_llm_key():
        print("  ⚠️  No LLM API key — skipping")
        return

    t0 = time.time()
    result = analyze_news_sentiment_sync("technology stocks earnings")
    elapsed = time.time() - t0

    assert isinstance(result, dict)
    assert "label" in result
    assert "rating" in result
    assert "rationale" in result
    if result["rating"] is not None:
        assert 1 <= result["rating"] <= 10

    _pass(
        f"rating={result['rating']}, label={result['label']}, "
        f"{elapsed:.1f}s"
    )
    print(f"    Rationale: {result['rationale'][:120]}")


def test_analyze_news_sentiment_no_results() -> None:
    """Topic query with no matching results returns graceful output."""
    _section("News Sentiment — No Results")

    result = analyze_news_sentiment_sync("xyzzy_no_such_topic_9999")
    assert isinstance(result, dict)
    assert result["label"] == "unknown"
    assert result["rating"] is None
    _pass("No-match query: label=unknown, rating=None (no LLM call)")


def test_all_tickers_sentiment() -> None:
    """Run sentiment analysis across all known tickers."""
    _section("All Known Tickers Sentiment (LLM)")

    if not _has_llm_key():
        print("  ⚠️  No LLM API key — skipping")
        return

    for ticker in KNOWN_TICKERS:
        t0 = time.time()
        result = analyze_ticker_sentiment_sync(ticker)
        elapsed = time.time() - t0

        assert isinstance(result, dict)
        assert result["ticker"] == ticker
        assert isinstance(result["rating"], int)
        assert 1 <= result["rating"] <= 10
        # Verify label matches rating
        if result["rating"] <= 4:
            expected_label = "Negative"
        elif result["rating"] == 5:
            expected_label = "Neutral"
        else:
            expected_label = "Positive"
        assert result["label"] == expected_label

        _pass(
            f"{ticker}: rating={result['rating']}, label={result['label']}, "
            f"{elapsed:.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────────
# Interactive mode
# ──────────────────────────────────────────────────────────────────────────

def interactive() -> None:
    """Prompt the user for a ticker / year / topic / text and run analysis."""
    _section("Interactive Mode")

    print("Options:")
    print("  1. Ticker sentiment  (e.g. AAPL)")
    print("  2. Year sentiment    (e.g. 2024)")
    print("  3. Topic sentiment   (e.g. inflation)")
    print("  4. Free text sentiment")
    print()

    choice = input("Choose (1-4): ").strip()

    if choice == "1":
        ticker = input("Ticker: ").strip().upper()
        result = analyze_ticker_sentiment_sync(ticker)
    elif choice == "2":
        year = int(input("Year: ").strip())
        result = analyze_year_sentiment_sync(year)
    elif choice == "3":
        topic = input("Topic: ").strip()
        result = analyze_news_sentiment_sync(topic)
    elif choice == "4":
        text = input("Text: ").strip()
        result = analyze_sentiment_sync(text)
    else:
        print("Invalid choice.")
        return

    print(json.dumps(result, indent=2, ensure_ascii=False))


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

TEST_GROUPS = {
    "schema": [test_tool_schemas],
    "data": [
        test_query_by_ticker_known,
        test_query_by_ticker_unknown,
        test_query_by_ticker_case_insensitive,
        test_query_by_year,
        test_query_by_topic,
        test_query_record_properties,
    ],
    "sentiment": [
        test_analyze_sentiment_free_text,
        test_analyze_ticker_sentiment_known,
        test_analyze_ticker_sentiment_unknown,
        test_analyze_year_sentiment,
        test_analyze_news_sentiment,
        test_analyze_news_sentiment_no_results,
    ],
    "full": [
        test_all_tickers_sentiment,
    ],
}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "interactive":
        interactive()
        return

    if mode == "all":
        groups = ["schema", "data", "sentiment"]
    elif mode in TEST_GROUPS:
        groups = [mode]
    else:
        print(f"Unknown mode: {mode}")
        print(f"Available: {', '.join(list(TEST_GROUPS.keys()) + ['all', 'interactive'])}")
        sys.exit(1)

    passed = 0
    failed = 0

    for group in groups:
        for test_fn in TEST_GROUPS[group]:
            try:
                test_fn()
                passed += 1
            except Exception as exc:
                _fail(f"{test_fn.__name__}: {exc}")
                failed += 1

    _section("Summary")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
