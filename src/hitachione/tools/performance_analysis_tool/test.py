"""
Test harness for the Performance Analysis Tool (Weaviate-backed).

Run:
    cd src/hitachione/tools/performance_analysis_tool
    python3 test.py all          # full suite
    python3 test.py data         # data retrieval only (no LLM)
    python3 test.py analyse      # full analysis (requires LLM key)
    python3 test.py schema       # show tool schema
    python3 test.py interactive  # interactive ticker input
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
    TOOL_SCHEMA,
    analyse_stock_performance,
    get_ticker_data,
)

# ── Tickers known to exist in the Weaviate collection ──
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


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

def test_tool_schema() -> None:
    """Validate the tool schema structure."""
    _section("Tool Schema")
    print(json.dumps(TOOL_SCHEMA, indent=2))

    assert TOOL_SCHEMA["type"] == "function"
    fn = TOOL_SCHEMA["function"]
    assert fn["name"] == "analyse_stock_performance"
    assert "ticker" in fn["parameters"]["properties"]
    assert "ticker" in fn["parameters"]["required"]
    _pass("Schema is valid")


def test_data_retrieval_known_ticker() -> None:
    """Retrieve data for known tickers and verify structure."""
    _section("Data Retrieval — Known Tickers")

    for ticker in ["AAPL", "TSLA", "JPM"]:
        t0 = time.time()
        data = get_ticker_data(ticker)
        elapsed = time.time() - t0

        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        for key in ("price_data", "earnings", "news"):
            assert key in data, f"Missing key '{key}' for {ticker}"

        total = sum(len(v) for v in data.values())
        _pass(
            f"{ticker}: {len(data['price_data'])} price, "
            f"{len(data['earnings'])} earnings, "
            f"{len(data['news'])} news  ({elapsed:.2f}s, {total} total)"
        )

        # At least one data source should have records
        assert total > 0, f"No data returned for known ticker {ticker}"


def test_data_retrieval_unknown_ticker() -> None:
    """Verify graceful handling of an unknown ticker."""
    _section("Data Retrieval — Unknown Ticker")

    data = get_ticker_data(UNKNOWN_TICKER)
    total = sum(len(v) for v in data.values())
    assert total == 0, f"Expected 0 records for {UNKNOWN_TICKER}, got {total}"
    _pass(f"{UNKNOWN_TICKER}: 0 records as expected")


def test_data_retrieval_case_insensitive() -> None:
    """Verify ticker is uppercased automatically."""
    _section("Data Retrieval — Case Insensitivity")

    data_upper = get_ticker_data("AAPL")
    data_lower = get_ticker_data("aapl")

    assert len(data_upper["price_data"]) == len(data_lower["price_data"]), \
        "Price data count differs between 'AAPL' and 'aapl'"
    _pass("'AAPL' and 'aapl' return same results")


def test_price_data_fields() -> None:
    """Verify price records contain expected fields."""
    _section("Price Data — Field Validation")

    data = get_ticker_data("TSLA")
    if not data["price_data"]:
        _fail("No price data for TSLA")
        return

    expected_fields = {"date", "open", "high", "low", "close"}
    for i, rec in enumerate(data["price_data"][:3]):
        present = set(rec.keys()) & expected_fields
        assert present == expected_fields, (
            f"Record {i} missing fields: {expected_fields - present}"
        )
    _pass(f"First {min(3, len(data['price_data']))} records have all OHLC fields")


def test_price_data_sorted() -> None:
    """Verify price records are sorted by date."""
    _section("Price Data — Sort Order")

    data = get_ticker_data("GOOGL")
    dates = [r["date"] for r in data["price_data"] if "date" in r]
    assert dates == sorted(dates), "Price data is not sorted by date"
    _pass(f"GOOGL: {len(dates)} price records sorted correctly")


def test_analyse_unknown_ticker() -> None:
    """Full analysis on unknown ticker returns None score."""
    _section("Full Analysis — Unknown Ticker")

    result = analyse_stock_performance(UNKNOWN_TICKER)
    assert result["ticker"] == UNKNOWN_TICKER
    assert result["performance_score"] is None
    assert result["data_summary"]["price_records"] == 0
    _pass(f"{UNKNOWN_TICKER}: score=None, outlook={result['outlook']}")


def test_analyse_known_ticker() -> None:
    """Full analysis on a known ticker returns valid structure."""
    _section("Full Analysis — Known Tickers (LLM)")

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        print("  ⚠️  No LLM API key — skipping LLM analysis tests")
        return

    for ticker in ["AAPL", "NVDA"]:
        t0 = time.time()
        result = analyse_stock_performance(ticker)
        elapsed = time.time() - t0

        assert isinstance(result, dict)
        assert result["ticker"] == ticker
        assert isinstance(result["performance_score"], int)
        assert 1 <= result["performance_score"] <= 10
        assert result["outlook"] in ("Bullish", "Bearish", "Volatile", "Sideways")
        assert len(result["justification"]) > 20
        assert result["data_summary"]["price_records"] > 0

        _pass(
            f"{ticker}: score={result['performance_score']}, "
            f"outlook={result['outlook']}, {elapsed:.1f}s"
        )
        print(f"    Justification: {result['justification'][:120]}...")


def test_analyse_multiple_tickers() -> None:
    """Analyse several tickers to confirm consistency."""
    _section("Full Analysis — All Known Tickers (LLM)")

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        print("  ⚠️  No LLM API key — skipping")
        return

    for ticker in KNOWN_TICKERS:
        result = analyse_stock_performance(ticker)
        score = result["performance_score"]
        outlook = result["outlook"]
        ds = result["data_summary"]
        _pass(
            f"{ticker:5s}: score={score:>2}, outlook={outlook:8s}  "
            f"(price={ds['price_records']}, earn={ds['earnings_records']}, "
            f"news={ds['news_records']})"
        )


# ──────────────────────────────────────────────────────────────────────────
# Interactive mode
# ──────────────────────────────────────────────────────────────────────────

def interactive() -> None:
    _section("Interactive Mode")
    print(f"Available tickers: {', '.join(KNOWN_TICKERS)}")
    print("Enter a ticker (or 'quit' to exit)\n")

    while True:
        try:
            ticker = input("Ticker> ").strip()
            if ticker.lower() in ("quit", "exit", "q"):
                break
            if not ticker:
                continue

            print(f"\nAnalysing {ticker.upper()}...")
            t0 = time.time()
            result = analyse_stock_performance(ticker)
            elapsed = time.time() - t0

            print(json.dumps(result, indent=2))
            print(f"({elapsed:.1f}s)\n")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 80)
    print("  Performance Analysis Tool (Weaviate) — Test Harness")
    print("=" * 80)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nModes:")
        print("  1. all         — Run all tests")
        print("  2. data        — Data retrieval tests only (no LLM)")
        print("  3. analyse     — Full analysis tests (requires LLM)")
        print("  4. schema      — Display tool schema")
        print("  5. interactive — Interactive ticker input")

        choice = input("\nSelect (1-5) or press Enter for 'all': ").strip()
        mode = {"1": "all", "2": "data", "3": "analyse", "4": "schema", "5": "interactive"}.get(choice, "all")

    if mode in ("all", "schema"):
        test_tool_schema()

    if mode in ("all", "data"):
        test_data_retrieval_known_ticker()
        test_data_retrieval_unknown_ticker()
        test_data_retrieval_case_insensitive()
        test_price_data_fields()
        test_price_data_sorted()

    if mode in ("all", "analyse"):
        test_analyse_unknown_ticker()
        test_analyse_known_ticker()
        test_analyse_multiple_tickers()

    if mode == "interactive":
        interactive()

    _section("Test Harness Complete")


if __name__ == "__main__":
    main()
