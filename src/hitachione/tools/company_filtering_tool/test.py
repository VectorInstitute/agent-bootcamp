"""
Test harness for the company filtering tool (Weaviate-backed).

This module provides comprehensive tests for the find_relevant_symbols tool.
Run this file to test the tool functionality.
"""

import sys
from typing import List
from dotenv import load_dotenv
from pathlib import Path

# Load .env file (project root is 5 levels up from this file)
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from tool import (
    find_relevant_symbols,
    find_relevant_sp500_symbols,
    get_all_symbols,
    get_company_mapping,
    TOOL_SCHEMA,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_symbol_extraction():
    """Test extracting all symbols from the Weaviate knowledge base."""
    print_section("Testing Symbol Extraction from Weaviate")
    
    print("Extracting all unique tickers from Weaviate collection...")
    print("(Iterates through Hitachi_finance_news collection)\n")
    
    try:
        import time
        start = time.time()
        
        symbols = get_all_symbols()
        elapsed = time.time() - start
        
        print(f"✓ Successfully extracted {len(symbols)} unique tickers in {elapsed:.2f}s")
        print(f"\nTickers: {', '.join(symbols)}")
        
        # Also show company mapping
        companies = get_company_mapping()
        print(f"\nCompany mapping:")
        for ticker, company in sorted(companies.items()):
            print(f"  {ticker}: {company}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def test_symbol_search():
    """Test the main symbol search functionality with LLM filtering."""
    print_section("Testing Tool Function with LLM Filtering")
    
    test_cases = [
        "List all the top automotive stocks of 2012",
        "Find technology companies from 2015",
        "Show me healthcare stocks",
        "top 5 tech stocks",
    ]
    
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  No LLM API key set - LLM filtering disabled")
        print("   Tool will return all symbols without filtering\n")
    else:
        print("✓ LLM API key found - LLM filtering enabled\n")
    
    for i, query in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {query}")
        print("-" * 80)
        
        try:
            symbols = find_relevant_symbols(query)
            
            print(f"✓ Returned {len(symbols)} filtered symbols")
            print(f"  Results: {', '.join(symbols[:15])}")
            if len(symbols) > 15:
                print(f"  ... and {len(symbols) - 15} more")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print()


def test_year_extraction():
    """Test year extraction from queries."""
    print_section("Legacy Test: Year Extraction")
    print("Note: This functionality is no longer used.")
    print("The tool now returns all symbols and lets the LLM do filtering.\n")


def test_keyword_detection():
    """Test keyword detection from queries."""
    print_section("Legacy Test: Keyword Detection")
    print("Note: This functionality is no longer used.")
    print("The tool now returns all symbols and lets the LLM do filtering.\n")


def test_tool_schema():
    """Display and validate the tool schema."""
    print_section("Tool Schema for OpenAI")
    
    import json
    print(json.dumps(TOOL_SCHEMA, indent=2))
    print("\nSchema validation: ✓ Valid JSON structure")


def run_interactive_test():
    """Run an interactive test where user can input queries."""
    print_section("Interactive Test Mode")
    
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  No LLM API key set")
        print("   Tool will return all symbols without LLM filtering\n")
    else:
        print("✓ LLM filtering enabled\n")
    
    print("Enter queries to test the tool (or 'quit' to exit)")
    print("Example: 'top 3 tech stocks of 2010'\n")
    
    # Load symbols once
    try:
        print("Loading all tickers from Weaviate...")
        all_symbols = get_all_symbols()
        companies = get_company_mapping()
        print(f"✓ Loaded {len(all_symbols)} unique tickers:")
        for t in all_symbols:
            print(f"  {t}: {companies.get(t, 'N/A')}")
        print()
    except Exception as e:
        print(f"✗ Error loading symbols: {e}")
        return
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nProcessing: '{query}'")
            
            import time
            start = time.time()
            symbols = find_relevant_symbols(query)
            elapsed = time.time() - start
            
            print(f"✓ Complete in {elapsed:.2f}s")
            print(f"  Filtered to {len(symbols)} symbols: {', '.join(symbols)}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def benchmark_query_performance():
    """Benchmark the performance of symbol extraction."""
    print_section("Performance Benchmarking")
    
    import time
    
    print("Benchmarking symbol extraction...\n")
    
    # First run (may need to read from disk)
    start_time = time.time()
    symbols = get_all_symbols()
    elapsed = time.time() - start_time
    
    print(f"First call: {elapsed:.3f}s ({len(symbols)} symbols)")
    
    # Second run (should use cache)
    start_time = time.time()
    symbols = get_all_symbols()
    elapsed = time.time() - start_time
    
    print(f"Cached call: {elapsed:.3f}s (from memory cache)")
    print()


def main():
    """Main test harness entry point."""
    print("\n" + "=" * 80)
    print("  Company Filtering Tool (Weaviate) - Test Harness")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable test modes:")
        print("  1. all       - Run all tests")
        print("  2. quick     - Run quick tests (schema only)")
        print("  3. extract   - Test symbol extraction")
        print("  4. interactive - Interactive query mode")
        print("  5. benchmark - Performance benchmarking")
        print("  6. schema    - Display tool schema")
        
        choice = input("\nSelect mode (1-6) or press Enter for 'all': ").strip()
        
        mode_map = {
            '1': 'all',
            '2': 'quick',
            '3': 'extract',
            '4': 'interactive',
            '5': 'benchmark',
            '6': 'schema',
        }
        
        mode = mode_map.get(choice, 'all')
    
    if mode in ['all', 'quick']:
        test_tool_schema()
    
    if mode in ['all', 'extract']:
        test_symbol_extraction()
        test_symbol_search()
    
    if mode == 'interactive':
        run_interactive_test()
    
    if mode == 'benchmark':
        benchmark_query_performance()
    
    if mode == 'schema':
        test_tool_schema()
    
    print_section("Test Harness Complete")
    print("To run specific tests, use: python test_sp500_tool.py <mode>")
    print("Available modes: all, quick, extract, interactive, benchmark, schema\n")


if __name__ == "__main__":
    main()
