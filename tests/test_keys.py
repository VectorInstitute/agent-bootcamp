#!/usr/bin/env python3
"""
Integration test for API keys.

This script tests all API keys to ensure they work correctly before
marking the participant as fully onboarded.
"""

import os
import sys
from pathlib import Path

import openai
import weaviate
from aieng_platform_onboard.utils import console
from dotenv import load_dotenv
from rich.panel import Panel
from rich.table import Table
from weaviate.classes.init import Auth


def test_gemini_api() -> tuple[bool, str]:
    """
    Test Gemini API (OpenAI-compatible).

    Returns
    -------
    tuple[bool, str]
        Tuple of (success, message).
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key or not base_url:
            return False, "Missing OPENAI_API_KEY or OPENAI_BASE_URL"

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Simple test: list models
        models = client.models.list()

        if not models:
            return False, "No models available"

        return True, f"Connected - {len(models.data)} models available"

    except Exception as e:
        return False, str(e)


def test_embedding_api() -> tuple[bool, str]:
    """
    Test Embedding API.

    Returns
    -------
    tuple[bool, str]
        Tuple of (success, message).
    """
    try:
        api_key = os.getenv("EMBEDDING_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL")

        if not api_key or not base_url:
            return False, "Missing EMBEDDING_API_KEY or EMBEDDING_BASE_URL"

        # Test with OpenAI client (Cloudflare Workers AI is OpenAI-compatible)
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Try to create an embedding
        response = client.embeddings.create(input="test", model="@cf/baai/bge-m3")

        # Check if we got a valid embedding
        if response.data and len(response.data) > 0:
            embedding_dim = len(response.data[0].embedding)
            return True, f"Connection successful (embedding dim: {embedding_dim})"

        return False, "No embedding data returned"

    except Exception as e:
        return False, str(e)


def test_weaviate() -> tuple[bool, str]:
    """
    Test Weaviate connection.

    Returns
    -------
    tuple[bool, str]
        Tuple of (success, message).
    """
    try:
        http_host = os.getenv("WEAVIATE_HTTP_HOST")
        api_key = os.getenv("WEAVIATE_API_KEY")
        http_secure = os.getenv("WEAVIATE_HTTP_SECURE", "true").lower() == "true"

        if not http_host or not api_key:
            return False, "Missing WEAVIATE_HTTP_HOST or WEAVIATE_API_KEY"

        # Connect to Weaviate
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=443 if http_secure else 80,
            http_secure=http_secure,
            grpc_host=os.getenv("WEAVIATE_GRPC_HOST", http_host),
            grpc_port=443 if http_secure else 50051,
            grpc_secure=http_secure,
            auth_credentials=Auth.api_key(api_key),
        )

        # Test connection
        is_ready = client.is_ready()
        client.close()

        if is_ready:
            return True, "Connection successful"
        return False, "Weaviate not ready"

    except Exception as e:
        return False, str(e)


def test_langfuse() -> tuple[bool, str]:
    """
    Test Langfuse connection.

    Returns
    -------
    tuple[bool, str]
        Tuple of (success, message).
    """
    try:
        # Langfuse test - just verify env vars are set
        # Full test would require actual Langfuse SDK usage
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST")

        if not secret_key or not public_key or not host:
            return False, "Missing LANGFUSE keys or host"

        # Basic format validation
        if not secret_key.startswith("sk-lf-"):
            return False, "Invalid LANGFUSE_SECRET_KEY format"

        if not public_key.startswith("pk-lf-"):
            return False, "Invalid LANGFUSE_PUBLIC_KEY format"

        return True, "Configuration valid"

    except Exception as e:
        return False, str(e)


def main() -> int:
    """
    Run all integration tests.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    # Load .env file
    env_path = Path(".env")
    if not env_path.exists():
        console.print("[red]✗ .env file not found[/red]")
        return 1

    load_dotenv(env_path)

    # Print header
    console.print(
        Panel.fit(
            "[bold cyan]API Keys Integration Test[/bold cyan]\n"
            "Verifying all API keys and connections",
            border_style="cyan",
        )
    )

    # Define tests
    tests = [
        ("Gemini API", test_gemini_api),
        ("Embedding API", test_embedding_api),
        ("Weaviate", test_weaviate),
        ("Langfuse", test_langfuse),
    ]

    # Run tests
    results = []
    all_passed = True

    console.print()
    for test_name, test_func in tests:
        console.print(f"[cyan]Testing {test_name}...[/cyan]")
        success, message = test_func()

        results.append(
            {
                "name": test_name,
                "success": success,
                "message": message,
            }
        )

        if success:
            console.print(f"  [green]✓ {message}[/green]")
        else:
            console.print(f"  [red]✗ {message}[/red]")
            all_passed = False

    # Display results table
    console.print()
    table = Table(title="Test Results", show_header=True, header_style="bold cyan")
    table.add_column("Service", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Message", style="dim")

    for result in results:
        status = (
            "[green]✓ Passed[/green]" if result["success"] else "[red]✗ Failed[/red]"
        )
        table.add_row(result["name"], status, result["message"])

    console.print(table)
    console.print()

    # Final verdict
    if all_passed:
        console.print(
            Panel.fit(
                "[green bold]✓ ALL TESTS PASSED[/green bold]\n"
                "All API keys are configured correctly",
                border_style="green",
                title="Success",
            )
        )
        return 0
    console.print(
        Panel.fit(
            "[red bold]✗ SOME TESTS FAILED[/red bold]\n"
            "Please check the errors above and contact your bootcamp admin",
            border_style="red",
            title="Failed",
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
