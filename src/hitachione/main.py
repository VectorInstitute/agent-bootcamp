"""Entry point for the multi-agent financial intelligence system.

Usage::

    python -m src.hitachione.main          # launch Gradio UI
    python -m src.hitachione.main --cli     # one-shot CLI mode
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `utils` etc. resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Intelligence Agent")
    parser.add_argument(
        "--cli", action="store_true", help="Run a single query from stdin"
    )
    parser.add_argument(
        "--query", type=str, default="", help="Query string (CLI mode)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Gradio server port"
    )
    args = parser.parse_args()

    if args.cli or args.query:
        from hitachione.agents.orchestrator import Orchestrator

        query = args.query or input("Enter query: ")
        orch = Orchestrator()
        answer = orch.run(query)
        print(answer.markdown)
        if answer.caveats:
            print("\nCaveats:")
            for c in answer.caveats:
                print(f"  - {c}")
        print(f"\nConfidence: {answer.confidence:.0%}")
    else:
        from hitachione.ui.app import build_app

        demo = build_app()
        demo.launch(share=True)


if __name__ == "__main__":
    main()
