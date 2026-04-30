"""Console entrypoints."""

import importlib
from types import ModuleType
from typing import Any


def _run_with_extra(module_path: str, attr: str, extra: str) -> None:
    """Import a data script module and invoke its Click-wrapped ``main``."""
    try:
        module: ModuleType = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            f"Missing optional dependency ({missing!r}). "
            f"Install with: pip install 'aieng-agents[{extra}]'"
        ) from exc
    except ImportError as exc:
        # Scripts may raise ``raise_missing_optional`` while importing optional deps.
        raise SystemExit(str(exc)) from exc

    target: Any = getattr(module, attr)
    target()


def pdf_to_hf_dataset_main() -> None:
    """Entry for ``pdf_to_hf_dataset`` console script."""
    _run_with_extra("aieng.agents.data.pdf_to_hf_dataset", "main", "data")


def chunk_hf_dataset_main() -> None:
    """Entry for ``chunk_hf_dataset`` console script."""
    _run_with_extra("aieng.agents.data.chunk_hf_dataset", "main", "data")
