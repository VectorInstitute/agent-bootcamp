"""Lightweight Langfuse tracing helpers.

If Langfuse keys are not set the helpers become no-ops so the system
runs cleanly without observability configured.

Uses the imperative Langfuse Python SDK span API:
  - ``Langfuse.start_span(name)`` creates a root span + auto-trace
  - ``LangfuseSpan.start_span(name)`` creates a nested child span
  - ``LangfuseSpan.update(output=…)`` attaches data
  - ``LangfuseSpan.end()`` closes the span
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── Try to import Langfuse; degrade gracefully ──────────────────────────
_langfuse = None
try:
    from langfuse import Langfuse
    from ..config.settings import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        _langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        if _langfuse.auth_check():
            logger.info("Langfuse tracing enabled (auth OK)")
        else:
            logger.warning("Langfuse auth check failed – tracing disabled")
            _langfuse = None
    else:
        logger.info("Langfuse keys not set – tracing disabled")
except Exception as exc:
    logger.debug("Langfuse unavailable: %s", exc)


# ── Public helpers ──────────────────────────────────────────────────────

class Tracer:
    """Thin wrapper around a Langfuse trace / span tree.

    Usage::

        tracer = Tracer.start("orchestrator_run", metadata={...})
        with tracer.span("intent_parse") as sp:
            sp.update(output={"intent": "rank"})
        tracer.end(output=final_answer)

    The root span acts as the top-level container.  All child spans
    created via ``tracer.span(...)`` are nested under it so the full
    Plan → Act → Observe → Reflect flow is visible in Langfuse.
    """

    def __init__(self, root_span: Any | None = None):
        self._root = root_span          # LangfuseSpan or None
        self._trace_id: str | None = None
        if root_span is not None:
            self._trace_id = root_span.trace_id

    # --- factory ---
    @classmethod
    def start(
        cls,
        name: str,
        *,
        user_id: str = "",
        metadata: dict | None = None,
    ) -> "Tracer":
        if _langfuse is None:
            return cls(None)
        try:
            root = _langfuse.start_span(name=name, metadata=metadata or {})
            # Attach trace-level info (name, user_id, tags)
            root.update_trace(
                name=name,
                user_id=user_id or None,
                metadata=metadata or {},
            )
            logger.info("Langfuse trace started: %s", root.trace_id)
            return cls(root)
        except Exception as exc:
            logger.warning("Langfuse trace start error: %s", exc)
            return cls(None)

    # --- span context-manager ---
    @contextmanager
    def span(self, name: str, **kwargs) -> Generator["_Span", None, None]:
        """Create a child span nested under the root."""
        sp = _Span.create(name, parent=self._root, **kwargs)
        try:
            yield sp
        except Exception as exc:
            sp.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            sp.finish()

    # --- finalise ---
    def end(self, *, output: Any = None):
        if self._root is not None:
            try:
                if output is not None:
                    self._root.update(output=output)
                self._root.end()
            except Exception as exc:
                logger.debug("Langfuse root span end error: %s", exc)
        if _langfuse is not None:
            try:
                _langfuse.flush()
            except Exception as exc:
                logger.debug("Langfuse flush error: %s", exc)

    @property
    def trace_id(self) -> str | None:
        return self._trace_id


class _Span:
    """One span inside a trace."""

    def __init__(self, name: str, lang_span: Any | None = None):
        self.name = name
        self._span = lang_span

    @classmethod
    def create(cls, name: str, parent: Any | None = None, **kwargs) -> "_Span":
        """Create a span as a child of *parent* (a ``LangfuseSpan``)."""
        if parent is None:
            return cls(name, None)
        try:
            child = parent.start_span(
                name=name,
                metadata=kwargs.get("metadata"),
            )
            return cls(name, child)
        except Exception as exc:
            logger.debug("Langfuse child span error (%s): %s", name, exc)
            return cls(name, None)

    def update(self, **kwargs):
        if self._span is not None:
            try:
                self._span.update(**kwargs)
            except Exception:
                pass

    def finish(self):
        if self._span is not None:
            try:
                self._span.end()
            except Exception:
                pass
