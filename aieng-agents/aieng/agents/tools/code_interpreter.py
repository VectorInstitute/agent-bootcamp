"""Code interpreter tool."""

import asyncio
import contextlib
import os
import random
from pathlib import Path
from typing import Any, Sequence

import httpx
from aieng.agents.async_utils import gather_with_progress
from e2b import TimeoutException
from e2b.exceptions import RateLimitException
from pydantic import BaseModel


try:
    from e2b_code_interpreter import AsyncSandbox
    from e2b_code_interpreter.models import serialize_results
except ModuleNotFoundError as exc:
    from aieng.agents._optional_extras import (
        EXTRA_CODE_INTERPRETER,
        raise_missing_optional,
    )

    raise_missing_optional(
        EXTRA_CODE_INTERPRETER, missing=getattr(exc, "name", None), from_exc=exc
    )


__all__ = ["CodeInterpreter", "CodeInterpreterOutput"]


class _CodeInterpreterOutputError(BaseModel):
    """Error from code interpreter."""

    name: str
    value: str
    traceback: str


class CodeInterpreterOutput(BaseModel):
    """JSON-serializable result of a sandbox code run."""

    stdout: list[str]
    stderr: list[str]
    results: list[dict[str, str]] | None = None
    error: _CodeInterpreterOutputError | None = None

    def __init__(self, stdout: list[str], stderr: list[str], **kwargs: Any) -> None:
        """Split stdout/stderr on newlines before validation."""
        stdout_processed: list[str] = []
        for line in stdout:
            stdout_processed.extend(line.splitlines())

        stderr_processed: list[str] = []
        for line in stderr:
            stderr_processed.extend(line.splitlines())

        super().__init__(stdout=stdout_processed, stderr=stderr_processed, **kwargs)


def _failure_json(name: str, message: str) -> str:
    """Serialize a :class:`CodeInterpreterOutput` error payload as JSON."""
    err = _CodeInterpreterOutputError(name=name, value=message, traceback="")
    out = CodeInterpreterOutput(
        stdout=[],
        stderr=[f"[code tool] {name}: {message}"],
        error=err,
    )
    return out.model_dump_json()


def _validate_str_dict(label: str, d: dict[str, str] | None) -> None:
    """Ensure ``d`` is ``None`` or a ``str``-to-``str`` mapping.

    Raises
    ------
    TypeError
        If any key or value is not a string.
    """
    if d is None:
        return
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, str):
            msg = f"{label} must map str -> str, got key {type(k).__name__!r}, value {type(v).__name__!r}"
            raise TypeError(msg)


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when ``condition`` is false."""
    if not condition:
        raise ValueError(message)


def _validate_code_interpreter_init(
    *,
    sandbox_timeout_seconds: int,
    code_execution_timeout_seconds: float | None,
    request_timeout_seconds: float | None,
    envs: dict[str, str] | None,
    metadata: dict[str, str] | None,
    sandbox_create_max_attempts: int,
    sandbox_create_retry_base_seconds: float,
    sandbox_create_retry_max_seconds: float,
) -> None:
    """Validate :class:`CodeInterpreter` constructor arguments.

    Raises
    ------
    ValueError
        If any timeout, retry, or sandbox-lifetime argument is out of range.
    TypeError
        If ``envs`` or ``metadata`` contains non-string keys or values.
    """
    _require(sandbox_timeout_seconds >= 1, "sandbox_timeout_seconds must be >= 1")
    _require(
        sandbox_timeout_seconds <= 86_400,
        "sandbox_timeout_seconds exceeds 24h cap; pass a lower value per E2B plan limits",
    )
    _validate_str_dict("envs", envs)
    _validate_str_dict("metadata", metadata)
    if code_execution_timeout_seconds is not None:
        _require(
            code_execution_timeout_seconds > 0,
            "code_execution_timeout_seconds must be positive when set",
        )
        _require(
            code_execution_timeout_seconds <= float(sandbox_timeout_seconds),
            "code_execution_timeout_seconds should not exceed sandbox_timeout_seconds "
            "(the HTTP read cannot outlive the VM)",
        )
    if request_timeout_seconds is not None:
        _require(
            request_timeout_seconds > 0,
            "request_timeout_seconds must be positive when set",
        )
    _require(
        sandbox_create_max_attempts >= 1, "sandbox_create_max_attempts must be >= 1"
    )
    _require(
        sandbox_create_retry_base_seconds > 0,
        "sandbox_create_retry_base_seconds must be positive",
    )
    _require(
        sandbox_create_retry_max_seconds > 0,
        "sandbox_create_retry_max_seconds must be positive",
    )
    _require(
        sandbox_create_retry_max_seconds >= sandbox_create_retry_base_seconds,
        "sandbox_create_retry_max_seconds must be >= sandbox_create_retry_base_seconds",
    )


def _resolved_code_execution_timeout(
    sandbox_timeout_seconds: int,
    code_execution_timeout_seconds: float | None,
) -> float:
    """Return explicit code-execution timeout or a default below VM lifetime."""
    if code_execution_timeout_seconds is None:
        return float(min(max(5, sandbox_timeout_seconds - 15), sandbox_timeout_seconds))
    return float(code_execution_timeout_seconds)


def _resolved_request_timeout(
    sandbox_timeout_seconds: int,
    request_timeout_seconds: float | None,
) -> float:
    """Return explicit httpx request timeout or sandbox lifetime plus headroom."""
    if request_timeout_seconds is None:
        return float(sandbox_timeout_seconds) + 120.0
    return float(request_timeout_seconds)


async def _upload_file(sandbox: AsyncSandbox, local_path: str | Path) -> str:
    """Upload a single file into the sandbox cwd; returns remote path."""
    path = Path(local_path)
    remote_path = path.name
    with path.open("rb") as file:
        await sandbox.files.write(remote_path, file)
    return remote_path


async def _upload_files(
    sandbox: AsyncSandbox, paths: Sequence[Path | str]
) -> list[str]:
    """Upload many paths (files or flattened directory trees)."""
    if not paths:
        return []

    coros = [_upload_file(sandbox, p) for p in paths]
    return list(
        await gather_with_progress(
            coros, description=f"Uploading {len(paths)} to sandbox"
        )
    )


def _enumerate_files(base_path: str | Path) -> list[Path]:
    """Return all files under ``base_path``; a file path yields itself."""
    if os.path.isfile(base_path):
        return [Path(base_path)]

    out: list[Path] = []
    for root, _, files in os.walk(base_path):
        for name in files:
            out.append(Path(root) / name)
    return out


class CodeInterpreter:
    """Run Python in an ephemeral E2B code-interpreter sandbox and return output.

    Parameters
    ----------
    local_files : Sequence[Path | str] | None, default=None
        Paths to upload after the sandbox is created (directories are flattened).
    sandbox_timeout_seconds : int, default=300
        Maximum lifetime of the sandbox VM, forwarded to
        ``AsyncSandbox.create(timeout=...)``. Per E2B, this is a wall-clock cap on
        how long the sandbox may exist (e.g. 300 s default; up to 24 h on Pro plans).
    code_execution_timeout_seconds : float | None, default=None
        HTTP read timeout for a single ``run_code`` / Jupyter ``execute`` stream,
        forwarded to ``AsyncSandbox.run_code(timeout=...)``. If ``None``, defaults
        to ``max(5, sandbox_timeout_seconds - 15)`` so the client tends to raise
        ``TimeoutException`` before the host SIGKILLs the VM.
    request_timeout_seconds : float | None, default=None
        Overall httpx timeout tuple budget for the execute request; if ``None``,
        uses ``float(sandbox_timeout_seconds) + 120.0``.
    template_name : str | None, default=None
        E2B template id for ``AsyncSandbox.create(template=...)``.
    envs : dict[str, str] | None, default=None
        Environment variables set **at sandbox creation** (visible to all code in
        that VM). Same lifetime as ``sandbox_timeout_seconds``.
    metadata : dict[str, str] | None, default=None
        Optional string tags for E2B observability (not shown to the model unless
        you log them yourself).
    allow_internet_access : bool, default=True
        Forwarded to ``AsyncSandbox.create(allow_internet_access=...)``.
    sandbox_create_max_attempts : int, default=12
        Total ``AsyncSandbox.create`` attempts when E2B returns
        ``RateLimitException`` (concurrent sandbox cap). Retries use exponential
        backoff with jitter between attempts.
    sandbox_create_retry_base_seconds : float, default=1.0
        Base delay for sandbox-create backoff (seconds).
    sandbox_create_retry_max_seconds : float, default=30.0
        Maximum delay between sandbox-create retries (seconds).
    return_errors_as_json : bool, default=True
        When ``True``, transport and execution timeouts are returned as
        :class:`CodeInterpreterOutput` JSON with ``error.name`` in
        ``{"ExecutionTimeout", "HttpTimeout", "StreamClosed", "RateLimitExceeded"}``
        instead of raising.
        When ``False``, those errors propagate to the host (e.g. ADK tool failure).

    Notes
    -----
    Each :meth:`run_code` creates a **new** sandbox; ``envs`` / ``metadata`` apply
    only to that sandbox instance.
    """

    def __init__(
        self,
        local_files: Sequence[Path | str] | None = None,
        *,
        sandbox_timeout_seconds: int = 300,
        code_execution_timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
        template_name: str | None = None,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        allow_internet_access: bool = True,
        sandbox_create_max_attempts: int = 12,
        sandbox_create_retry_base_seconds: float = 1.0,
        sandbox_create_retry_max_seconds: float = 30.0,
        return_errors_as_json: bool = True,
    ) -> None:
        """Configure sandbox creation defaults used for every :meth:`run_code` call."""
        _validate_code_interpreter_init(
            sandbox_timeout_seconds=sandbox_timeout_seconds,
            code_execution_timeout_seconds=code_execution_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            envs=envs,
            metadata=metadata,
            sandbox_create_max_attempts=sandbox_create_max_attempts,
            sandbox_create_retry_base_seconds=sandbox_create_retry_base_seconds,
            sandbox_create_retry_max_seconds=sandbox_create_retry_max_seconds,
        )

        self.sandbox_timeout_seconds = sandbox_timeout_seconds
        self._code_execution_timeout_seconds = _resolved_code_execution_timeout(
            sandbox_timeout_seconds, code_execution_timeout_seconds
        )
        self._request_timeout_seconds = _resolved_request_timeout(
            sandbox_timeout_seconds, request_timeout_seconds
        )
        self.template_name = template_name
        self.envs = envs
        self.metadata = metadata
        self.allow_internet_access = allow_internet_access
        self.sandbox_create_max_attempts = sandbox_create_max_attempts
        self.sandbox_create_retry_base_seconds = sandbox_create_retry_base_seconds
        self.sandbox_create_retry_max_seconds = sandbox_create_retry_max_seconds
        self.return_errors_as_json = return_errors_as_json

        self.local_files: list[Path] = []
        if local_files:
            for path in local_files:
                self.local_files.extend(_enumerate_files(path))

    async def _create_sandbox(self) -> AsyncSandbox:
        """Create an E2B sandbox, retrying on concurrent-cap ``RateLimitException``.

        Uses exponential backoff with multiplicative jitter between attempts,
        configured via ``sandbox_create_*`` instance attributes.

        Raises
        ------
        RateLimitException
            When all create attempts are exhausted.
        """
        for attempt in range(1, self.sandbox_create_max_attempts + 1):
            try:
                return await AsyncSandbox.create(
                    timeout=self.sandbox_timeout_seconds,
                    template=self.template_name,
                    metadata=self.metadata,
                    envs=self.envs,
                    allow_internet_access=self.allow_internet_access,
                )
            except RateLimitException:
                if attempt >= self.sandbox_create_max_attempts:
                    raise
                delay = min(
                    self.sandbox_create_retry_base_seconds * (2 ** (attempt - 1)),
                    self.sandbox_create_retry_max_seconds,
                )
                await asyncio.sleep(delay * (0.5 + random.random() * 0.5))
        msg = "sandbox create retry loop exhausted without raising"
        raise AssertionError(msg)

    async def run_code(self, code: str) -> str:
        """Execute Python code in a fresh sandbox and return output of execution.

        Parameters
        ----------
        code
            Complete Python source for one Jupyter-style execution.

        Returns
        -------
        str
            Serialized :class:`CodeInterpreterOutput` JSON with the following
            fields:
            - stdout: list of strings, each representing a line of stdout.
            - stderr: list of strings, each representing a line of stderr.
            - results: list of dictionaries, each representing a result.
            - error: dictionary representing the error, if any.

        Raises
        ------
        TimeoutException
            If the code execution timeout is exceeded and
            ``self.return_errors_as_json`` is ``False``. Otherwise, this is
            returned as serialized JSON error output.
        httpx.TimeoutException
            If the request timeout is exceeded and
            ``self.return_errors_as_json`` is ``False``. Otherwise, this is
            returned as serialized JSON error output.
        httpx.RemoteProtocolError
            If the HTTP response stream ended early and
            ``self.return_errors_as_json`` is ``False``. Otherwise, this is
            returned as serialized JSON error output.
        RateLimitException
            If E2B's concurrent sandbox cap is still exceeded after all create
            attempts and ``self.return_errors_as_json`` is ``False``. Otherwise,
            returned as serialized JSON error output with
            ``error.name="RateLimitExceeded"``.

        Notes
        -----
        Sandboxes are **not** reused: variables and downloaded files do not exist on the
        next call.
        """
        sbx: AsyncSandbox | None = None
        try:
            try:
                sbx = await self._create_sandbox()
            except RateLimitException as exc:
                if self.return_errors_as_json:
                    return _failure_json(
                        "RateLimitExceeded",
                        f"{exc} — E2B concurrent sandbox limit still exceeded after "
                        f"{self.sandbox_create_max_attempts} create attempt(s). "
                        "Retry the tool call after other sandboxes finish, or reduce "
                        "simultaneous code execution in the room.",
                    )
                raise

            await _upload_files(sbx, self.local_files)

            result = await sbx.run_code(
                code,
                on_error=lambda error: print(error.traceback),
                timeout=self._code_execution_timeout_seconds,
                request_timeout=self._request_timeout_seconds,
            )
            response = CodeInterpreterOutput.model_validate_json(result.logs.to_json())
            if result.error is not None:
                response.error = _CodeInterpreterOutputError.model_validate_json(
                    result.error.to_json()
                )
            if result.results:
                response.results = serialize_results(result.results)
            return response.model_dump_json()
        except TimeoutException as exc:
            if self.return_errors_as_json:
                return _failure_json(
                    "ExecutionTimeout",
                    f"{exc} (code read budget ~{self._code_execution_timeout_seconds:g}s; "
                    f"sandbox VM up to ~{self.sandbox_timeout_seconds}s). Retry with less work "
                    "per run—e.g. fewer downloads or model fits, smaller loops, or split logic "
                    "across multiple tool calls (each run starts from a fresh sandbox).",
                )
            raise
        except httpx.TimeoutException as exc:
            if self.return_errors_as_json:
                return _failure_json(
                    "HttpTimeout",
                    f"{type(exc).__name__}: {exc} — the HTTP client timed out waiting for the "
                    "sandbox. Retry with less work per run or split across multiple tool calls.",
                )
            raise
        except httpx.RemoteProtocolError as exc:
            if self.return_errors_as_json:
                return _failure_json(
                    "StreamClosed",
                    "HTTP response stream ended early — often the sandbox hit its "
                    f"wall-clock limit (~{self.sandbox_timeout_seconds}s) during a long "
                    "download or compute step, or the network dropped. Retry with a smaller "
                    f"per-run workload. Detail: {exc}",
                )
            raise
        finally:
            if sbx is not None:
                with contextlib.suppress(Exception):
                    await sbx.kill()
