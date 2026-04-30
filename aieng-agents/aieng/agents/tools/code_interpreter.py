"""Code interpreter tool."""

import contextlib
import os
from pathlib import Path
from typing import Any, Sequence

import httpx
from aieng.agents.async_utils import gather_with_progress
from e2b import TimeoutException
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
    err = _CodeInterpreterOutputError(name=name, value=message, traceback="")
    out = CodeInterpreterOutput(
        stdout=[],
        stderr=[f"[code tool] {name}: {message}"],
        error=err,
    )
    return out.model_dump_json()


def _validate_str_dict(label: str, d: dict[str, str] | None) -> None:
    if d is None:
        return
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, str):
            msg = f"{label} must map str -> str, got key {type(k).__name__!r}, value {type(v).__name__!r}"
            raise TypeError(msg)


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
    return_errors_as_json : bool, default=True
        When ``True``, transport and execution timeouts are returned as
        :class:`CodeInterpreterOutput` JSON with ``error.name`` in
        ``{"ExecutionTimeout", "HttpTimeout", "StreamClosed"}`` instead of raising.
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
        return_errors_as_json: bool = True,
    ) -> None:
        """Configure sandbox creation defaults used for every :meth:`run_code` call."""
        if sandbox_timeout_seconds < 1:
            msg = "sandbox_timeout_seconds must be >= 1"
            raise ValueError(msg)
        if sandbox_timeout_seconds > 86_400:
            msg = "sandbox_timeout_seconds exceeds 24h cap; pass a lower value per E2B plan limits"
            raise ValueError(msg)

        _validate_str_dict("envs", envs)
        _validate_str_dict("metadata", metadata)

        if code_execution_timeout_seconds is not None:
            if code_execution_timeout_seconds <= 0:
                msg = "code_execution_timeout_seconds must be positive when set"
                raise ValueError(msg)
            if code_execution_timeout_seconds > float(sandbox_timeout_seconds):
                msg = (
                    "code_execution_timeout_seconds should not exceed sandbox_timeout_seconds "
                    "(the HTTP read cannot outlive the VM)"
                )
                raise ValueError(msg)

        if request_timeout_seconds is not None and request_timeout_seconds <= 0:
            msg = "request_timeout_seconds must be positive when set"
            raise ValueError(msg)

        self.sandbox_timeout_seconds = sandbox_timeout_seconds
        if code_execution_timeout_seconds is None:
            self._code_execution_timeout_seconds = float(
                max(5, sandbox_timeout_seconds - 15)
            )
        else:
            self._code_execution_timeout_seconds = float(code_execution_timeout_seconds)

        if request_timeout_seconds is None:
            self._request_timeout_seconds = float(sandbox_timeout_seconds) + 120.0
        else:
            self._request_timeout_seconds = float(request_timeout_seconds)

        self.template_name = template_name
        self.envs = envs
        self.metadata = metadata
        self.allow_internet_access = allow_internet_access
        self.return_errors_as_json = return_errors_as_json

        self.local_files: list[Path] = []
        if local_files:
            for path in local_files:
                self.local_files.extend(_enumerate_files(path))

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
            If the code execution timeout is exceeded.
        httpx.TimeoutException
            If the request timeout is exceeded.
        httpx.RemoteProtocolError
            If the HTTP response stream ended early.

        Notes
        -----
        Sandboxes are **not** reused: variables and downloaded files do not exist on the
        next call.
        """
        sbx: AsyncSandbox | None = None
        try:
            sbx = await AsyncSandbox.create(
                timeout=self.sandbox_timeout_seconds,
                template=self.template_name,
                metadata=self.metadata,
                envs=self.envs,
                allow_internet_access=self.allow_internet_access,
            )
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
