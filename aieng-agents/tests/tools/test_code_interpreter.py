"""Test code interpreter tool."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aieng.agents import pretty_print
from aieng.agents.tools.code_interpreter import CodeInterpreter, CodeInterpreterOutput
from e2b.exceptions import RateLimitException


FAST_RETRY_KWARGS = {
    "sandbox_create_max_attempts": 2,
    "sandbox_create_retry_base_seconds": 0.01,
    "sandbox_create_retry_max_seconds": 0.02,
}

PANDAS_VERSION_SCRIPT = """\
import os
import pandas as pd
print(pd.__version__)
"""

PANDAS_READ_FILE_SCRIPT = """\
import pandas as pd
from pathlib import Path

assert Path("example_a.csv").exists()
df = pd.read_csv("example_a.csv")
print(df.sum()["y"])
"""


@pytest.mark.asyncio
async def test_code_interpreter() -> None:
    """Test running a Python command in the interpreter."""
    session = CodeInterpreter(sandbox_timeout_seconds=15)

    response = await session.run_code(PANDAS_VERSION_SCRIPT)
    response_typed = CodeInterpreterOutput.model_validate_json(response)
    assert response_typed.error is None

    pretty_print(response_typed)
    pd_version_major, *_ = response_typed.stdout[0].strip().split(".")
    assert int(pd_version_major) >= 2


@pytest.mark.asyncio
async def test_jupyter_command() -> None:
    """Test running a Python command in the interpreter."""
    session = CodeInterpreter(sandbox_timeout_seconds=15)

    response = await session.run_code("!pip freeze")
    response_typed = CodeInterpreterOutput.model_validate_json(response)

    pretty_print(response_typed)


@pytest.mark.asyncio
async def test_code_interpreter_upload_file() -> None:
    """Test running a Python command in the interpreter."""
    example_paths = [Path("aieng-agents/tests/example_files/example_a.csv")]
    for _path in example_paths:
        assert _path.exists()

    session = CodeInterpreter(sandbox_timeout_seconds=15, local_files=example_paths)
    response = await session.run_code(PANDAS_READ_FILE_SCRIPT)
    response_typed = CodeInterpreterOutput.model_validate_json(response)

    pretty_print(response_typed)
    assert int(response_typed.stdout[0]) == 126


class _FakeRunResult:
    class _Logs:
        @staticmethod
        def to_json() -> str:
            return '{"stdout":["1\\n"],"stderr":[]}'

    logs = _Logs()
    error = None
    results = None


class _FakeSandbox:
    sandbox_id = "fake-sandbox"

    async def run_code(self, code: str, **kwargs: Any) -> _FakeRunResult:
        return _FakeRunResult()

    async def kill(self) -> None:
        return None

    class Files:
        @staticmethod
        async def write(path: str, file: Any) -> None:
            return None


@pytest.mark.asyncio
async def test_create_sandbox_retries_on_rate_limit() -> None:
    """RateLimitException on create is retried before run_code proceeds."""
    attempts = {"count": 0}

    async def fake_create(**kwargs: Any) -> _FakeSandbox:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RateLimitException("429: concurrent sandbox limit")
        return _FakeSandbox()

    with patch(
        "aieng.agents.tools.code_interpreter.AsyncSandbox.create",
        side_effect=fake_create,
    ):
        session = CodeInterpreter(
            sandbox_timeout_seconds=15,
            **{**FAST_RETRY_KWARGS, "sandbox_create_max_attempts": 5},
        )
        response = await session.run_code("print(1)")

    assert attempts["count"] == 3
    response_typed = CodeInterpreterOutput.model_validate_json(response)
    assert response_typed.error is None
    assert response_typed.stdout == ["1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_errors_as_json", "expect_json_error"),
    [(True, True), (False, False)],
)
async def test_create_sandbox_rate_limit_exhausted(
    return_errors_as_json: bool,
    expect_json_error: bool,
) -> None:
    """Exhausted create retries return JSON by default or raise when configured."""
    with patch(
        "aieng.agents.tools.code_interpreter.AsyncSandbox.create",
        new=AsyncMock(side_effect=RateLimitException("429: concurrent sandbox limit")),
    ):
        session = CodeInterpreter(
            sandbox_timeout_seconds=15,
            return_errors_as_json=return_errors_as_json,
            **FAST_RETRY_KWARGS,
        )
        if expect_json_error:
            response = await session.run_code("print(1)")
            response_typed = CodeInterpreterOutput.model_validate_json(response)
            assert response_typed.error is not None
            assert response_typed.error.name == "RateLimitExceeded"
            assert "429" in response_typed.error.value
        else:
            with pytest.raises(RateLimitException, match="429"):
                await session.run_code("print(1)")
