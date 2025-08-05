"""Test code interpreter tool."""

from pathlib import Path

import pytest

from src.utils import pretty_print
from src.utils.tools.code_interpreter import (
    CodeInterpreter,
    CodeInterpreterOutput,
)


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
async def test_code_interpreter():
    """Test running a Python command in the interpreter."""
    session = CodeInterpreter(timeout_seconds=15)

    response = await session.run_code(PANDAS_VERSION_SCRIPT)
    response_typed = CodeInterpreterOutput.model_validate_json(response)
    assert response_typed.error is None

    pretty_print(response_typed)
    pd_version_major, *_ = response_typed.stdout[0].strip().split(".")
    assert int(pd_version_major) >= 2


@pytest.mark.asyncio
async def test_jupyter_command():
    """Test running a Python command in the interpreter."""
    session = CodeInterpreter(timeout_seconds=15)

    response = await session.run_code("!pip freeze")
    response_typed = CodeInterpreterOutput.model_validate_json(response)

    pretty_print(response_typed)


@pytest.mark.asyncio
async def test_code_interpreter_upload_file():
    """Test running a Python command in the interpreter."""
    example_paths = [Path("tests/tool_tests/example_files/example_a.csv")]
    for _path in example_paths:
        assert _path.exists()

    session = CodeInterpreter(timeout_seconds=15, local_files=example_paths)
    response = await session.run_code(PANDAS_READ_FILE_SCRIPT)
    response_typed = CodeInterpreterOutput.model_validate_json(response)

    pretty_print(response_typed)
    assert int(response_typed.stdout[0]) == 126