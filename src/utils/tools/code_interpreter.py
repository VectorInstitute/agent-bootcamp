"""Code interpreter tool."""

from pathlib import Path
from typing import Sequence

from e2b_code_interpreter import AsyncSandbox
from pydantic import BaseModel

from ..async_utils import gather_with_progress


class _CodeInterpreterOutputError(BaseModel):
    """Error from code interpreter."""

    name: str
    value: str
    traceback: str


class CodeInterpreterOutput(BaseModel):
    """Output from code interpreter."""

    stdout: list[str]
    stderr: list[str]
    error: _CodeInterpreterOutputError | None = None

    def __init__(self, stdout: list[str], stderr: list[str], **kwargs):
        """Split lines in stdout and stderr."""
        stdout_processed = []
        for _line in stdout:
            stdout_processed.extend(_line.splitlines())

        stderr_processed = []
        for _line in stderr:
            stderr_processed.extend(_line.splitlines())

        super().__init__(stdout=stdout_processed, stderr=stderr_processed, **kwargs)


async def _upload_file(sandbox: "AsyncSandbox", local_path: "str | Path") -> str:
    """Upload file to sandbox.

    Returns
    -------
        str, denoting the remote path.
    """
    path = Path(local_path)
    remote_path = f"{path.name}"
    with open(local_path, "rb") as file:
        await sandbox.files.write(remote_path, file)

    return remote_path


async def _upload_files(
    sandbox: "AsyncSandbox", paths: Sequence[Path | str]
) -> list[str]:
    """Upload files to the sandbox.

    Parameters
    ----------
        paths: Sequence[pathlib.Path | str]
            Files to upload to the sandbox.

    Returns
    -------
        list[str]
        List of remote paths, one per file.
    """
    if not paths:
        return []

    file_upload_coros = [_upload_file(sandbox, _path) for _path in paths]
    remote_paths = await gather_with_progress(
        file_upload_coros, description=f"Uploading {len(paths)} to sandbox"
    )
    return list(remote_paths)


class CodeInterpreter:
    """Code Interpreter tool for the agent."""

    def __init__(
        self,
        local_files: "Sequence[Path | str]| None" = None,
        timeout_seconds: int = 30,
    ):
        """Configure your Code Interpreter session.

        Note that the sandbox is not persistent, and each run_code will
        execute in a fresh sandbox! (e.g., variables need to be re-declared each time.)

        Parameters
        ----------
            local_files : list[pathlib.Path | str] | None
                Optionally, specify a list of local files (as paths)
                to upload to sandbox working directory.
            timeout_seconds : int
                Limit executions to this duration.
        """
        self.timeout_seconds = timeout_seconds
        self.local_files = local_files if local_files else []

    async def run_code(self, code: str) -> str:
        """Run the given Python code in a sandbox environment.

        Parameters
        ----------
            code : str
                Python logic to execute.
        """
        sbx = await AsyncSandbox.create(timeout=self.timeout_seconds)
        await _upload_files(sbx, self.local_files)

        try:
            result = await sbx.run_code(
                code, on_error=lambda error: print(error.traceback)
            )
            response = CodeInterpreterOutput.model_validate_json(result.logs.to_json())

            error = result.error
            if error is not None:
                response.error = _CodeInterpreterOutputError.model_validate_json(
                    error.to_json()
                )

            return response.model_dump_json()
        finally:
            await sbx.kill()
