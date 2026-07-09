"""Subprocess runner for legacy GalactICS executables."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from galacticsics.legacy.paths import require_binary


class LegacyRunError(RuntimeError):
    """Raised when a legacy executable exits with non-zero status."""


@dataclass
class LegacyRunResult:
    """
    Result of a legacy subprocess invocation.

    Attributes
    ----------
    command : list[str]
        argv passed to the subprocess.
    returncode : int
        Process exit code.
    stdout : str
        Captured standard output (may be empty).
    stderr : str
        Captured standard error (may be empty).
    cwd : Path
        Working directory used for the run.
    """

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    cwd: Path


class LegacyRunner:
    """
    Run legacy GalactICS binaries in an isolated working directory.

    The runner never modifies the legacy source tree. It executes pre-built
    binaries from ``legacy/bin/`` with a caller-supplied ``cwd`` (typically a
    temporary directory containing generated ``in.*`` input files).

    Parameters
    ----------
    cwd : path-like
        Working directory for subprocess execution. Input files (``in.dbh``,
        ``in.gendenspsi``, etc.) must already exist in this directory unless
        passed via ``stdin``.

    Examples
    --------
    >>> from pathlib import Path
    >>> from galacticsics.legacy.runner import LegacyRunner
    >>> runner = LegacyRunner("/tmp/my_model")
    >>> result = runner.run("dbh", stdin_path=Path("/tmp/my_model/in.dbh"))
    >>> result.returncode
    0
    """

    def __init__(self, cwd: Path | str) -> None:
        self.cwd = Path(cwd).resolve()

    def run(
        self,
        binary: str,
        *,
        args: Sequence[str] = (),
        stdin_path: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> LegacyRunResult:
        """
        Execute a legacy binary.

        Parameters
        ----------
        binary : str
            Executable name in ``legacy/bin/`` (e.g. ``"dbh"``).
        args : sequence of str, optional
            Additional command-line arguments after the executable path.
        stdin_path : Path, optional
            If given, file contents are piped to standard input (equivalent
            to ``dbh < in.dbh`` in the original workflow).
        env : mapping, optional
            Extra environment variables merged onto ``os.environ``.
        timeout : float, optional
            Maximum wall time in seconds.

        Returns
        -------
        LegacyRunResult
            Captured process output and metadata.

        Raises
        ------
        LegacyRunError
            If the process exits with a non-zero return code.
        FileNotFoundError
            If the binary or ``stdin_path`` is missing.
        """
        exe = require_binary(binary)
        cmd = [str(exe), *args]
        stdin_file = None
        try:
            if stdin_path is not None:
                stdin_file = open(stdin_path, "rb")
            proc = subprocess.run(
                cmd,
                cwd=self.cwd,
                stdin=stdin_file,
                capture_output=True,
                env=None if env is None else {**__import__("os").environ, **dict(env)},
                timeout=timeout,
                check=False,
            )
        finally:
            if stdin_file is not None:
                stdin_file.close()

        result = LegacyRunResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout.decode(errors="replace"),
            stderr=proc.stderr.decode(errors="replace"),
            cwd=self.cwd,
        )
        if proc.returncode != 0:
            raise LegacyRunError(
                f"{binary} failed (code {proc.returncode}) in {self.cwd}\n"
                f"stderr:\n{result.stderr[:2000]}"
            )
        return result
