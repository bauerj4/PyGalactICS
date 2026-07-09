"""Helpers for launching MPI benchmark workers from notebooks or scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def mpirun_env(venv_bin: Path | None = None) -> dict[str, str]:
    """
    Environment variables for mpirun subprocesses (WSL / Jupyter safe).

    Parameters
    ----------
    venv_bin : Path, optional
        When set, prepend this directory to ``PATH`` so workers and
        ``orted`` find the same Python/OpenMPI helpers as the notebook kernel.
    """
    env = os.environ.copy()
    if venv_bin is not None:
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    env.setdefault("OMPI_MCA_btl_base_warn_component_unused", "0")
    return env


def mpirun_command(
    n_ranks: int,
    worker_args: list[str],
    *,
    module: str = "ntropy.benchmark.mpi_force_bench",
    python: str | None = None,
) -> list[str]:
    """
    Build an ``mpirun`` command line for an ntropy benchmark worker module.

    Parameters
    ----------
    n_ranks : int
        MPI rank count.
    worker_args : list of str
        Arguments passed to the worker after the module name.
    module : str
        Python module invoked as ``python -m <module>``.
    python : str, optional
        Python executable (defaults to ``sys.executable``).

    Returns
    -------
    cmd : list of str
    """
    py = python or sys.executable
    return [
        "mpirun",
        "--bind-to",
        "none",
        "--mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-n",
        str(n_ranks),
        py,
        "-m",
        module,
        *worker_args,
    ]


def _run_mpirun_worker(
    n_ranks: int,
    worker_args: list[str],
    *,
    module: str,
    cwd: Path | str,
    python: str | None = None,
    venv_bin: Path | None = None,
    timeout_s: float | None = 600.0,
    label: str = "mpirun worker",
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = mpirun_command(n_ranks, worker_args, module=module, python=python)
    env = mpirun_env(venv_bin)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=capture_output,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"{label} failed (exit {result.returncode}, ranks={n_ranks})\n"
            f"command: {' '.join(cmd)}"
        )
        if capture_output:
            msg += f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        else:
            msg += "\n(see live output above)"
        raise RuntimeError(msg)
    return result


def run_mpirun_benchmark(
    n_ranks: int,
    worker_args: list[str],
    *,
    cwd: Path | str,
    python: str | None = None,
    venv_bin: Path | None = None,
    timeout_s: float | None = 600.0,
) -> subprocess.CompletedProcess[str]:
    """
    Run the MPI force-benchmark worker and return the completed process.

    Raises
    ------
    RuntimeError
        When ``mpirun`` exits non-zero; stderr/stdout are included in the message.
    """
    return _run_mpirun_worker(
        n_ranks,
        worker_args,
        module="ntropy.benchmark.mpi_force_bench",
        cwd=cwd,
        python=python,
        venv_bin=venv_bin,
        timeout_s=timeout_s,
        label="mpirun force benchmark",
    )


def run_mpirun_simulation(
    n_ranks: int,
    worker_args: list[str],
    *,
    cwd: Path | str,
    python: str | None = None,
    venv_bin: Path | None = None,
    timeout_s: float | None = 86400.0,
) -> subprocess.CompletedProcess[str]:
    """
    Run the MPI simulation worker (energy-drift notebook runs).

    Progress bars stream to the terminal (``capture_output=False``).

    Raises
    ------
    RuntimeError
        When ``mpirun`` exits non-zero; stderr/stdout are included in the message.
    """
    return _run_mpirun_worker(
        n_ranks,
        worker_args,
        module="ntropy.benchmark.mpi_simulation_worker",
        cwd=cwd,
        python=python,
        venv_bin=venv_bin,
        timeout_s=timeout_s,
        label="mpirun simulation",
        capture_output=False,
    )
