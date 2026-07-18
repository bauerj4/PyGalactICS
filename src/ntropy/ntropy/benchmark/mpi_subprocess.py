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
    env["PYTHONUNBUFFERED"] = "1"
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
    # Invoke main() via -c rather than ``python -m``.  The editable/namespace
    # layout of ntropy can leave the module already imported when ``-m`` runs,
    # so the ``__main__`` guard never fires and the worker exits 0 with no work.
    bootstrap = (
        "import sys; "
        f"from {module} import main; "
        "raise SystemExit(main(sys.argv))"
    )
    return [
        "mpirun",
        "--oversubscribe",
        "--bind-to",
        "none",
        "--mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-n",
        str(n_ranks),
        py,
        "-c",
        bootstrap,
        *worker_args,
    ]


def _stream_process_output(
    proc: subprocess.Popen[str],
    *,
    log_path: Path | None = None,
) -> tuple[str, str]:
    """
    Read merged stdout/stderr line-by-line; optionally tee to ``log_path``.

    Lines are echoed to this process's stdout as they arrive (live progress
    in notebooks) and appended to ``log_path`` when given.

    Parameters
    ----------
    proc : subprocess.Popen
        Running process opened with ``stdout=PIPE, stderr=STDOUT, text=True``.
    log_path : Path, optional
        File to tee the merged stream into (parent dirs are created).

    Returns
    -------
    stdout, stderr : str
        Full merged output and an empty string (stderr is folded into
        stdout by the caller's pipe setup).
    """
    stdout_chunks: list[str] = []
    assert proc.stdout is not None
    log_file = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
        for line in proc.stdout:
            stdout_chunks.append(line)
            print(line, end="", flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        proc.wait()
    finally:
        if log_file is not None:
            log_file.close()
    merged = "".join(stdout_chunks)
    return merged, ""


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
    extra_env: dict[str, str] | None = None,
    log_path: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Launch an ntropy worker module under ``mpirun`` and wait for it.

    Applies the WSL/Jupyter-safe environment from :func:`mpirun_env` and
    guards against OpenMP oversubscription: when ``OMP_NUM_THREADS`` is not
    already set (directly or via ``extra_env``), it defaults to
    ``cores // n_ranks`` so ranks × threads never exceeds the machine.

    Parameters
    ----------
    n_ranks : int
        MPI rank count.
    worker_args : list of str
        Arguments forwarded to the worker's ``main(argv)``.
    module : str
        Worker module; its ``main`` is bootstrapped via ``python -c``
        (see :func:`mpirun_command` for why ``-m`` is avoided).
    cwd : path
        Working directory for the workers.
    python : str, optional
        Python executable (defaults to ``sys.executable``).
    venv_bin : Path, optional
        Prepended to ``PATH`` so workers resolve the notebook's venv.
    timeout_s : float, optional
        Kill the run after this many seconds.
    label : str
        Human-readable name used in error messages.
    capture_output : bool
        True: capture and return output silently. False: stream lines
        live (long simulations).
    extra_env : dict, optional
        Extra environment variables (e.g. ``OMP_NUM_THREADS`` for hybrid
        MPI × OpenMP sweeps).
    log_path : path, optional
        Tee worker stdout/stderr to this file.

    Returns
    -------
    result : subprocess.CompletedProcess
        Completed process with merged output in ``stdout``.

    Raises
    ------
    RuntimeError
        When ``mpirun`` exits non-zero; the tail of the log (or captured
        output) is included in the message.
    """
    cmd = mpirun_command(n_ranks, worker_args, module=module, python=python)
    env = mpirun_env(venv_bin)
    if extra_env:
        env.update(extra_env)
    # Guard against thread oversubscription: without an explicit setting each
    # rank would spawn one OpenMP thread per core (ranks × cores threads
    # total), which anti-scales badly.  Default to cores / ranks per rank.
    if "OMP_NUM_THREADS" not in env:
        cores = os.cpu_count() or 1
        env["OMP_NUM_THREADS"] = str(max(1, cores // max(1, n_ranks)))
    log = Path(log_path) if log_path is not None else None
    if capture_output:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                (result.stdout or "") + (result.stderr or ""),
                encoding="utf-8",
            )
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            merged_out, _ = _stream_process_output(proc, log_path=log)
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if timeout_s is not None and proc.returncode is None:
            proc.wait(timeout=timeout_s)
        result = subprocess.CompletedProcess(
            proc.args,
            proc.returncode if proc.returncode is not None else proc.wait(),
            merged_out,
            "",
        )
    if result.returncode != 0:
        msg = (
            f"{label} failed (exit {result.returncode}, ranks={n_ranks})\n"
            f"command: {' '.join(cmd)}"
        )
        if log is not None and log.is_file():
            msg += f"\nlog: {log}\n{log.read_text()[-4000:]}"
        elif capture_output:
            msg += f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        else:
            err_log = Path(cwd) / "evolve_error.log"
            if err_log.is_file():
                msg += f"\n\n{err_log.read_text()[-4000:]}"
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
    extra_env: dict[str, str] | None = None,
    log_path: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run the MPI force-benchmark worker and return the completed process.

    Thin wrapper over :func:`_run_mpirun_worker` targeting
    ``ntropy.benchmark.mpi_force_bench`` with output captured (benchmarks
    are quiet; results land in the worker's JSON output file).

    Parameters
    ----------
    n_ranks : int
        MPI rank count.
    worker_args : list of str
        Positional arguments for the force-bench worker:
        ``<state.npz> <method> <theta> <n_repeat> <out.json>
        [mpi_local_trees]``.
    cwd : path
        Working directory for the workers.
    python : str, optional
        Python executable (defaults to ``sys.executable``).
    venv_bin : Path, optional
        Prepended to ``PATH`` for the workers.
    timeout_s : float, optional
        Kill the benchmark after this many seconds (default 600).
    extra_env : dict, optional
        Extra environment variables for the workers (e.g.
        ``{"OMP_NUM_THREADS": "4"}`` for hybrid MPI × OpenMP sweeps).
    log_path : path, optional
        When set, write captured stdout/stderr to this file.

    Returns
    -------
    result : subprocess.CompletedProcess
        Completed process; benchmark timings are in the worker's JSON file.

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
        extra_env=extra_env,
        log_path=log_path,
    )


def run_mpirun_simulation(
    n_ranks: int,
    worker_args: list[str],
    *,
    cwd: Path | str,
    python: str | None = None,
    venv_bin: Path | None = None,
    timeout_s: float | None = 86400.0,
    extra_env: dict[str, str] | None = None,
    log_path: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run the MPI simulation worker (energy-drift notebook and campaign runs).

    Merges worker stdout/stderr and streams lines live (``capture_output=False``)
    so tqdm progress from rank 0 is visible. When ``log_path`` is set, the
    same stream is also written to that file (e.g. the campaign runner's
    ``evolve_mpirun.log``).

    Parameters
    ----------
    n_ranks : int
        MPI rank count.
    worker_args : list of str
        Arguments for ``ntropy.benchmark.mpi_simulation_worker`` — either
        the legacy ``<state.npz> <config.json> <out.json> [final.npz]``
        form or ``--campaign-dir <work_dir> [--label NAME]``.
    cwd : path
        Working directory for the workers.
    python : str, optional
        Python executable (defaults to ``sys.executable``).
    venv_bin : Path, optional
        Prepended to ``PATH`` for the workers.
    timeout_s : float, optional
        Kill the simulation after this many seconds (default 24 h).
    extra_env : dict, optional
        Extra environment variables (e.g. ``OMP_NUM_THREADS`` per rank).
    log_path : path, optional
        Tee the live output stream to this file as well.

    Returns
    -------
    result : subprocess.CompletedProcess
        Completed process with merged output in ``stdout``.

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
        extra_env=extra_env,
        log_path=log_path,
    )
