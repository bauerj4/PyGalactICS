"""Live progress reporting for DBH campaigns (notebooks and CLI)."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

Stage = str


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 0 or not (seconds < 1e12):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def stream_ntropy_jsonl(
    path: Path,
    *,
    n_steps_total: int,
    stop_event: threading.Event,
    poll_s: float = 0.25,
    heartbeat_s: float = 30.0,
) -> None:
    """
    Tail ``evolution/diagnostics.progress.jsonl`` and print ntropy lines live.

    Works in Jupyter when ``mpirun`` buffers worker stdout; rank 0 flushes each
    diagnostic record to disk as the simulation runs.

    Resets the read offset when the file shrinks (worker truncates on restart).
    Without that reset, stale lines from a prior run print instantly and the
    tailer then ignores all new progress.
    """
    from ntropy.integrators.tiered_progress import format_ntropy_progress_from_dict

    offset = 0
    last_step = -1
    last_line_time = time.monotonic()
    last_heartbeat = 0.0
    while not stop_event.is_set():
        if not path.is_file():
            last_heartbeat = _emit_heartbeat_if_due(
                n_steps_total,
                last_step,
                last_line_time,
                last_heartbeat,
                heartbeat_s,
                waiting_for_start=True,
            )
            time.sleep(poll_s)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(poll_s)
            continue
        if size < offset:
            offset = 0
            last_step = -1
            last_line_time = time.monotonic()
        if size <= offset:
            last_heartbeat = _emit_heartbeat_if_due(
                n_steps_total,
                last_step,
                last_line_time,
                last_heartbeat,
                heartbeat_s,
                waiting_for_start=size == 0 and last_step < 0,
            )
            time.sleep(poll_s)
            continue
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = int(rec.get("step", -1))
            if step <= last_step:
                continue
            last_step = step
            last_line_time = time.monotonic()
            print(
                format_ntropy_progress_from_dict(rec, n_steps_total=n_steps_total),
                flush=True,
            )
        time.sleep(poll_s)


def _emit_heartbeat_if_due(
    n_steps_total: int,
    last_step: int,
    last_line_time: float,
    last_heartbeat: float,
    heartbeat_s: float,
    *,
    waiting_for_start: bool,
) -> float:
    now = time.monotonic()
    if now - last_line_time < heartbeat_s or now - last_heartbeat < heartbeat_s:
        return last_heartbeat
    if waiting_for_start and last_step < 0:
        print("[ntropy] waiting for MPI worker to start…", flush=True)
    elif last_step >= 0:
        waiting = int(now - last_line_time)
        print(
            f"[ntropy] still computing step {last_step + 1}/{n_steps_total} "
            f"({waiting}s since last update)…",
            flush=True,
        )
    return now


@dataclass
class CampaignProgress:
    """
    Timestamped campaign logger with optional tqdm bars.

    Parameters
    ----------
    enabled : bool
        When ``False``, all methods are no-ops.
    stream_legacy : bool
        Stream legacy Fortran **stderr** live (stdout is always captured so
        particle samplers do not flood the notebook).
    """

    enabled: bool = True
    stream_legacy: bool = True
    campaign_name: str = ""
    work_root: str = ""
    stages: list[Stage] = field(default_factory=list)
    _model_index: int = 0
    _model_total: int = 0
    _model_label: str = ""
    _model_hash: str = ""
    _stage: Stage | None = None
    _stage_t0: float = 0.0
    _campaign_t0: float = field(default_factory=time.perf_counter)
    _completed_models: int = 0
    _model_durations: list[float] = field(default_factory=list)
    _model_bar: Any = None
    _stage_bar: Any = None

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"[{_ts()}] {message}", flush=True)

    def banner(
        self,
        *,
        name: str,
        work_root: str,
        stages: list[Stage],
        n_models: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.campaign_name = name
        self.work_root = work_root
        self.stages = stages
        self._model_total = n_models
        self._campaign_t0 = time.perf_counter()
        self.log("=" * 72)
        self.log(f"Campaign {name!r} → {work_root}")
        self.log(f"Stages: {', '.join(stages)} | models: {n_models}")
        if extra:
            for key, value in extra.items():
                self.log(f"  {key}: {value}")
        self.log("=" * 72)

    def start_model(self, index: int, total: int, label: str, model_hash: str, path: str) -> None:
        if not self.enabled:
            return
        self._model_index = index
        self._model_total = total
        self._model_label = label
        self._model_hash = model_hash
        eta = self._eta_models_remaining()
        eta_s = f" | ETA all models ~{_fmt_seconds(eta)}" if eta is not None else ""
        self.log("-" * 72)
        self.log(
            f"Model [{index}/{total}] {label}  hash={model_hash[:12]}…{eta_s}\n"
            f"  dir: {path}"
        )
        self._close_bars()
        self._model_bar = self._make_bar(
            total=len(self.stages),
            desc=f"{label[:28]}",
            unit="stage",
            position=0,
        )

    def skip_model(self, label: str, reason: str) -> None:
        self.log(f"Skip {label}: {reason}")

    def start_stage(self, stage: Stage, *, detail: str = "") -> None:
        if not self.enabled:
            return
        self._stage = stage
        self._stage_t0 = time.perf_counter()
        msg = f"▶ {stage}"
        if detail:
            msg += f" — {detail}"
        self.log(msg)
        if self._stage_bar is not None:
            self._stage_bar.close()
            self._stage_bar = None

    def skip_stage(self, stage: Stage, *, reason: str) -> None:
        if not self.enabled:
            return
        self.log(f"⊙ {stage} skipped ({reason})")
        if self._model_bar is not None:
            self._model_bar.update(1)

    def legacy_command(self, binary: str, cwd: str) -> None:
        self.log(f"  legacy: {binary}  (cwd={cwd})")

    def mpi_launch(self, n_ranks: int, label: str) -> None:
        self.log(f"  MPI evolve: mpirun -n {n_ranks} — {label} (live output below)")

    def end_stage(self, stage: Stage, elapsed: float, summary: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        parts = [f"✓ {stage} finished in {_fmt_seconds(elapsed)}"]
        if summary:
            for key in (
                "rtidal",
                "solve_seconds",
                "sample_seconds",
                "evolve_seconds",
                "dE_over_E0",
                "n_energies",
                "n_ranks",
                "mpi",
            ):
                if key in summary and summary[key] is not None:
                    val = summary[key]
                    if isinstance(val, float):
                        parts.append(f"{key}={val:.4g}")
                    else:
                        parts.append(f"{key}={val}")
        self.log("  " + " | ".join(parts))
        if self._model_bar is not None:
            self._model_bar.update(1)
        self._stage = None

    def end_model(self, elapsed: float) -> None:
        if not self.enabled:
            return
        self._completed_models += 1
        self._model_durations.append(elapsed)
        self.log(f"Model done in {_fmt_seconds(elapsed)}")
        self._close_bars()

    def finish(self) -> None:
        if not self.enabled:
            return
        total = time.perf_counter() - self._campaign_t0
        self._close_bars()
        self.log("=" * 72)
        self.log(
            f"Campaign complete: {self._completed_models} model(s) in {_fmt_seconds(total)}"
        )
        self.log("=" * 72)

    def _eta_models_remaining(self) -> float | None:
        if not self._model_durations or self._model_total <= self._model_index:
            return None
        avg = sum(self._model_durations) / len(self._model_durations)
        remaining = self._model_total - self._model_index + 1
        return avg * remaining

    def _make_bar(self, *, total: int, desc: str, unit: str, position: int = 0) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        try:
            return tqdm(
                total=total,
                desc=desc,
                unit=unit,
                leave=True,
                position=position,
                file=sys.stdout,
                mininterval=0.5,
            )
        except Exception:
            return None

    def _close_bars(self) -> None:
        for bar in (self._model_bar, self._stage_bar):
            if bar is not None:
                bar.close()
        self._model_bar = None
        self._stage_bar = None

    @property
    def stream_output(self) -> bool:
        return self.enabled and self.stream_legacy
