"""Live ntropy progress lines for tiered integration."""

from __future__ import annotations

import json
import os
import sys
from typing import TextIO

from ntropy.analysis.tiered_diagnostics import StepDiagnostics


def format_ntropy_progress_line(
    record: StepDiagnostics,
    *,
    n_steps_total: int,
) -> str:
    """
    One-line sync-point summary (tiered bin / activity).

    Example::

        step  12/409 | t=0.0029 Gyr | active 49.1% (98234) | dE/E0=3.2e-05 |
        bins 1200,45000,98000,52000,3800,0,0 | disk μb=1.2 halo μb=2.8
    """
    type_bits = []
    for label, info in sorted(record.by_type.items()):
        type_bits.append(f"{label} μb={info['mean_bin']:.1f}")
    type_str = " ".join(type_bits) if type_bits else "—"
    bins_str = ",".join(str(c) for c in record.bin_counts)
    return (
        f"step {record.step:4d}/{n_steps_total} | "
        f"t={record.t_gyr:.4f} Gyr | "
        f"active {record.active_fraction * 100:5.1f}% ({record.n_active}) | "
        f"dE/E0={record.dE_over_E0:.2e} | "
        f"bins {bins_str} | {type_str}"
    )


def format_ntropy_progress_from_dict(rec: dict, *, n_steps_total: int) -> str:
    """Format a JSONL record (see ``ntropy_progress_dict``) as a progress line."""
    type_bits = []
    for label, info in sorted(rec.get("by_type", {}).items()):
        type_bits.append(f"{label} μb={info['mean_bin']:.1f}")
    type_str = " ".join(type_bits) if type_bits else "—"
    bins_str = ",".join(str(c) for c in rec.get("bin_counts", []))
    active_frac = float(rec.get("active_fraction", 0.0))
    return (
        f"step {int(rec['step']):4d}/{n_steps_total} | "
        f"t={float(rec['t_gyr']):.4f} Gyr | "
        f"active {active_frac * 100:5.1f}% ({int(rec.get('n_active', 0))}) | "
        f"dE/E0={float(rec.get('dE_over_E0', 0.0)):.2e} | "
        f"bins {bins_str} | {type_str}"
    )


def ntropy_progress_dict(record: StepDiagnostics) -> dict:
    """JSON-serializable snapshot for ``diagnostics.progress.jsonl``."""
    return {
        "step": record.step,
        "t_gyr": record.t_gyr,
        "t_code": record.t_code,
        "n_active": record.n_active,
        "n_particles": record.n_particles,
        "active_fraction": record.active_fraction,
        "dE_over_E0": record.dE_over_E0,
        "bin_counts": record.bin_counts,
        "by_type": record.by_type,
        "mean_accel": record.mean_accel,
    }


class NtropyProgressReporter:
    """
    Print flushed ntropy progress lines during tiered evolution.

    Works in notebooks and ``mpirun`` subprocesses where ``tqdm`` is often
    invisible or buffered.
    """

    def __init__(
        self,
        *,
        n_steps_total: int,
        n_particles: int,
        end_time_gyr: float,
        label: str | None = None,
        print_every: int = 1,
        stream: TextIO | None = None,
        progress_jsonl: str | None = None,
    ) -> None:
        self.n_steps_total = n_steps_total
        self.n_particles = n_particles
        self.end_time_gyr = end_time_gyr
        self.label = label or "tiered evolve"
        self.print_every = max(1, print_every)
        self.stream = stream or sys.stderr
        self.progress_jsonl = progress_jsonl
        self._jsonl_only = os.environ.get("NTROPY_PROGRESS_JSONL_ONLY") == "1"
        self._last_printed_step = -1

    def banner(self, *, dt_base: float) -> None:
        print(
            f"[ntropy] {self.label}: N={self.n_particles:,}, "
            f"{self.n_steps_total} fine substeps → {self.end_time_gyr:.4g} Gyr "
            f"(dt_base={dt_base:.5f})",
            file=self.stream,
            flush=True,
        )

    def note(self, message: str) -> None:
        print(f"[ntropy] {message}", file=self.stream, flush=True)

    def update(self, record: StepDiagnostics) -> None:
        if record.step % self.print_every != 0 and record.step != self.n_steps_total:
            return
        if record.step == self._last_printed_step:
            return
        self._last_printed_step = record.step
        if self.progress_jsonl is not None:
            with open(self.progress_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(ntropy_progress_dict(record)) + "\n")
                f.flush()
        if not self._jsonl_only:
            line = format_ntropy_progress_line(record, n_steps_total=self.n_steps_total)
            print(line, file=self.stream, flush=True)

    def finish(self) -> None:
        print(
            f"[ntropy] {self.label}: finished {self.n_steps_total} substeps",
            file=self.stream,
            flush=True,
        )
