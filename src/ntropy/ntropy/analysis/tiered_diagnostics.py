"""ntropy per-substep timestep-bin and activity diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState
from ntropy.units import code_time_to_gyr


@dataclass
class TypeBinDiagnostics:
    """
    Timestep-bin summary for one particle type on a recorded substep.

    Attributes
    ----------
    n_particles : int
        Particle count for this type.
    mean_bin : float
        Mean integer bin index.
    min_bin, max_bin : int
        Bin range.
    bin_counts : list of int
        Histogram over global bin indices ``0 … max_bin``.
    """

    n_particles: int
    mean_bin: float
    min_bin: int
    max_bin: int
    bin_counts: list[int]


@dataclass
class StepDiagnostics:
    """
    Aggregate diagnostics for one fine substep (tiered sync-point).

    Attributes
    ----------
    step : int
        Fine substep index (0 = initial).
    t_code : float
        Elapsed time [code units].
    t_gyr : float
        Elapsed time [Gyr].
    energy : float
        Total energy [code units].
    dE_over_E0 : float
        ``|E - E0| / |E0|`` relative to the run's initial energy.
    n_active : int
        Particles kicked on this substep.
    n_particles : int
        Total particle count.
    active_fraction : float
        ``n_active / n_particles``.
    bin_counts : list of int
        Global histogram, length ``max_bin + 1``.
    by_type : dict
        Maps type label → per-type bin summary dict.
    mean_accel : float or None
        Mean acceleration magnitude over all particles (when recorded).
    """

    step: int
    t_code: float
    t_gyr: float
    energy: float
    dE_over_E0: float
    n_active: int
    n_particles: int
    active_fraction: float
    bin_counts: list[int]
    by_type: dict[str, dict[str, Any]]
    mean_accel: float | None = None


@dataclass
class TieredDiagnosticsLog:
    """
    Full diagnostic history for a tiered leapfrog run.

    Attributes
    ----------
    dt_base : float
        Fine substep size [code units].
    max_bin : int
        Coarsest bin index.
    e0 : float
        Initial total energy.
    steps : list of StepDiagnostics
        Recorded substeps (empty when ``jsonl_path`` streaming is enabled).
    jsonl_path : Path or None
        When set, each record is appended here instead of ``steps``.
    n_recorded : int
        Number of substeps written (RAM or stream).
    """

    dt_base: float
    max_bin: int
    e0: float
    steps: list[StepDiagnostics] = field(default_factory=list)
    jsonl_path: Path | None = None
    n_recorded: int = 0

    def record(self, step_diag: StepDiagnostics) -> None:
        """Store one substep record in RAM or on the streaming JSONL path."""
        if self.jsonl_path is not None:
            append_diagnostics_jsonl(self.jsonl_path, step_diag)
        else:
            self.steps.append(step_diag)
        self.n_recorded += 1


def _bin_histogram(bins: np.ndarray, max_bin: int) -> np.ndarray:
    counts = np.bincount(bins.astype(int), minlength=max_bin + 1)
    return counts[: max_bin + 1]


def collect_step_diagnostics(
    *,
    step: int,
    bins: np.ndarray,
    type_id: np.ndarray,
    registry: TypeRegistry,
    energy: float,
    e0: float,
    n_active: int,
    dt_base: float,
    max_bin: int,
    acc: np.ndarray | None = None,
) -> StepDiagnostics:
    """
    Build one substep diagnostic record from the current particle state.

    Parameters
    ----------
    step : int
        Fine substep index.
    bins : ndarray, shape (N,)
        Current per-particle timestep bins.
    type_id : ndarray, shape (N,)
        Particle type ids.
    registry : TypeRegistry
        Type metadata.
    energy : float
        Total energy at this substep.
    e0 : float
        Initial energy for drift normalization.
    n_active : int
        Number of active particles this substep.
    dt_base : float
        Base timestep [code units].
    max_bin : int
        Maximum bin index.
    acc : ndarray, shape (N, 3), optional
        Accelerations for mean |a| diagnostic.

    Returns
    -------
    record : StepDiagnostics
    """
    n = len(bins)
    bin_counts = _bin_histogram(bins, max_bin)
    t_code = step * dt_base
    by_type: dict[str, dict[str, Any]] = {}

    for label, spec in registry.types.items():
        mask = type_id == spec.id
        if not np.any(mask):
            continue
        tb = bins[mask]
        tcounts = _bin_histogram(tb, max_bin)
        by_type[label] = {
            "n_particles": int(mask.sum()),
            "mean_bin": float(tb.mean()),
            "min_bin": int(tb.min()),
            "max_bin": int(tb.max()),
            "bin_counts": tcounts.tolist(),
        }

    mean_accel = None
    if acc is not None:
        mean_accel = float(np.linalg.norm(acc, axis=1).mean())

    return StepDiagnostics(
        step=step,
        t_code=t_code,
        t_gyr=code_time_to_gyr(t_code),
        energy=energy,
        dE_over_E0=abs(energy - e0) / max(abs(e0), 1e-30),
        n_active=n_active,
        n_particles=n,
        active_fraction=n_active / max(n, 1),
        bin_counts=bin_counts.tolist(),
        by_type=by_type,
        mean_accel=mean_accel,
    )


def write_particle_bin_dump(
    path: Path,
    state: ParticleState,
    *,
    acc: np.ndarray | None = None,
) -> None:
    """
    Write per-particle bin and kinematic diagnostics to ``.npz``.

    Parameters
    ----------
    path : Path
        Output ``.npz`` path.
    state : ParticleState
        Current snapshot (must include ``timestep_bin``).
    acc : ndarray, optional
        Acceleration array for ``accel_mag`` column.
    """
    if state.timestep_bin is None:
        raise ValueError("particle bin dump requires state.timestep_bin")
    payload: dict[str, np.ndarray] = {
        "timestep_bin": state.timestep_bin.astype(np.int32),
        "type_id": state.type_id.astype(np.int32) if state.type_id is not None else np.zeros(state.n, dtype=np.int32),
        "mass": state.mass,
        "eps": state.eps,
        "pos": state.pos,
        "vel": state.vel,
    }
    if state.tags is not None:
        payload["tags"] = np.asarray(state.tags).astype("U16")
    if acc is not None:
        payload["accel_mag"] = np.linalg.norm(acc, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def append_diagnostics_jsonl(path: Path, record: StepDiagnostics) -> None:
    """Append one diagnostics record as a JSON line (streaming evolve logs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def _read_diagnostics_jsonl(jsonl_path: Path) -> list[StepDiagnostics]:
    steps: list[StepDiagnostics] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(StepDiagnostics(**json.loads(line)))
    return steps


def write_diagnostics_log(log: TieredDiagnosticsLog, output_dir: Path) -> Path:
    """
    Write aggregate diagnostics to ``diagnostics.jsonl`` and ``diagnostics.csv``.

    Returns
    -------
    jsonl_path : Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log.jsonl_path or (output_dir / "diagnostics.jsonl")
    if log.jsonl_path is None:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in log.steps:
                f.write(json.dumps(asdict(rec)) + "\n")

    steps = log.steps if log.steps else _read_diagnostics_jsonl(jsonl_path)

    csv_path = output_dir / "diagnostics.csv"
    if steps:
        # Flatten per-type mean_bin into columns for easy pandas load
        type_labels = sorted(
            {k for rec in steps for k in rec.by_type}
        )
        fieldnames = [
            "step", "t_code", "t_gyr", "energy", "dE_over_E0",
            "n_active", "n_particles", "active_fraction", "mean_accel",
            *[f"bin_{b}" for b in range(log.max_bin + 1)],
            *[f"{lbl}_mean_bin" for lbl in type_labels],
            *[f"{lbl}_n" for lbl in type_labels],
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in steps:
                row: dict[str, Any] = {
                    "step": rec.step,
                    "t_code": rec.t_code,
                    "t_gyr": rec.t_gyr,
                    "energy": rec.energy,
                    "dE_over_E0": rec.dE_over_E0,
                    "n_active": rec.n_active,
                    "n_particles": rec.n_particles,
                    "active_fraction": rec.active_fraction,
                    "mean_accel": rec.mean_accel,
                }
                for b, c in enumerate(rec.bin_counts):
                    row[f"bin_{b}"] = c
                for lbl in type_labels:
                    info = rec.by_type.get(lbl, {})
                    row[f"{lbl}_mean_bin"] = info.get("mean_bin")
                    row[f"{lbl}_n"] = info.get("n_particles")
                writer.writerow(row)

    meta_path = output_dir / "diagnostics_meta.json"
    meta_path.write_text(
        json.dumps({"dt_base": log.dt_base, "max_bin": log.max_bin, "e0": log.e0}, indent=2)
    )
    return jsonl_path


def load_diagnostics_csv(path: Path) -> list[dict[str, Any]]:
    """Load ``diagnostics.csv`` as a list of row dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_diagnostics_log(output_dir: Path) -> TieredDiagnosticsLog:
    """Load diagnostics written by :func:`write_diagnostics_log`."""
    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "diagnostics_meta.json").read_text())
    steps: list[StepDiagnostics] = []
    jsonl = output_dir / "diagnostics.jsonl"
    for line in jsonl.read_text().splitlines():
        if line.strip():
            steps.append(StepDiagnostics(**json.loads(line)))
    return TieredDiagnosticsLog(
        dt_base=meta["dt_base"],
        max_bin=meta["max_bin"],
        e0=meta["e0"],
        steps=steps,
    )


def diagnostics_dataframe(path: Path):
    """
    Load ``diagnostics.csv`` as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path : Path
        Directory containing ``diagnostics.csv``, or path to the CSV file.

    Returns
    -------
    DataFrame
    """
    import pandas as pd

    path = Path(path)
    csv_path = path / "diagnostics.csv" if path.is_dir() else path
    return pd.read_csv(csv_path)


def plot_tiered_diagnostics(df, *, max_bin: int | None = None, ax=None):
    """
    Four-panel ntropy tiered diagnostic figure from a diagnostics DataFrame.

    Panels: |ΔE/E₀|, active fraction, bin-count heatmap, per-type mean bin.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`diagnostics_dataframe`.
    max_bin : int, optional
        Inferred from ``bin_*`` columns when ``None``.
    ax : array of Axes, optional
        Shape ``(2, 2)`` axes to draw on.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    bin_cols = [c for c in df.columns if c.startswith("bin_")]
    if max_bin is None:
        max_bin = len(bin_cols) - 1
    type_mean_cols = [c for c in df.columns if c.endswith("_mean_bin")]

    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    else:
        fig = ax.ravel()[0].figure

    ax[0, 0].semilogy(df["t_gyr"], df["dE_over_E0"], "k-", lw=1)
    ax[0, 0].set_xlabel("t [Gyr]")
    ax[0, 0].set_ylabel("|ΔE/E₀|")
    ax[0, 0].set_title("Energy drift")

    ax[0, 1].plot(df["t_gyr"], df["active_fraction"], "C0-", lw=1)
    ax[0, 1].set_xlabel("t [Gyr]")
    ax[0, 1].set_ylabel("active fraction")
    ax[0, 1].set_title("Kicked particles per fine step")

    heat = df[bin_cols].to_numpy().T
    im = ax[1, 0].imshow(
        heat,
        aspect="auto",
        origin="lower",
        extent=[df["t_gyr"].iloc[0], df["t_gyr"].iloc[-1], -0.5, max_bin + 0.5],
        cmap="viridis",
    )
    ax[1, 0].set_xlabel("t [Gyr]")
    ax[1, 0].set_ylabel("timestep bin")
    ax[1, 0].set_title("Global bin histogram")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, label="N particles")

    for col in type_mean_cols:
        label = col.replace("_mean_bin", "")
        ax[1, 1].plot(df["t_gyr"], df[col], label=label, lw=1)
    ax[1, 1].set_xlabel("t [Gyr]")
    ax[1, 1].set_ylabel("mean bin")
    ax[1, 1].set_title("Per-type mean timestep bin")
    if type_mean_cols:
        ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig

