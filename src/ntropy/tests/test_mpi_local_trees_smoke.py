"""Smoke tests: Gadget-style MPI local trees with galacticsics (Python) ICs.

Samples a spherical NFW halo and a disk+halo galaxy via the **galacticsics**
Python physics backend, checks virial equilibrium, compares LET vs replicated
Barnes–Hut forces under ``mpirun``, and evolves briefly with
``force.mpi_local_trees=True``.  Simulation stdout/stderr is teed to a temp log.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from ntropy.benchmark.mpi_subprocess import mpirun_env, run_mpirun_simulation
from ntropy.forces.bhtree_c import extension_available
from ntropy.integrations.galacticsics import (
    galacticsics_available,
    nfw_halo_model_fast,
    sample_galacticsics_galaxy,
    sample_galacticsics_halo,
)
from ntropy.particles import ParticleState
from ntropy.softening import virial_diagnostic
from ntropy.units import GYR_PER_CODE_TIME

pytestmark = [
    pytest.mark.skipif(not galacticsics_available(), reason="galacticsics unavailable"),
    pytest.mark.skipif(not extension_available(), reason="C Barnes–Hut extension not built"),
    pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available"),
]

_REPO = Path(__file__).resolve().parents[3]
_MPI_WORKER = Path(__file__).resolve().parent / "mpi_local_tree_worker.py"
_STEPS_PER_GYR = 2500
_SMOKE_DURATION_GYR = 0.1
_SMOKE_N_STEPS = int(_SMOKE_DURATION_GYR * _STEPS_PER_GYR)
_SMOKE_DT = (1.0 / _STEPS_PER_GYR) / GYR_PER_CODE_TIME


def _extra_env() -> dict[str, str]:
    env = mpirun_env(Path(sys.executable).resolve().parent)
    env["OMP_NUM_THREADS"] = "1"
    # Prefer the in-tree packages under mpirun.
    env["PYTHONPATH"] = f"{_REPO / 'src'}:{_REPO / 'src' / 'ntropy'}:" + env.get(
        "PYTHONPATH", ""
    )
    return env


def _force_parity(
    state: ParticleState,
    tmp_path: Path,
    *,
    tag: str,
    n_ranks: int = 4,
    theta: float = 0.5,
) -> tuple[float, float]:
    state_path = tmp_path / f"{tag}_force.npz"
    out_path = tmp_path / f"{tag}_force_out.npz"
    log_path = tmp_path / f"{tag}_force.log"
    np.savez(state_path, pos=state.pos, mass=state.mass, eps=state.eps)

    import subprocess

    cmd = [
        "mpirun",
        "--oversubscribe",
        "-n",
        str(n_ranks),
        sys.executable,
        str(_MPI_WORKER),
        str(state_path),
        str(out_path),
        str(theta),
    ]
    result = subprocess.run(
        cmd,
        cwd=_REPO,
        env=_extra_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(
        (result.stdout or "") + (result.stderr or ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"force parity mpirun failed ({result.returncode}); log={log_path}\n"
            f"{log_path.read_text()[-2000:]}"
        )
    with np.load(out_path) as data:
        acc_let = data["acc_let"]
        acc_rep = data["acc_rep"]
    rel = np.linalg.norm(acc_let - acc_rep, axis=1) / np.maximum(
        np.linalg.norm(acc_rep, axis=1), 1e-30
    )
    return float(np.median(rel)), float(rel.max())


def _mpi_evolve(
    state: ParticleState,
    tmp_path: Path,
    *,
    tag: str,
    n_ranks: int = 4,
    max_rel_energy: float = 0.05,
) -> tuple[float, float, Path]:
    state_path = tmp_path / f"{tag}_state.npz"
    config_path = tmp_path / f"{tag}_config.json"
    out_path = tmp_path / f"{tag}_energies.json"
    final_path = tmp_path / f"{tag}_final.npz"
    log_path = tmp_path / f"{tag}_evolve.log"

    np.savez(
        state_path,
        pos=state.pos,
        vel=state.vel,
        mass=state.mass,
        eps=state.eps,
    )
    config_path.write_text(
        json.dumps(
            {
                "label": tag,
                "integrator": {
                    "type": "leapfrog",
                    "order": 2,
                    "dt": _SMOKE_DT,
                    "n_steps": _SMOKE_N_STEPS,
                },
                "force": {
                    "method": "bh_c",
                    "theta": 0.5,
                    "mpi_local_trees": True,
                },
                "parallel": {"enabled": True, "n_workers": n_ranks},
            },
            indent=2,
        )
    )

    run_mpirun_simulation(
        n_ranks,
        [
            str(state_path),
            str(config_path),
            str(out_path),
            str(final_path),
        ],
        cwd=_REPO,
        venv_bin=Path(sys.executable).resolve().parent,
        timeout_s=1800.0,
        extra_env={"OMP_NUM_THREADS": "1", "PYTHONPATH": _extra_env()["PYTHONPATH"]},
        log_path=log_path,
    )
    assert log_path.is_file() and log_path.stat().st_size > 0

    energies = np.asarray(json.loads(out_path.read_text())["energies"], dtype=float)
    rel = np.abs(energies - energies[0]) / max(abs(energies[0]), 1e-30)
    assert rel.max() < max_rel_energy, (
        f"{tag}: max |ΔE/E0|={rel.max():.3e} exceeds {max_rel_energy}; "
        f"see {log_path}"
    )

    with np.load(final_path) as data:
        final = ParticleState.from_arrays(
            data["pos"], data["vel"], data["mass"], data["eps"]
        )
    virial = virial_diagnostic(
        final.pos, final.vel, final.mass, final.eps, rtol=0.55
    )
    return float(rel.max()), float(virial["virial_ratio"]), log_path


@pytest.mark.physics_python
def test_mpi_local_trees_spherical_halo_galacticsics(tmp_path: Path):
    """Spherical NFW from galacticsics Python backend stays near equilibrium under LET MPI."""
    sample = sample_galacticsics_halo(
        nfw_halo_model_fast(),
        n_particles=2048,
        seed=-42,
        work_dir=tmp_path / "halo_work",
        eps=0.04,
        solve=True,
        cleanup=False,
    )
    state = sample.state
    assert state.n == 2048

    virial = virial_diagnostic(
        state.pos, state.vel, state.mass, state.eps, rtol=0.35
    )
    assert 0.65 <= virial["virial_ratio"] <= 1.15, virial
    assert virial["is_virial_equilibrium"]

    med, mx = _force_parity(state, tmp_path, tag="halo")
    assert med < 0.05, f"halo LET vs replicated median rel={med:.3e} max={mx:.3e}"

    dE, final_virial, log_path = _mpi_evolve(
        state, tmp_path, tag="halo", max_rel_energy=0.05
    )
    assert 0.5 < final_virial < 1.4
    # Ensure the evolve log captured tqdm / config output.
    log_text = log_path.read_text()
    assert "halo" in log_text or "leapfrog" in log_text.lower() or "dE" in log_text


@pytest.mark.physics_python
def test_mpi_local_trees_disk_halo_galacticsics(tmp_path: Path):
    """Disk+halo from galacticsics Python backend: force parity + short LET evolve."""
    from galacticsics.models import GalaxyModel
    from galacticsics.sampling.sampler import SampleConfig

    model = GalaxyModel.reference_disk_halo()
    config = SampleConfig(
        n_disk=512,
        n_halo=1536,
        n_bulge=0,
        seed_halo=-7,
        seed_disk=-11,
        run_diskdf=True,
        center=True,
        use_openmp=True,
    )
    # Prefer precomputed reference artifacts when present (fast); else Python solve.
    ref = _REPO / "tests" / "generated" / "reference"
    sample = sample_galacticsics_galaxy(
        model,
        config,
        work_dir=tmp_path / "disk_halo_work",
        artifact_dir=ref if (ref / "dbh.dat").is_file() else None,
        solve=not (ref / "dbh.dat").is_file(),
        eps_by_component={"halo": 0.05, "disk": 0.02},
    )
    state = sample.state
    assert state.n == 512 + 1536
    assert set(sample.components) >= {"halo", "disk"}

    virial = virial_diagnostic(
        state.pos, state.vel, state.mass, state.eps, rtol=0.55
    )
    # Truncated disk+halo from a lowered DF is not perfectly virialised.
    assert 0.5 <= virial["virial_ratio"] <= 1.6, virial

    med, mx = _force_parity(state, tmp_path, tag="disk_halo")
    assert med < 0.08, f"disk+halo LET median rel={med:.3e} max={mx:.3e}"

    dE, final_virial, log_path = _mpi_evolve(
        state, tmp_path, tag="disk_halo", max_rel_energy=0.08
    )
    assert 0.4 < final_virial < 1.8
    assert dE < 0.08
    assert "bh_c" in log_path.read_text() or "leapfrog" in log_path.read_text().lower()
