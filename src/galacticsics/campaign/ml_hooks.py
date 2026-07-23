"""Campaign hooks for ML encoding and training export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.representations.learned import LearnedEncoderConfig, build_encoder
from galacticsics.representations.particle_features import particle_state_to_batch


def encode_run_directory(
    run_dir: Path | str,
    *,
    encoder_config: LearnedEncoderConfig | None = None,
    particles_path: Path | str | None = None,
    output_name: str = "latent.npz",
) -> dict[str, Any]:
    """
    Encode one campaign run directory and write ``latent.npz``.

    Parameters
    ----------
    run_dir : path
        Per-model work directory (contains particles or ``merged.dat``).
    encoder_config : LearnedEncoderConfig, optional
        Defaults to mean-pool baseline.
    particles_path : path, optional
        Explicit particle file; otherwise searches ``merged.dat`` / components.
    output_name : str
        Filename under ``run_dir / ml``.

    Returns
    -------
    summary : dict
        Encoding metadata written alongside the latent array.
    """
    run_dir = Path(run_dir)
    ml_dir = run_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(run_dir, particles_path)
    batch = particle_state_to_batch(state)
    encoder = build_encoder(encoder_config)
    rep = encoder.encode(batch)

    out_path = ml_dir / output_name
    np.savez(
        out_path,
        latent=rep.latent,
        backend=rep.config.backend.value,
        trained=rep.metadata.trained if rep.metadata else False,
        preprocessing=rep.metadata.preprocessing if rep.metadata else {},
    )

    summary = {
        "latent_path": str(out_path),
        "latent_dim": int(rep.latent.shape[0]),
        "backend": rep.config.backend.value,
        "trained": rep.metadata.trained if rep.metadata else False,
        "n_particles": batch.n_particles,
    }
    (ml_dir / "encode_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def encode_evolution_checkpoints(
    run_dir: Path | str,
    *,
    encoder_config: LearnedEncoderConfig | None = None,
    particle_glob: str = "evolution/particles/step_*.npz",
) -> list[dict[str, Any]]:
    """
    Encode particle dumps written during tiered evolution.

    Parameters
    ----------
    run_dir : path
        Campaign run directory.
    encoder_config : LearnedEncoderConfig, optional
    particle_glob : str
        Glob relative to ``run_dir`` for per-step ``.npz`` dumps.

    Returns
    -------
    summaries : list of dict
        One entry per encoded checkpoint.
    """
    run_dir = Path(run_dir)
    encoder = build_encoder(encoder_config)
    summaries: list[dict[str, Any]] = []

    for npz_path in sorted(run_dir.glob(particle_glob)):
        state = _load_state_from_npz(npz_path)
        batch = particle_state_to_batch(state)
        rep = encoder.encode(batch)
        step = npz_path.stem.replace("step_", "")
        out_path = run_dir / "ml" / f"latent_step_{step}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            latent=rep.latent,
            step=step,
            backend=rep.config.backend.value,
            trained=rep.metadata.trained if rep.metadata else False,
        )
        summaries.append(
            {
                "step": step,
                "latent_path": str(out_path),
                "latent_dim": int(rep.latent.shape[0]),
            }
        )
    return summaries


def _load_state(run_dir: Path, particles_path: Path | str | None):
    if particles_path is not None:
        return _load_state_from_path(Path(particles_path))
    from galacticsics.ml.training_data import _load_particle_state

    return _load_particle_state(run_dir)


def _load_state_from_path(path: Path):
    if path.suffix == ".npz":
        return _load_state_from_npz(path)
    from galacticsics.sampling.particles import ParticleSet

    return _particle_set_to_state(ParticleSet.from_ascii(path, component=path.stem))


def _particle_set_to_state(ps):
    from ntropy.particles import ParticleState

    n = len(ps.data)
    pos = np.column_stack([ps.data["x"], ps.data["y"], ps.data["z"]])
    vel = np.column_stack([ps.data["vx"], ps.data["vy"], ps.data["vz"]])
    mass = ps.data["mass"].copy()
    eps = np.full(n, 0.01, dtype=float)
    return ParticleState.from_arrays(pos, vel, mass, eps)


def _load_state_from_npz(path: Path):
    from ntropy.particles import ParticleState

    data = np.load(path)
    return ParticleState.from_arrays(
        data["pos"],
        data["vel"],
        data["mass"],
        data["eps"],
        type_id=data["type_id"] if "type_id" in data else None,
    )
