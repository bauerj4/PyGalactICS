"""Export particle feature tensors for ML / transformer training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from galacticsics.representations.learned import (
    EncoderBackend,
    LearnedEncoderConfig,
    build_encoder,
    particles_to_learned_artifact,
    torch_available,
)
from galacticsics.representations.particle_features import (
    FEATURE_NAMES,
    particle_state_to_batch,
)


def _load_particles(path: Path):
    from ntropy.io.particles import read_particles_ascii, read_type_ids
    from ntropy.particles import ParticleState

    data = read_particles_ascii(path)
    n = len(data)
    pos = np.column_stack([data["x"], data["y"], data["z"]])
    vel = np.column_stack([data["vx"], data["vy"], data["vz"]])
    mass = data["mass"].copy()
    eps = np.full(n, 0.01, dtype=float)
    type_id = None
    if "type_id" in data.dtype.names:
        type_id = data["type_id"].astype(np.int32)
    types_path = path.with_suffix(".types")
    if type_id is None and types_path.is_file():
        type_id = read_type_ids(types_path)
    return ParticleState.from_arrays(pos, vel, mass, eps, type_id=type_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export particle ML features")
    parser.add_argument(
        "--particles",
        type=Path,
        default=Path("particles.dat"),
        help="ntropy particle file (.dat or .npz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("particle_features.npz"),
        help="Output .npz with features, type_id, mask",
    )
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=None,
        help="Optional .npz with mean-pooled baseline latent",
    )
    parser.add_argument(
        "--backend",
        choices=[b.value for b in EncoderBackend if b != EncoderBackend.PERCEIVER],
        default=EncoderBackend.MEAN_POOL.value,
        help="Encoder backend for optional latent demo",
    )
    args = parser.parse_args(argv)

    state = _load_particles(args.particles)
    batch = particle_state_to_batch(state)
    np.savez(
        args.output,
        feature_names=np.array(FEATURE_NAMES),
        **batch.as_dict(),
    )
    print(f"Wrote {batch.n_particles} tokens × {batch.n_features} features → {args.output}")

    cfg = LearnedEncoderConfig(backend=EncoderBackend(args.backend), d_model=64)
    artifact = particles_to_learned_artifact(
        state, encoder=build_encoder(cfg), config=cfg
    )
    rep = artifact.data
    if args.latent_output:
        np.savez(
            args.latent_output,
            latent=rep.latent,
            d_model=rep.config.d_model,
            backend=rep.config.backend.value,
            trained=rep.metadata.trained if rep.metadata else False,
        )
        print(f"Latent dim {rep.latent.shape[0]} ({cfg.backend.value}) → {args.latent_output}")

    if torch_available() and cfg.backend == EncoderBackend.TRANSFORMER:
        print(
            f"Transformer (untrained) latent norm: {np.linalg.norm(rep.latent):.4f}"
        )
    elif cfg.backend == EncoderBackend.FIELD:
        print(f"Field encoder latent norm: {np.linalg.norm(rep.latent):.4f}")
    elif not torch_available() and cfg.backend in (
        EncoderBackend.TRANSFORMER,
        EncoderBackend.FIELD,
    ):
        print("PyTorch not installed — field encoder uses numpy fallback")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
