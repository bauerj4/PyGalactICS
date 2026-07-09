"""Compare dynamical model backends for a solved GalactICS model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from galacticsics.builder import GalaxyBuilder
from galacticsics.integrations.galpy import GalactICSPotential
from galacticsics.models import GalaxyModel
from galacticsics.potential.evaluate import evaluate_potential


def rotation_curve(potential, r_vals: np.ndarray, z: float = 0.0) -> np.ndarray:
    vc = np.zeros_like(r_vals)
    for i, r in enumerate(r_vals):
        if r <= 0:
            continue
        psi_r = evaluate_potential(potential, r, z)
        psi_dr = evaluate_potential(potential, r + 0.01, z)
        vc[i] = np.sqrt(max(0.0, 2.0 * (psi_dr - psi_r) / 0.01))
    return vc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare dynamical model representations")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("tests/generated/reference"),
        help="Directory with dbh.dat",
    )
    parser.add_argument("--r-max", type=float, default=20.0)
    args = parser.parse_args(argv)

    builder = GalaxyBuilder(
        model=GalaxyModel.reference_disk_halo(),
        model_dir=args.model_dir,
    ).load_artifacts()
    harmonic = builder.potential
    r = np.linspace(0.5, args.r_max, 40)

    vc_native = rotation_curve(harmonic, r)
    print("Native harmonic V_c(R) at z=0:")
    for ri, vi in zip(r[::5], vc_native[::5]):
        print(f"  R={ri:5.1f} kpc  Vc={vi:.3f} (100 km/s)")

    try:
        gp = GalactICSPotential.from_harmonic(harmonic).to_galpy()
        vc_galpy = np.array([gp.vcirc(ri) for ri in r])
        rms = np.sqrt(np.mean((vc_galpy - vc_native) ** 2))
        print(f"\ngalpy RMS difference vs native: {rms:.4f} (100 km/s)")
    except ImportError:
        print("\ngalpy not installed (pip install galacticsics[galpy])")

    try:
        from galacticsics.integrations.agama import agama_available, harmonic_to_agama

        if agama_available():
            ap = harmonic_to_agama(harmonic)
            print("agama potential created OK")
    except ImportError:
        print("agama not installed")

    try:
        from galacticsics.integrations.gala import gala_available, harmonic_to_gala

        if gala_available():
            harmonic_to_gala(harmonic)
            print("gala potential created OK")
    except ImportError:
        print("gala not installed")

    try:
        from ntropy.ics.plummer import sample_plummer

        from galacticsics.representations.learned import mean_pool_encoder, torch_available
        from galacticsics.representations.particle_features import particle_state_to_batch

        batch = particle_state_to_batch(sample_plummer(seed=0))
        rep = mean_pool_encoder(batch)
        print(f"\nlearned (mean-pool) latent dim: {rep.latent.shape[0]}")
        if torch_available():
            from galacticsics.integrations.torch_encoder import ParticleTransformerEncoder

            enc = ParticleTransformerEncoder(d_model=64, n_layers=1)
            print("transformer encoder scaffold OK")
    except ImportError as exc:
        print(f"\nML representation helpers: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
