#!/usr/bin/env python3
"""Generate a Morton-token galaxy from a trained VAE or transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("generated_ic.npz"))
    parser.add_argument("--model", choices=("vae", "transformer"), default="transformer")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--theta-json",
        type=Path,
        default=None,
        help="JSON object of conditioning parameters (keys match training theta_keys)",
    )
    parser.add_argument(
        "--evolve-gyr",
        type=float,
        default=None,
        help="Optional short ntropy evolve for smoke check",
    )
    args = parser.parse_args(argv)

    import torch

    from galacticsics.ml.morton.dataset import DEFAULT_THETA_KEYS
    from galacticsics.ml.morton.tokenize import particles_from_tokens
    from galacticsics.ml.models.morton_transformer import MortonTransformer, MortonTransformerConfig
    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig

    blob = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    theta_keys = blob.get("theta_keys", DEFAULT_THETA_KEYS)
    theta_raw = json.loads(args.theta_json.read_text()) if args.theta_json else {}
    theta_vec = np.asarray([float(theta_raw.get(k, 0.0)) for k in theta_keys], dtype=np.float64)
    theta = torch.as_tensor(theta_vec[None, :], device=args.device)

    if args.model == "vae":
        cfg = SequenceVAEConfig(**{k: v for k, v in blob["config"].items() if k in SequenceVAEConfig.__dataclass_fields__})
        model = SequenceVAE(cfg).to(args.device)
        model.load_state_dict(blob["model"])
        tokens = model.generate(theta, n=args.n)
    else:
        cfg = MortonTransformerConfig(
            **{k: v for k, v in blob["config"].items() if k in MortonTransformerConfig.__dataclass_fields__}
        )
        model = MortonTransformer(cfg).to(args.device)
        model.load_state_dict(blob["model"])
        tokens = model.generate(theta, n=args.n)

    # Attach bbox defaults for reconstruction (unit box if missing)
    tok0 = {
        "c": tokens["c"][0],
        "dm": tokens["dm"][0],
        "dx": tokens["dx"][0],
        "v": tokens["v"][0],
        "box_min": np.array([-30.0, -30.0, -30.0]),
        "box_size": np.asarray(60.0),
        "bits": np.asarray(10),
    }
    pos, vel, cid = particles_from_tokens(tok0)
    mass = np.full(pos.shape[0], 1.0 / pos.shape[0])
    eps = np.full(pos.shape[0], 0.1)
    np.savez_compressed(
        args.out,
        pos=pos,
        vel=vel,
        mass=mass,
        eps=eps,
        type_id=cid.astype(np.int32),
        theta=theta_vec,
        theta_keys=np.asarray(theta_keys),
    )
    print(f"wrote {args.out}  N={pos.shape[0]}")

    if args.evolve_gyr is not None and args.evolve_gyr > 0:
        from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
        from ntropy.particles import ParticleState
        from ntropy.simulation import Simulation

        state = ParticleState.from_arrays(pos, vel, mass, eps, type_id=cid.astype(np.int32))
        cfg_run = RunConfig()
        cfg_run.integrator = IntegratorConfig(dt=0.01, end_time_gyr=float(args.evolve_gyr))
        cfg_run.force = ForceConfig(method="bh")
        cfg_run.parallel = ParallelConfig(enabled=False)
        cfg_run.output.write_final = False
        cfg_run.output.every = 0
        result = Simulation(cfg_run, state=state.copy()).run()
        print(f"smoke evolve {args.evolve_gyr} Gyr: steps={len(result.energies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
