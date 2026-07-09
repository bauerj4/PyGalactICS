#!/usr/bin/env python3
"""Generate Plummer IC and run a short stability simulation."""

from __future__ import annotations

import json
from pathlib import Path

from ntropy.config import load_config
from ntropy.ics.plummer import sample_plummer
from ntropy.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "examples" / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    particle_file = DATA / "plummer_128.dat"
    state = sample_plummer(seed=42)
    state.write_ascii(particle_file)

    config_path = CONFIGS / "plummer_short.json"
    cfg = load_config(config_path)
    cfg.particles.file = str(particle_file.relative_to(config_path.parent))
    cfg.output.dir = str((DATA / "run_plummer").relative_to(config_path.parent))
    patched = config_path.parent / "plummer_short_local.json"
    patched.write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "particles": {"file": cfg.particles.file},
                "softening": {"default": 0.02, "per_particle": False},
                "force": {"method": "bh", "theta": 0.5},
                "integrator": {"type": "leapfrog", "dt": 0.005, "n_steps": 50},
                "parallel": {"enabled": True, "n_workers": 2},
                "output": {"dir": cfg.output.dir, "every": 25, "write_final": True},
                "analysis": {"density_bins": 20, "r_max": 5.0},
            },
            indent=2,
        )
        + "\n"
    )
    result = run_simulation(load_config(patched))
    print(f"Plummer run complete. Final energy: {result.energies[-1]:.6e}")


if __name__ == "__main__":
    main()
