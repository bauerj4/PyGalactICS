#!/usr/bin/env python3
"""Plot dynamical-consistency A₂(t) before/after vs data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _arm_series(arm: dict) -> tuple[np.ndarray, np.ndarray] | None:
    t = arm.get("t_gyr") or arm.get("snap_t_gyr")
    a2 = arm.get("a2_t")
    if t is None or a2 is None:
        return None
    return np.asarray(t, dtype=float), np.asarray(a2, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdict", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=str, default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str, default=r"Dynamical consistency: $A_2(t)$")
    args = p.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {
        "data": "C0",
        "fft_recon": "C3",
        "fft_recon_keep_bulge": "C2",
        "fft_recon_shell_bulge": "C1",
        "fft_recon_input_dens_shell": "C4",
    }
    styles = {
        "data": "-",
        "fft_recon": "--",
        "fft_recon_keep_bulge": "-.",
        "fft_recon_shell_bulge": ":",
        "fft_recon_input_dens_shell": "-",
    }
    for i, path in enumerate(args.verdict):
        v = json.loads(Path(path).read_text())
        prefix = (labels[i] + " ") if i < len(labels) else ""
        for name, arm in (v.get("arms") or {}).items():
            ser = _arm_series(arm)
            if ser is None:
                continue
            t, a2 = ser
            ax.plot(
                t,
                a2,
                styles.get(name, "-"),
                color=colors.get(name),
                lw=2.0 if name == "data" else 1.6,
                label=f"{prefix}{name}",
            )
            if arm.get("a2_pre") is not None:
                print(f"{prefix}{name}: {arm['a2_pre']:.3f}→{arm['a2_post']:.3f}")
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"$A_2$ (disk)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
