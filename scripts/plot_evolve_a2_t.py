#!/usr/bin/env python3
"""Post-process evolve_component_slices verdict → A2(t) + paper copies.

Can relabel arms and replot face-on/profile titles from saved maps without
re-evolving (e.g. nobulge data arm = quiet IC control).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_component_slices import (  # noqa: E402
    _load_maps_npz,
    _plot_a2_t,
    _plot_faceon_grid,
    _plot_profiles,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-prefix", type=str, default="fig_evolve_2gyr")
    p.add_argument(
        "--relabel-data",
        type=str,
        default=None,
        help="Override display label for the data arm (e.g. 'quiet IC (control)').",
    )
    p.add_argument(
        "--replot-faceon-data",
        action="store_true",
        help="Rebuild faceon/profiles for data arm from component_maps_data.npz.",
    )
    p.add_argument(
        "--write-verdict",
        action="store_true",
        help="Persist updated arm_labels (and figure paths) back to verdict.json.",
    )
    args = p.parse_args()
    verdict = args.run / "verdict.json"
    report = json.loads(verdict.read_text())
    arm_labels = dict(report.get("arm_labels") or {})
    if args.relabel_data:
        arm_labels["data"] = str(args.relabel_data)
    # Sensible defaults when older verdicts lack arm_labels.
    arm_labels.setdefault("data", "data dump")
    arm_labels.setdefault("residual_f0", "GalactICS f0 + morph dens residual")
    arm_labels.setdefault("fft_recon", "FFT recon")
    report["arm_labels"] = arm_labels

    if args.replot_faceon_data and "data" in report.get("arms", {}):
        row = report["arms"]["data"]
        maps_path = Path((row.get("figures") or {}).get("maps_npz") or "")
        if not maps_path.is_file():
            maps_path = args.run / "component_maps_data.npz"
        if not maps_path.is_file():
            raise SystemExit(f"missing data maps npz for replot: {maps_path}")
        snaps, times, _ = _load_maps_npz(maps_path)
        n_disk = int(report.get("n_disk") or 0)
        n_tot = int(report.get("n_evolve") or 0)
        lab = arm_labels["data"]
        face_png = args.run / "faceon_components_data.png"
        prof_png = args.run / "profiles_components_data.png"
        _plot_faceon_grid(
            face_png,
            snaps,
            times,
            arm=lab,
            title=(
                rf"{lab}: face-on $\Sigma$ by component "
                rf"($N_{{\rm disk}}={n_disk:,}$, $N_{{\rm tot}}={n_tot:,}$)"
            ),
        )
        _plot_profiles(
            prof_png,
            snaps,
            times,
            arm=lab,
            title=(
                rf"{lab}: density profiles "
                rf"($N_{{\rm disk}}={n_disk:,}$, $N_{{\rm tot}}={n_tot:,}$)"
            ),
        )
        row.setdefault("figures", {})
        row["figures"]["faceon"] = str(face_png)
        row["figures"]["profiles"] = str(prof_png)
        row["figures"]["maps_npz"] = str(maps_path)
        report["figures"]["data_faceon"] = str(face_png)
        report["figures"]["data_profiles"] = str(prof_png)
        print("replot", face_png)
        print("replot", prof_png)

    png = args.run / "a2_t.png"
    _plot_a2_t(png, report)
    report.setdefault("figures", {})["a2_t"] = str(png)
    print("wrote", png)

    if args.write_verdict:
        verdict.write_text(json.dumps(report, indent=2) + "\n")
        print("updated", verdict)

    if args.paper_figures is not None:
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        prefix = args.paper_prefix.rstrip("_")
        shutil.copy2(png, args.paper_figures / f"{prefix}_a2_t.png")
        print("paper", f"{prefix}_a2_t.png")
        for tag, row in report.get("arms", {}).items():
            if not row.get("ok"):
                continue
            for key, suffix in (
                ("faceon", f"{prefix}_faceon_{tag}.png"),
                ("profiles", f"{prefix}_profiles_{tag}.png"),
            ):
                figs = row.get("figures") or {}
                src = Path(figs[key]) if figs.get(key) else Path()
                if src.is_file():
                    shutil.copy2(src, args.paper_figures / suffix)
                    print("paper", suffix)
        print("paper dir", args.paper_figures)


if __name__ == "__main__":
    main()
