#!/usr/bin/env python3
"""Score residual_f0 dens+vel vs barred morph/data dump (t=0 and optional final).

Writes kinetic profile panel + JSON MSE summary for ⟨v_φ⟩, σ_R, σ_φ, σ_z.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ood_theta_df_compare import _disk_kinematics  # noqa: E402
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402


def _load_parts(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        pos = np.asarray(z["pos"], dtype=np.float64)
        vel = np.asarray(z["vel"], dtype=np.float64)
        mass = np.asarray(z["mass"], dtype=np.float64)
        files = set(z.files)
        if "component_id" in files:
            cid = np.asarray(z["component_id"])
        else:
            tags = z["tags"] if "tags" in files else None
            type_id = z["type_id"] if "type_id" in files else None
            cid = _component_ids(tags, type_id, pos.shape[0])
        out = {
            "pos": pos,
            "vel": vel,
            "mass": mass,
            "component_id": cid,
        }
        if "eps" in files:
            out["eps"] = np.asarray(z["eps"], dtype=np.float64)
    return out


def _a2_rd(parts: dict, r_eval: float = 2.0) -> float:
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=float(r_eval),
    )
    return float(fout["a_m_over_a0_at_r"])


def _mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    d = np.asarray(a[mask], dtype=np.float64) - np.asarray(b[mask], dtype=np.float64)
    return float(np.mean(d * d))


def _score_vs_ref(parts: dict, ref: dict) -> dict:
    k = _disk_kinematics(parts)
    kr = _disk_kinematics(ref)
    cnt = np.asarray(k["counts"])
    cntr = np.asarray(kr["counts"])
    ok = (cnt >= 20) & (cntr >= 20) & np.isfinite(k["mean_vphi"]) & np.isfinite(
        kr["mean_vphi"]
    )
    out = {
        "r_mid": [float(x) for x in k["r_mid"]],
        "kin": {kk: [None if not np.isfinite(v) else float(v) for v in k[kk]] for kk in (
            "mean_vphi", "sig_r", "sig_phi", "sig_z"
        )},
        "ref_kin": {
            kk: [None if not np.isfinite(v) else float(v) for v in kr[kk]]
            for kk in ("mean_vphi", "sig_r", "sig_phi", "sig_z")
        },
        "mse": {
            "mean_vphi": _mse(k["mean_vphi"], kr["mean_vphi"], ok),
            "sig_r": _mse(k["sig_r"], kr["sig_r"], ok),
            "sig_phi": _mse(k["sig_phi"], kr["sig_phi"], ok),
            "sig_z": _mse(k["sig_z"], kr["sig_z"], ok),
        },
        "a2_rd": _a2_rd(parts),
        "ref_a2_rd": _a2_rd(ref),
        "n_bins_ok": int(np.count_nonzero(ok)),
    }
    out["mse"]["kinetic_mean"] = float(
        np.nanmean([out["mse"][k] for k in ("mean_vphi", "sig_r", "sig_phi", "sig_z")])
    )
    return out


def _plot_kin(
    path: Path,
    scores: dict[str, dict],
    *,
    title: str,
) -> None:
    keys = [
        ("mean_vphi", r"$\langle v_\varphi\rangle$"),
        ("sig_r", r"$\sigma_R$"),
        ("sig_phi", r"$\sigma_\varphi$"),
        ("sig_z", r"$\sigma_z$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
    # Use first score's ref
    ref_key = next(iter(scores))
    ref = scores[ref_key]["ref_kin"]
    r = np.asarray(scores[ref_key]["r_mid"], dtype=float)
    for ax, (kk, ylab) in zip(axes.ravel(), keys):
        yr = np.asarray(
            [np.nan if v is None else v for v in ref[kk]], dtype=float
        )
        ax.plot(r, yr, "k-", lw=2.0, label="data dump", zorder=3)
        for name, sc in scores.items():
            y = np.asarray(
                [np.nan if v is None else v for v in sc["kin"][kk]], dtype=float
            )
            ax.plot(r, y, lw=1.6, label=name)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel(r"$R$ [kpc]")
    axes[1, 1].set_xlabel(r"$R$ [kpc]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", type=Path, required=True, help="Barred morph/data dump")
    p.add_argument("--ic", type=Path, required=True, help="residual IC / t0 particles")
    p.add_argument("--final", type=Path, default=None, help="Optional final particles")
    p.add_argument("--baseline-ic", type=Path, default=None, help="Optional f0-vel IC")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--paper-fig", type=Path, default=None)
    p.add_argument("--label", type=str, default="morphvel")
    p.add_argument("--baseline-label", type=str, default="f0vel")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ref = _load_parts(args.ref)
    # Morph dump may lack component_id — treat all as disk if missing tags handled upstream.
    if "component_id" not in ref or ref["component_id"] is None:
        raise SystemExit("ref dump must include component_id")

    scores_t0: dict[str, dict] = {}
    ic = _load_parts(args.ic)
    scores_t0[args.label] = _score_vs_ref(ic, ref)
    if args.baseline_ic is not None and args.baseline_ic.is_file():
        base = _load_parts(args.baseline_ic)
        scores_t0[args.baseline_label] = _score_vs_ref(base, ref)

    fig_t0 = args.out / "kin_profiles_t0.png"
    _plot_kin(
        fig_t0,
        scores_t0,
        title=rf"t=0 disk kinetics vs data dump ({args.label})",
    )

    summary: dict = {"t0": scores_t0, "figures": {"t0": str(fig_t0)}}
    if args.final is not None and args.final.is_file():
        scores_f = {args.label: _score_vs_ref(_load_parts(args.final), ref)}
        fig_f = args.out / "kin_profiles_final.png"
        _plot_kin(
            fig_f,
            scores_f,
            title=rf"final disk kinetics vs data dump ({args.label})",
        )
        summary["final"] = scores_f
        summary["figures"]["final"] = str(fig_f)

    (args.out / "kinetics_score.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.paper_fig is not None:
        args.paper_fig.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(fig_t0, args.paper_fig)
        summary["figures"]["paper_t0"] = str(args.paper_fig)
        if "final" in summary["figures"]:
            dst_f = args.paper_fig.with_name(
                args.paper_fig.name.replace("_t0", "_final").replace(
                    "kin_t0", "kin_final"
                )
            )
            if dst_f == args.paper_fig:
                dst_f = args.paper_fig.with_name(
                    args.paper_fig.stem + "_final" + args.paper_fig.suffix
                )
            shutil.copy2(summary["figures"]["final"], dst_f)
            summary["figures"]["paper_final"] = str(dst_f)

    # Console scoreboard
    print("=== kinetic MSE vs data dump (t=0) ===", flush=True)
    for name, sc in scores_t0.items():
        m = sc["mse"]
        print(
            f"  {name}: A2(Rd)={sc['a2_rd']:.3f}  "
            f"⟨vφ⟩MSE={m['mean_vphi']:.4g}  "
            f"σR={m['sig_r']:.4g}  σφ={m['sig_phi']:.4g}  σz={m['sig_z']:.4g}  "
            f"mean={m['kinetic_mean']:.4g}",
            flush=True,
        )
    if "final" in summary:
        for name, sc in summary["final"].items():
            m = sc["mse"]
            print(
                f"  {name} final: A2(Rd)={sc['a2_rd']:.3f}  "
                f"⟨vφ⟩MSE={m['mean_vphi']:.4g}  mean={m['kinetic_mean']:.4g}",
                flush=True,
            )
    print(f"  wrote {args.out / 'kinetics_score.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
