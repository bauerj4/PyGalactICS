#!/usr/bin/env python3
"""Search GalactICS corpus ICs for the closest equilibrium f0 to a barred dump.

Scores each available ``ic_state.npz`` (structural θ from ``model.json``) against
a barred morph/data dump on **axisymmetric** dens + kinematic profiles only —
not bar morphology.  That answers: which quiet GalactICS DF is the best
starting point before any residual / full-dyn transport.

Primary score (lower better)::

    score = w_dens * med|log10 Σ_disk| + w_bulge * med|log10 ρ_bulge|
          + w_kin  * mean MSE(⟨vφ⟩, σ_R, σ_φ, σ_z)

Example::

    .venv/bin/python scripts/closest_galactics_f0.py \\
      --dump runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz \\
      --same-campaign 906c4af73543 \\
      --out runs/ml/field_maps/closest_f0_full_dyn_2026-08-02/search_906c4 \\
      --label 906c4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_component_slices import _profile_comp  # noqa: E402
from evolve_resample_compare import _full_com, _stratified_down  # noqa: E402
from ood_theta_compare import THETA_DIST_KEYS, _theta_from_model  # noqa: E402
from ood_theta_df_compare import (  # noqa: E402
    _disk_kinematics,
    _med_abs_log_resid,
)
from score_residual_f0_kinetics import _a2_rd, _load_parts  # noqa: E402

CORPUS_DEFAULT = Path("runs/mw_morton_corpus_v2")


def _iter_corpus_ics(corpus: Path) -> list[dict]:
    rows: list[dict] = []
    for d in sorted(corpus.iterdir()):
        if not d.is_dir():
            continue
        ic = d / "ic_state.npz"
        model_p = d / "model.json"
        if not ic.is_file() or not model_p.is_file():
            continue
        model = json.loads(model_p.read_text())
        theta = _theta_from_model(model)
        has_bulge = bool(model.get("bulge", {}).get("enabled", False)) and float(
            model.get("bulge", {}).get("v0", 0.0) or 0.0
        ) > 0.0
        rows.append(
            {
                "hash": d.name,
                "ic_path": ic,
                "model_path": model_p,
                "theta": theta,
                "has_bulge": has_bulge,
                "disk_mass": float(theta.get("disk.mass", np.nan)),
                "Rd": float(theta.get("disk.scale_length", np.nan)),
            }
        )
    return rows


def _axisym_bundle(parts: dict) -> dict:
    pos = parts["pos"]
    mass = parts["mass"]
    cid = parts["component_id"]
    profiles = {
        "disk": _profile_comp(pos, mass, cid, "disk", 0, r_max=15.0),
        "bulge": _profile_comp(pos, mass, cid, "bulge", 2, r_max=6.0),
        "halo": _profile_comp(pos, mass, cid, "halo", 1, r_max=40.0),
    }
    kin = _disk_kinematics(parts)
    disk_m = float(mass[cid == 0].sum()) if np.any(cid == 0) else 0.0
    bulge_m = float(mass[cid == 2].sum()) if np.any(cid == 2) else 0.0
    halo_m = float(mass[cid == 1].sum()) if np.any(cid == 1) else 0.0
    return {
        "profiles": profiles,
        "kin": kin,
        "masses": {"disk": disk_m, "bulge": bulge_m, "halo": halo_m, "total": disk_m + bulge_m + halo_m},
        "a2_rd": _a2_rd(parts),
        "n_disk": int(np.count_nonzero(cid == 0)),
        "n_bulge": int(np.count_nonzero(cid == 2)),
        "n_halo": int(np.count_nonzero(cid == 1)),
    }


def _score_ic_vs_dump(
    ic_b: dict,
    dump_b: dict,
    *,
    w_dens: float = 1.0,
    w_bulge: float = 0.5,
    w_halo: float = 0.15,
    w_kin: float = 1.0,
    w_mass: float = 0.25,
) -> dict:
    pd, pi = dump_b["profiles"], ic_b["profiles"]
    dens_disk = _med_abs_log_resid(
        np.asarray(pd["disk"]["y"]),
        np.asarray(pi["disk"]["y"]),
        counts=np.asarray(pd["disk"]["counts"]),
        min_count=20,
    )
    dens_bulge = _med_abs_log_resid(
        np.asarray(pd["bulge"]["y"]),
        np.asarray(pi["bulge"]["y"]),
        counts=np.asarray(pd["bulge"]["counts"]),
        min_count=10,
    )
    dens_halo = _med_abs_log_resid(
        np.asarray(pd["halo"]["y"]),
        np.asarray(pi["halo"]["y"]),
        counts=np.asarray(pd["halo"]["counts"]),
        min_count=20,
    )

    kd, ki = dump_b["kin"], ic_b["kin"]
    ok = (
        (np.asarray(kd["counts"]) >= 20)
        & (np.asarray(ki["counts"]) >= 20)
        & np.isfinite(kd["mean_vphi"])
        & np.isfinite(ki["mean_vphi"])
    )

    def _mse(a, b):
        if not np.any(ok):
            return float("nan")
        d = np.asarray(a)[ok] - np.asarray(b)[ok]
        return float(np.mean(d * d))

    kin_mse = {
        "mean_vphi": _mse(ki["mean_vphi"], kd["mean_vphi"]),
        "sig_r": _mse(ki["sig_r"], kd["sig_r"]),
        "sig_phi": _mse(ki["sig_phi"], kd["sig_phi"]),
        "sig_z": _mse(ki["sig_z"], kd["sig_z"]),
    }
    kin_mean = float(np.nanmean(list(kin_mse.values())))

    # Relative mass mismatch (disk + bulge + total).
    md, mi = dump_b["masses"], ic_b["masses"]

    def _rel(a, b, floor=1e-6):
        if not np.isfinite(a) or not np.isfinite(b) or abs(a) < floor:
            return float("nan")
        return float(abs(b - a) / abs(a))

    mass_rel = {
        "disk": _rel(md["disk"], mi["disk"]),
        "bulge": _rel(md["bulge"], mi["bulge"]),
        "total": _rel(md["total"], mi["total"]),
    }
    mass_mean = float(np.nanmean([mass_rel["disk"], mass_rel["bulge"], mass_rel["total"]]))

    # Normalize kin MSE to O(0.1–1) scale similar to dens med|log10|.
    # Typical ⟨vφ⟩ MSE ~ 0.01–1; take sqrt so score is comparable.
    kin_term = float(np.sqrt(max(kin_mean, 0.0))) if np.isfinite(kin_mean) else 1.0
    dens_term = dens_disk if np.isfinite(dens_disk) else 1.0
    bulge_term = dens_bulge if np.isfinite(dens_bulge) else 0.0
    # If dump has no bulge particles, ignore bulge dens term.
    if dump_b["n_bulge"] < 100:
        bulge_term = 0.0
        w_bulge_eff = 0.0
    else:
        w_bulge_eff = w_bulge
    halo_term = dens_halo if np.isfinite(dens_halo) else 0.0
    mass_term = mass_mean if np.isfinite(mass_mean) else 0.0

    score = (
        w_dens * dens_term
        + w_bulge_eff * bulge_term
        + w_halo * halo_term
        + w_kin * kin_term
        + w_mass * mass_term
    )
    return {
        "score": float(score),
        "dens_med_abs_log": {
            "disk": dens_disk,
            "bulge": dens_bulge,
            "halo": dens_halo,
        },
        "kin_mse": kin_mse,
        "kin_mse_mean": kin_mean,
        "kin_term": kin_term,
        "mass_rel": mass_rel,
        "mass_term": mass_term,
        "a2_rd_ic": float(ic_b["a2_rd"]),
        "a2_rd_dump": float(dump_b["a2_rd"]),
        "n_bins_kin_ok": int(np.count_nonzero(ok)),
        "weights": {
            "w_dens": w_dens,
            "w_bulge": w_bulge_eff,
            "w_halo": w_halo,
            "w_kin": w_kin,
            "w_mass": w_mass,
        },
    }


def _downsample(parts: dict, n_tot: int, rng: np.random.Generator) -> dict:
    if parts["pos"].shape[0] <= n_tot:
        return parts
    return _stratified_down(parts, n_tot, rng)


def _plot_rank(
    path: Path,
    ranked: list[dict],
    *,
    same_hash: str | None,
    title: str,
    top_k: int = 15,
) -> None:
    top = ranked[:top_k]
    labels = [r["hash"][:6] for r in top]
    scores = [r["metrics"]["score"] for r in top]
    colors = []
    for r in top:
        if same_hash and r["hash"] == same_hash:
            colors.append("C3")
        elif r.get("rank", 99) == 0:
            colors.append("C2")
        else:
            colors.append("C0")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(range(len(top)), scores, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("axisym dens+kin score (↓ better)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if same_hash:
        ax.plot([], [], color="C3", lw=6, label=f"same-campaign ({same_hash[:6]})")
        ax.plot([], [], color="C2", lw=6, label="best")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(
    path: Path,
    dump_b: dict,
    same_b: dict | None,
    best_b: dict,
    *,
    same_label: str,
    best_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    # dens disk / bulge / halo
    for ax, name, ylab in zip(
        axes[0],
        ("disk", "bulge", "halo"),
        (r"$\Sigma_{\mathrm{disk}}(R)$", r"$\rho_{\mathrm{bulge}}(r)$", r"$\rho_{\mathrm{halo}}(r)$"),
    ):
        pr = dump_b["profiles"][name]
        r = np.asarray(pr["r_mid"])
        y = np.asarray(pr["y"])
        ax.plot(r, y, "k-", lw=2.0, label="data dump")
        if same_b is not None:
            ys = np.asarray(same_b["profiles"][name]["y"])
            ax.plot(r, ys, "C3--", lw=1.6, label=same_label)
        yb = np.asarray(best_b["profiles"][name]["y"])
        # best may have different r grid — interpolate if needed
        rb = np.asarray(best_b["profiles"][name]["r_mid"])
        if rb.shape == r.shape and np.allclose(rb, r):
            ax.plot(r, yb, "C2-", lw=1.6, label=best_label)
        else:
            ax.plot(rb, yb, "C2-", lw=1.6, label=best_label)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"$R$ or $r$ [kpc]")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, alpha=0.25)

    for ax, key, ylab in zip(
        axes[1],
        ("mean_vphi", "sig_r", "sig_z"),
        (r"$\langle v_\varphi\rangle$", r"$\sigma_R$", r"$\sigma_z$"),
    ):
        r = np.asarray(dump_b["kin"]["r_mid"])
        ax.plot(r, dump_b["kin"][key], "k-", lw=2.0, label="data dump")
        if same_b is not None:
            ax.plot(r, same_b["kin"][key], "C3--", lw=1.6, label=same_label)
        ax.plot(r, best_b["kin"][key], "C2-", lw=1.6, label=best_label)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"$R$ [kpc]")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True, help="Barred morph/data dump")
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    p.add_argument(
        "--same-campaign",
        type=str,
        default=None,
        help="Corpus hash of the 'same campaign IC' baseline (for comparison).",
    )
    p.add_argument("--extra-ic", type=Path, nargs="*", default=[], help="Extra IC paths (e.g. nobulge)")
    p.add_argument("--extra-label", type=str, nargs="*", default=[], help="Labels for --extra-ic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", type=str, default="dump")
    p.add_argument("--n-tot", type=int, default=400_000, help="Downsample for scoring speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--w-dens", type=float, default=1.0)
    p.add_argument("--w-bulge", type=float, default=0.5)
    p.add_argument("--w-halo", type=float, default=0.15)
    p.add_argument("--w-kin", type=float, default=1.0)
    p.add_argument("--w-mass", type=float, default=0.25)
    p.add_argument(
        "--require-bulge",
        action="store_true",
        help="Skip ICs without bulge when dump has bulge.",
    )
    p.add_argument(
        "--allow-nobulge",
        action="store_true",
        help="Include no-bulge ICs even if dump has bulge.",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    dump = _full_com(_load_parts(args.dump))
    dump = _downsample(dump, args.n_tot, rng)
    dump_b = _axisym_bundle(dump)
    print(
        f"=== closest f0 search label={args.label} dump={args.dump} "
        f"N={dump['pos'].shape[0]} a2_rd={dump_b['a2_rd']:.3f} "
        f"Mdisk={dump_b['masses']['disk']:.3f} Mbulge={dump_b['masses']['bulge']:.3f} ===",
        flush=True,
    )

    rows = _iter_corpus_ics(args.corpus)
    # Optional extras (nobulge custom IC, etc.)
    for i, xp in enumerate(args.extra_ic):
        lab = args.extra_label[i] if i < len(args.extra_label) else xp.parent.name
        model_p = xp.parent / "model.json"
        theta = {}
        has_bulge = False
        if model_p.is_file():
            model = json.loads(model_p.read_text())
            theta = _theta_from_model(model)
            has_bulge = bool(model.get("bulge", {}).get("enabled", False))
        rows.append(
            {
                "hash": f"extra:{lab}",
                "ic_path": Path(xp),
                "model_path": model_p if model_p.is_file() else None,
                "theta": theta,
                "has_bulge": has_bulge,
                "disk_mass": float(theta.get("disk.mass", np.nan)) if theta else float("nan"),
                "Rd": float(theta.get("disk.scale_length", np.nan)) if theta else float("nan"),
            }
        )

    dump_has_bulge = dump_b["n_bulge"] >= 100
    ranked: list[dict] = []
    for i, row in enumerate(rows):
        if dump_has_bulge and args.require_bulge and not row["has_bulge"]:
            continue
        if dump_has_bulge and (not args.allow_nobulge) and (not row["has_bulge"]):
            # Default: prefer bulge-matched when dump has bulge; still score all
            # with bulge. Skip pure disk+halo unless --allow-nobulge.
            if not str(row["hash"]).startswith("extra:"):
                # corpus always has bulge in mw_morton; extras may not
                pass
        try:
            ic = _full_com(_load_parts(row["ic_path"]))
            ic = _downsample(ic, args.n_tot, rng)
            ic_b = _axisym_bundle(ic)
            metrics = _score_ic_vs_dump(
                ic_b,
                dump_b,
                w_dens=args.w_dens,
                w_bulge=args.w_bulge,
                w_halo=args.w_halo,
                w_kin=args.w_kin,
                w_mass=args.w_mass,
            )
        except Exception as e:  # noqa: BLE001 — keep search robust
            print(f"  skip {row['hash']}: {e}", flush=True)
            continue
        ranked.append(
            {
                **row,
                "ic_path": str(row["ic_path"]),
                "model_path": str(row["model_path"]) if row["model_path"] else None,
                "metrics": metrics,
                "masses_ic": ic_b["masses"],
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(rows)}] {row['hash'][:12]} "
                f"score={metrics['score']:.4f} "
                f"dens={metrics['dens_med_abs_log']['disk']:.4f} "
                f"kin√mse={metrics['kin_term']:.4f}",
                flush=True,
            )

    ranked.sort(key=lambda r: r["metrics"]["score"])
    for i, r in enumerate(ranked):
        r["rank"] = i

    same_row = None
    if args.same_campaign:
        for r in ranked:
            if r["hash"] == args.same_campaign:
                same_row = r
                break

    best = ranked[0] if ranked else None
    summary = {
        "label": args.label,
        "dump": str(args.dump),
        "corpus": str(args.corpus),
        "n_candidates": len(ranked),
        "n_tot_score": args.n_tot,
        "dump_axisym": {
            "masses": dump_b["masses"],
            "a2_rd": dump_b["a2_rd"],
            "n_disk": dump_b["n_disk"],
            "n_bulge": dump_b["n_bulge"],
            "n_halo": dump_b["n_halo"],
        },
        "weights": {
            "w_dens": args.w_dens,
            "w_bulge": args.w_bulge,
            "w_halo": args.w_halo,
            "w_kin": args.w_kin,
            "w_mass": args.w_mass,
        },
        "best": None,
        "same_campaign": None,
        "improvement_vs_same": None,
        "ranked": [
            {
                "rank": r["rank"],
                "hash": r["hash"],
                "ic_path": r["ic_path"],
                "theta": r["theta"],
                "has_bulge": r["has_bulge"],
                "score": r["metrics"]["score"],
                "dens_disk": r["metrics"]["dens_med_abs_log"]["disk"],
                "dens_bulge": r["metrics"]["dens_med_abs_log"]["bulge"],
                "kin_mse_mean": r["metrics"]["kin_mse_mean"],
                "kin_term": r["metrics"]["kin_term"],
                "mass_rel": r["metrics"]["mass_rel"],
            }
            for r in ranked
        ],
        "elapsed_s": float(time.time() - t0),
    }

    if best is not None:
        summary["best"] = {
            "hash": best["hash"],
            "ic_path": best["ic_path"],
            "theta": best["theta"],
            "metrics": best["metrics"],
            "masses_ic": best["masses_ic"],
        }
    if same_row is not None:
        summary["same_campaign"] = {
            "hash": same_row["hash"],
            "rank": same_row["rank"],
            "ic_path": same_row["ic_path"],
            "theta": same_row["theta"],
            "metrics": same_row["metrics"],
            "masses_ic": same_row["masses_ic"],
        }
        if best is not None:
            sc_same = same_row["metrics"]["score"]
            sc_best = best["metrics"]["score"]
            dens_same = same_row["metrics"]["dens_med_abs_log"]["disk"]
            dens_best = best["metrics"]["dens_med_abs_log"]["disk"]
            kin_same = same_row["metrics"]["kin_mse_mean"]
            kin_best = best["metrics"]["kin_mse_mean"]
            summary["improvement_vs_same"] = {
                "score_same": sc_same,
                "score_best": sc_best,
                "score_ratio_best_over_same": sc_best / sc_same if sc_same > 0 else None,
                "dens_disk_same": dens_same,
                "dens_disk_best": dens_best,
                "dens_disk_delta": dens_best - dens_same,
                "kin_mse_same": kin_same,
                "kin_mse_best": kin_best,
                "kin_mse_delta": kin_best - kin_same,
                "best_is_same": best["hash"] == same_row["hash"],
            }

    (args.out / "closest_f0_rank.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Markdown summary
    lines = [
        f"# Closest GalactICS $f_0$ — {args.label}",
        "",
        f"Dump: `{args.dump}`",
        f"Candidates scored: **{len(ranked)}** (corpus `{args.corpus}`).",
        "",
        "## Best θ",
        "",
    ]
    if best is not None:
        th = best["theta"]
        lines += [
            f"- **hash:** `{best['hash']}`",
            f"- **ic:** `{best['ic_path']}`",
            f"- **score:** {best['metrics']['score']:.4f}",
            f"- dens med$|\\log_{{10}}|$ disk: {best['metrics']['dens_med_abs_log']['disk']:.4f}",
            f"- kin MSE mean: {best['metrics']['kin_mse_mean']:.4f}",
            "",
            "| θ key | value |",
            "|-------|-------|",
        ]
        for k in THETA_DIST_KEYS:
            if k in th:
                lines.append(f"| `{k}` | {th[k]} |")
    lines += ["", "## vs same-campaign IC", ""]
    if same_row is not None and summary["improvement_vs_same"] is not None:
        imp = summary["improvement_vs_same"]
        lines += [
            f"- same-campaign `{same_row['hash']}` rank **#{same_row['rank']+1}** / {len(ranked)}",
            f"- score same→best: {imp['score_same']:.4f} → {imp['score_best']:.4f}"
            + (f" (ratio {imp['score_ratio_best_over_same']:.3f})" if imp['score_ratio_best_over_same'] else ""),
            f"- Σ disk med$|\\log|$ same→best: {imp['dens_disk_same']:.4f} → {imp['dens_disk_best']:.4f}",
            f"- kin MSE same→best: {imp['kin_mse_same']:.4f} → {imp['kin_mse_best']:.4f}",
            f"- best is same-campaign: **{imp['best_is_same']}**",
        ]
    else:
        lines.append("_same-campaign hash not provided or not found._")
    lines += ["", "## Top 10", "", "| rank | hash | score | dens disk | kin√mse | disk.M | Rd | halo.v0 | bulge.v0 |",
              "|------|------|-------|-----------|---------|--------|----|---------|----------|"]
    for r in ranked[:10]:
        th = r["theta"]
        lines.append(
            f"| {r['rank']+1} | `{r['hash'][:12]}` | {r['metrics']['score']:.4f} | "
            f"{r['metrics']['dens_med_abs_log']['disk']:.4f} | {r['metrics']['kin_term']:.4f} | "
            f"{th.get('disk.mass', float('nan'))} | {th.get('disk.scale_length', float('nan'))} | "
            f"{th.get('halo.v0', float('nan'))} | {th.get('bulge.v0', float('nan'))} |"
        )
    lines.append("")
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    # Plots
    _plot_rank(
        args.out / "rank_bar.png",
        ranked,
        same_hash=args.same_campaign,
        title=rf"Closest $f_0$ rank — {args.label}",
        top_k=min(args.top_k, len(ranked)),
    )

    # Reload best / same at full score resolution for profile plots
    if best is not None:
        best_parts = _full_com(_load_parts(Path(best["ic_path"])))
        best_parts = _downsample(best_parts, args.n_tot, rng)
        best_b = _axisym_bundle(best_parts)
        same_b = None
        same_lab = "same-campaign"
        if same_row is not None:
            same_parts = _full_com(_load_parts(Path(same_row["ic_path"])))
            same_parts = _downsample(same_parts, args.n_tot, rng)
            same_b = _axisym_bundle(same_parts)
            same_lab = f"same ({same_row['hash'][:6]})"
        best_lab = f"best ({best['hash'][:6]})"
        _plot_profiles(
            args.out / "profiles_best_vs_same.png",
            dump_b,
            same_b,
            best_b,
            same_label=same_lab,
            best_label=best_lab,
            title=rf"$f_0$ axisym dens+kin vs data — {args.label}",
        )

    print(
        f"=== done best={best['hash'] if best else None} "
        f"same_rank={same_row['rank']+1 if same_row else None} "
        f"elapsed={time.time()-t0:.1f}s out={args.out} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
