#!/usr/bin/env python3
"""Path-LOO latent gen evolve vs data + new-θ sampling (particle_retrieve).

Part 1: rebuild full-N path-LOO ICs for 906c4 / 54a8 and evolve 2 Gyr vs data
dump (disk=1e6, gpu_bh, A₂(R_d)). Part 2: sample barred z on new θ, compare to
GalactICS f0(θ), optionally evolve.

Does **not** use full_dyn_replace graft.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.campaign.analysis import dens_array_log10  # noqa: E402
from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_vector  # noqa: E402
from galacticsics.ml.fields.feature_library import (  # noqa: E402
    _hash_from_path,
    encode_snapshot_features,
    load_frozen_teacher_bundle,
)
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402
from latent_theta_gen import (  # noqa: E402
    COMP_NAMES,
    COUNT,
    _component_masses,
    _decode_particle_retrieve,
    _dense_path_loo_neighbor,
    _ic_path_for_dump,
    _load_parts,
)
from ntropy.analysis.disk_density import (  # noqa: E402
    bin_midplane_surface_density,
    bin_plane_density,
    disk_azimuthal_fourier,
)
from ntropy.analysis.density import bin_spherical_density  # noqa: E402
from ood_theta_df_compare import _disk_kinematics  # noqa: E402
from score_residual_f0_kinetics import _a2_rd, _score_vs_ref  # noqa: E402

TEACHER = Path("runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
OUT_DEFAULT = Path("runs/ml/field_maps/latent_theta_gen_2026-08-02")
PAPER = Path("papers/mnras_noneq_ics/figures")

EVAL = {
    "906c4": {
        "hash": "906c4af73543",
        "data": Path(
            "runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz"
        ),
        "short": "906c4",
    },
    "54a8": {
        "hash": "54a8faf836a0",
        "data": Path(
            "runs/mw_morton_corpus_v2/54a8faf836a0/evolution/particles/step_001800.npz"
        ),
        "short": "54a8",
    },
}


def _save_parts(path: Path, parts: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos=np.asarray(parts["pos"], dtype=np.float64),
        vel=np.asarray(parts["vel"], dtype=np.float64),
        mass=np.asarray(parts["mass"], dtype=np.float64),
        component_id=np.asarray(parts["component_id"]),
        eps=np.asarray(
            parts.get("eps", np.full(len(parts["pos"]), 0.05, dtype=np.float64))
        ),
    )


def _stratified_to_n(parts: dict, n_tot: int, rng: np.random.Generator) -> dict:
    """Match evolve_component_slices mix disk:halo:bulge ≈ 4:2:1."""
    cid = parts["component_id"]
    targets = {
        0: int(round(n_tot * COUNT["disk"])),
        1: int(round(n_tot * COUNT["halo"])),
        2: int(round(n_tot * COUNT["bulge"])),
    }
    # Fix rounding so sum == n_tot.
    targets[0] += n_tot - sum(targets.values())
    idx = []
    for c, nt in targets.items():
        sel = np.where(cid == c)[0]
        if len(sel) == 0 or nt <= 0:
            continue
        if len(sel) >= nt:
            take = rng.choice(sel, size=nt, replace=False)
        else:
            take = rng.choice(sel, size=nt, replace=True)
        idx.append(take)
    idx = np.concatenate(idx) if idx else np.arange(min(n_tot, len(cid)))
    out = {k: np.asarray(parts[k])[idx] for k in ("pos", "vel", "mass", "component_id")}
    out["eps"] = np.asarray(
        parts.get("eps", np.full(len(parts["pos"]), 0.05))
    )[idx]
    # Preserve component mass totals.
    for c, name in COMP_NAMES.items():
        m = out["component_id"] == c
        if not np.any(m):
            continue
        src_m = parts["component_id"] == c
        tgt = float(parts["mass"][src_m].sum()) if np.any(src_m) else 0.0
        s = float(out["mass"][m].sum())
        if s > 0 and tgt > 0:
            out["mass"][m] *= tgt / s
    return out


def build_path_loo_ic(
    *,
    eval_path: Path,
    run_hash: str,
    teacher,
    cfg,
    stats,
    pca_mean: np.ndarray,
    pca_w: np.ndarray,
    ranked: list,
    n_tot: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    ic_path = _ic_path_for_dump(eval_path)
    if ic_path is None:
        raise RuntimeError(f"no ic_state for {eval_path}")
    f0 = _load_parts(ic_path, None, rng)
    mass_prior = _component_masses(f0)
    _feat, z_raw, a2_enc = encode_snapshot_features(
        eval_path, teacher=teacher, cfg=cfg, stats=stats, enc_grid=4
    )
    z_eval = (np.asarray(z_raw, dtype=np.float64) - pca_mean) @ pca_w
    nn_path, zd, a2_nn = _dense_path_loo_neighbor(
        eval_path,
        run_hash=run_hash,
        z_eval=z_eval,
        a2_enc=float(a2_enc),
        ranked=ranked,
        pca_mean=pca_mean,
        pca_w=pca_w,
        teacher=teacher,
        cfg=cfg,
        stats=stats,
    )
    gen = _decode_particle_retrieve(nn_path, mass_prior, None, rng)
    gen = _stratified_to_n(gen, n_tot, rng)
    meta = {
        "eval_path": str(eval_path),
        "ic_path": str(ic_path),
        "nn_path": str(nn_path),
        "z_dist": float(zd),
        "a2_enc": float(a2_enc),
        "a2_nn": float(a2_nn),
        "a2_rd_gen": _a2_rd(gen, 2.0),
        "mass": mass_prior,
        "n": int(len(gen["pos"])),
        "decode_mode": "particle_retrieve",
    }
    return gen, meta


def run_evolve(
    *,
    out: Path,
    data_path: Path,
    gen_npz: Path,
    paper_prefix: str,
    n_disk: int,
    evolve_gyr: float,
    force: str,
    omp: int,
    a2_r_eval: float,
    extra_label: str,
    skip_data: bool = False,
    reuse_data_from: Path | None = None,
    data_arm_label: str = "data dump",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/evolve_component_slices.py"),
        "--out",
        str(out),
        "--teacher",
        str(TEACHER),
        "--data-path",
        str(data_path),
        "--methods",
        "",
        "--n-disk",
        str(n_disk),
        "--evolve-gyr",
        str(evolve_gyr),
        "--dt",
        "0.01",
        "--force",
        force,
        "--omp",
        str(omp),
        "--a2-r-eval",
        str(a2_r_eval),
        "--faceon-times",
        (
            "0,0.25,0.5,1.0,2.0"
            if evolve_gyr >= 2.0
            else ("0,0.25,0.5,1.0" if evolve_gyr >= 1.0 else "0,0.25,0.5")
        ),
        "--extra-arm-npz",
        str(gen_npz),
        "--extra-arm-name",
        "latent_gen",
        "--extra-arm-label",
        extra_label,
        "--data-arm-label",
        data_arm_label,
        "--save-final-particles",
        "--paper-figures",
        str(PAPER),
        "--paper-prefix",
        paper_prefix,
        "--timeout-s",
        "1e9",
    ]
    if skip_data:
        cmd.append("--skip-data")
    if reuse_data_from is not None:
        cmd.extend(["--reuse-data-from", str(reuse_data_from)])
    log = out / "evolve.log"
    print(f"=== evolve → {out} ===", flush=True)
    with log.open("w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"evolve failed rc={proc.returncode} log={log}")
    return out


def promote_evolve_figs(prefix: str, short: str) -> None:
    """Normalize paper names to fig_latent_theta_evolve_loo_path_{short}_*."""
    mapping = {
        f"{prefix}_a2_t.png": f"fig_latent_theta_evolve_loo_path_{short}_a2_t.png",
        f"{prefix}_faceon_data.png": (
            f"fig_latent_theta_evolve_loo_path_{short}_faceon_data.png"
        ),
        f"{prefix}_faceon_latent_gen.png": (
            f"fig_latent_theta_evolve_loo_path_{short}_faceon_latent_gen.png"
        ),
        f"{prefix}_profiles_data.png": (
            f"fig_latent_theta_evolve_loo_path_{short}_profiles_data.png"
        ),
        f"{prefix}_profiles_latent_gen.png": (
            f"fig_latent_theta_evolve_loo_path_{short}_profiles_latent_gen.png"
        ),
    }
    # Also pick up data_vs compare if written under run dir later.
    for src_name, dst_name in mapping.items():
        src = PAPER / src_name
        if src.is_file():
            shutil.copy2(src, PAPER / dst_name)
            print(f"  paper ← {dst_name}", flush=True)


def score_kin_pair(
    gen_t0: Path,
    data_t0: Path,
    gen_f: Path | None,
    data_f: Path | None,
    out_png: Path,
    title: str,
) -> dict:
    g0 = _load_parts(gen_t0, None, np.random.default_rng(0))
    d0 = _load_parts(data_t0, None, np.random.default_rng(0))
    # Align N for fair dens noise if needed — score uses kinematics profiles.
    sc0 = _score_vs_ref(g0, d0)
    payload = {
        "t0": {
            "a2_rd_gen": _a2_rd(g0, 2.0),
            "a2_rd_ref": _a2_rd(d0, 2.0),
            "kin_mean_mse": sc0["mse"]["kinetic_mean"],
            "kin_mse": sc0["mse"],
        }
    }
    rows = [("t=0 gen", g0), ("t=0 data", d0)]
    if gen_f and gen_f.is_file() and data_f and data_f.is_file():
        gf = _load_parts(gen_f, None, np.random.default_rng(0))
        df = _load_parts(data_f, None, np.random.default_rng(0))
        scf = _score_vs_ref(gf, df)
        payload["t_final"] = {
            "a2_rd_gen": _a2_rd(gf, 2.0),
            "a2_rd_ref": _a2_rd(df, 2.0),
            "kin_mean_mse": scf["mse"]["kinetic_mean"],
            "kin_mse": scf["mse"],
        }
        rows += [("t=end gen", gf), ("t=end data", df)]
    # Lightweight multi-curve kin panel.
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharex=True)
    keys = [
        ("mean_vphi", r"$\langle v_\varphi\rangle$"),
        ("sig_r", r"$\sigma_R$"),
        ("sig_phi", r"$\sigma_\varphi$"),
        ("sig_z", r"$\sigma_z$"),
    ]
    colors = {
        "t=0 gen": "C0",
        "t=0 data": "0.25",
        "t=end gen": "C3",
        "t=end data": "0.55",
    }
    ls = {
        "t=0 gen": "-",
        "t=0 data": "--",
        "t=end gen": "-",
        "t=end data": "--",
    }
    for ax, (kk, ylab) in zip(axes.ravel(), keys):
        for lab, parts in rows:
            k = _disk_kinematics(parts)
            r = np.asarray(k["r_mid"])
            y = np.asarray(k[kk])
            ok = np.asarray(k["counts"]) >= 20
            ax.plot(r[ok], y[ok], color=colors[lab], ls=ls[lab], lw=1.6, label=lab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].set_xlabel(r"$R$ [kpc]")
    axes[1, 1].set_xlabel(r"$R$ [kpc]")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return payload


def _faceon_disk(parts: dict, half: float = 12.0, bins: int = 160) -> np.ndarray:
    disk = parts["component_id"] == 0
    pos = parts["pos"][disk]
    mass = parts["mass"][disk]
    com = np.average(pos, axis=0, weights=mass)
    dens = bin_plane_density(
        pos - com,
        mass,
        axes=(0, 1),
        n_bins=bins,
        half_extent=half,
    ).density
    return np.asarray(dens, dtype=np.float64)


def plot_newtheta_t0(
    *,
    barred: dict,
    quiet: dict,
    out_faceon: Path,
    out_profiles: Path,
    title: str,
    rd: float = 2.0,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, parts, lab in (
        (axes[0], quiet, r"$f_0(\theta)$ / quiet $z$"),
        (axes[1], barred, r"new $\theta$ + barred $z$"),
    ):
        dens = _faceon_disk(parts)
        show, vmin_s, vmax_s, _ = dens_array_log10(dens, vmax_pct=98.0)
        ax.imshow(
            show,
            origin="lower",
            cmap="magma",
            vmin=vmin_s,
            vmax=vmax_s,
            extent=[-12, 12, -12, 12],
        )
        ax.set_title(lab)
        ax.set_xlabel(r"$x$ [kpc]")
        ax.set_ylabel(r"$y$ [kpc]")
        a2 = _a2_rd(parts, float(rd))
        ax.text(
            0.02,
            0.98,
            rf"$A_2(R_d)={a2:.3f}$",
            transform=ax.transAxes,
            va="top",
            color="w",
            fontsize=9,
        )
    fig.suptitle(title)
    fig.tight_layout()
    out_faceon.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_faceon, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    for parts, lab, c, ls in (
        (quiet, r"$f_0(\theta)$ / quiet $z$", "0.35", "--"),
        (barred, r"barred $z$", "C0", "-"),
    ):
        disk = parts["component_id"] == 0
        pos_d = parts["pos"][disk]
        m_d = parts["mass"][disk]
        com = np.average(pos_d, axis=0, weights=m_d)
        pos_c = pos_d - com
        prof = bin_midplane_surface_density(
            pos_c, m_d, r_max=12.0, n_bins=24, z_max=0.5
        )
        r = np.asarray(prof.r_mid)
        sig = np.asarray(prof.sigma)
        cnt = np.asarray(prof.counts)
        ok = cnt > 0
        axes[0, 0].semilogy(r[ok], np.maximum(sig[ok], 1e-30), color=c, ls=ls, lw=1.7, label=lab)
        # disk_azimuthal_fourier recenters by default; pass already-centered coords
        # with recenter=False to avoid a no-op second COM.
        fout = disk_azimuthal_fourier(
            pos_c,
            m_d,
            m=2,
            r_max=12.0,
            n_bins=24,
            z_max=0.5,
            min_count=10,
            r_eval=float(rd),
            recenter=False,
        )
        axes[0, 1].plot(
            np.asarray(fout["r_mid"]),
            np.asarray(fout["a_m_over_a0"]),
            color=c,
            ls=ls,
            lw=1.7,
            label=lab,
        )
        k = _disk_kinematics(parts)
        kok = np.asarray(k["counts"]) >= 20
        axes[1, 0].plot(
            np.asarray(k["r_mid"])[kok],
            np.asarray(k["mean_vphi"])[kok],
            color=c,
            ls=ls,
            lw=1.7,
            label=lab,
        )
        axes[1, 1].plot(
            np.asarray(k["r_mid"])[kok],
            np.asarray(k["sig_r"])[kok],
            color=c,
            ls=ls,
            lw=1.7,
            label=lab,
        )
    axes[0, 0].set_ylabel(r"disk $\Sigma(R)$")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].set_ylabel(r"$A_2/A_0$")
    axes[1, 0].set_ylabel(r"$\langle v_\varphi\rangle$")
    axes[1, 1].set_ylabel(r"$\sigma_R$")
    for ax in axes.ravel():
        ax.set_xlabel(r"$R$ [kpc]")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title + r" — dens/kin vs quiet $f_0$")
    fig.tight_layout()
    fig.savefig(out_profiles, dpi=150)
    plt.close(fig)
    return {
        "a2_rd_barred": _a2_rd(barred, float(rd)),
        "a2_rd_quiet": _a2_rd(quiet, float(rd)),
        "kin_mse_vs_f0": _score_vs_ref(barred, quiet)["mse"]["kinetic_mean"],
    }


def pick_new_thetas(ranked: list, n: int = 4) -> list[dict]:
    """OOD structural θ ≠ 906c4/54a8 with a barred dump available."""
    anchors = []
    by_hash: dict[str, list] = {}
    for r in ranked:
        h = r.get("run_hash") or _hash_from_path(r["path"])
        th = r.get("theta") or {}
        if not th:
            continue
        tv = theta_vector(th, keys=DEFAULT_THETA_KEYS)
        by_hash.setdefault(h, []).append((float(r["a2"]), Path(r["path"]), th, tv, r))
        if h in ("906c4af73543", "54a8faf836a0"):
            anchors.append(tv)
    if not anchors:
        raise RuntimeError("missing anchor θ")
    anchors_a = np.stack(anchors)
    cands = []
    for h, rows in by_hash.items():
        if h in ("906c4af73543", "54a8faf836a0"):
            continue
        rows = sorted(rows, key=lambda x: -x[0])
        a2_max, path_bar, th, tv, row = rows[0]
        if a2_max < 0.22:
            continue
        # need a quiet-ish same hash or use ic_state as f0
        ic = Path(f"runs/mw_morton_corpus_v2/{h}/ic_state.npz")
        if not ic.is_file():
            continue
        dtheta = float(min(np.linalg.norm(tv - a) for a in anchors_a))
        cands.append(
            {
                "hash": h,
                "dtheta": dtheta,
                "a2_max": a2_max,
                "bar_path": path_bar,
                "ic_path": ic,
                "theta": th,
                "theta_vec": tv,
            }
        )
    cands.sort(key=lambda c: (-c["dtheta"], -c["a2_max"]))
    # diversify by disk.mass
    picked = []
    masses = set()
    for c in cands:
        md = float((c["theta"] or {}).get("disk.mass") or -1)
        if md in masses and len(picked) < n:
            continue
        picked.append(c)
        masses.add(md)
        if len(picked) >= n:
            break
    return picked


def build_newtheta_barred(
    *,
    cand: dict,
    teacher,
    cfg,
    stats,
    pca_mean,
    pca_w,
    ranked,
    n_tot: int,
    rng,
    z_mode: str = "barred",
) -> tuple[dict, dict, dict]:
    """particle_retrieve nearest barred library neighbor for θ family.

    Quiet mode returns GalactICS ``f0(θ)`` (structural quiet baseline).
    """
    rd = float((cand.get("theta") or {}).get("disk.scale_length") or 2.0)
    ic = _load_parts(cand["ic_path"], None, rng)
    mass_prior = _component_masses(ic)
    quiet = _stratified_to_n(ic, n_tot, rng)
    if z_mode != "barred":
        meta = {
            "hash": cand["hash"],
            "z_mode": "quiet",
            "z_src": str(cand["ic_path"]),
            "nn_path": str(cand["ic_path"]),
            "z_dist": 0.0,
            "a2_enc": _a2_rd(quiet, rd),
            "a2_nn": _a2_rd(quiet, rd),
            "a2_rd_gen": _a2_rd(quiet, rd),
            "a2_rd_f0": _a2_rd(quiet, rd),
            "theta": {
                k: cand["theta"].get(k)
                for k in (
                    "disk.mass",
                    "disk.scale_length",
                    "disk_kinematics.toomre_q_target",
                    "halo.v0",
                    "bulge.v0",
                )
            },
            "dtheta_vs_anchors": float(cand["dtheta"]),
        }
        return quiet, quiet, meta

    z_src = cand["bar_path"]
    _feat, z_raw, a2_enc = encode_snapshot_features(
        z_src, teacher=teacher, cfg=cfg, stats=stats, enc_grid=4
    )
    z_eval = (np.asarray(z_raw, dtype=np.float64) - pca_mean) @ pca_w
    nn_path, zd, a2_nn = _dense_path_loo_neighbor(
        z_src,
        run_hash=cand["hash"],
        z_eval=z_eval,
        a2_enc=float(a2_enc),
        ranked=ranked,
        pca_mean=pca_mean,
        pca_w=pca_w,
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        t_min=0.3,
    )
    gen = _decode_particle_retrieve(nn_path, mass_prior, None, rng)
    gen = _stratified_to_n(gen, n_tot, rng)
    meta = {
        "hash": cand["hash"],
        "z_mode": z_mode,
        "z_src": str(z_src),
        "nn_path": str(nn_path),
        "z_dist": float(zd),
        "a2_enc": float(a2_enc),
        "a2_nn": float(a2_nn),
        "a2_rd_gen": _a2_rd(gen, rd),
        "a2_rd_f0": _a2_rd(quiet, rd),
        "theta": {
            k: cand["theta"].get(k)
            for k in (
                "disk.mass",
                "disk.scale_length",
                "disk_kinematics.toomre_q_target",
                "halo.v0",
                "bulge.v0",
            )
        },
        "dtheta_vs_anchors": float(cand["dtheta"]),
    }
    return gen, quiet, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--evolve-gyr", type=float, default=2.0)
    p.add_argument("--newtheta-evolve-gyr", type=float, default=1.0)
    p.add_argument("--force", type=str, default="gpu_bh")
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-part1", action="store_true")
    p.add_argument("--skip-part2", action="store_true")
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument("--systems", type=str, default="906c4,54a8")
    p.add_argument("--n-newtheta", type=int, default=4)
    p.add_argument("--n-newtheta-evolve", type=int, default=2)
    args = p.parse_args()

    out = args.out
    gates = out / "gates"
    samples = out / "samples"
    logs = out / "logs"
    for d in (gates, samples, logs, out / "figs"):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_tot = int(round(args.n_disk * 7 / 4))
    print(f"=== latent_theta evolve suite n_disk={args.n_disk} N={n_tot} ===", flush=True)

    # PCA basis from existing library codes (keys: mean, pca_w).
    lib_codes = out / "feature_library_codes.npz"
    if not lib_codes.is_file():
        lib_codes = Path("runs/ml/field_maps/latent_theta_gen_2026-08-02/feature_library_codes.npz")
    codes = np.load(lib_codes, allow_pickle=True)
    pca_mean = np.asarray(
        codes["mean"] if "mean" in codes.files else codes["pca_mean"],
        dtype=np.float64,
    )
    pca_w = np.asarray(
        codes["pca_w"] if "pca_w" in codes.files else codes["pca_components"],
        dtype=np.float64,
    )
    if pca_w.shape[0] != pca_mean.shape[0] and pca_w.shape[1] == pca_mean.shape[0]:
        pca_w = pca_w.T
    ranked = json.loads(RANK.read_text())
    teacher, cfg, stats = load_frozen_teacher_bundle(TEACHER)

    suite = {"part1": {}, "part2": {}, "n_disk": args.n_disk, "n_tot": n_tot}

    if not args.skip_part1:
        for key in [s.strip() for s in args.systems.split(",") if s.strip()]:
            spec = EVAL[key]
            print(f"\n===== PART1 {key} =====", flush=True)
            gen, meta = build_path_loo_ic(
                eval_path=spec["data"],
                run_hash=spec["hash"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                pca_mean=pca_mean,
                pca_w=pca_w,
                ranked=ranked,
                n_tot=n_tot,
                rng=rng,
            )
            gen_path = samples / f"loo_path_{key}_fullN_gen.npz"
            _save_parts(gen_path, gen)
            (logs / f"loo_path_{key}_fullN_meta.json").write_text(
                json.dumps(meta, indent=2)
            )
            print(
                f"  wrote {gen_path.name} N={meta['n']} "
                f"A2Rd={meta['a2_rd_gen']:.3f} nn={Path(meta['nn_path']).name}",
                flush=True,
            )
            # Also score t0 dens/kin vs data quickly.
            data_parts = _stratified_to_n(
                _load_parts(spec["data"], None, rng), n_tot, rng
            )
            data_t0 = samples / f"data_{key}_t0_sub.npz"
            _save_parts(data_t0, data_parts)
            kin_png = PAPER / f"fig_latent_theta_evolve_loo_path_{key}_kin.png"
            kin = score_kin_pair(
                gen_path,
                data_t0,
                None,
                None,
                kin_png,
                title=f"path-LOO gen vs data kin ({key}, t=0)",
            )
            suite["part1"][key] = {"meta": meta, "kin_t0": kin, "gen_path": str(gen_path)}

            if args.skip_evolve:
                continue
            evo_out = gates / f"evolve_2gyr_loo_path_{key}"
            t0 = time.time()
            run_evolve(
                out=evo_out,
                data_path=spec["data"],
                gen_npz=gen_path,
                paper_prefix=f"fig_latent_theta_evolve_loo_path_{key}",
                n_disk=args.n_disk,
                evolve_gyr=args.evolve_gyr,
                force=args.force,
                omp=args.omp,
                a2_r_eval=2.0,
                extra_label=f"path-LOO latent gen ({key})",
            )
            promote_evolve_figs(f"fig_latent_theta_evolve_loo_path_{key}", key)
            # Prefer canonical names already matching prefix.
            verdict = json.loads((evo_out / "verdict.json").read_text())
            # kin at final if particles saved
            kin_final_png = (
                PAPER / f"fig_latent_theta_evolve_loo_path_{key}_kin_t0_tend.png"
            )
            kin2 = score_kin_pair(
                evo_out / "particles_latent_gen_t0.npz",
                evo_out / "particles_data_t0.npz",
                evo_out / "particles_latent_gen_final.npz",
                evo_out / "particles_data_final.npz",
                kin_final_png,
                title=f"path-LOO gen vs data kin ({key}, t=0 & t={args.evolve_gyr:g} Gyr)",
            )
            # Copy compare profiles if present
            cmp = evo_out / "profiles_components_data_vs_latent_gen.png"
            if cmp.is_file():
                shutil.copy2(
                    cmp,
                    PAPER
                    / f"fig_latent_theta_evolve_loo_path_{key}_profiles_data_vs_gen.png",
                )
            suite["part1"][key].update(
                {
                    "evolve": {
                        "out": str(evo_out),
                        "wall_s": time.time() - t0,
                        "arms": {
                            tag: {
                                "a2_pre": row.get("a2_pre"),
                                "a2_post": row.get("a2_post"),
                                "com_drift_kpc": row.get("com_drift_kpc"),
                            }
                            for tag, row in verdict.get("arms", {}).items()
                            if row.get("ok")
                        },
                    },
                    "kin_evolve": kin2,
                }
            )

    if not args.skip_part2:
        print("\n===== PART2 new θ =====", flush=True)
        cands = pick_new_thetas(ranked, n=args.n_newtheta)
        suite["part2"]["candidates"] = []
        evolve_budget = args.n_newtheta_evolve
        for i, cand in enumerate(cands):
            tag = cand["hash"][:5]
            print(
                f"  newθ[{i}] {cand['hash'][:8]} dθ={cand['dtheta']:.2f} "
                f"A2max={cand['a2_max']:.3f} Md={cand['theta'].get('disk.mass')} "
                f"Rd={cand['theta'].get('disk.scale_length')} "
                f"Q={cand['theta'].get('disk_kinematics.toomre_q_target')}",
                flush=True,
            )
            rd = float(cand["theta"].get("disk.scale_length") or 2.0)
            try:
                barred, quiet_f0, meta = build_newtheta_barred(
                    cand=cand,
                    teacher=teacher,
                    cfg=cfg,
                    stats=stats,
                    pca_mean=pca_mean,
                    pca_w=pca_w,
                    ranked=ranked,
                    n_tot=n_tot,
                    rng=rng,
                    z_mode="barred",
                )
                quiet_z, _, meta_q = build_newtheta_barred(
                    cand=cand,
                    teacher=teacher,
                    cfg=cfg,
                    stats=stats,
                    pca_mean=pca_mean,
                    pca_w=pca_w,
                    ranked=ranked,
                    n_tot=n_tot,
                    rng=rng,
                    z_mode="quiet",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP build: {exc}", flush=True)
                continue
            # Prefer f0 baseline for quiet panel; also keep quiet-z sample.
            quiet = quiet_f0
            bpath = samples / f"newtheta_{tag}_barred_gen.npz"
            qpath = samples / f"newtheta_{tag}_f0_quiet.npz"
            qzpath = samples / f"newtheta_{tag}_quiet_z_gen.npz"
            _save_parts(bpath, barred)
            _save_parts(qpath, quiet)
            _save_parts(qzpath, quiet_z)
            face = PAPER / f"fig_latent_theta_newtheta_{tag}_faceon_t0.png"
            prof = PAPER / f"fig_latent_theta_newtheta_{tag}_profiles_t0.png"
            t0sc = plot_newtheta_t0(
                barred=barred,
                quiet=quiet,
                out_faceon=face,
                out_profiles=prof,
                title=(
                    rf"new $\theta$ {cand['hash'][:8]} "
                    rf"(Md={meta['theta'].get('disk.mass')}, "
                    rf"Rd={meta['theta'].get('disk.scale_length')}, "
                    rf"Q={meta['theta'].get('disk_kinematics.toomre_q_target')})"
                ),
                rd=rd,
            )
            t0sc["a2_rd_quiet_z"] = _a2_rd(quiet_z, rd)
            t0sc["a2_rd_f0"] = _a2_rd(quiet_f0, rd)
            meta["quiet_z"] = {
                "nn_path": meta_q.get("nn_path"),
                "a2_rd_gen": meta_q.get("a2_rd_gen"),
            }
            row = {
                "hash": cand["hash"],
                "tag": tag,
                "meta": meta,
                "t0": t0sc,
                "figs": {"faceon": str(face), "profiles": str(prof)},
            }
            if (
                not args.skip_evolve
                and evolve_budget > 0
                and meta["a2_rd_gen"] > 0.15
            ):
                evo_out = gates / f"evolve_newtheta_{tag}_barred"
                # Evolve barred gen; data arm = quiet f0 control
                run_evolve(
                    out=evo_out,
                    data_path=qpath,
                    gen_npz=bpath,
                    paper_prefix=f"fig_latent_theta_newtheta_{tag}",
                    n_disk=args.n_disk,
                    evolve_gyr=args.newtheta_evolve_gyr,
                    force=args.force,
                    omp=args.omp,
                    a2_r_eval=float(
                        meta["theta"].get("disk.scale_length") or 2.0
                    ),
                    extra_label=f"newθ barred z ({tag})",
                    data_arm_label=r"f0(θ) quiet baseline",
                )
                # rename a2 panel
                src = PAPER / f"fig_latent_theta_newtheta_{tag}_a2_t.png"
                if src.is_file():
                    shutil.copy2(
                        src,
                        PAPER / f"fig_latent_theta_newtheta_{tag}_evolve_a2_t.png",
                    )
                v = json.loads((evo_out / "verdict.json").read_text())
                row["evolve"] = {
                    "out": str(evo_out),
                    "gyr": args.newtheta_evolve_gyr,
                    "arms": {
                        t: {
                            "a2_pre": r.get("a2_pre"),
                            "a2_post": r.get("a2_post"),
                        }
                        for t, r in v.get("arms", {}).items()
                        if r.get("ok")
                    },
                }
                evolve_budget -= 1
            suite["part2"]["candidates"].append(row)

    (logs / "evolve_suite_verdict.json").write_text(json.dumps(suite, indent=2, default=str))
    print("=== DONE suite ===", flush=True)
    print(json.dumps(suite, indent=2, default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
