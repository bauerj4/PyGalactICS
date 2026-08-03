#!/usr/bin/env python3
"""
Fine-tune a trained Morton VAE with the virial consistency term, then
regenerate face-on PNGs (data | sample | evolved) under vae_prod_examples.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(8)

from galacticsics.ml.morton.dataset import DEFAULT_THETA_KEYS, MortonSnapshotDataset, collate_morton_batch
from galacticsics.ml.morton.index import write_snapshot_manifest
from galacticsics.ml.morton.tokenize import center_phase_space, subsample_stratified
from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig
from ntropy.analysis.disk_density import disk_azimuthal_fourier
from ntropy.forces.bhtree_c import compute_forces_bh_c
from ntropy.softening import virial_diagnostic

OUT = Path("runs/ml/vae_prod_examples")
WORK = Path("runs/mw_morton_corpus_v2")
CKPT = OUT / "sequence_vae_prod.pt"
DEVICE = "cpu"
N_TRAIN = 2048
MAX_SNAP = 80
EPOCHS = 24
PATIENCE = 10
BATCH = 2
N_GEN = 100_000
N_EVOLVE = 8_000
EVOLVE_GYR = 0.06
DT = 0.03
SEED = 0
LR = 3e-4

EXAMPLES = [
    {
        "name": "bar_54a8_late",
        "run": "54a8faf836a0",
        "dump": "evolution/particles/step_001700.npz",
        "label": "barred disk (A₂≈0.5)",
    },
    {
        "name": "quiet_ic_081e",
        "run": "081ed8af4b2b",
        "dump": "ic_state.npz",
        "label": "quiet IC (A₂≈0.01)",
    },
]


def _collate(batch):
    b = collate_morton_batch(batch)
    return {
        "c": torch.as_tensor(b["c"], dtype=torch.long),
        "dm": torch.as_tensor(b["dm"], dtype=torch.float32),
        "dx": torch.as_tensor(b["dx"], dtype=torch.float32),
        "v": torch.as_tensor(b["v"], dtype=torch.float32),
        "theta": torch.as_tensor(b["theta"], dtype=torch.float32),
    }


def _theta_from_model(run_dir: Path, t_gyr: float) -> np.ndarray:
    raw = json.loads((run_dir / "model.json").read_text())
    theta: dict[str, float] = {}
    for comp in ("halo", "disk", "bulge", "disk_kinematics"):
        block = raw.get(comp) or {}
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(v, (int, float)) and k != "enabled":
                    theta[f"{comp}.{k}"] = float(v)
    theta["t_gyr"] = float(t_gyr) if np.isfinite(t_gyr) else 0.0
    return np.asarray([float(theta.get(k, 0.0)) for k in DEFAULT_THETA_KEYS], dtype=np.float64)


def _t_gyr_from_dump(path: Path) -> float:
    if path.name.startswith("ic"):
        return 0.0
    try:
        step = int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return float("nan")
    csv_path = path.parents[2] / "evolution" / "diagnostics.csv"
    if csv_path.is_file():
        import csv

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row.get("step", -1))) == step and "t_gyr" in row:
                    return float(row["t_gyr"])
    return step * 0.001


def _a2(pos, mass, type_id) -> float:
    m = (type_id == 0) | (type_id == 3)
    if m.sum() < 500:
        m = np.ones(len(type_id), dtype=bool)
    out = disk_azimuthal_fourier(
        pos[m], mass[m], m=2, n_bins=10, r_max=12.0, z_max=0.5, min_count=40
    )
    return float(out["a_m_over_a0_median"])


def _virial_ratio(pos, vel, mass, eps, *, max_particles=4096) -> float:
    return float(
        virial_diagnostic(
            pos, vel, mass, eps, max_particles=max_particles, rng=np.random.default_rng(0)
        )["virial_ratio"]
    )


def _faceon_panel(ax, pos, type_id, title: str, n_show: int = 20000):
    rng = np.random.default_rng(0)
    colors = {0: ("disk", "C0"), 1: ("halo", "C1"), 2: ("bulge", "C2"), 3: ("disk", "C0")}
    for cid, (name, color) in colors.items():
        idx = np.flatnonzero(type_id == cid)
        if idx.size == 0:
            continue
        if idx.size > n_show // 3:
            idx = rng.choice(idx, n_show // 3, replace=False)
        ax.scatter(pos[idx, 0], pos[idx, 1], s=0.25, alpha=0.3, c=color, label=name, rasterized=True)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def _load_snapshot(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    pos = np.asarray(data["pos"], dtype=np.float64)
    vel = np.asarray(data["vel"], dtype=np.float64)
    mass = np.asarray(data["mass"], dtype=np.float64)
    tid = np.asarray(data["type_id"], dtype=np.int32)
    ml = np.zeros_like(tid)
    ml[tid == 3] = 0
    ml[tid == 1] = 1
    ml[tid == 2] = 2
    if "tags" in data.files:
        tags = data["tags"]
        ml = np.zeros(len(tags), dtype=np.int32)
        ml[tags == "disk"] = 0
        ml[tags == "halo"] = 1
        ml[tags == "bulge"] = 2
    pos, vel = center_phase_space(pos, vel, mass)
    eps = np.asarray(data["eps"], dtype=np.float64) if "eps" in data.files else np.full(len(pos), 0.1)
    return {"pos": pos, "vel": vel, "mass": mass, "type_id": ml, "eps": eps}


def leapfrog_bh_c(pos, vel, mass, eps, *, dt=DT, n_steps=None, theta=0.7):
    n_steps = int(n_steps if n_steps is not None else max(1, int(round(EVOLVE_GYR / dt))))
    acc = compute_forces_bh_c(pos, mass, eps, theta=theta)
    vel = vel + 0.5 * dt * acc
    for _ in range(n_steps):
        pos = pos + dt * vel
        acc = compute_forces_bh_c(pos, mass, eps, theta=theta)
        vel = vel + dt * acc
    vel = vel - 0.5 * dt * acc
    return pos, vel


def main() -> int:
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint {CKPT}")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg_dict = dict(blob["config"])
    # Enable virial (and keep morphology weights)
    cfg_dict["lambda_virial"] = 2.0
    cfg_dict["virial_n_sub"] = 256
    cfg_dict["lambda_virial_target"] = 1.5
    cfg_dict["virial_target_ratio"] = 1.0
    cfg_dict["virial_eps"] = 0.1
    # Drop unknown keys for forward-compat
    known = {f.name for f in SequenceVAEConfig.__dataclass_fields__.values()}
    cfg = SequenceVAEConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = SequenceVAE(cfg)
    model.load_state_dict(blob["model"], strict=False)
    model.to(DEVICE)

    manifest = write_snapshot_manifest(WORK, OUT / "manifest.json")
    ds = MortonSnapshotDataset(
        manifest,
        n_particles=N_TRAIN,
        split=None,
        seed=SEED,
        max_snapshots=MAX_SNAP,
        center=True,
        augment=True,
        order="random",
    )
    print(
        f"fine-tune virial on {len(ds)} snaps  N={N_TRAIN}  epochs≤{EPOCHS}  "
        f"λ_virial={cfg.lambda_virial}  n_sub={cfg.virial_n_sub}",
        flush=True,
    )
    print("preloading…", flush=True)
    ds.preload()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    stall = 0
    model.train()
    for epoch in range(EPOCHS):
        ds.set_epoch(1000 + epoch)
        losses = []
        last = None
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
            metrics = model.loss(batch, out)
            opt.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = metrics
            losses.append(float(metrics["loss"].detach()))
        mean_loss = float(np.mean(losses))
        row = {
            "epoch": float(epoch + 1),
            "loss": mean_loss,
            "chamfer": float(last["chamfer"].detach()),
            "virial": float(last["virial"].detach()),
            "ratio_pred": float(last["ratio_pred"].detach()),
            "ratio_data": float(last["ratio_data"].detach()),
            "nax": float(last["nonaxisym"].detach()),
        }
        history.append(row)
        print(
            f"  ft {epoch + 1}/{EPOCHS}  loss={mean_loss:.2f}  ch={row['chamfer']:.2f}  "
            f"vir={row['virial']:.3f}  2K/|W|={row['ratio_pred']:.2f}/{row['ratio_data']:.2f}  "
            f"nax={row['nax']:.2f}",
            flush=True,
        )
        if mean_loss < best_loss * 0.998:
            stall = 0
        else:
            stall += 1
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stall >= PATIENCE:
            print(f"  fine-tune converged @ epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    out_ckpt = OUT / "sequence_vae_prod_virial.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "theta_keys": blob.get("theta_keys", ds.theta_keys),
            "history": history,
            "best_loss": best_loss,
            "base_ckpt": str(CKPT),
        },
        out_ckpt,
    )
    (OUT / "train_history_virial_ft.json").write_text(json.dumps(history, indent=2))
    print(f"wrote {out_ckpt}  best_loss={best_loss:.2f}", flush=True)

    model.eval()
    summary = []
    for ex in EXAMPLES:
        run_dir = WORK / ex["run"]
        dump_path = run_dir / ex["dump"]
        if not dump_path.is_file():
            continue
        print(f"\n=== {ex['name']} (virial ft) ===", flush=True)
        data = _load_snapshot(dump_path)
        t_gyr = _t_gyr_from_dump(dump_path)
        theta = _theta_from_model(run_dir, t_gyr)
        a2_data = _a2(data["pos"], data["mass"], data["type_id"])
        with torch.no_grad():
            th = torch.as_tensor(theta[None], dtype=torch.float32, device=DEVICE)
            tok = model.generate(th, n=N_GEN, chunk_size=4096)
        gen = {
            "pos": tok["dx"][0].astype(np.float64),
            "vel": tok["v"][0].astype(np.float64),
            "type_id": tok["c"][0].astype(np.int32),
            "mass": np.full(N_GEN, 1.0 / N_GEN),
            "eps": np.full(N_GEN, 0.08),
        }
        gen["pos"], gen["vel"] = center_phase_space(gen["pos"], gen["vel"], gen["mass"])
        a2_gen = _a2(gen["pos"], gen["mass"], gen["type_id"])
        vr_gen = _virial_ratio(gen["pos"], gen["vel"], gen["mass"], gen["eps"])
        rng = np.random.default_rng(SEED)
        ev_idx = subsample_stratified(gen["type_id"].astype(np.int64), N_EVOLVE, rng=rng)
        sub_pos = gen["pos"][ev_idx].copy()
        sub_vel = gen["vel"][ev_idx].copy()
        sub_tid = gen["type_id"][ev_idx]
        sub_mass = np.full(N_EVOLVE, 1.0 / N_EVOLVE)
        sub_eps = gen["eps"][ev_idx].copy()
        vr_pre = _virial_ratio(sub_pos, sub_vel, sub_mass, sub_eps, max_particles=N_EVOLVE)
        epos, evel = leapfrog_bh_c(sub_pos, sub_vel, sub_mass, sub_eps)
        a2_evo = _a2(epos, sub_mass, sub_tid)
        vr_evo = _virial_ratio(epos, evel, sub_mass, sub_eps, max_particles=N_EVOLVE)
        com = float(np.linalg.norm(epos.mean(0)))
        print(
            f"  A2 data/gen/evo={a2_data:.3f}/{a2_gen:.3f}/{a2_evo:.3f}  "
            f"Q gen/pre/evo={vr_gen:.2f}/{vr_pre:.2f}/{vr_evo:.2f}  COM={com:.3f}",
            flush=True,
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        d_idx = rng.choice(len(data["pos"]), size=min(40000, len(data["pos"])), replace=False)
        _faceon_panel(axes[0], data["pos"][d_idx], data["type_id"][d_idx], f"data  A₂={a2_data:.2f}")
        _faceon_panel(
            axes[1], gen["pos"], gen["type_id"], f"VAE+virial  A₂={a2_gen:.2f}  Q={vr_gen:.2f}"
        )
        _faceon_panel(
            axes[2], epos, sub_tid, f"OpenMP BH {EVOLVE_GYR} Gyr  A₂={a2_evo:.2f}  Q={vr_evo:.2f}"
        )
        axes[0].legend(loc="upper right", fontsize=7, markerscale=3)
        fig.suptitle(f"{ex['label']} · virial fine-tune · {ex['run']}", y=1.02)
        fig.tight_layout()
        png = OUT / f"{ex['name']}_faceon.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {png}", flush=True)
        np.savez_compressed(
            OUT / f"{ex['name']}_particles.npz",
            data_pos=data["pos"][d_idx],
            data_type=data["type_id"][d_idx],
            gen_pos=gen["pos"],
            gen_type=gen["type_id"],
            evo_pos=epos,
            evo_type=sub_tid,
            a2_data=a2_data,
            a2_gen=a2_gen,
            a2_evo=a2_evo,
            virial_gen=vr_gen,
            virial_pre=vr_pre,
            virial_evo=vr_evo,
            com=com,
        )
        summary.append(
            {
                "example": ex["name"],
                "a2_data": a2_data,
                "a2_gen": a2_gen,
                "a2_evo": a2_evo,
                "virial_gen": vr_gen,
                "virial_pre": vr_pre,
                "virial_evo": vr_evo,
                "com": com,
            }
        )
    (OUT / "virial_ft_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'virial_ft_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
