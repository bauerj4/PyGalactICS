#!/usr/bin/env python3
"""Phase B: learn midplane dens contrast δ from (f0, evolved) pairs.

Default: morph-conditioned ``CondDelta`` (3ch) — soft morph contrast hint
plus f0 dens/axisym → sharp m=2 contrast target. Soft hint is a blurred
evolved deposit (train-time; no teacher required). Inference feeds teacher
or deposit morph contrast via ``morph_source=phase_b``.

Also supports f0-only ``TinyDelta`` (``--arch tiny``) for ablation.

Primary metric at gate time remains particle A₂(R_d); training adds an
optional soft map-A₂(R_d) loss on ax·(1+pred).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galacticsics.ml.fields.binning import (  # noqa: E402
    MultiScaleSliceConfig,
    bin_multiscale_slice_stacks,
)
from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.fields.residual_delta import (  # noqa: E402
    build_delta_model,
    midplane_contrast_from_dens,
)
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402


def _load(path: Path):
    with np.load(path, allow_pickle=True) as d:
        pos = np.asarray(d["pos"], dtype=np.float64)
        vel = np.asarray(d["vel"], dtype=np.float64)
        mass = np.asarray(d["mass"], dtype=np.float64)
        cid = _component_ids(d.get("tags"), d.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    return pos, vel, mass, cid


def _midplane_dens(pos, vel, mass, cid, cfg) -> np.ndarray:
    maps = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    stack, _meta = maps["disk"]
    g = cfg.grid_for("disk")
    dens_i = g.moment_keys.index("dens")
    n_mom = len(g.moment_keys)
    iz = g.n_z // 2
    dens = np.asarray(stack[iz * n_mom + dens_i], dtype=np.float32)
    if dens.ndim != 2:
        raise ValueError(f"expected 2D midplane dens, got shape {dens.shape}")
    return dens


def _map_a2_at_r(dens: torch.Tensor, *, r_eval: float, r_max: float) -> torch.Tensor:
    """Differentiable-ish ring A₂(R≈r_eval) on a batch of dens maps (B,H,W)."""
    # Use complex m=2 moments in a radial annulus around r_eval.
    b, h, w = dens.shape
    device = dens.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dens.dtype),
        torch.arange(w, device=device, dtype=dens.dtype),
        indexing="ij",
    )
    cx = (h - 1) / 2.0
    scale = (2.0 * float(r_max)) / float(h)
    x = (xx - cx) * scale
    y = (yy - cx) * scale
    R = torch.sqrt(x * x + y * y)
    phi = torch.atan2(y, x)
    lo = max(0.5, float(r_eval) - 0.75)
    hi = float(r_eval) + 0.75
    mask = ((R >= lo) & (R < hi)).to(dens.dtype)
    c2 = torch.cos(2.0 * phi)
    s2 = torch.sin(2.0 * phi)
    m = dens * mask
    a0 = m.sum(dim=(-2, -1)).clamp_min(1e-8)
    a2c = (m * c2).sum(dim=(-2, -1))
    a2s = (m * s2).sum(dim=(-2, -1))
    a2 = torch.sqrt(a2c * a2c + a2s * a2s)
    # plane_density uses 2*|c2| style; match relative amplitude scale ≈ 2 a2/a0
    return (2.0 * a2 / a0).clamp(0.0, 2.0)


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        cfg,
        *,
        eps: float = 1e-8,
        contrast_clip: float = 3.0,
        m2_only: bool = True,
        r_max: float = 14.0,
        hint_blur_sigma: float = 2.0,
        conditioned: bool = True,
        r_eval: float = 2.0,
        augment_blur: bool = True,
    ):
        self.pairs = pairs
        self.cfg = cfg
        self.eps = eps
        self.contrast_clip = float(contrast_clip)
        self.m2_only = bool(m2_only)
        self.r_max = float(r_max)
        self.hint_blur_sigma = float(hint_blur_sigma)
        self.conditioned = bool(conditioned)
        self.r_eval = float(r_eval)
        self.augment_blur = bool(augment_blur)
        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None

    def preload(self) -> None:
        self._cache = []
        for ic, ev in self.pairs:
            dens0 = _midplane_dens(*_load(ic), self.cfg)
            dens1 = _midplane_dens(*_load(ev), self.cfg)
            self._cache.append((dens0, dens1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if self._cache is None:
            ic, ev = self.pairs[i]
            dens0 = _midplane_dens(*_load(ic), self.cfg)
            dens1 = _midplane_dens(*_load(ev), self.cfg)
        else:
            dens0, dens1 = self._cache[i]
        from galacticsics.ml.fields.resample import _axisym_and_rbin

        ax0, _ = _axisym_and_rbin(np.maximum(dens0.astype(np.float64), 0.0))
        y = midplane_contrast_from_dens(
            dens1,
            eps=self.eps,
            contrast_clip=self.contrast_clip,
            m2_only=self.m2_only,
            r_max=self.r_max,
            blur_sigma=0.0,
        )
        chans = [
            np.log1p(np.maximum(dens0, 0.0)),
            np.log1p(np.maximum(ax0.astype(np.float32), 0.0)),
        ]
        if self.conditioned:
            if self.augment_blur and self.hint_blur_sigma > 0:
                # Mix identity + soft hints so deposit/teacher/blend all in-domain.
                u = float(np.random.random())
                if u < 0.2:
                    sig = 0.0
                elif u < 0.45:
                    sig = 1.0
                else:
                    sig = float(
                        np.random.uniform(1.0, max(self.hint_blur_sigma, 1.0) * 1.75)
                    )
            else:
                sig = float(self.hint_blur_sigma)
            hint = midplane_contrast_from_dens(
                dens1,
                eps=self.eps,
                contrast_clip=self.contrast_clip,
                m2_only=self.m2_only,
                r_max=self.r_max,
                blur_sigma=sig,
            )
            chans.append(hint)
        x = np.stack(chans, axis=0).astype(np.float32)
        w = (ax0 / (float(np.mean(ax0)) + self.eps)).astype(np.float32)[None]
        w = np.clip(w, 0.05, 10.0)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None]),
            torch.from_numpy(w),
            torch.from_numpy(ax0.astype(np.float32)),
            torch.tensor(self.r_eval, dtype=torch.float32),
        )


def discover_pairs(
    corpus: Path,
    max_pairs: int,
    *,
    min_a2: float = 0.0,
    exclude_hashes: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Discover (ic, evolved) pairs; optionally keep only evolved A₂ ≥ min_a2."""
    from ntropy.analysis.disk_density import disk_azimuthal_fourier

    exclude_hashes = exclude_hashes or set()
    pairs = []
    for run in sorted(corpus.iterdir()):
        if run.name in exclude_hashes:
            continue
        ic = run / "ic_state.npz"
        ev_dir = run / "evolution" / "particles"
        if not ic.is_file() or not ev_dir.is_dir():
            continue
        steps = sorted(ev_dir.glob("step_*.npz"))
        if not steps:
            continue
        ev = steps[-1]
        if min_a2 > 0.0:
            try:
                with np.load(ev, allow_pickle=True) as d:
                    pos = np.asarray(d["pos"], dtype=np.float64)
                    mass = np.asarray(d["mass"], dtype=np.float64)
                    cid = _component_ids(d.get("tags"), d.get("type_id"), pos.shape[0])
                disk = cid == 0
                if not np.any(disk):
                    continue
                fout = disk_azimuthal_fourier(
                    pos[disk],
                    mass[disk],
                    m=2,
                    r_max=12.0,
                    n_bins=12,
                    z_max=0.5,
                    min_count=10,
                )
                a2 = float(fout["a_m_over_a0_median"])
                if not np.isfinite(a2) or a2 < float(min_a2):
                    continue
            except Exception:
                continue
        pairs.append((ic, ev))
        if len(pairs) >= max_pairs:
            break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path("runs/mw_morton_corpus_v2"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-pairs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--contrast-clip", type=float, default=3.0)
    p.add_argument("--hint-blur-sigma", type=float, default=2.0)
    p.add_argument(
        "--arch",
        choices=("cond", "tiny"),
        default="cond",
        help="cond=3ch morph-hint U-Net; tiny=2ch f0-only.",
    )
    p.add_argument(
        "--a2-loss-weight",
        type=float,
        default=0.15,
        help="Weight on soft map A₂(R_d) matching loss (0 disables).",
    )
    p.add_argument("--a2-r-eval", type=float, default=2.0)
    p.add_argument(
        "--min-a2",
        type=float,
        default=0.12,
        help="Keep only pairs whose evolved median A₂ ≥ this (barred filter).",
    )
    p.add_argument(
        "--holdout",
        type=str,
        default="906c4af73543,54a8faf836a0,ac023abb258d",
        help="Comma-separated corpus hashes excluded from training (gate systems).",
    )
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    want = str(args.device).lower()
    if want.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "FATAL: --device cuda but torch.cuda.is_available() is False. "
                "Re-run outside the sandbox (Shell required_permissions: [\"all\"]); "
                "sandbox strips /dev/nvidia* and silently falls back to CPU."
            )
        device = torch.device(args.device)
        print(
            f"CUDA OK: device={device} name={torch.cuda.get_device_name(0)} "
            f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GiB",
            flush=True,
        )
        # Touch CUDA so nvidia-smi shows this PID before preload.
        _ = torch.zeros(1, device=device)
    else:
        device = torch.device("cpu")
        print("WARNING: training on CPU (device=cpu)", flush=True)

    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=64, include_potential=False
    )
    holdout = {h.strip() for h in str(args.holdout).split(",") if h.strip()}
    pairs = discover_pairs(
        args.corpus,
        args.max_pairs,
        min_a2=float(args.min_a2),
        exclude_hashes=holdout,
    )
    if len(pairs) < 4:
        raise SystemExit(f"need ≥4 pairs, found {len(pairs)} (min_a2={args.min_a2})")
    n_val = max(2, len(pairs) // 5)
    # Deterministic split by sorted order already; take ends as val.
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
    conditioned = args.arch == "cond"
    print(
        f"pairs train={len(train_pairs)} val={len(val_pairs)} "
        f"min_a2={args.min_a2} arch={args.arch} holdout={sorted(holdout)}",
        flush=True,
    )

    print("preloading train…", flush=True)
    ds_tr = PairDataset(
        train_pairs,
        cfg,
        contrast_clip=args.contrast_clip,
        hint_blur_sigma=args.hint_blur_sigma,
        conditioned=conditioned,
        r_eval=args.a2_r_eval,
        augment_blur=True,
    )
    ds_tr.preload()
    print("preloading val…", flush=True)
    ds_va = PairDataset(
        val_pairs,
        cfg,
        contrast_clip=args.contrast_clip,
        hint_blur_sigma=args.hint_blur_sigma,
        conditioned=conditioned,
        r_eval=args.a2_r_eval,
        augment_blur=False,
    )
    ds_va.preload()

    dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    n_in = 3 if conditioned else 2
    model = build_delta_model(arch=args.arch, width=int(args.width), n_in=n_in).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    hist = []
    best_va = float("inf")
    best_state = None
    a2_w = float(args.a2_loss_weight)
    r_max_map = float(cfg.grid_for("disk").r_max)

    def _batch_loss(x, y, w, ax0):
        pred = model(x)
        err = (pred - y) ** 2
        mse = (err * w).sum() / w.sum().clamp_min(1e-8)
        loss = mse
        extras = {"mse_w": float(mse.detach().item())}
        if a2_w > 0.0:
            dens_tgt = ax0 * (1.0 + y[:, 0].clamp(-3, 3))
            dens_pred = ax0 * (1.0 + pred[:, 0].clamp(-3, 3))
            a2_t = _map_a2_at_r(dens_tgt, r_eval=float(args.a2_r_eval), r_max=r_max_map)
            a2_p = _map_a2_at_r(
                dens_pred, r_eval=float(args.a2_r_eval), r_max=r_max_map
            )
            a2_loss = ((a2_p - a2_t) ** 2).mean()
            loss = loss + a2_w * a2_loss
            extras["a2_loss"] = float(a2_loss.detach().item())
            extras["a2_pred_mean"] = float(a2_p.mean().detach().item())
            extras["a2_tgt_mean"] = float(a2_t.mean().detach().item())
        return loss, extras

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        n = 0
        for x, y, w, ax0, _re in dl:
            x, y, w, ax0 = x.to(device), y.to(device), w.to(device), ax0.to(device)
            loss, _ = _batch_loss(x, y, w, ax0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_tr += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        sched.step()
        model.eval()
        with torch.no_grad():
            xv = torch.stack([ds_va[i][0] for i in range(len(ds_va))]).to(device)
            yv = torch.stack([ds_va[i][1] for i in range(len(ds_va))]).to(device)
            wv = torch.stack([ds_va[i][2] for i in range(len(ds_va))]).to(device)
            axv = torch.stack([ds_va[i][3] for i in range(len(ds_va))]).to(device)
            loss_va_t, ex = _batch_loss(xv, yv, wv, axv)
            loss_va = float(loss_va_t.item())
            pred_v = model(xv)
            loss_va_uw = float(((pred_v - yv) ** 2).mean().item())
        row = {
            "epoch": ep,
            "train_loss": loss_tr / max(n, 1),
            "val_loss": loss_va,
            "val_mse": loss_va_uw,
            **{f"val_{k}": v for k, v in ex.items()},
        }
        hist.append(row)
        print(
            f"ep {ep:03d} train={row['train_loss']:.4g} "
            f"val={row['val_loss']:.4g} mse={row['val_mse']:.4g}"
            + (
                f" a2p={ex.get('a2_pred_mean', float('nan')):.3f}"
                f" a2t={ex.get('a2_tgt_mean', float('nan')):.3f}"
                if a2_w > 0
                else ""
            ),
            flush=True,
        )
        if loss_va < best_va:
            best_va = loss_va
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = args.out / "delta_midplane.pt"
    meta = {
        "model": model.state_dict(),
        "hist": hist,
        "cfg": "progressive_64",
        "arch": args.arch,
        "n_in": n_in,
        "width": int(args.width),
        "input": (
            "log1p(dens), log1p(axisym), morph_contrast_hint"
            if conditioned
            else "log1p(dens), log1p(axisym)"
        ),
        "target": f"clip(m2_contrast,±{args.contrast_clip})",
        "hint_blur_sigma": float(args.hint_blur_sigma) if conditioned else 0.0,
        "a2_loss_weight": a2_w,
        "a2_r_eval": float(args.a2_r_eval),
        "best_val_loss": best_va,
        "min_a2": float(args.min_a2),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "holdout": sorted(holdout),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available() and device.type == "cuda"),
        "cuda_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and device.type == "cuda"
            else None
        ),
    }
    torch.save(meta, ckpt)
    (args.out / "SUMMARY.md").write_text(
        "# Phase B δ (morph-conditioned)\n\n"
        f"arch=`{args.arch}` n_in={n_in} width={args.width}\n\n"
        f"**device=`{device}` cuda={torch.cuda.is_available() and device.type == 'cuda'}**"
        + (
            f" ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available() and device.type == "cuda"
            else ""
        )
        + "\n\n"
        f"pairs train={len(train_pairs)} val={len(val_pairs)} "
        f"epochs={args.epochs} min_a2={args.min_a2}\n\n"
        f"hint_blur_sigma={args.hint_blur_sigma} a2_loss_weight={a2_w}\n\n"
        f"best val loss={best_va:.4g}\n"
        f"final val MSE={hist[-1]['val_mse']:.4g}\n"
        f"ckpt=`{ckpt}`\n"
        f"holdout={sorted(holdout)}\n"
    )
    (args.out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(f"wrote {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
