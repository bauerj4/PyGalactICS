#!/usr/bin/env python3
"""
Lightweight RealNVP / Gaussian-mix prior on PCA feature-library codes.

Does NOT synthesize skips — only learns p(z|kind); decode still retrieves /
amplifies teacher features from the library (frozen crisp AE).

    OMP_NUM_THREADS=4 python scripts/train_code_flow_prior.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.feature_library import (
    FeatureLibraryConfig,
    TeacherFeatureLibrary,
    encode_snapshot_features,
    load_frozen_teacher_bundle,
    scale_residual,
)
from galacticsics.ml.fields.normalize import denormalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from ntropy.analysis.disk_density import disk_azimuthal_fourier


CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
LOG = Path("runs/ml/field_maps/MARATHON_6H.md")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.n1 = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.n1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * (dim - self.n1)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, reverse=False):
        x1, x2 = x[:, : self.n1], x[:, self.n1 :]
        h = self.net(x1)
        log_s, t = h.chunk(2, dim=-1)
        log_s = 0.5 * torch.tanh(log_s)
        if reverse:
            y2 = (x2 - t) * torch.exp(-log_s)
            logdet = -log_s.sum(-1)
        else:
            y2 = x2 * torch.exp(log_s) + t
            logdet = log_s.sum(-1)
        return torch.cat([x1, y2], dim=-1), logdet


class RealNVP(nn.Module):
    def __init__(self, dim: int, n_layers: int = 6, hidden: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden) for _ in range(n_layers)])
        self.register_buffer("perm", torch.arange(dim))

    def _permute(self, x, reverse=False):
        if reverse:
            inv = torch.argsort(self.perm)
            return x[:, inv]
        return x[:, self.perm]

    def forward(self, x, reverse=False):
        logdet = torch.zeros(x.shape[0], device=x.device)
        layers = reversed(list(self.layers)) if reverse else self.layers
        # alternate flips via roll of perm each layer
        z = x
        for i, layer in enumerate(layers):
            if i % 2 == 1:
                z = torch.flip(z, dims=[-1])
            z, ld = layer(z, reverse=reverse)
            logdet = logdet + ld
            if i % 2 == 1:
                z = torch.flip(z, dims=[-1])
        return z, logdet

    def log_prob(self, x):
        z, logdet = self.forward(x, reverse=False)
        log_pz = -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * math.log(2 * math.pi))
        return log_pz + logdet

    def sample(self, n: int, device="cpu"):
        z = torch.randn(n, self.layers[0].dim, device=device)
        x, _ = self.forward(z, reverse=True)
        return x


def _a2_disk(parts):
    disk = parts["component_id"] == 0
    return float(
        disk_azimuthal_fourier(
            parts["pos"][disk],
            parts["mass"][disk],
            m=2,
            r_max=12.0,
            n_bins=12,
            z_max=0.5,
            min_count=10,
        )["a_m_over_a0_median"]
    )


def _disk_collapse(stack, n_z, n_mom):
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _stratified(ranked, n_bar, n_quiet, n_mid, bar_floor, quiet_ceil):
    seen, picked = set(), []

    def add(row):
        if row["path"] in seen:
            return
        seen.add(row["path"])
        picked.append(row)

    bars = [r for r in ranked if r["a2"] >= bar_floor]
    quiets = [r for r in ranked if r["a2"] <= quiet_ceil]
    for r in bars[:n_bar]:
        add(r)
    for r in reversed(quiets[-n_quiet:]):
        add(r)
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid > 0:
        idx = np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int)
        for i in idx:
            add(rest[int(i)])
    for r in ranked[:12]:
        add(r)
    return picked


def build_library(args) -> TeacherFeatureLibrary:
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _stratified(
        ranked, args.n_bar, args.n_quiet, args.n_mid, args.bar_floor, args.quiet_ceil
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    lib_cfg = FeatureLibraryConfig(
        enc_grid=args.enc_grid, bar_floor=args.bar_floor, quiet_ceil=args.quiet_ceil
    )
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding library snaps={len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis = (
            list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
            if is_bar and args.n_rot_bar > 1
            else [None]
        )
        for phi in phis:
            feat, z, a2 = encode_snapshot_features(
                row["path"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                phi=None if phi is None else float(phi),
                enc_grid=args.enc_grid,
            )
            feats.append(feat)
            zs.append(z)
            a2s.append(a2)
            meta.append({"path": row["path"], "rank_a2": float(row["a2"]), "data_a2": float(a2)})
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    Z = np.stack(zs, axis=0)
    A2 = np.asarray(a2s, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(args.n_pc, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_pc].T
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=Zc @ W,
        a2=A2,
        meta=meta,
        pca_mean=Z.mean(0),
        pca_w=W,
        lib_cfg=lib_cfg,
    )


def train_flow(codes: np.ndarray, epochs: int, lr: float, device="cpu") -> RealNVP:
    x = torch.as_tensor(codes, dtype=torch.float32, device=device)
    # standardize
    mu, std = x.mean(0), x.std(0).clamp_min(1e-3)
    x_n = (x - mu) / std
    flow = RealNVP(dim=x.shape[1], n_layers=6, hidden=128).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        nll = -flow.log_prob(x_n).mean()
        nll.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  flow ep {ep+1}/{epochs} nll={float(nll):.3f}", flush=True)
    flow._mu = mu
    flow._std = std
    return flow


def sample_flow_z(flow: RealNVP, n: int, rng: np.random.Generator) -> np.ndarray:
    with torch.no_grad():
        x_n = flow.sample(n)
        x = x_n * flow._std + flow._mu
    return x.cpu().numpy()


def eval_flow_retrieve(lib, flow, *, n_samples, n_resample, rng, amplify=True, alpha_lo=1.08, alpha_hi=1.28):
    disk_g = lib.cfg.grid_for("disk")
    rows = []
    for kind in ("barred", "quiet"):
        for j in range(n_samples):
            if kind == "quiet":
                feat, meta = lib.sample_features(kind="quiet", method="uniform_knn", rng=rng)
            else:
                z = sample_flow_z(flow, 1, rng)[0]
                feat, meta = lib.sample_features(kind="barred", z=z, rng=rng, a2_power=3.5, k=4)
                if amplify:
                    base = lib._quiet_mean
                    alpha = float(rng.uniform(alpha_lo, alpha_hi))
                    feat = scale_residual(base, feat, alpha)
                    meta = {**meta, "method": "flow_z_amplify", "alpha": alpha}
            out = lib.decode_features(feat)
            den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts = resample_particles_from_multiscale(
                den, cfg=lib.cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
            rows.append({"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta})
            print(f"  flow_{'amp' if amplify else 'ret'} {kind}#{j} A₂p={a2_part:.3f}", flush=True)
    bar = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "barred"]))
    qui = float(np.mean([r["a2_part"] for r in rows if r["kind"] == "quiet"]))
    return {"bar_mean": bar, "quiet_mean": qui, "hits_target": bar >= 0.33 and qui <= 0.05, "samples": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_code_flow_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=32)
    p.add_argument("--n-quiet", type=int, default=18)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=5)
    p.add_argument("--bar-floor", type=float, default=0.22)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=48)
    p.add_argument("--flow-epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=21)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    # train flow on barred codes only
    bar_codes = lib.codes[lib.bar_pool]
    print(f"training RealNVP on {len(bar_codes)} barred codes dim={bar_codes.shape[1]}…", flush=True)
    flow = train_flow(bar_codes, epochs=args.flow_epochs, lr=args.lr)
    torch.save(
        {"state": flow.state_dict(), "mu": flow._mu, "std": flow._std, "dim": bar_codes.shape[1]},
        args.out / "code_flow.pt",
    )

    verdict = {"approach": "PCA-code RealNVP prior + retrieve/amplify"}
    for name, amplify in (("flow_retrieve", False), ("flow_z_amplify", True)):
        verdict[name] = eval_flow_retrieve(
            lib, flow, n_samples=args.n_samples, n_resample=args.n_resample, rng=rng, amplify=amplify
        )
        print(
            f"→ {name}: bar={verdict[name]['bar_mean']:.3f} quiet={verdict[name]['quiet_mean']:.3f}",
            flush=True,
        )

    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a") as f:
        for name in ("flow_retrieve", "flow_z_amplify"):
            r = verdict[name]
            hits = "YES" if r["hits_target"] else ("near" if r["bar_mean"] >= 0.30 and r["quiet_mean"] <= 0.05 else "no")
            f.write(
                f"| {ts} | {name} | {r['bar_mean']:.3f} | {r['quiet_mean']:.3f} | {hits} | {args.out.name} |\n"
            )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in verdict.items() if isinstance(v, dict) and "bar_mean" in v}, indent=2))
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
