#!/usr/bin/env python3
"""Train a conditional Morton set VAE until train loss converges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="snapshot_manifest.json")
    parser.add_argument("--out", type=Path, default=Path("runs/ml/morton_vae"))
    parser.add_argument("--n-particles", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18, help="Stop after this many epochs without improvement")
    parser.add_argument("--min-delta", type=float, default=0.002, help="Relative improvement to reset patience")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--order", choices=("morton", "random"), default="random")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--lambda-virial", type=float, default=2.0)
    parser.add_argument("--virial-n-sub", type=int, default=256)
    parser.add_argument("--lambda-virial-target", type=float, default=1.5)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from galacticsics.ml.morton.dataset import MortonSnapshotDataset, collate_morton_batch
    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig

    ds = MortonSnapshotDataset(
        args.manifest,
        n_particles=args.n_particles,
        split="train",
        seed=args.seed,
        max_snapshots=args.max_snapshots,
        center=True,
        augment=not args.no_augment,
        order=args.order,
    )
    if len(ds) == 0:
        # Fall back to all records when manifest has no train split labels
        ds = MortonSnapshotDataset(
            args.manifest,
            n_particles=args.n_particles,
            split=None,
            seed=args.seed,
            max_snapshots=args.max_snapshots,
            center=True,
            augment=not args.no_augment,
        )
    if len(ds) == 0:
        raise SystemExit(f"no snapshots in {args.manifest}")

    print("preloading snapshots…", flush=True)
    ds.preload()

    def _collate(batch):
        np_batch = collate_morton_batch(batch)
        return {
            "c": torch.as_tensor(np_batch["c"], dtype=torch.long),
            "dm": torch.as_tensor(np_batch["dm"], dtype=torch.float32),
            "dx": torch.as_tensor(np_batch["dx"], dtype=torch.float32),
            "v": torch.as_tensor(np_batch["v"], dtype=torch.float32),
            "theta": torch.as_tensor(np_batch["theta"], dtype=torch.float32),
        }

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    cfg = SequenceVAEConfig(
        n_particles=args.n_particles,
        theta_dim=len(ds.theta_keys),
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        n_layers=2,
        n_heads=4,
        n_decode_layers=0,
        lambda_recon=1.0,
        lambda_chamfer=1.0,
        lambda_ce=1.0,
        lambda_mix=1.0,
        lambda_index=0.0,
        lambda_sigma=2.0,
        lambda_rho=1.0,
        lambda_vphi=1.0,
        lambda_nonaxisym=6.0,
        lambda_maps=6.0,
        lambda_virial=args.lambda_virial,
        virial_n_sub=args.virial_n_sub,
        lambda_virial_target=args.lambda_virial_target,
        map_n_pix=32,
        profile_n_bins=16,
        profile_r_max_sph=40.0,
        chamfer_max_n=512,
        enc_attn_n=256,
    )
    model = SequenceVAE(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"SequenceVAE params={n_params:,}  device={args.device}  N={args.n_particles}  "
        f"snaps={len(ds)}  center=1  augment={ds.augment}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    stall = 0
    model.train()
    for epoch in range(args.max_epochs):
        ds.set_epoch(epoch)
        losses = []
        last = None
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
            metrics = model.loss(batch, out)
            opt.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = metrics
            losses.append(float(metrics["loss"].detach().cpu()))
        mean_loss = float(np.mean(losses))
        row = {
            "epoch": float(epoch + 1),
            "loss": mean_loss,
            "mse_dx": float(last["mse_dx"].detach().cpu()),
            "chamfer": float(last["chamfer"].detach().cpu()),
            "ce": float(last["ce"].detach().cpu()),
            "ce_acc": float(last["ce_acc"].detach().cpu()),
            "mix_mse": float(last["mix_mse"].detach().cpu()),
            "kl": float(last["kl"].detach().cpu()),
            "nax": float(last["nonaxisym"].detach().cpu()),
            "virial": float(last["virial"].detach().cpu()),
            "ratio_pred": float(last["ratio_pred"].detach().cpu()),
            "ratio_data": float(last["ratio_data"].detach().cpu()),
            "x_scale": float(model.x_scale.detach().cpu()),
        }
        history.append(row)
        print(
            f"epoch {epoch + 1}/{args.max_epochs}  loss={mean_loss:.4f}  "
            f"dx={row['mse_dx']:.1f}  ch={row['chamfer']:.2f}  "
            f"ce={row['ce']:.3f}  acc={row['ce_acc']:.3f}  mix={row['mix_mse']:.4f}  "
            f"nax={row['nax']:.3f}  vir={row['virial']:.3f}  "
            f"2K/|W|={row['ratio_pred']:.2f}/{row['ratio_data']:.2f}",
            flush=True,
        )
        improved = mean_loss < best_loss * (1.0 - args.min_delta)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if improved:
            stall = 0
        else:
            stall += 1
            if stall >= args.patience:
                print(
                    f"converged: no relative improvement ≥{args.min_delta:.3g} "
                    f"for {args.patience} epochs (best loss={best_loss:.4f})",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = args.out / "sequence_vae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "theta_keys": ds.theta_keys,
            "history": history,
            "best_loss": best_loss,
        },
        ckpt,
    )
    (args.out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"wrote {ckpt}  best_loss={best_loss:.4f}  epochs={len(history)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
