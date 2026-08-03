#!/usr/bin/env python3
"""Train a conditional Morton autoregressive transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="snapshot_manifest.json")
    parser.add_argument("--out", type=Path, default=Path("runs/ml/morton_transformer"))
    parser.add_argument("--n-particles", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", choices=("morton", "random"), default="morton")
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from galacticsics.ml.morton.dataset import MortonSnapshotDataset, collate_morton_batch
    from galacticsics.ml.models.morton_transformer import MortonTransformer, MortonTransformerConfig

    ds = MortonSnapshotDataset(
        args.manifest,
        n_particles=args.n_particles,
        split="train",
        order=args.order,
        seed=args.seed,
    )
    if len(ds) == 0:
        raise SystemExit(f"no train snapshots in {args.manifest}")

    def _collate(batch):
        np_batch = collate_morton_batch(batch)
        return {k: torch.as_tensor(np_batch[k]) for k in ("c", "dm", "dx", "v", "theta")}

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    cfg = MortonTransformerConfig(n_particles=args.n_particles, theta_dim=len(ds.theta_keys))
    model = MortonTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
            metrics = model.loss(batch, out)
            opt.zero_grad()
            metrics["loss"].backward()
            opt.step()
            losses.append(float(metrics["loss"].detach().cpu()))
        print(f"epoch {epoch + 1}/{args.epochs}  loss={np.mean(losses):.4f}")

    ckpt = args.out / "morton_transformer.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__, "theta_keys": ds.theta_keys, "order": args.order},
        ckpt,
    )
    print(f"wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
