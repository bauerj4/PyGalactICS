#!/usr/bin/env python3
"""Benchmark Python Poisson solve (dbh) for a GalaxyModel or campaign work dir."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from galacticsics.campaign.serialize import model_from_dict
from galacticsics.campaign.spec import format_dbh_grid_summary, preview_dbh_model
from galacticsics.potential.frequencies_tabulate import tabulate_frequencies
from galacticsics.potential.solver import solve_potential


def _load_model(args) -> tuple[object, Path | None]:
    if args.work_dir:
        work = Path(args.work_dir)
        model = model_from_dict(json.loads((work / "model.json").read_text()))
        return model, work
    model = preview_dbh_model(
        base=args.base,
        coarse=not args.fine,
        dr=args.dr,
        nr=args.nr,
        lmax=args.lmax,
    )
    return model, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work_dir", nargs="?", help="Existing campaign dir with model.json")
    p.add_argument("--base", default="milky_way_disk_halo")
    p.add_argument("--fine", action="store_true", help="Use production grid (no coarse cap)")
    p.add_argument("--dr", type=float, default=None)
    p.add_argument("--nr", type=int, default=None)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--npsi", type=int, default=1000)
    p.add_argument("--nint", type=int, default=20)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--out", type=Path, default=None, help="Write dbh artifacts here")
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Set GALACTICSICS_POISSON_GPU=1 for CuPy batched polar fill",
    )
    args = p.parse_args()

    model, work_hint = _load_model(args)
    out = args.out or work_hint or Path("notebooks/artifacts/solve_bench")
    out.mkdir(parents=True, exist_ok=True)

    if args.n_workers is not None:
        os.environ["GALACTICSICS_SOLVE_WORKERS"] = str(args.n_workers)
    if args.gpu:
        os.environ["GALACTICSICS_POISSON_GPU"] = "1"
        os.environ.setdefault("GALACTICSICS_POISSON_THREADS", "0")

    print(format_dbh_grid_summary(model))
    times: list[float] = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        solve_potential(
            model,
            work_dir=out,
            cleanup=False,
            backend="python",
            npsi=args.npsi,
            nint=args.nint,
            max_iter=args.max_iter,
            n_workers=args.n_workers,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"run {i + 1}/{args.repeat}: solve+freq {elapsed:.2f}s")

    t_freq = time.perf_counter()
    tabulate_frequencies(out)
    freq_s = time.perf_counter() - t_freq
    print(f"retabulate freqdbh only: {freq_s:.3f}s")
    print(f"mean solve: {sum(times) / len(times):.2f}s  out={out}")


if __name__ == "__main__":
    main()
