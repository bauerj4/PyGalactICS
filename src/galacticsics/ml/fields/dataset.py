"""On-the-fly field-map Dataset over Morton corpus snapshot dumps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.ml.fields.binning import (
    MultiScaleSliceConfig,
    MultiScaleVoxelConfig,
    SliceMapConfig,
    VoxelMapConfig,
    bin_multiscale_slice_stacks,
    bin_multiscale_voxel_stacks,
    bin_vertical_slice_stack,
    bin_voxel_stack,
    dens_channel_indices_component,
)
from galacticsics.ml.fields.normalize import (
    FieldNormStats,
    append_phi_channels,
    estimate_norm_stats,
    estimate_norm_stats_component,
    normalize_stack,
)
from galacticsics.ml.fields.potential import (
    plummer_potential_multiscale,
    plummer_potential_on_xy_slices,
)
from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import _component_ids


def _load_snapshot_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _select_bar_heavy_records(
    records: list[SnapshotRecord],
    *,
    n: int,
    bar_frac: float,
    bar_a2_floor: float,
    a2_rank_path: Path | str | None,
    seed: int,
) -> list[SnapshotRecord]:
    """
    Oversample barred dumps when subsampling the corpus for dens/morph FT.

    Uses ``corpus_particle_a2_rank.json`` (path→a2) when available; otherwise
    falls back to uniform stride (no bar metadata).
    """
    n = max(1, int(n))
    bar_frac = float(np.clip(bar_frac, 0.0, 1.0))
    path_a2: dict[str, float] = {}
    candidates: list[Path] = []
    if a2_rank_path is not None:
        candidates.append(Path(a2_rank_path))
    candidates.append(Path("runs/ml/field_maps/corpus_particle_a2_rank.json"))
    for cand in candidates:
        if cand.is_file():
            raw = json.loads(cand.read_text())
            rows = raw if isinstance(raw, list) else raw.get("records", raw.get("rows", []))
            for row in rows:
                p = str(row.get("path") or row.get("dump") or "")
                if not p:
                    continue
                try:
                    path_a2[p] = float(row.get("a2") or row.get("particle_a2") or 0.0)
                except (TypeError, ValueError):
                    continue
            break
    if not path_a2:
        step = max(1, len(records) // n)
        return records[::step][:n]

    def _a2(rec: SnapshotRecord) -> float:
        p = str(rec.path)
        if p in path_a2:
            return path_a2[p]
        for k, v in path_a2.items():
            if p.endswith(k) or k.endswith(p):
                return v
        return 0.0

    barred = [r for r in records if _a2(r) >= float(bar_a2_floor)]
    quiet = [r for r in records if _a2(r) < float(bar_a2_floor)]
    n_bar = int(round(n * bar_frac))
    n_quiet = n - n_bar
    rng = np.random.default_rng(int(seed))

    def _take(pool: list[SnapshotRecord], k: int) -> list[SnapshotRecord]:
        if k <= 0 or not pool:
            return []
        if len(pool) <= k:
            idx = rng.choice(len(pool), size=k, replace=True)
            return [pool[i] for i in idx]
        scored = sorted(pool, key=_a2, reverse=True)
        top = scored[: max(k, min(len(scored), 3 * k))]
        idx = rng.choice(len(top), size=k, replace=False)
        return [top[i] for i in idx]

    picked = _take(barred, n_bar) + _take(quiet, n_quiet)
    rng.shuffle(picked)
    return picked[:n]


class _SnapshotMixin:
    """Shared manifest load / phase-space prep for field datasets."""

    def _init_records(
        self,
        records: list[SnapshotRecord] | Path | str,
        *,
        split: str | None,
        max_snapshots: int | None,
        sources: tuple[str, ...] | None,
        bar_frac: float | None = None,
        bar_a2_floor: float = 0.20,
        a2_rank_path: Path | str | None = None,
        seed: int = 0,
    ) -> None:
        if isinstance(records, (str, Path)):
            raw = json.loads(Path(records).read_text())
            self.records = [SnapshotRecord(**r) for r in raw["records"]]
        else:
            self.records = list(records)
        if split is not None:
            self.records = [r for r in self.records if r.split == split]
        if sources is not None:
            allow = set(sources)
            self.records = [r for r in self.records if r.source in allow]
        if max_snapshots is not None and len(self.records) > int(max_snapshots):
            bf = float(bar_frac) if bar_frac is not None else None
            if bf is not None and bf > 0.0:
                self.records = _select_bar_heavy_records(
                    self.records,
                    n=int(max_snapshots),
                    bar_frac=bf,
                    bar_a2_floor=float(bar_a2_floor),
                    a2_rank_path=a2_rank_path,
                    seed=int(seed),
                )
            else:
                step = max(1, len(self.records) // int(max_snapshots))
                self.records = self.records[::step][: int(max_snapshots)]

    def preload(self) -> None:
        cache: dict[str, dict[str, np.ndarray]] = {}
        for rec in self.records:
            if rec.path not in cache:
                cache[rec.path] = _load_snapshot_arrays(Path(rec.path))
        self._cache = cache

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _arrays(self, path: str) -> dict[str, np.ndarray]:
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        return _load_snapshot_arrays(Path(path))

    def _phase_space(self, arrays: dict[str, np.ndarray], rng: np.random.Generator):
        """Load arrays into one shared COM frame (+ optional common rotation).

        Never recenters disk/bulge/halo independently — that would misalign
        components.  See :func:`~galacticsics.ml.fields.frame.prepare_shared_frame`.
        """
        pos = np.asarray(arrays["pos"], dtype=np.float64)
        vel = np.asarray(arrays["vel"], dtype=np.float64)
        mass = np.asarray(arrays["mass"], dtype=np.float64)
        cid = _component_ids(arrays.get("tags"), arrays.get("type_id"), pos.shape[0])
        pos, vel, _meta = prepare_shared_frame(
            pos,
            vel,
            mass,
            center=self.center,
            rotate=self.augment,
            rng=rng if self.augment else None,
        )
        return pos, vel, mass, cid


class FieldSliceDataset(_SnapshotMixin):
    """
    Shared-geometry slice stacks (legacy). Prefer :class:`MultiScaleFieldDataset`.
    """

    def __init__(
        self,
        records: list[SnapshotRecord] | Path | str,
        *,
        cfg: SliceMapConfig | None = None,
        split: str | None = "train",
        seed: int = 0,
        max_snapshots: int | None = None,
        sources: tuple[str, ...] | None = None,
        center: bool = True,
        augment: bool | None = None,
        include_potential: bool = False,
        potential_n_sub: int = 2048,
        potential_eps: float = 0.05,
        norm_stats: FieldNormStats | None = None,
    ) -> None:
        self._init_records(
            records, split=split, max_snapshots=max_snapshots, sources=sources
        )
        self.cfg = cfg or SliceMapConfig()
        self.seed = int(seed)
        self.center = bool(center)
        self.augment = (split == "train") if augment is None else bool(augment)
        self.include_potential = bool(include_potential)
        self.potential_n_sub = int(potential_n_sub)
        self.potential_eps = float(potential_eps)
        self.norm_stats = norm_stats
        self._epoch = 0
        self._cache: dict[str, dict[str, np.ndarray]] | None = None

    def raw_stack(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        rec = self.records[index]
        rng = np.random.default_rng(
            self.seed + self._epoch * 1_000_003 + index * 9973
        )
        arrays = self._arrays(rec.path)
        pos, vel, mass, cid = self._phase_space(arrays, rng)
        stack, meta = bin_vertical_slice_stack(pos, vel, mass, cid, cfg=self.cfg)
        if self.include_potential:
            phi = plummer_potential_on_xy_slices(
                pos,
                mass,
                cid,
                cfg=self.cfg,
                eps=self.potential_eps,
                n_sub=self.potential_n_sub,
                rng=rng,
            )
            stack = append_phi_channels(stack, phi)
            names = list(meta["channel_names"]) + [f"phi/z{i}" for i in range(phi.shape[0])]
            meta["channel_names"] = np.asarray(names, dtype=object)
        meta["run_hash"] = rec.run_hash
        meta["path"] = rec.path
        meta["t_gyr"] = float(rec.t_gyr) if np.isfinite(rec.t_gyr) else 0.0
        return stack, meta

    def fit_norm_stats(self, n_samples: int = 16) -> FieldNormStats:
        old_aug = self.augment
        self.augment = False
        stacks = [self.raw_stack(i)[0] for i in range(min(int(n_samples), len(self)))]
        self.augment = old_aug
        n_phi = int(self.cfg.n_z) if self.include_potential else 0
        self.norm_stats = estimate_norm_stats(stacks, cfg=self.cfg, n_phi=n_phi)
        return self.norm_stats

    def __getitem__(self, index: int) -> dict[str, Any]:
        stack, meta = self.raw_stack(index)
        if self.norm_stats is None:
            raise RuntimeError("call fit_norm_stats() before indexing, or pass norm_stats=")
        return {
            "stack": normalize_stack(stack, self.norm_stats).astype(np.float32),
            "stack_raw": stack.astype(np.float32),
            "run_hash": meta["run_hash"],
            "path": meta["path"],
            "t_gyr": meta["t_gyr"],
            "channel_names": meta["channel_names"],
        }


class MultiScaleFieldDataset(_SnapshotMixin):
    """
    Per-component native-resolution slice stacks for multi-tower CNN training.

    Each item is ``stacks[name]`` (normalized) plus raw stacks / metadata.
    """

    def __init__(
        self,
        records: list[SnapshotRecord] | Path | str,
        *,
        cfg: MultiScaleSliceConfig | None = None,
        split: str | None = "train",
        seed: int = 0,
        max_snapshots: int | None = None,
        sources: tuple[str, ...] | None = None,
        center: bool = True,
        augment: bool | None = None,
        include_potential: bool | None = None,
        potential_n_sub: int = 1024,
        potential_eps: float = 0.05,
        norm_stats: dict[str, FieldNormStats] | None = None,
        theta_keys: list[str] | None = None,
        bar_frac: float | None = None,
        bar_a2_floor: float = 0.20,
        a2_rank_path: Path | str | None = None,
    ) -> None:
        self._init_records(
            records,
            split=split,
            max_snapshots=max_snapshots,
            sources=sources,
            bar_frac=bar_frac,
            bar_a2_floor=bar_a2_floor,
            a2_rank_path=a2_rank_path,
            seed=seed,
        )
        self.cfg = cfg or MultiScaleSliceConfig.smoke_defaults()
        self.seed = int(seed)
        self.center = bool(center)
        self.augment = (split == "train") if augment is None else bool(augment)
        if include_potential is None:
            include_potential = self.cfg.include_potential
        self.include_potential = bool(include_potential)
        self.potential_n_sub = int(potential_n_sub)
        self.potential_eps = float(potential_eps)
        self.norm_stats = norm_stats
        self.theta_keys = list(theta_keys or DEFAULT_THETA_KEYS)
        self._epoch = 0
        self._cache: dict[str, dict[str, np.ndarray]] | None = None

    def raw_maps(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rec = self.records[index]
        rng = np.random.default_rng(
            self.seed + self._epoch * 1_000_003 + index * 9973
        )
        arrays = self._arrays(rec.path)
        pos, vel, mass, cid = self._phase_space(arrays, rng)
        binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=self.cfg)
        maps: dict[str, np.ndarray] = {k: v[0] for k, v in binned.items()}
        metas = {k: v[1] for k, v in binned.items()}
        if self.include_potential:
            phis = plummer_potential_multiscale(
                pos,
                mass,
                cid,
                cfg=self.cfg,
                eps=self.potential_eps,
                n_sub=self.potential_n_sub,
                rng=rng,
            )
            for name, phi in phis.items():
                maps[name] = append_phi_channels(maps[name], phi)
        t_gyr = resolve_snapshot_t_gyr(rec.path, rec.t_gyr)
        meta = {
            "run_hash": rec.run_hash,
            "path": rec.path,
            "t_gyr": float(t_gyr),
            "theta": theta_from_record(
                rec.theta, t_gyr=t_gyr, keys=self.theta_keys
            ),
            "component_meta": metas,
        }
        return maps, meta

    def fit_norm_stats(self, n_samples: int = 12) -> dict[str, FieldNormStats]:
        old_aug = self.augment
        self.augment = False
        per_comp: dict[str, list[np.ndarray]] = {g.name: [] for g in self.cfg.grids}
        for i in range(min(int(n_samples), len(self))):
            maps, _ = self.raw_maps(i)
            for name, stack in maps.items():
                per_comp[name].append(stack)
        self.augment = old_aug
        n_phi = 0  # Φ channels counted via grid.n_z when include_potential
        stats: dict[str, FieldNormStats] = {}
        for g in self.cfg.grids:
            n_phi_g = g.n_z if self.include_potential else 0
            stats[g.name] = estimate_norm_stats_component(
                per_comp[g.name], g, n_phi=n_phi_g
            )
        self.norm_stats = stats
        return stats

    def dens_indices(self) -> dict[str, list[int]]:
        return {g.name: dens_channel_indices_component(g) for g in self.cfg.grids}

    def __getitem__(self, index: int) -> dict[str, Any]:
        maps, meta = self.raw_maps(index)
        if self.norm_stats is None:
            raise RuntimeError("call fit_norm_stats() before indexing, or pass norm_stats=")
        stacks = {
            name: normalize_stack(stack, self.norm_stats[name]).astype(np.float32)
            for name, stack in maps.items()
        }
        return {
            "stacks": stacks,
            "stacks_raw": {k: v.astype(np.float32) for k, v in maps.items()},
            "run_hash": meta["run_hash"],
            "path": meta["path"],
            "t_gyr": meta["t_gyr"],
            "theta": np.asarray(meta["theta"], dtype=np.float32),
            "theta_keys": self.theta_keys,
        }


def collate_field_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate shared-geometry :class:`FieldSliceDataset` items."""
    return {
        "stack": np.stack([b["stack"] for b in batch], axis=0),
        "run_hash": [b["run_hash"] for b in batch],
        "t_gyr": np.asarray([b["t_gyr"] for b in batch], dtype=np.float64),
        "path": [b["path"] for b in batch],
    }


def collate_multiscale_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate :class:`MultiScaleFieldDataset` items into stacked per-component tensors."""
    names = list(batch[0]["stacks"].keys())
    stacks = {
        name: np.stack([b["stacks"][name] for b in batch], axis=0) for name in names
    }
    out: dict[str, Any] = {
        "stacks": stacks,
        "run_hash": [b["run_hash"] for b in batch],
        "t_gyr": np.asarray([b["t_gyr"] for b in batch], dtype=np.float64),
        "path": [b["path"] for b in batch],
    }
    if "theta" in batch[0]:
        out["theta"] = np.stack([b["theta"] for b in batch], axis=0)
    return out


def bin_voxel_from_path(
    path: Path | str,
    *,
    cfg: VoxelMapConfig | None = None,
    center: bool = True,
) -> np.ndarray:
    """Load one NPZ and return a shared cubic voxel stack (global COM frame)."""
    arrays = _load_snapshot_arrays(Path(path))
    pos = np.asarray(arrays["pos"], dtype=np.float64)
    vel = np.asarray(arrays["vel"], dtype=np.float64)
    mass = np.asarray(arrays["mass"], dtype=np.float64)
    cid = _component_ids(arrays.get("tags"), arrays.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=center)
    stack, _ = bin_voxel_stack(pos, vel, mass, cid, cfg=cfg)
    return stack


def bin_multiscale_voxels_from_path(
    path: Path | str,
    *,
    cfg: MultiScaleVoxelConfig | None = None,
    center: bool = True,
) -> dict[str, np.ndarray]:
    """Load one NPZ and return per-component voxels on a shared global COM frame."""
    arrays = _load_snapshot_arrays(Path(path))
    pos = np.asarray(arrays["pos"], dtype=np.float64)
    vel = np.asarray(arrays["vel"], dtype=np.float64)
    mass = np.asarray(arrays["mass"], dtype=np.float64)
    cid = _component_ids(arrays.get("tags"), arrays.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=center)
    raw = bin_multiscale_voxel_stacks(pos, vel, mass, cid, cfg=cfg)
    return {k: v[0] for k, v in raw.items()}
