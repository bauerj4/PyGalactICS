"""On-the-fly Morton sequence Dataset (subsamples each access; no disk cache)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record, theta_vector
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import (
    center_phase_space,
    random_rotate_z,
    subsample_stratified,
    tokenize_morton,
)

# Re-export for existing imports (``from ...morton.dataset import DEFAULT_THETA_KEYS``).
__all__ = ["DEFAULT_THETA_KEYS", "MortonSnapshotDataset", "collate_morton_batch"]


def _load_snapshot_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    return out


def _theta_vector(theta: dict[str, float], keys: list[str]) -> np.ndarray:
    """Deprecated alias — use :func:`galacticsics.ml.conditioning.theta_vector`."""
    return theta_vector(theta, keys)


class MortonSnapshotDataset:
    """
    On-the-fly Morton sequence dataset backed by a snapshot manifest.

    Each ``__getitem__`` loads one full N-body snapshot from disk, draws a
    random stratified subset of ``n_particles``, Morton-sorts (or randomly
    permutes) the subset, and returns token arrays plus the conditioning
    vector ``θ``.

    Parameters
    ----------
    records : list of SnapshotRecord or path
        Manifest JSON from :func:`~galacticsics.ml.morton.index.write_snapshot_manifest`
        or an in-memory record list.
    n_particles : int
        Subsample size ``N`` per item (keep small under GPU memory pressure).
    split : {'train', 'val', 'test', None}
        Filter by ``SnapshotRecord.split``.  ``None`` keeps all records.
    order : {'morton', 'random'}
        Particle ordering for tokenization (ablation for the AR transformer).
    bits : int
        Morton key resolution passed to :func:`~galacticsics.ml.morton.tokenize.tokenize_morton`.
    theta_keys : list of str, optional
        Ordered keys harvested from ``record.theta`` / ``model.json``.
    seed : int
        Base RNG seed; per-item streams use ``seed``, ``epoch``, and ``index``.
    center : bool
        Subtract mass-weighted COM (positions and velocities) before tokenize.
    augment : bool or None
        If ``True``, apply a random in-plane (about-``z``) rotation each access.
        Default: ``True`` when ``split='train'``, else ``False``.

    Returns
    -------
    item : dict
        ``c`` (N,), ``dm`` (N,), ``dx`` (N,3), ``v`` (N,3), ``theta`` (K,),
        plus metadata (``run_hash``, ``t_gyr``, box fields).

    Notes
    -----
    Snapshots are **not** cached as sequences on disk: the corpus particle
    dumps are the source of truth, which avoids a second large on-disk
    representation while the evolve campaign is still writing data.

    Call :meth:`set_epoch` each epoch so subsample / rotation draws change.
    """

    def __init__(
        self,
        records: list[SnapshotRecord] | Path | str,
        *,
        n_particles: int = 4096,
        split: str | None = "train",
        order: Literal["morton", "random"] = "morton",
        bits: int = 10,
        theta_keys: list[str] | None = None,
        seed: int = 0,
        max_snapshots: int | None = None,
        sources: tuple[str, ...] | None = None,
        center: bool = True,
        augment: bool | None = None,
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
            # Keep a deterministic spread across the manifest (ICs + dumps).
            step = max(1, len(self.records) // int(max_snapshots))
            self.records = self.records[::step][: int(max_snapshots)]
        self.n_particles = int(n_particles)
        self.order = order
        self.bits = int(bits)
        self.theta_keys = list(theta_keys or DEFAULT_THETA_KEYS)
        self.seed = int(seed)
        self.center = bool(center)
        if augment is None:
            self.augment = split == "train"
        else:
            self.augment = bool(augment)
        self._epoch = 0
        self._cache: dict[str, dict[str, np.ndarray]] | None = None

    def preload(self) -> None:
        """Load all snapshot arrays into RAM (recommended for long train runs)."""
        cache: dict[str, dict[str, np.ndarray]] = {}
        for rec in self.records:
            if rec.path not in cache:
                cache[rec.path] = _load_snapshot_arrays(Path(rec.path))
        self._cache = cache

    def set_epoch(self, epoch: int) -> None:
        """Advance the RNG epoch so train augmentations / subsamples reshuffle."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        rng = np.random.default_rng(
            self.seed
            + self._epoch * 1_000_003
            + index * 9973
            + (hash(rec.path) % 10_000)
        )
        if self._cache is not None and rec.path in self._cache:
            arrays = self._cache[rec.path]
        else:
            arrays = _load_snapshot_arrays(Path(rec.path))
        pos = np.asarray(arrays["pos"], dtype=np.float64)
        vel = np.asarray(arrays["vel"], dtype=np.float64)
        mass = arrays.get("mass")
        if mass is not None:
            mass = np.asarray(mass, dtype=np.float64)
        tags = arrays.get("tags")
        type_id = arrays.get("type_id")
        n = pos.shape[0]
        # Build component ids before subsample
        from galacticsics.ml.morton.tokenize import _component_ids

        cid = _component_ids(tags, type_id, n)
        if self.center:
            pos, vel = center_phase_space(pos, vel, mass)
        if self.augment:
            pos, vel = random_rotate_z(pos, vel, rng)
        idx = subsample_stratified(cid, self.n_particles, rng=rng)
        tokens = tokenize_morton(
            pos[idx],
            vel[idx],
            component_id=cid[idx],
            bits=self.bits,
            order=self.order,
            rng=rng,
        )
        t_gyr = resolve_snapshot_t_gyr(rec.path, rec.t_gyr)
        return {
            "c": tokens["c"],
            "dm": tokens["dm"],
            "dx": tokens["dx"],
            "v": tokens["v"],
            "theta": theta_from_record(rec.theta, t_gyr=t_gyr, keys=self.theta_keys),
            "theta_keys": self.theta_keys,
            "run_hash": rec.run_hash,
            "t_gyr": float(t_gyr),
            "box_min": tokens["box_min"],
            "box_size": tokens["box_size"],
            "bits": tokens["bits"],
        }


def collate_morton_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stack a list of dataset items into batched numpy arrays (torch-free).

    Parameters
    ----------
    batch : list of dict
        Items from :class:`MortonSnapshotDataset`.

    Returns
    -------
    dict
        Stacked ``c``, ``dm``, ``dx``, ``v``, ``theta`` with leading batch
        axis, plus ``run_hash`` (list) and ``t_gyr`` (1-D array).
    """
    keys = ("c", "dm", "dx", "v", "theta")
    out: dict[str, Any] = {k: np.stack([b[k] for b in batch], axis=0) for k in keys}
    out["run_hash"] = [b["run_hash"] for b in batch]
    out["t_gyr"] = np.asarray([b["t_gyr"] for b in batch], dtype=np.float64)
    return out
