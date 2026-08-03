"""Tests for Morton tokenization and on-the-fly dataset indexing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from galacticsics.ml.morton.dataset import MortonSnapshotDataset
from galacticsics.ml.morton.index import SnapshotRecord, build_snapshot_index, write_snapshot_manifest
from galacticsics.ml.morton.tokenize import (
    center_phase_space,
    particles_from_tokens,
    random_rotate_z,
    rotate_about_z,
    subsample_stratified,
    tokenize_morton,
)


@pytest.mark.essential
def test_center_phase_space_zeros_com():
    pos = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]])
    mass = np.array([1.0, 1.0, 2.0])
    pc, vc = center_phase_space(pos, vel, mass)
    w = mass / mass.sum()
    assert np.allclose((pc * w[:, None]).sum(axis=0), 0.0, atol=1e-12)
    assert np.allclose((vc * w[:, None]).sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.essential
def test_rotate_about_z_preserves_z_and_norms():
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(50, 3))
    vel = rng.normal(size=(50, 3))
    rp, rv = rotate_about_z(pos, vel, phi=np.pi / 3)
    assert np.allclose(rp[:, 2], pos[:, 2])
    assert np.allclose(rv[:, 2], vel[:, 2])
    assert np.allclose(np.linalg.norm(rp, axis=1), np.linalg.norm(pos, axis=1))
    rp2, _ = random_rotate_z(pos, vel, rng)
    assert rp2.shape == pos.shape


@pytest.mark.essential
def test_dataset_centers_com(tmp_path: Path):
    run = tmp_path / "centerme"
    run.mkdir()
    (run / "model.json").write_text(
        json.dumps(
            {
                "disk": {"mass": 10.0, "scale_length": 2.5, "scale_height": 0.25, "enabled": True},
                "halo": {"v0": 3.0, "a": 30.0, "enabled": True},
                "bulge": {"v0": 1.0, "a": 0.5, "enabled": True},
            }
        )
    )
    n = 210
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(n, 3)) + np.array([20.0, -10.0, 5.0])
    vel = rng.normal(size=(n, 3)) + np.array([1.0, 2.0, -0.5])
    mass = np.ones(n)
    tags = np.array(["disk"] * 120 + ["halo"] * 60 + ["bulge"] * 30)
    np.savez(run / "ic_state.npz", pos=pos, vel=vel, mass=mass, tags=tags)
    from galacticsics.ml.morton.index import write_snapshot_manifest

    manifest = write_snapshot_manifest(tmp_path)
    ds = MortonSnapshotDataset(
        manifest, n_particles=64, split=None, seed=0, center=True, augment=False
    )
    item = ds[0]
    # Subsample COM need not be exactly 0, but should be near origin vs raw +20 offset
    assert np.linalg.norm(item["dx"].mean(axis=0)) < 5.0


@pytest.mark.essential
def test_sequence_vae_generate_has_spread_and_components():
    torch = pytest.importorskip("torch")
    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig

    cfg = SequenceVAEConfig(n_particles=32, theta_dim=4, d_model=32, latent_dim=8, n_heads=2)
    model = SequenceVAE(cfg)
    theta = torch.zeros(2, 4)
    with torch.no_grad():
        out = model.generate(theta, n=64, chunk_size=32)
    assert out["dx"].std() > 0.05
    assert set(np.unique(out["c"])).issubset({0, 1, 2})



@pytest.mark.essential
def test_tokenize_morton_monotonic_keys():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(200, 3))
    vel = rng.normal(size=(200, 3))
    tags = np.array(["disk"] * 100 + ["halo"] * 70 + ["bulge"] * 30)
    tok = tokenize_morton(pos, vel, tags=tags, order="morton")
    assert np.all(np.diff(tok["keys"].astype(np.int64)) >= 0)
    assert tok["c"].shape == (200,)
    assert tok["dx"].shape == (200, 3)


@pytest.mark.essential
def test_tokenize_roundtrip_positions_close():
    from galacticsics.ml.morton.morton_keys import sort_by_morton

    rng = np.random.default_rng(1)
    pos = rng.uniform(-10, 10, size=(80, 3))
    vel = rng.normal(size=(80, 3)) * 0.1
    tok = tokenize_morton(pos, vel, order="morton", bits=10)
    pos2, vel2, _ = particles_from_tokens(tok)
    assert np.allclose(pos2, pos[sort_by_morton(pos)], atol=1e-5)
    assert np.allclose(vel2, tok["v"])


@pytest.mark.essential
def test_subsample_stratified_counts():
    cid = np.array([0] * 100 + [1] * 50 + [2] * 25)
    rng = np.random.default_rng(2)
    idx = subsample_stratified(cid, 70, rng=rng)
    assert idx.shape == (70,)
    assert len(np.unique(idx)) == 70


@pytest.mark.essential
def test_snapshot_index_and_dataset(tmp_path: Path):
    run = tmp_path / "abc123"
    run.mkdir()
    (run / "model.json").write_text(
        json.dumps(
            {
                "disk": {"mass": 17.0, "scale_length": 2.5, "scale_height": 0.25, "enabled": True},
                "halo": {"v0": 3.7, "a": 33.0, "enabled": True},
                "bulge": {"v0": 2.0, "a": 0.5, "enabled": True},
            }
        )
    )
    n = 300
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(n, 3))
    vel = rng.normal(size=(n, 3))
    mass = np.ones(n) / n
    eps = np.full(n, 0.1)
    type_id = np.array([0] * 150 + [1] * 100 + [2] * 50, dtype=np.int32)
    np.savez(run / "ic_state.npz", pos=pos, vel=vel, mass=mass, eps=eps, type_id=type_id)
    part = run / "evolution" / "particles"
    part.mkdir(parents=True)
    np.savez_compressed(
        part / "step_000100.npz",
        pos=pos,
        vel=vel,
        mass=mass,
        eps=eps,
        type_id=type_id,
        timestep_bin=np.zeros(n, dtype=np.int32),
    )

    records = build_snapshot_index(tmp_path)
    assert len(records) == 2
    manifest = write_snapshot_manifest(tmp_path)
    assert manifest.is_file()

    ds = MortonSnapshotDataset(manifest, n_particles=64, split=None, seed=0)
    assert len(ds) == 2
    item = ds[0]
    assert item["c"].shape == (64,)
    assert item["dx"].shape == (64, 3)
    assert item["theta"].shape[0] == len(ds.theta_keys)
