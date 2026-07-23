"""Tests for DBH campaign grid expansion."""

from __future__ import annotations

import json

from galacticsics.campaign.runner import _default_force_method, _write_evolve_config
from galacticsics.campaign.spec import GridSpec, expand_grid, model_hash
from galacticsics.models import GalaxyModel


def test_expand_grid_factorial():
    spec = GridSpec(
        base="reference_disk_halo",
        axes={"halo.v0": [3.5, 3.7]},
        omit_components=[[], ["bulge"]],
        coarse_grid=True,
    )
    models = expand_grid(spec)
    assert len(models) == 4
    labels = {label for label, _ in models}
    assert any("omit_bulge" in label for label in labels)


def test_model_hash_stable():
    m = GalaxyModel.reference_disk_halo()
    assert model_hash(m) == model_hash(m)


def test_apply_dbh_grid_override():
    from galacticsics.campaign.spec import apply_dbh_grid, format_dbh_grid_summary, preview_dbh_model
    from galacticsics.models import GalaxyModel

    m = preview_dbh_model(coarse=True)
    assert m.grid.nr == 4000
    m2 = apply_dbh_grid(m, nr=2000)
    assert m2.grid.nr == 2000
    assert "nr=2000" in format_dbh_grid_summary(m2)


def test_expand_grid_uses_grid_spec_overrides():
    spec = GridSpec(
        base="milky_way_disk_halo",
        coarse_grid=True,
        grid_nr=3000,
        axes={"halo.v0": [3.7]},
        omit_components=[[]],
    )
    _, model = expand_grid(spec)[0]
    assert model.grid.nr == 3000
    spec = GridSpec(base="milky_way_disk_halo", axes={}, coarse_grid=True)
    _, model = expand_grid(spec)[0]
    assert model.grid.nr <= 4000


def test_default_force_method_prefers_bh_c_when_built():
    from ntropy.forces.bhtree_c import extension_available

    expected = "bh_c" if extension_available() else "bh"
    assert _default_force_method() == expected


def test_coarse_grid_clamps_high_lmax_override():
    from galacticsics.campaign.spec import COARSE_GRID_LMAX_CAP, preview_dbh_model

    model = preview_dbh_model(coarse=True, lmax=8)
    assert model.grid.lmax == COARSE_GRID_LMAX_CAP
    assert model.grid.nr == 4000


def test_assert_dbh_grid_rejects_too_coarse_nr():
    from galacticsics.campaign.spec import apply_dbh_grid, assert_dbh_grid_diskdf_compatible
    from galacticsics.models import GalaxyModel

    model = apply_dbh_grid(GalaxyModel.milky_way_disk_halo(), dr=0.5, nr=400, lmax=8)
    try:
        assert_dbh_grid_diskdf_compatible(model)
    except ValueError as exc:
        assert "nr=400" in str(exc)
    else:
        raise AssertionError("expected ValueError for nr=400")


def test_expand_grid_rejects_incompatible_override():
    import pytest

    from galacticsics.campaign.spec import GridSpec, expand_grid

    spec = GridSpec(
        base="milky_way_disk_halo",
        coarse_grid=False,
        grid_dr=0.5,
        grid_nr=400,
        grid_lmax=8,
        axes={"halo.v0": [3.7]},
        omit_components=[[]],
    )
    with pytest.raises(ValueError, match="incompatible with diskdf"):
        expand_grid(spec)


def test_expand_grid_clamps_lmax_with_coarse_and_override():
    spec = GridSpec(
        base="milky_way_disk_halo",
        coarse_grid=True,
        grid_lmax=8,
        axes={"halo.v0": [3.7]},
        omit_components=[[]],
    )
    _, model = expand_grid(spec)[0]
    assert model.grid.lmax == 4


def test_write_evolve_config_sets_force_method(tmp_path):
    from ntropy.config import BhOptimizationsConfig
    from ntropy.forces.bhtree_c import extension_available
    from ntropy.particle_types import TypeRegistry

    registry = TypeRegistry.default_galaxy()
    bh = BhOptimizationsConfig.from_preset("optimized")  # type: ignore[arg-type]
    path = _write_evolve_config(
        tmp_path,
        registry,
        end_time_gyr=0.1,
        diagnostics_every=1,
        particle_dump_every=50,
        mpi_ranks=2,
        force_rebuild_every=5,
        dt_base=0.05,
        timestep_eta=0.02,
        max_timestep_bin=5,
        bh_optimizations=bh,
    )
    cfg = json.loads(path.read_text())
    assert cfg["force"]["method"] == _default_force_method()
    assert cfg["force"]["rebuild_every"] == 5
    assert cfg["integrator"]["dt_base"] == 0.05
    assert cfg["integrator"]["timestep"]["eta"] == 0.02
    assert cfg["integrator"]["timestep"]["max_bin"] == 5
    if extension_available():
        assert cfg["force"]["bh_optimizations"]["preset"] == "optimized"
