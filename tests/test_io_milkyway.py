"""I/O tests against generated reference artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from galacticsics.io import (
    read_component_masses,
    read_disk_correction,
    read_frequency_table,
    read_harmonic_potential,
    write_harmonic_potential,
)
from galacticsics.io.formats import read_rtidal, read_toomre_q
from galacticsics.artifacts.verify import verify_artifact_consistency
from tests.constants import ATOL, RTOL


def test_read_harmonic_potential_reference(dbh_path, reference_model):
    pot = read_harmonic_potential(dbh_path)
    assert pot.nr == reference_model.grid.nr
    assert pot.lmax == reference_model.grid.lmax
    assert pot.dr == pytest.approx(reference_model.grid.dr, rel=0, abs=ATOL)
    assert pot.flags.disk is True
    assert pot.flags.halo is True
    assert pot.model.disk.mass == pytest.approx(reference_model.disk.mass, rel=RTOL)
    assert pot.model.halo.v0 == pytest.approx(reference_model.halo.v0, rel=RTOL)


def test_harmonic_round_trip(tmp_path, dbh_path):
    pot = read_harmonic_potential(dbh_path)
    out = tmp_path / "dbh_roundtrip.dat"
    write_harmonic_potential(pot, out)
    pot2 = read_harmonic_potential(out)
    np.testing.assert_allclose(pot.apot, pot2.apot, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(pot.fr, pot2.fr, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(pot.adens, pot2.adens, rtol=RTOL, atol=ATOL)


@pytest.mark.physics_python
def test_read_harmonic_potential_preserves_flags_at_high_lmax(tmp_path) -> None:
    """Six-column harmonic rows must not be mistaken for component flags (lmax >= 8)."""
    from dataclasses import replace

    from galacticsics.models import GalaxyModel, PotentialGrid
    from galacticsics.physics.backend import PhysicsBackendKind
    from galacticsics.potential.solver import solve_potential

    model = replace(
        GalaxyModel.reference_disk_halo(),
        grid=PotentialGrid(dr=0.2, nr=36, lmax=8),
    )
    work = tmp_path / "l8"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    pot = read_harmonic_potential(work / "dbh.dat")
    assert pot.lmax == 8
    assert pot.flags.disk is True
    assert pot.flags.halo is True
    assert pot.n_harmonics == 5


def test_read_component_masses(mr_path):
    masses = read_component_masses(mr_path)
    dm, dr = masses["disk"]
    hm, hr = masses["halo"]
    assert dm > 0 and hm > 0
    assert hm > dm
    assert dr > 0 and hr > 0


def test_read_disk_correction(cordbh_path, reference_model):
    corr = read_disk_correction(cordbh_path)
    assert corr.sigma_r0 == pytest.approx(reference_model.disk_kinematics.sigma_r0, rel=RTOL)
    assert corr.sigma_r_scale == pytest.approx(reference_model.disk_kinematics.sigma_r_scale, rel=RTOL)
    assert len(corr.radius) >= 6
    assert corr.f_d_at(0.0) == pytest.approx(1.0, rel=1e-2)


def test_cordbh_is_valid(reference_artifacts_dir, cordbh_path, tmp_path):
    from galacticsics.io.formats import cordbh_is_valid, read_disk_correction

    assert cordbh_is_valid(cordbh_path)
    empty = tmp_path / "cordbh.dat"
    empty.write_text("")
    assert not cordbh_is_valid(empty)
    assert not cordbh_is_valid(reference_artifacts_dir / "missing.dat")

    # Collapsed diskdf floor (f_d ~ 1e-3) must be rejected.
    bad = tmp_path / "bad_cordbh.dat"
    good = read_disk_correction(cordbh_path)
    lines = [f"# {good.sigma_r0:17.7f} {good.sigma_r_scale:17.7f} {len(good.radius)-1}"]
    for r in good.radius:
        lines.append(f" {r:17.7f} {0.001:17.7f} {0.001:17.7f}")
    bad.write_text("\n".join(lines) + "\n")
    assert not cordbh_is_valid(bad)


def test_cordbh_needs_refresh_after_dbh(tmp_path):
    import time

    from galacticsics.io.formats import cordbh_needs_refresh

    work = tmp_path / "run"
    work.mkdir()
    cordbh = work / "cordbh.dat"
    dbh = work / "dbh.dat"
    cordbh.write_text("x")
    assert not cordbh_needs_refresh(work)
    time.sleep(0.01)
    dbh.write_text("y")
    assert cordbh_needs_refresh(work)


def test_read_frequency_table(freqdbh_path):
    freq = read_frequency_table(freqdbh_path)
    assert freq.radius.shape[0] > 100
    assert freq.omega(5.0) > 0.0
    assert freq.kappa(5.0) > 0.0


def test_read_rtidal_and_toomre(reference_artifacts_dir):
    rtidal = read_rtidal(reference_artifacts_dir / "rtidal.dat")
    toomre = read_toomre_q(reference_artifacts_dir / "toomre2.5")
    assert rtidal > 0
    assert 0.5 < toomre < 3.0


def test_artifact_consistency(reference_artifacts_dir, reference_model):
    report = verify_artifact_consistency(reference_artifacts_dir, model=reference_model)
    assert report.ok, report.summary()


def test_manifest_matches_model(reference_artifacts_dir, reference_model):
    manifest = json.loads((reference_artifacts_dir / "manifest.json").read_text())
    assert manifest["grid"]["nr"] == reference_model.grid.nr
    assert manifest["disk"]["mass_galactics"] == pytest.approx(reference_model.disk.mass)
