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
