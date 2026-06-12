"""Halo-first workflow and harmonic merge tests."""

from __future__ import annotations

import numpy as np
import pytest

from galacticsics.io.formats import (
    merge_harmonic_potentials,
    read_halo_harmonics,
    read_harmonic_potential,
)
from galacticsics.models import GalaxyModel
from galacticsics.fitting.halo_particles import estimate_nfw_from_particles


def test_model_halo_only_disables_baryons(reference_model):
    halo = reference_model.with_halo_only()
    assert halo.halo is not None and halo.halo.enabled
    assert halo.disk is not None and not halo.disk.enabled
    assert halo.bulge is None or not halo.bulge.enabled


def test_model_baryons_only_disables_halo(reference_model):
    baryon = reference_model.with_baryons_only()
    assert baryon.halo is not None and not baryon.halo.enabled
    assert baryon.disk is not None and baryon.disk.enabled


def test_read_halo_harmonics_generated(reference_artifacts_dir):
    halo = read_halo_harmonics(reference_artifacts_dir / "h.dat")
    full = read_harmonic_potential(reference_artifacts_dir / "dbh.dat")
    assert halo.model.grid.nr == full.model.grid.nr
    assert halo.flags.halo
    assert halo.apot.shape == full.apot.shape


def test_merge_harmonics_generated(reference_artifacts_dir):
    halo = read_halo_harmonics(reference_artifacts_dir / "h.dat")
    full = read_harmonic_potential(reference_artifacts_dir / "dbh.dat")
    merged = merge_harmonic_potentials(halo, full, model=full.model)
    assert np.all(np.isfinite(merged.apot))
    assert merged.apot.shape == halo.apot.shape


def test_estimate_nfw_from_synthetic_particles():
    """Fit returns positive parameters on a smooth enclosed-mass profile."""
    a = 15.0
    m_scale = 80.0
    r = np.logspace(-1, 2.2, 200)
    from galacticsics.fitting.halo_particles import _nfw_mass_enclosed

    m_enc = _nfw_mass_enclosed(r, m_scale, a)
    x, y, z = r, np.zeros_like(r), np.zeros_like(r)
    dm = np.diff(m_enc, prepend=0.0)
    dm[dm <= 0] = m_scale * 1e-4
    fit = estimate_nfw_from_particles(x, y, z, dm, center=False, n_bins=25)
    assert fit.a > 0 and fit.v0 > 0
    assert fit.rms_residual < 0.05


def test_from_physical_matches_reference(reference_model):
    physical = GalaxyModel.from_physical()
    assert physical.disk.mass == pytest.approx(reference_model.disk.mass, rel=1e-3)
    assert physical.halo.v0 == pytest.approx(reference_model.halo.v0, rel=1e-3)
