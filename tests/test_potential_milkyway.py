"""Potential evaluation tests for generated reference model."""

from __future__ import annotations

import math

import pytest

from galacticsics.io import read_harmonic_potential
from galacticsics.potential import evaluate_force, evaluate_potential
from galacticsics.units import velocity_to_kms


def test_potential_at_solar_neighborhood(dbh_path):
    pot = read_harmonic_potential(dbh_path)
    psi = evaluate_potential(pot, 8.0, 0.0)
    assert psi > 0.0


def test_force_midplane(dbh_path):
    pot = read_harmonic_potential(dbh_path)
    fr, fz, psi = evaluate_force(pot, 8.0, 0.0)
    assert abs(fz) < 0.1
    assert fr < 0.0


def test_circular_velocity_scale(dbh_path):
    """v_circ(R=8 kpc) should be in the 150–350 km/s range."""
    pot = read_harmonic_potential(dbh_path)
    r = 8.0
    fr, _, _ = evaluate_force(pot, r, 0.0)
    v_circ = math.sqrt(max(0.0, -fr * r))
    v_kms = velocity_to_kms(v_circ)
    assert 150.0 < v_kms < 350.0


def test_potential_force_consistency(dbh_path):
    """Numerical dPsi/dR matches radial force component at R=5, z=0.5."""
    pot = read_harmonic_potential(dbh_path)
    r, z = 5.0, 0.5
    eps = 1e-3
    psi_p = evaluate_potential(pot, r + eps, z)
    psi_m = evaluate_potential(pot, r - eps, z)
    dpsi_dr = (psi_p - psi_m) / (2 * eps)
    fr, _, _ = evaluate_force(pot, r, z)
    assert dpsi_dr == pytest.approx(fr, rel=0.15, abs=0.05)
