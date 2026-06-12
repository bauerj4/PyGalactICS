"""Tests for Plummer IC generator and Abel DF."""

from __future__ import annotations

import numpy as np

from ntropy.ics.plummer import plummer_df, plummer_density, sample_plummer
from ntropy.ics.spherical import abel_df_plummer, eddington_df
from ntropy.softening import kinetic_energy, softened_potential_energy


def test_abel_matches_analytic_plummer_df():
    mass, a = 50.0, 1.0
    r_grid = np.logspace(-2, 2, 300) * a
    rho = plummer_density(r_grid, mass, a)
    e_grid, f_num, _ = eddington_df(r_grid, rho)
    f_ana = plummer_df(mass, a, e_grid)
    f_abel = abel_df_plummer(mass, a, e_grid)
    mask = f_ana > 1e-10
    np.testing.assert_allclose(f_abel[mask], f_ana[mask], rtol=0.05)
    np.testing.assert_allclose(f_num[mask], f_ana[mask], rtol=0.3)


def test_plummer_virial_ratio():
    state = sample_plummer(seed=42)
    ke = kinetic_energy(state.vel, state.mass)
    pe = softened_potential_energy(state.pos, state.mass, state.eps)
    ratio = ke / abs(pe) if pe != 0 else 0.0
    assert 0.2 < ratio < 2.0
