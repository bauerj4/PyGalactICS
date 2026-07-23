"""Tests for virial-theorem diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from ntropy.ics.plummer import sample_plummer
from ntropy.softening import kinetic_energy, virial_diagnostic

pytestmark = pytest.mark.essential


def test_plummer_virial_equilibrium():
    state = sample_plummer(seed=42)
    diag = virial_diagnostic(state.pos, state.vel, state.mass, state.eps, rtol=0.55)
    assert diag["is_virial_equilibrium"]
    assert 0.2 < diag["ke_over_abs_pe"] < 2.0
    assert 0.4 < diag["virial_ratio"] < 2.5


def test_virial_subsamples_large_n():
    from ntropy.ics.plummer import PlummerParams

    state = sample_plummer(PlummerParams(n_particles=200), seed=0)
    # Artificially small cap forces subsampling path
    diag = virial_diagnostic(
        state.pos, state.vel, state.mass, state.eps, max_particles=64, rng=np.random.default_rng(1)
    )
    assert diag["subsampled"]
    assert diag["n_used"] == 64
    assert diag["n_particles"] == 200
    assert np.isfinite(diag["virial_ratio"])
    # Same-subset T and W → O(1) ratio; the old full-T / subset-W bug gave ~N/n_sub.
    assert 0.2 < diag["virial_ratio"] < 5.0
    assert diag["kinetic_energy"] < kinetic_energy(state.vel, state.mass)