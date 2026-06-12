"""Parity tests: SciPy numerics vs legacy tabulated values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from galacticsics.io import read_disk_correction
from galacticsics.numerics import legendre_even_l, natural_cubic_spline, simpson_integrate
from tests.constants import RTOL


def test_cubic_spline_matches_cordbh_nodes(cordbh_path):
    corr = read_disk_correction(cordbh_path)
    for r, fd in zip(corr.radius, corr.f_d):
        assert corr.f_d_at(r) == pytest.approx(fd, rel=1e-10)


def test_legendre_at_mu_one():
    p, _ = legendre_even_l(1.0, lmax=4)
    assert p[0] == pytest.approx(1.0, abs=1e-12)  # P_0
    assert p[1] == pytest.approx(1.0, abs=1e-12)  # P_2(1)


def test_simpson_integrate_polynomial():
    result = simpson_integrate(lambda x: x**2, 0.0, 1.0, n=128)
    scipy_ref = integrate.quad(lambda x: x**2, 0.0, 1.0)[0]
    assert result == pytest.approx(scipy_ref, rel=RTOL)
    assert result == pytest.approx(1.0 / 3.0, rel=RTOL)
