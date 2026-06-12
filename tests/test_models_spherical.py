"""Unit tests for spherical galaxy model fixtures."""

from __future__ import annotations

import pytest

from tests.fixtures.spherical import nfw_halo_only, nfw_plus_bulge, sersic_bulge_only


def test_nfw_halo_only_fixture():
    m = nfw_halo_only()
    assert m.halo.enabled
    assert m.disk is None or not m.disk.enabled
    assert m.grid.lmax == 0


def test_sersic_bulge_only_fixture():
    m = sersic_bulge_only()
    assert m.bulge.enabled
    assert m.halo is None or not (m.halo and m.halo.enabled)


def test_nfw_plus_bulge_fixture():
    m = nfw_plus_bulge()
    assert m.halo.enabled
    assert m.bulge.enabled
    assert m.psi0 == pytest.approx(m.halo.v0**2)
