"""Unit system and physical-unit constructors."""

from __future__ import annotations

import pytest

from galacticsics import DEFAULT_UNITS, UnitSystem
from galacticsics.models import GalaxyModel
from galacticsics.units import (
    length_from_kpc,
    length_to_kpc,
    mass_from_msun,
    mass_to_msun,
    velocity_from_kms,
    velocity_to_kms,
)


def test_default_unit_system():
    assert DEFAULT_UNITS.velocity_kms == 100.0
    assert DEFAULT_UNITS.mass_msun == pytest.approx(2.325e9)
    assert DEFAULT_UNITS.length_kpc == 1.0


def test_mass_velocity_conversions():
    assert mass_to_msun(1.0) == pytest.approx(2.325e9)
    assert mass_from_msun(2.325e9) == pytest.approx(1.0)
    assert velocity_to_kms(2.2) == pytest.approx(220.0)
    assert velocity_from_kms(220.0) == pytest.approx(2.2)


def test_length_conversions():
    assert length_to_kpc(8.0) == pytest.approx(8.0)
    assert length_from_kpc(33.0) == pytest.approx(33.0)


def test_from_physical_round_trip():
    model = GalaxyModel.from_physical(
        disk_mass_msun=4.0e10,
        halo_v0_kms=200.0,
        halo_a_kpc=20.0,
    )
    assert mass_to_msun(model.disk.mass) == pytest.approx(4.0e10, rel=1e-9)
    assert velocity_to_kms(model.halo.v0) == pytest.approx(200.0)
    assert length_to_kpc(model.halo.a) == pytest.approx(20.0)


def test_custom_unit_system():
    units = UnitSystem(velocity_kms=50.0, mass_msun=1e9, length_kpc=2.0)
    assert units.velocity_from_kms(100.0) == pytest.approx(2.0)
    assert units.mass_from_msun(5e9) == pytest.approx(5.0)
