"""Spherical galaxy test model builders."""

from __future__ import annotations

from galacticsics.models import GalaxyModel


def nfw_halo_only() -> GalaxyModel:
    return GalaxyModel.nfw_halo_only()


def sersic_bulge_only() -> GalaxyModel:
    return GalaxyModel.sersic_bulge_only()


def nfw_plus_bulge() -> GalaxyModel:
    return GalaxyModel.nfw_plus_bulge()
