"""Serialize GalaxyModel to/from JSON-compatible dicts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from galacticsics.models import (
    BlackHole,
    DiskKinematics,
    ExponentialDisk,
    GalaxyModel,
    GasDisk,
    NFWHalo,
    PotentialGrid,
    SersicBulge,
    Sech2Disk,
)


def model_to_dict(model: GalaxyModel) -> dict[str, Any]:
    return asdict(model)


def model_from_dict(data: dict[str, Any]) -> GalaxyModel:
    def _optional(cls, key: str):
        raw = data.get(key)
        return None if raw is None else cls(**raw)

    return GalaxyModel(
        halo=_optional(NFWHalo, "halo"),
        disk=_optional(ExponentialDisk, "disk"),
        disk2=_optional(Sech2Disk, "disk2"),
        gas=_optional(GasDisk, "gas"),
        bulge=_optional(SersicBulge, "bulge"),
        black_hole=BlackHole(**data.get("black_hole", {})),
        grid=PotentialGrid(**data.get("grid", {})),
        disk_kinematics=DiskKinematics(**data.get("disk_kinematics", {})),
    )
