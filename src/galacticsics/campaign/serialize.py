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
        if raw is None:
            return None
        values = dict(raw)
        # Model JSON written before the projected-column correction described
        # the legacy midplane-normalized density law.  Preserve that meaning
        # instead of silently changing old campaigns when they are reloaded.
        if cls is ExponentialDisk and "vertical_density_mode" not in values:
            values["vertical_density_mode"] = "legacy_midplane"
        return cls(**values)

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
