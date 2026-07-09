"""Parameter grid specification and expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from galacticsics.models import GalaxyModel


@dataclass
class GridSpec:
    """
    DBH parameter grid definition.

    Attributes
    ----------
    name : str
        Campaign label.
    base : str
        Base model factory name (e.g. ``milky_way_disk_halo``).
    axes : dict
        Dot-path keys → list of values, e.g. ``{"halo.v0": [3.4, 3.7]}``.
    omit_components : list of list of str
        Component omission patterns applied via ``enabled=False``.
    grid_mode : str
        ``factorial`` (default) or ``list`` (use ``models`` directly).
    models : list of dict, optional
        Explicit model overrides when ``grid_mode == "list"``.
    max_points : int
        Cap on factorial grid size.
    coarse_grid : bool
        When true, use reduced ``nr`` for screening solves.
    grid_dr, grid_nr, grid_lmax : optional
        Explicit DBH Poisson grid overrides [kpc], [bins], [multipole order].
        Applied after ``coarse_grid`` coarsening when expanding the grid.
        With ``coarse_grid=True``, ``lmax`` is capped at 4 so legacy ``diskdf``
        produces a valid ``cordbh.dat``.
    """

    name: str = "mw_grid"
    base: str = "milky_way_disk_halo"
    axes: dict[str, list[Any]] = field(default_factory=dict)
    omit_components: list[list[str]] = field(default_factory=list)
    grid_mode: str = "factorial"
    models: list[dict[str, Any]] = field(default_factory=list)
    max_points: int = 500
    coarse_grid: bool = True
    grid_dr: float | None = None
    grid_nr: int | None = None
    grid_lmax: int | None = None


def _base_model(name: str) -> GalaxyModel:
    factories = {
        "milky_way_disk_halo": GalaxyModel.milky_way_disk_halo,
        "reference_disk_halo": GalaxyModel.reference_disk_halo,
    }
    if name not in factories:
        raise ValueError(f"unknown base model {name!r}")
    return factories[name]()


def _set_nested(model: GalaxyModel, path: str, value: Any) -> GalaxyModel:
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(f"axis path must be component.field, got {path!r}")
    comp, attr = parts
    obj = getattr(model, comp)
    if obj is None:
        raise ValueError(f"model has no component {comp!r}")
    return replace(model, **{comp: replace(obj, **{attr: value})})


def _apply_omit(model: GalaxyModel, omit: list[str]) -> GalaxyModel:
    updates: dict[str, Any] = {}
    mapping = {
        "halo": "halo",
        "disk": "disk",
        "disk2": "disk2",
        "gas": "gas",
        "bulge": "bulge",
        "black_hole": "black_hole",
    }
    for name in omit:
        comp = mapping.get(name)
        if comp is None:
            raise ValueError(f"unknown component to omit: {name!r}")
        obj = getattr(model, comp)
        if obj is None:
            continue
        updates[comp] = replace(obj, enabled=False)
    return replace(model, **updates) if updates else model


# ``diskdf`` fails with NaN splines on coarse grids when lmax is too high (nr≤4000).
COARSE_GRID_LMAX_CAP = 4


def _coarsen_grid(model: GalaxyModel) -> GalaxyModel:
    from galacticsics.models import PotentialGrid

    g = model.grid
    return replace(
        model,
        grid=PotentialGrid(dr=max(g.dr, 0.05), nr=min(g.nr, 4000), lmax=min(g.lmax, 4)),
    )


def _clamp_coarse_lmax(model: GalaxyModel) -> GalaxyModel:
    """Cap multipole order so legacy ``diskdf`` + ``gendisk`` succeed on coarse grids."""
    from galacticsics.models import PotentialGrid

    g = model.grid
    if g.lmax <= COARSE_GRID_LMAX_CAP:
        return model
    return replace(
        model,
        grid=PotentialGrid(dr=g.dr, nr=g.nr, lmax=COARSE_GRID_LMAX_CAP),
    )


def apply_dbh_grid(
    model: GalaxyModel,
    *,
    dr: float | None = None,
    nr: int | None = None,
    lmax: int | None = None,
) -> GalaxyModel:
    """Apply explicit ``dbh`` Poisson grid overrides to a model."""
    if dr is None and nr is None and lmax is None:
        return model
    from galacticsics.models import PotentialGrid

    g = model.grid
    return replace(
        model,
        grid=PotentialGrid(
            dr=g.dr if dr is None else dr,
            nr=g.nr if nr is None else nr,
            lmax=g.lmax if lmax is None else lmax,
        ),
    )


def _apply_grid_spec(model: GalaxyModel, spec: GridSpec) -> GalaxyModel:
    if spec.coarse_grid:
        model = _coarsen_grid(model)
    model = apply_dbh_grid(model, dr=spec.grid_dr, nr=spec.grid_nr, lmax=spec.grid_lmax)
    if spec.coarse_grid:
        model = _clamp_coarse_lmax(model)
    return model


def preview_dbh_model(
    base: str = "milky_way_disk_halo",
    *,
    coarse: bool = True,
    dr: float | None = None,
    nr: int | None = None,
    lmax: int | None = None,
) -> GalaxyModel:
    """Build a model with the same DBH grid rules as :func:`expand_grid`."""
    spec = GridSpec(
        base=base,
        coarse_grid=coarse,
        grid_dr=dr,
        grid_nr=nr,
        grid_lmax=lmax,
    )
    return _apply_grid_spec(_base_model(base), spec)


def format_dbh_grid_summary(model: GalaxyModel) -> str:
    """One-line summary of the legacy ``dbh`` Poisson grid."""
    g = model.grid
    r_max = g.nr * g.dr
    return (
        f"DBH grid: dr={g.dr:g} kpc, nr={g.nr}, lmax={g.lmax} "
        f"(r_max ≈ {r_max:g} kpc; cost scales ~ nr × lmax²)"
    )


def model_hash(model: GalaxyModel) -> str:
    """Short hash for work directory naming."""
    from galacticsics.campaign.serialize import model_to_dict

    blob = json.dumps(model_to_dict(model), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def expand_grid(spec: GridSpec) -> list[tuple[str, GalaxyModel]]:
    """
    Expand a grid spec into (label, GalaxyModel) pairs.

    Returns
    -------
    list of (label, model)
    """
    results: list[tuple[str, GalaxyModel]] = []

    if spec.grid_mode == "list":
        from galacticsics.campaign.serialize import model_from_dict

        for i, raw in enumerate(spec.models):
            model = _apply_grid_spec(model_from_dict(raw), spec)
            results.append((f"model_{i:04d}", model))
        return results

    base = _apply_grid_spec(_base_model(spec.base), spec)

    axis_keys = sorted(spec.axes.keys())
    axis_vals = [spec.axes[k] for k in axis_keys]
    combos = list(product(*axis_vals)) if axis_vals else [()]

    if len(combos) * max(1, len(spec.omit_components) or 1) > spec.max_points:
        raise ValueError(
            f"grid would exceed max_points={spec.max_points}; "
            f"reduce axes or raise max_points"
        )

    omit_patterns = spec.omit_components or [[]]
    for combo in combos:
        model = base
        label_parts: list[str] = []
        for key, val in zip(axis_keys, combo):
            model = _set_nested(model, key, val)
            label_parts.append(f"{key.replace('.', '_')}={val}")
        for omit in omit_patterns:
            m = _apply_omit(model, omit)
            omit_label = "omit_" + "_".join(omit) if omit else "full"
            label = "__".join(label_parts + [omit_label]) if label_parts else omit_label
            results.append((label, m))

    return results


def load_grid_spec(path: Path | str) -> GridSpec:
    """Load grid spec from JSON or YAML file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML required for YAML grid specs: pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return GridSpec(**raw)
