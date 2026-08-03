"""Parameter grid specification and expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.models import GalaxyModel

# GalaxyModel sub-objects editable via dot-path patch / grid axes.
PATCHABLE_COMPONENTS = (
    "halo",
    "disk",
    "disk2",
    "gas",
    "bulge",
    "black_hole",
    "disk_kinematics",
)


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
        Dot-path keys → list of values, e.g. ``{"halo.v0": [3.4, 3.7]}`` or
        ``{"disk_kinematics.sigma_r0": [0.85, 1.0]}``,
        ``{"disk_kinematics.toomre_q_target": [1.2, 1.5]}``.
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
        With ``coarse_grid=True``, ``lmax`` is capped at 4 so Python ``diskdf``
        produces a valid ``cordbh.dat``.
    physics_backend : str, optional
        IC numerics backend: ``"python"`` (default) or ``"legacy"`` (Fortran
        ``legacy/bin`` subprocesses). Set ``GALACTICSICS_PHYSICS_BACKEND=legacy``
        to override globally.
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
    physics_backend: str = "python"
    patch: dict[str, Any] = field(default_factory=dict)


def _base_model(name: str) -> GalaxyModel:
    factories = {
        "milky_way_disk_halo": GalaxyModel.milky_way_disk_halo,
        "milky_way_disk_halo_bulge": GalaxyModel.milky_way_disk_halo_bulge,
        "reference_disk_halo": GalaxyModel.reference_disk_halo,
    }
    if name not in factories:
        raise ValueError(f"unknown base model {name!r}")
    return factories[name]()


def _default_component(comp: str):
    """Instantiate a missing patchable component (e.g. bulge on disk+halo bases)."""
    from galacticsics.models import SersicBulge

    if comp == "bulge":
        return SersicBulge(n_sersic=4.0, ppp=0.5, v0=2.0, a=0.5, enabled=True)
    raise ValueError(f"model has no component {comp!r}")


def _set_nested(model: GalaxyModel, path: str, value: Any) -> GalaxyModel:
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(f"axis path must be component.field, got {path!r}")
    comp, attr = parts
    obj = getattr(model, comp)
    if obj is None:
        obj = _default_component(comp)
        model = replace(model, **{comp: obj})
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
    nr = min(g.nr, 4000)
    if model.disk and model.disk.enabled and nr < 4000:
        nr = 4000
    return replace(
        model,
        grid=PotentialGrid(dr=max(g.dr, 0.05), nr=nr, lmax=min(g.lmax, 4)),
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


def assert_dbh_grid_diskdf_compatible(model: GalaxyModel) -> None:
    """
    Raise when the Poisson grid is too coarse for Python ``diskdf``.

    Collapsed ``cordbh.dat`` (``f_d`` clipped to ``1e-3``) and near-static disk
    ICs are the usual symptom of violating these limits.
    """
    if not (model.disk and model.disk.enabled):
        return
    g = model.grid
    problems: list[str] = []
    if g.nr < 1000:
        problems.append(f"nr={g.nr} is too small (use coarse_grid=True or nr≥4000)")
    if g.nr <= 4000 and g.lmax > COARSE_GRID_LMAX_CAP:
        problems.append(
            f"lmax={g.lmax} is too high for nr={g.nr} (cap lmax at {COARSE_GRID_LMAX_CAP})"
        )
    if problems:
        raise ValueError(
            "DBH grid is incompatible with diskdf: "
            + "; ".join(problems)
            + ". "
            + format_dbh_grid_summary(model)
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


def _apply_patch(model: GalaxyModel, patch: dict[str, Any]) -> GalaxyModel:
    for key, value in patch.items():
        model = _set_nested(model, key, value)
    return model


def model_patch_defaults(base: str = "milky_way_disk_halo") -> dict[str, Any]:
    """
    Flatten default physical parameters for ``base`` into dot-path patch keys.

    Used to populate notebook / JSON ``base_model.patch``.  Units: lengths [kpc],
    velocities [100 km/s], masses [GalactICS mass units ≈ 2.325×10⁹ M☉].
    Poisson grid (``grid.dr``, ``grid.nr``, ``grid.lmax``) is configured separately
    under ``base_model.grid``.
    """
    model = _base_model(base)
    patch: dict[str, Any] = {}
    for comp in PATCHABLE_COMPONENTS:
        obj = getattr(model, comp, None)
        if obj is None or not is_dataclass(obj):
            continue
        for f in fields(obj):
            if f.name == "enabled":
                continue
            patch[f"{comp}.{f.name}"] = getattr(obj, f.name)
    return patch


def model_component_parameter_table(
    model: GalaxyModel,
) -> "pd.DataFrame":
    """All patchable component fields on a :class:`~galacticsics.models.GalaxyModel`."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for comp in PATCHABLE_COMPONENTS:
        obj = getattr(model, comp, None)
        if obj is None or not is_dataclass(obj):
            continue
        for f in fields(obj):
            if f.name == "enabled":
                continue
            rows.append(
                {
                    "parameter": f"{comp}.{f.name}",
                    "value": getattr(obj, f.name),
                    "enabled": getattr(obj, "enabled", True),
                }
            )
    return pd.DataFrame(rows)


def _apply_grid_spec(model: GalaxyModel, spec: GridSpec) -> GalaxyModel:
    if spec.coarse_grid:
        model = _coarsen_grid(model)
    model = apply_dbh_grid(model, dr=spec.grid_dr, nr=spec.grid_nr, lmax=spec.grid_lmax)
    if spec.coarse_grid:
        model = _clamp_coarse_lmax(model)
    if spec.patch:
        model = _apply_patch(model, spec.patch)
    assert_dbh_grid_diskdf_compatible(model)
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


def models_from_work_root(
    work_root: Path | str,
    *,
    require_sample: bool = True,
) -> list[tuple[str, GalaxyModel]]:
    """
    Load ``(label, model)`` pairs from existing campaign model directories.

    Used to resume a corpus after a prior run selected a different random
    subsample (or after a crash). Directories must contain ``model.json``.
    When ``require_sample`` is True, only dirs with ``.done_sample`` or a
    ``disk`` particle file are included.
    """
    from galacticsics.campaign.serialize import model_from_dict

    work_root = Path(work_root)
    if not work_root.is_dir():
        return []
    results: list[tuple[str, GalaxyModel]] = []
    for path in sorted(work_root.iterdir()):
        if not path.is_dir():
            continue
        model_path = path / "model.json"
        if not model_path.is_file():
            continue
        if require_sample and not (
            (path / ".done_sample").is_file() or (path / "disk").is_file()
        ):
            continue
        try:
            model = model_from_dict(json.loads(model_path.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        results.append((f"resume_{path.name}", model))
    return results


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
    omit_patterns = spec.omit_components or [[]]
    total = len(combos) * max(1, len(omit_patterns))
    selected_combos = combos
    if total > spec.max_points:
        # Deterministic subsample for ML corpora (factorial grids explode quickly).
        # Use sha256 — NOT Python's hash(), which is randomized per process and would
        # reshuffle the corpus on every campaign relaunch (breaking skip_done).
        seed = int(hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n_keep = max(1, spec.max_points // max(1, len(omit_patterns)))
        idx = rng.choice(len(combos), size=min(n_keep, len(combos)), replace=False)
        selected_combos = [combos[i] for i in sorted(idx.tolist())]

    for combo in selected_combos:
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


def expand_grids(specs: list[GridSpec]) -> list[tuple[str, GalaxyModel]]:
    """
    Expand one or more grid specs and merge with de-duplication by model hash.

    Labels are prefixed with ``{spec.name}::`` when more than one spec is given
    so suite provenance stays visible in manifests.
    """
    if not specs:
        return []
    multi = len(specs) > 1
    seen: set[str] = set()
    results: list[tuple[str, GalaxyModel]] = []
    for spec in specs:
        for label, model in expand_grid(spec):
            h = model_hash(model)
            if h in seen:
                continue
            seen.add(h)
            full_label = f"{spec.name}::{label}" if multi else label
            results.append((full_label, model))
    return results


def grids_as_list_spec(
    name: str,
    pairs: list[tuple[str, GalaxyModel]],
    *,
    base: str = "milky_way_disk_halo_bulge",
    coarse_grid: bool = True,
    physics_backend: str = "python",
) -> GridSpec:
    """Pack expanded ``(label, model)`` pairs into a ``grid_mode='list'`` spec."""
    from galacticsics.campaign.serialize import model_to_dict

    return GridSpec(
        name=name,
        base=base,
        grid_mode="list",
        models=[model_to_dict(model) for _, model in pairs],
        coarse_grid=coarse_grid,
        physics_backend=physics_backend,
        max_points=max(len(pairs), 1),
    )


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
