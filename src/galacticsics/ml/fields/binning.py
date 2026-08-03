"""Bin particle snapshots into vertical-slice channel stacks or voxels.

**Shared center (required)**

Callers must place the *full* snapshot on one mass-weighted global COM frame
before calling these depositors
(:func:`galacticsics.ml.fields.frame.prepare_shared_frame`).  Binning never
recenters per component — that would misalign disk / bulge / halo.

**Per-component scale (design constraint)**

Disk, bulge, and halo live on very different physical scales.  A single shared
FOV either wastes resolution on the bulge or truncates the halo.  Prefer
:class:`MultiScaleSliceConfig` / :class:`MultiScaleVoxelConfig` with independent
``r_max``, ``n_pix`` / ``n_bins``, and ``z`` spacing per component:

| Component | Typical FOV | Resolution bias | ``z`` spacing |
|-----------|-------------|-----------------|---------------|
| bulge     | ~4 kpc      | fine voxels     | uniform       |
| disk      | ~12–15 kpc  | midplane focus  | denser near ``z=0`` |
| halo      | ~40–80 kpc  | coarser         | uniform, thick |

CNN encoders fuse these via separate towers (native resolution) or by
rescaling to a common canvas (:func:`fuse_slice_maps_to_common`).

**Scaling (CPU smoke → production)**

| Knob | Baseline (32²) | Near-prod smoke | Scale-up |
|------|----------------|-----------------|----------|
| disk ``n_pix`` | 32 | 64–96 | 128–256 |
| bulge ``n_pix`` | 32 | 48–64 | 128 |
| halo ``n_pix`` | 24 | 32 | 64 |
| ``n_z`` | 4–8 | 8–12 | 16–32 |
| Moments | dens + ⟨v⟩ | +σ (and optional β) | +skew / A₂ maps |
| Potential | off or subsample Plummer | same | BH/tree Φ on grid |
| Model | shallow multi-tower CNN | U-Net + dens/moment heads | 3-D CNN on voxels |

Memory for one component float32 stack is roughly
``4 · n_z · n_mom · n_pix²`` (slices) or ``4 · n_mom · n_xy² · n_z`` (voxels)
with ``n_mom = len(MOMENT_KEYS)`` (7 by default: dens, ⟨v⟩, σ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from galacticsics.ml.morton.tokenize import COMPONENT_IDS, ID_TO_COMPONENT

# Channel layout per z-slab (slices) or per voxel cell:
# dens, mean velocity (vx,vy,vz), velocity dispersion (sx,sy,sz),
# optional mass-weighted anisotropy proxy β = 1 - 2 σ_z² / (σ_x² + σ_y²).
MOMENT_KEYS_BASE = ("dens", "vx", "vy", "vz")
MOMENT_KEYS_DISP = ("sx", "sy", "sz")
MOMENT_KEYS_ANISO = ("beta",)
MOMENT_KEYS = MOMENT_KEYS_BASE + MOMENT_KEYS_DISP  # default: 7 channels
MOMENT_KEYS_FULL = MOMENT_KEYS_BASE + MOMENT_KEYS_DISP + MOMENT_KEYS_ANISO

# Per-offset kind for normalize / loss: dens|vel|disp|aniso
MOMENT_KINDS: dict[str, str] = {
    "dens": "dens",
    "vx": "vel",
    "vy": "vel",
    "vz": "vel",
    "sx": "disp",
    "sy": "disp",
    "sz": "disp",
    "beta": "aniso",
}

DEFAULT_COMPONENTS = ("disk", "halo", "bulge")
ZSpacing = Literal["uniform", "midplane"]
MomentSet = Literal["base", "disp", "full"]


def resolve_moment_keys(moment_set: MomentSet = "disp") -> tuple[str, ...]:
    """Select channel layout: dens+⟨v⟩, +σ, or +β anisotropy."""
    if moment_set == "base":
        return MOMENT_KEYS_BASE
    if moment_set == "full":
        return MOMENT_KEYS_FULL
    return MOMENT_KEYS


def moment_kind(key: str) -> str:
    return MOMENT_KINDS.get(key, "vel")


@dataclass(frozen=True)
class ComponentSliceGrid:
    """
    Vertical-slice map geometry for **one** component.

    Parameters
    ----------
    name : str
        ``disk``, ``halo``, or ``bulge``.
    n_pix : int
        Pixels per side on the face-on ``(x, y)`` plane.
    n_z : int
        Number of slabs along ``z``.
    r_max : float
        Half-width of the square ``(x, y)`` domain [kpc].
    z_max : float
        Half-height of the ``z`` stack [kpc]; slabs cover ``[-z_max, z_max]``.
    z_spacing : {"uniform", "midplane"}
        ``midplane`` clusters Δz near ``z=0`` (disk); ``uniform`` for bulge/halo.
    moment_set : {"base", "disp", "full"}
        Channel layout (see :func:`resolve_moment_keys`).
    """

    name: str
    n_pix: int = 64
    n_z: int = 8
    r_max: float = 15.0
    z_max: float = 2.0
    z_spacing: ZSpacing = "uniform"
    moment_set: MomentSet = "disp"

    @property
    def moment_keys(self) -> tuple[str, ...]:
        return resolve_moment_keys(self.moment_set)

    @property
    def n_mom(self) -> int:
        return len(self.moment_keys)

    @property
    def n_moment_channels(self) -> int:
        return int(self.n_z) * self.n_mom


@dataclass(frozen=True)
class MultiScaleSliceConfig:
    """
    Per-component slice grids (different FOV / resolution / Δz).

    Use :meth:`smoke_defaults` for the current CPU near-prod sizes, or
    :meth:`baseline_32_defaults` to reproduce the original 32² smoke.
    """

    grids: tuple[ComponentSliceGrid, ...]
    include_potential: bool = False

    @staticmethod
    def smoke_defaults(
        *,
        include_potential: bool = True,
        moment_set: MomentSet = "disp",
    ) -> MultiScaleSliceConfig:
        """Higher-res multi-scale grids (disk 64², finer bulge, coarser halo)."""
        return MultiScaleSliceConfig(
            grids=(
                ComponentSliceGrid(
                    "bulge",
                    n_pix=48,
                    n_z=8,
                    r_max=4.0,
                    z_max=4.0,
                    z_spacing="uniform",
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "disk",
                    n_pix=64,
                    n_z=10,
                    r_max=12.0,
                    z_max=1.5,
                    z_spacing="midplane",
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "halo",
                    n_pix=32,
                    n_z=4,
                    r_max=40.0,
                    z_max=40.0,
                    z_spacing="uniform",
                    moment_set=moment_set,
                ),
            ),
            include_potential=include_potential,
        )

    @staticmethod
    def baseline_32_defaults(
        *,
        include_potential: bool = False,
        moment_set: MomentSet = "base",
    ) -> MultiScaleSliceConfig:
        """Original 32² dens+⟨v⟩ smoke (for A₂ regression vs baseline)."""
        return MultiScaleSliceConfig(
            grids=(
                ComponentSliceGrid(
                    "bulge",
                    n_pix=32,
                    n_z=8,
                    r_max=4.0,
                    z_max=4.0,
                    z_spacing="uniform",
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "disk",
                    n_pix=32,
                    n_z=8,
                    r_max=12.0,
                    z_max=1.5,
                    z_spacing="midplane",
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "halo",
                    n_pix=24,
                    n_z=4,
                    r_max=40.0,
                    z_max=40.0,
                    z_spacing="uniform",
                    moment_set=moment_set,
                ),
            ),
            include_potential=include_potential,
        )

    @staticmethod
    def progressive_defaults(
        *,
        disk_n_pix: int = 96,
        include_potential: bool = False,
        moment_set: MomentSet = "disp",
        disk_n_z: int | None = None,
        bulge_n_z: int | None = None,
        halo_n_z: int | None = None,
        bulge_z_max: float | None = None,
        bulge_z_spacing: str | None = None,
        bulge_n_pix: int | None = None,
    ) -> MultiScaleSliceConfig:
        """
        Progressive multi-scale grids: ``n_pix`` and ``n_z`` both scale with disk.

        Halo stays coarse vs the disk but ``n_pix`` / ``n_z`` grow with disk
        resolution so non-axisymmetric halo residuals are not forcibly smoothed.
        Explicit ``*_n_z`` overrides replace the tier defaults (useful for
        midplane stacking experiments without changing in-plane resolution).
        ``bulge_z_max`` / ``bulge_z_spacing`` / ``bulge_n_pix`` tune the bulge
        tower for cusp recovery without forcing a full disk-resolution bump.
        """
        disk_n_pix = int(disk_n_pix)
        # Prefer even grids for U-Net stride-2 friendliness.
        if disk_n_pix >= 96:
            bulge_n = max(48, ((disk_n_pix * 2) // 3) // 2 * 2)
        else:
            bulge_n = max(48, (disk_n_pix // 2) // 2 * 2)
        if bulge_n_pix is not None:
            bulge_n = int(bulge_n_pix)
            if bulge_n % 2:
                bulge_n += 1
        halo_n = max(32, (disk_n_pix // 3) // 2 * 2)
        # Vertical tiers: crisp@128 uses 14/12/8; hires_z@≥160 bumps midplane stack.
        if disk_n_pix <= 96:
            disk_nz, bulge_nz, halo_nz = 12, 10, 6
        elif disk_n_pix <= 128:
            disk_nz, bulge_nz, halo_nz = 14, 12, 8
        elif disk_n_pix <= 160:
            disk_nz, bulge_nz, halo_nz = 18, 14, 10
        else:
            disk_nz, bulge_nz, halo_nz = 20, 16, 12
        if disk_n_z is not None:
            disk_nz = int(disk_n_z)
        if bulge_n_z is not None:
            bulge_nz = int(bulge_n_z)
        if halo_n_z is not None:
            halo_nz = int(halo_n_z)
        b_zmax = 4.0 if bulge_z_max is None else float(bulge_z_max)
        b_zsp = "uniform" if bulge_z_spacing is None else str(bulge_z_spacing)
        return MultiScaleSliceConfig(
            grids=(
                ComponentSliceGrid(
                    "bulge",
                    n_pix=bulge_n,
                    n_z=bulge_nz,
                    r_max=4.0,
                    z_max=b_zmax,
                    z_spacing=b_zsp,  # type: ignore[arg-type]
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "disk",
                    n_pix=disk_n_pix,
                    n_z=disk_nz,
                    r_max=12.0,
                    z_max=1.5,
                    z_spacing="midplane",
                    moment_set=moment_set,
                ),
                ComponentSliceGrid(
                    "halo",
                    n_pix=halo_n,
                    n_z=halo_nz,
                    r_max=40.0,
                    z_max=40.0,
                    z_spacing="uniform",
                    moment_set=moment_set,
                ),
            ),
            include_potential=include_potential,
        )

    @staticmethod
    def cusp_bulge_defaults(
        *,
        disk_n_pix: int = 128,
        include_potential: bool = False,
        moment_set: MomentSet = "disp",
        bulge_n_pix: int = 128,
        bulge_n_z: int = 32,
        bulge_z_max: float = 4.0,
        disk_n_z: int | None = None,
        halo_n_z: int | None = None,
    ) -> MultiScaleSliceConfig:
        """
        Disk/halo match the FFT-long progressive teacher; bulge is finer.

        Default bulge ``128² × 32`` over ``r,z ≤ 4`` kpc → Δx ≈ 0.0625 kpc,
        mean Δz ≈ 0.25 kpc (≈ 3–4 voxels across a typical ``a≈0.4`` kpc
        Hernquist scale), vs FFT-long ``84×12`` (Δx≈0.095, Δz≈0.67).

        Architecture-incompatible with FFT-long bulge weights — warm-start
        disk/halo towers only (``load_compatible_towers``) or train from scratch.
        """
        return MultiScaleSliceConfig.progressive_defaults(
            disk_n_pix=disk_n_pix,
            include_potential=include_potential,
            moment_set=moment_set,
            disk_n_z=disk_n_z,
            halo_n_z=halo_n_z,
            bulge_n_pix=bulge_n_pix,
            bulge_n_z=bulge_n_z,
            bulge_z_max=bulge_z_max,
            bulge_z_spacing="uniform",
        )

    def grid_for(self, name: str) -> ComponentSliceGrid:
        for g in self.grids:
            if g.name == name:
                return g
        raise KeyError(f"no slice grid for component {name!r}")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.grids)


@dataclass(frozen=True)
class ComponentVoxelGrid:
    """
    Anisotropic voxel grid for **one** component.

    Disk uses a flat box (``r_z ≪ r_xy``); bulge/halo use near-cubic boxes at
    different FOVs.
    """

    name: str
    n_xy: int = 32
    n_z: int = 24
    r_xy: float = 15.0
    r_z: float = 15.0
    moment_set: MomentSet = "disp"

    @property
    def moment_keys(self) -> tuple[str, ...]:
        return resolve_moment_keys(self.moment_set)

    @property
    def n_mom(self) -> int:
        return len(self.moment_keys)

    @property
    def n_moment_channels(self) -> int:
        return self.n_mom


@dataclass(frozen=True)
class MultiScaleVoxelConfig:
    """Per-component voxel grids with independent box size and resolution."""

    grids: tuple[ComponentVoxelGrid, ...]

    @staticmethod
    def smoke_defaults(*, moment_set: MomentSet = "disp") -> MultiScaleVoxelConfig:
        return MultiScaleVoxelConfig(
            grids=(
                ComponentVoxelGrid(
                    "bulge", n_xy=32, n_z=32, r_xy=4.0, r_z=4.0, moment_set=moment_set
                ),
                ComponentVoxelGrid(
                    "disk", n_xy=48, n_z=16, r_xy=12.0, r_z=1.5, moment_set=moment_set
                ),
                ComponentVoxelGrid(
                    "halo", n_xy=28, n_z=28, r_xy=40.0, r_z=40.0, moment_set=moment_set
                ),
            )
        )

    @staticmethod
    def cusp_bulge_defaults(*, moment_set: MomentSet = "disp") -> MultiScaleVoxelConfig:
        """Near-cubic fine voxels for the bulge cusp; disk/halo stay anisotropic."""
        return MultiScaleVoxelConfig(
            grids=(
                ComponentVoxelGrid(
                    "bulge", n_xy=64, n_z=64, r_xy=4.0, r_z=4.0, moment_set=moment_set
                ),
                ComponentVoxelGrid(
                    "disk", n_xy=48, n_z=16, r_xy=12.0, r_z=1.5, moment_set=moment_set
                ),
                ComponentVoxelGrid(
                    "halo", n_xy=28, n_z=28, r_xy=40.0, r_z=40.0, moment_set=moment_set
                ),
            )
        )

    def grid_for(self, name: str) -> ComponentVoxelGrid:
        for g in self.grids:
            if g.name == name:
                return g
        raise KeyError(f"no voxel grid for component {name!r}")

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.grids)


@dataclass(frozen=True)
class SliceMapConfig:
    """
    Shared-geometry vertical-slice stack (legacy / fused common canvas).

    Prefer :class:`MultiScaleSliceConfig` when components need different scales.
    """

    n_pix: int = 32
    n_z: int = 4
    r_max: float = 15.0
    z_max: float = 2.0
    components: tuple[str, ...] = DEFAULT_COMPONENTS
    include_potential: bool = False
    moment_set: MomentSet = "disp"

    @property
    def moment_keys(self) -> tuple[str, ...]:
        return resolve_moment_keys(self.moment_set)

    @property
    def n_mom(self) -> int:
        return len(self.moment_keys)


@dataclass(frozen=True)
class VoxelMapConfig:
    """Shared cubic voxel grid (legacy). Prefer :class:`MultiScaleVoxelConfig`."""

    n_bins: int = 24
    r_max: float = 15.0
    components: tuple[str, ...] = DEFAULT_COMPONENTS
    moment_set: MomentSet = "disp"

    @property
    def moment_keys(self) -> tuple[str, ...]:
        return resolve_moment_keys(self.moment_set)

    @property
    def n_mom(self) -> int:
        return len(self.moment_keys)


def z_edges_for_grid(grid: ComponentSliceGrid) -> np.ndarray:
    """Build ``z`` slab edges for a component grid (uniform or midplane-focused)."""
    n_z = int(grid.n_z)
    z_max = float(grid.z_max)
    if grid.z_spacing == "uniform":
        return np.linspace(-z_max, z_max, n_z + 1)
    # sinh clustering: finer Δz near the midplane (disk).
    a = 2.5
    u = np.linspace(-1.0, 1.0, n_z + 1)
    return z_max * np.sinh(a * u) / np.sinh(a)


def channel_names_for_component_slices(grid: ComponentSliceGrid) -> list[str]:
    names: list[str] = []
    for iz in range(grid.n_z):
        for m in grid.moment_keys:
            names.append(f"{grid.name}/z{iz}/{m}")
    return names


def channel_names_slices(cfg: SliceMapConfig) -> list[str]:
    """Human-readable channel names for a shared-geometry slice stack."""
    names: list[str] = []
    keys = cfg.moment_keys
    for comp in cfg.components:
        for iz in range(cfg.n_z):
            for m in keys:
                names.append(f"{comp}/z{iz}/{m}")
    return names


def channel_names_voxels(cfg: VoxelMapConfig) -> list[str]:
    """Human-readable channel names for a shared voxel stack."""
    names: list[str] = []
    keys = cfg.moment_keys
    for comp in cfg.components:
        for m in keys:
            names.append(f"{comp}/{m}")
    return names


def channel_names_for_component_voxels(grid: ComponentVoxelGrid) -> list[str]:
    return [f"{grid.name}/{m}" for m in grid.moment_keys]


def _component_mask(component_id: np.ndarray, name: str) -> np.ndarray:
    cid = COMPONENT_IDS[name]
    return np.asarray(component_id, dtype=np.int64) == int(cid)


def _xy_edges(n_pix: int, r_max: float) -> np.ndarray:
    return np.linspace(-float(r_max), float(r_max), int(n_pix) + 1)


def _z_edges(n_z: int, z_max: float) -> np.ndarray:
    return np.linspace(-float(z_max), float(z_max), int(n_z) + 1)


def _deposit_moments_2d(
    x: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    vel: np.ndarray,
    *,
    xy_edges: np.ndarray,
    moment_keys: tuple[str, ...],
    pixel_area: float,
) -> np.ndarray:
    """
    Deposit dens / ⟨v⟩ / σ / optional β into ``(n_mom, ny, nx)`` for one slab.

    ``histogram2d`` uses ``(x, y)`` → first axis ``x``, second ``y``.
    Empty pixels keep ``dens=0`` and moment channels ``0`` (finite tensors for
    AE training).  When reducing maps to radial ⟨v⟩ profiles, **dens-weight**
    and ignore ``dens==0`` pixels — equal-pixel means of zero-filled empties
    pull ⟨v_φ⟩→0 as ``N`` drops (outer disk).
    """
    n_mom = len(moment_keys)
    n_pix = len(xy_edges) - 1
    out = np.zeros((n_mom, n_pix, n_pix), dtype=np.float64)
    mass_hist, _, _ = np.histogram2d(x, y, bins=[xy_edges, xy_edges], weights=m)
    dens = mass_hist / max(pixel_area, 1e-30)

    means = np.zeros((3, n_pix, n_pix), dtype=np.float64)
    secs = np.zeros((3, n_pix, n_pix), dtype=np.float64)
    for axis in range(3):
        wh, _, _ = np.histogram2d(
            x, y, bins=[xy_edges, xy_edges], weights=m * vel[:, axis]
        )
        wh2, _, _ = np.histogram2d(
            x, y, bins=[xy_edges, xy_edges], weights=m * vel[:, axis] ** 2
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            means[axis] = np.where(mass_hist > 0, wh / mass_hist, 0.0)
            secs[axis] = np.where(mass_hist > 0, wh2 / mass_hist, 0.0)

    var = np.maximum(secs - means**2, 0.0)
    sigma = np.sqrt(var)

    key_to_arr = {
        "dens": dens,
        "vx": means[0],
        "vy": means[1],
        "vz": means[2],
        "sx": sigma[0],
        "sy": sigma[1],
        "sz": sigma[2],
    }
    if "beta" in moment_keys:
        # Anisotropy proxy: β ≈ 1 − 2 σ_z² / (σ_x² + σ_y²); 0 when planar σ→0.
        s2xy = var[0] + var[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(s2xy > 1e-30, 1.0 - 2.0 * var[2] / s2xy, 0.0)
        key_to_arr["beta"] = np.clip(beta, -2.0, 2.0)

    for i, key in enumerate(moment_keys):
        out[i] = key_to_arr[key]
    return out


def _bin_one_component_slices(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    mask_c: np.ndarray,
    grid: ComponentSliceGrid,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_pix = int(grid.n_pix)
    n_z = int(grid.n_z)
    keys = grid.moment_keys
    n_mom = len(keys)
    stack = np.zeros((n_z * n_mom, n_pix, n_pix), dtype=np.float64)
    xy_edges = _xy_edges(n_pix, grid.r_max)
    z_edges = z_edges_for_grid(grid)
    pixel_area = float((xy_edges[1] - xy_edges[0]) ** 2)

    if np.any(mask_c):
        for iz in range(n_z):
            zlo, zhi = float(z_edges[iz]), float(z_edges[iz + 1])
            if iz == n_z - 1:
                mask_z = mask_c & (pos[:, 2] >= zlo) & (pos[:, 2] <= zhi)
            else:
                mask_z = mask_c & (pos[:, 2] >= zlo) & (pos[:, 2] < zhi)
            if not np.any(mask_z):
                continue
            slab = _deposit_moments_2d(
                pos[mask_z, 0],
                pos[mask_z, 1],
                mass[mask_z],
                vel[mask_z],
                xy_edges=xy_edges,
                moment_keys=keys,
                pixel_area=pixel_area,
            )
            stack[iz * n_mom : (iz + 1) * n_mom] = slab

    meta = {
        "xy_edges": xy_edges.astype(np.float64),
        "z_edges": z_edges.astype(np.float64),
        "channel_names": np.asarray(channel_names_for_component_slices(grid), dtype=object),
        "component": grid.name,
        "r_max": float(grid.r_max),
        "z_max": float(grid.z_max),
        "n_pix": n_pix,
        "n_z": n_z,
        "z_spacing": grid.z_spacing,
        "moment_keys": np.asarray(keys, dtype=object),
        "moment_set": grid.moment_set,
    }
    return stack.astype(np.float32), meta


def bin_multiscale_slice_stacks(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: MultiScaleSliceConfig | None = None,
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Deposit dens + ⟨v⟩ + σ (and optional β) on **per-component** slice grids.

    Coordinates must already share one origin (see
    :func:`galacticsics.ml.fields.frame.prepare_shared_frame`).  This function
    never recenters per component.

    Returns
    -------
    maps : dict
        ``component → (stack, meta)`` where ``stack`` has shape
        ``(n_z · n_mom, n_pix, n_pix)`` in that component's native resolution.
    """
    cfg = cfg or MultiScaleSliceConfig.smoke_defaults(include_potential=False)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    cid = np.asarray(component_id, dtype=np.int64).reshape(-1)

    out: dict[str, tuple[np.ndarray, dict]] = {}
    for grid in cfg.grids:
        mask_c = _component_mask(cid, grid.name)
        out[grid.name] = _bin_one_component_slices(pos, vel, mass, mask_c, grid)
    return out


def fuse_slice_maps_to_common(
    maps: dict[str, tuple[np.ndarray, dict]],
    *,
    n_pix: int = 32,
    order: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Bilinear-resize each component stack to a shared ``n_pix`` and concatenate.

    Useful for a single shared CNN when multi-tower fusion is not needed.
    Physical FOV differences are **not** preserved in pixel space — only
    morphology at each component's native FOV after rescaling.
    """
    from scipy.ndimage import zoom

    comps = order or tuple(maps.keys())
    channels: list[np.ndarray] = []
    names: list[str] = []
    for comp in comps:
        stack, meta = maps[comp]
        h, w = stack.shape[-2], stack.shape[-1]
        if h != n_pix or w != n_pix:
            zoom_yx = (n_pix / h, n_pix / w)
            resized = np.stack(
                [zoom(stack[c], zoom_yx, order=1) for c in range(stack.shape[0])],
                axis=0,
            ).astype(np.float32)
        else:
            resized = stack.astype(np.float32)
        channels.append(resized)
        ch_names = list(meta.get("channel_names", []))
        if len(ch_names) != resized.shape[0]:
            ch_names = [f"{comp}/ch{i}" for i in range(resized.shape[0])]
        names.extend(ch_names)
    return np.concatenate(channels, axis=0), names


def bin_vertical_slice_stack(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: SliceMapConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Deposit per-component moments on a **shared** z-slice grid.

    Prefer :func:`bin_multiscale_slice_stacks` when FOVs should differ.
    """
    cfg = cfg or SliceMapConfig()
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    cid = np.asarray(component_id, dtype=np.int64).reshape(-1)

    n_pix = int(cfg.n_pix)
    n_z = int(cfg.n_z)
    n_comp = len(cfg.components)
    keys = cfg.moment_keys
    n_mom = len(keys)
    stack = np.zeros((n_comp * n_z * n_mom, n_pix, n_pix), dtype=np.float64)

    xy_edges = _xy_edges(n_pix, cfg.r_max)
    z_edges = _z_edges(n_z, cfg.z_max)
    pixel_area = float((xy_edges[1] - xy_edges[0]) ** 2)

    for ic, comp in enumerate(cfg.components):
        mask_c = _component_mask(cid, comp)
        if not np.any(mask_c):
            continue
        for iz in range(n_z):
            zlo, zhi = float(z_edges[iz]), float(z_edges[iz + 1])
            if iz == n_z - 1:
                mask_z = mask_c & (pos[:, 2] >= zlo) & (pos[:, 2] <= zhi)
            else:
                mask_z = mask_c & (pos[:, 2] >= zlo) & (pos[:, 2] < zhi)
            if not np.any(mask_z):
                continue
            slab = _deposit_moments_2d(
                pos[mask_z, 0],
                pos[mask_z, 1],
                mass[mask_z],
                vel[mask_z],
                xy_edges=xy_edges,
                moment_keys=keys,
                pixel_area=pixel_area,
            )
            base = (ic * n_z + iz) * n_mom
            stack[base : base + n_mom] = slab

    meta = {
        "xy_edges": xy_edges.astype(np.float64),
        "z_edges": z_edges.astype(np.float64),
        "channel_names": np.asarray(channel_names_slices(cfg), dtype=object),
        "moment_keys": np.asarray(keys, dtype=object),
        "moment_set": cfg.moment_set,
    }
    return stack.astype(np.float32), meta


def _deposit_moments_3d(
    flat: np.ndarray,
    m: np.ndarray,
    vel: np.ndarray,
    *,
    n_z: int,
    n_xy: int,
    cell_vol: float,
    moment_keys: tuple[str, ...],
) -> np.ndarray:
    """Deposit dens / ⟨v⟩ / σ / optional β into ``(n_mom, n_z, n_xy, n_xy)``."""
    n_cells = n_z * n_xy * n_xy
    mass_dep = np.bincount(flat, weights=m, minlength=n_cells).reshape(n_z, n_xy, n_xy)
    dens = mass_dep / max(cell_vol, 1e-30)
    means = np.zeros((3, n_z, n_xy, n_xy), dtype=np.float64)
    secs = np.zeros((3, n_z, n_xy, n_xy), dtype=np.float64)
    for k in range(3):
        wh = np.bincount(
            flat, weights=m * vel[:, k], minlength=n_cells
        ).reshape(n_z, n_xy, n_xy)
        wh2 = np.bincount(
            flat, weights=m * vel[:, k] ** 2, minlength=n_cells
        ).reshape(n_z, n_xy, n_xy)
        with np.errstate(divide="ignore", invalid="ignore"):
            means[k] = np.where(mass_dep > 0, wh / mass_dep, 0.0)
            secs[k] = np.where(mass_dep > 0, wh2 / mass_dep, 0.0)
    var = np.maximum(secs - means**2, 0.0)
    sigma = np.sqrt(var)
    key_to_arr = {
        "dens": dens,
        "vx": means[0],
        "vy": means[1],
        "vz": means[2],
        "sx": sigma[0],
        "sy": sigma[1],
        "sz": sigma[2],
    }
    if "beta" in moment_keys:
        s2xy = var[0] + var[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(s2xy > 1e-30, 1.0 - 2.0 * var[2] / s2xy, 0.0)
        key_to_arr["beta"] = np.clip(beta, -2.0, 2.0)
    out = np.zeros((len(moment_keys), n_z, n_xy, n_xy), dtype=np.float64)
    for i, key in enumerate(moment_keys):
        out[i] = key_to_arr[key]
    return out


def _bin_one_component_voxels(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    mask_c: np.ndarray,
    grid: ComponentVoxelGrid,
) -> tuple[np.ndarray, dict]:
    n_xy = int(grid.n_xy)
    n_z = int(grid.n_z)
    keys = grid.moment_keys
    n_mom = len(keys)
    stack = np.zeros((n_mom, n_z, n_xy, n_xy), dtype=np.float64)
    edges_xy = np.linspace(-float(grid.r_xy), float(grid.r_xy), n_xy + 1)
    edges_z = np.linspace(-float(grid.r_z), float(grid.r_z), n_z + 1)
    cell_vol = float(
        (edges_xy[1] - edges_xy[0]) ** 2 * (edges_z[1] - edges_z[0])
    )

    if np.any(mask_c):
        ix = np.clip(np.searchsorted(edges_xy, pos[:, 0], side="right") - 1, -1, n_xy)
        iy = np.clip(np.searchsorted(edges_xy, pos[:, 1], side="right") - 1, -1, n_xy)
        iz = np.clip(np.searchsorted(edges_z, pos[:, 2], side="right") - 1, -1, n_z)
        inside = (
            mask_c
            & (ix >= 0)
            & (ix < n_xy)
            & (iy >= 0)
            & (iy < n_xy)
            & (iz >= 0)
            & (iz < n_z)
        )
        if np.any(inside):
            flat = iz[inside] * (n_xy * n_xy) + ix[inside] * n_xy + iy[inside]
            stack = _deposit_moments_3d(
                flat,
                mass[inside],
                vel[inside],
                n_z=n_z,
                n_xy=n_xy,
                cell_vol=cell_vol,
                moment_keys=keys,
            )

    meta = {
        "edges_xy": edges_xy.astype(np.float64),
        "edges_z": edges_z.astype(np.float64),
        "channel_names": np.asarray(channel_names_for_component_voxels(grid), dtype=object),
        "component": grid.name,
        "r_xy": float(grid.r_xy),
        "r_z": float(grid.r_z),
        "n_xy": n_xy,
        "n_z": n_z,
        "moment_keys": np.asarray(keys, dtype=object),
        "moment_set": grid.moment_set,
    }
    return stack.astype(np.float32), meta


def bin_multiscale_voxel_stacks(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: MultiScaleVoxelConfig | None = None,
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Deposit dens + ⟨v⟩ + σ on **per-component** anisotropic voxel grids.

    Returns
    -------
    maps : dict
        ``component → (stack, meta)`` with ``stack`` shape
        ``(n_mom, n_z, n_xy, n_xy)``.
    """
    cfg = cfg or MultiScaleVoxelConfig.smoke_defaults()
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    cid = np.asarray(component_id, dtype=np.int64).reshape(-1)

    out: dict[str, tuple[np.ndarray, dict]] = {}
    for grid in cfg.grids:
        mask_c = _component_mask(cid, grid.name)
        out[grid.name] = _bin_one_component_voxels(pos, vel, mass, mask_c, grid)
    return out


def bin_voxel_stack(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    component_id: np.ndarray,
    *,
    cfg: VoxelMapConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Deposit per-component moments on a shared cubic voxel grid.
    """
    cfg = cfg or VoxelMapConfig()
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    cid = np.asarray(component_id, dtype=np.int64).reshape(-1)

    n = int(cfg.n_bins)
    n_comp = len(cfg.components)
    keys = cfg.moment_keys
    n_mom = len(keys)
    stack = np.zeros((n_comp * n_mom, n, n, n), dtype=np.float64)
    edges = np.linspace(-float(cfg.r_max), float(cfg.r_max), n + 1)
    cell_vol = float((edges[1] - edges[0]) ** 3)

    ix = np.clip(np.searchsorted(edges, pos[:, 0], side="right") - 1, -1, n)
    iy = np.clip(np.searchsorted(edges, pos[:, 1], side="right") - 1, -1, n)
    iz = np.clip(np.searchsorted(edges, pos[:, 2], side="right") - 1, -1, n)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n)

    for ic, comp in enumerate(cfg.components):
        mask = inside & _component_mask(cid, comp)
        if not np.any(mask):
            continue
        # Legacy spatial order (ix, iy, iz); do not reuse anisotropic (iz, ix, iy).
        flat = ix[mask] * (n * n) + iy[mask] * n + iz[mask]
        m = mass[mask]
        v = vel[mask]
        mass_dep = np.bincount(flat, weights=m, minlength=n**3).reshape(n, n, n)
        dens = mass_dep / max(cell_vol, 1e-30)
        means = np.zeros((3, n, n, n), dtype=np.float64)
        secs = np.zeros((3, n, n, n), dtype=np.float64)
        for k in range(3):
            wh = np.bincount(flat, weights=m * v[:, k], minlength=n**3).reshape(n, n, n)
            wh2 = np.bincount(
                flat, weights=m * v[:, k] ** 2, minlength=n**3
            ).reshape(n, n, n)
            with np.errstate(divide="ignore", invalid="ignore"):
                means[k] = np.where(mass_dep > 0, wh / mass_dep, 0.0)
                secs[k] = np.where(mass_dep > 0, wh2 / mass_dep, 0.0)
        var = np.maximum(secs - means**2, 0.0)
        sigma = np.sqrt(var)
        key_to_arr: dict[str, np.ndarray] = {
            "dens": dens,
            "vx": means[0],
            "vy": means[1],
            "vz": means[2],
            "sx": sigma[0],
            "sy": sigma[1],
            "sz": sigma[2],
        }
        if "beta" in keys:
            s2xy = var[0] + var[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = np.where(s2xy > 1e-30, 1.0 - 2.0 * var[2] / s2xy, 0.0)
            key_to_arr["beta"] = np.clip(beta, -2.0, 2.0)
        base = ic * n_mom
        for k, key in enumerate(keys):
            stack[base + k] = key_to_arr[key]

    meta = {
        "edges": edges.astype(np.float64),
        "channel_names": np.asarray(channel_names_voxels(cfg), dtype=object),
        "moment_keys": np.asarray(keys, dtype=object),
        "moment_set": cfg.moment_set,
    }
    return stack.astype(np.float32), meta


def dens_channel_indices(cfg: SliceMapConfig) -> list[int]:
    """Indices of density channels in a shared slice stack."""
    n_mom = cfg.n_mom
    n_z = int(cfg.n_z)
    idx: list[int] = []
    for ic in range(len(cfg.components)):
        for iz in range(n_z):
            idx.append((ic * n_z + iz) * n_mom)
    return idx


def dens_channel_indices_component(grid: ComponentSliceGrid) -> list[int]:
    """Density channel indices within one component's native slice stack."""
    n_mom = grid.n_mom
    return [iz * n_mom for iz in range(int(grid.n_z))]


def moment_channel_indices_component(
    grid: ComponentSliceGrid, key: str
) -> list[int]:
    """Indices of a named moment (e.g. ``sx``) across z-slabs."""
    keys = grid.moment_keys
    if key not in keys:
        raise KeyError(f"{key!r} not in moment_keys={keys}")
    off = keys.index(key)
    n_mom = len(keys)
    return [iz * n_mom + off for iz in range(int(grid.n_z))]


def component_label_for_channel(cfg: SliceMapConfig, channel: int) -> int:
    """ML component id for a shared slice-stack channel index."""
    n_mom = cfg.n_mom
    n_z = int(cfg.n_z)
    block = n_z * n_mom
    ic = int(channel) // block
    name = cfg.components[ic]
    return int(COMPONENT_IDS[name])


def scale_summary(cfg: MultiScaleSliceConfig | MultiScaleVoxelConfig) -> list[dict]:
    """Serializable summary of per-component FOV / resolution choices."""
    rows: list[dict] = []
    if isinstance(cfg, MultiScaleSliceConfig):
        for g in cfg.grids:
            rows.append(
                {
                    "component": g.name,
                    "kind": "slice",
                    "n_pix": g.n_pix,
                    "n_z": g.n_z,
                    "r_max_kpc": g.r_max,
                    "z_max_kpc": g.z_max,
                    "z_spacing": g.z_spacing,
                    "dx_kpc": 2.0 * g.r_max / g.n_pix,
                    "mean_dz_kpc": 2.0 * g.z_max / g.n_z,
                    "moment_set": g.moment_set,
                    "n_mom": g.n_mom,
                    "moment_keys": list(g.moment_keys),
                }
            )
    else:
        for g in cfg.grids:
            rows.append(
                {
                    "component": g.name,
                    "kind": "voxel",
                    "n_xy": g.n_xy,
                    "n_z": g.n_z,
                    "r_xy_kpc": g.r_xy,
                    "r_z_kpc": g.r_z,
                    "dxy_kpc": 2.0 * g.r_xy / g.n_xy,
                    "dz_kpc": 2.0 * g.r_z / g.n_z,
                    "moment_set": g.moment_set,
                    "n_mom": g.n_mom,
                    "moment_keys": list(g.moment_keys),
                }
            )
    return rows


def radial_vphi_from_deposit_slab(
    dens: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    r_max: float,
    n_bins: int = 28,
    r_max_prof: float | None = None,
    min_mass: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dens-weighted radial ``⟨v_φ⟩(R)`` from one deposited dens/vx/vy slab.

    Coordinates follow ``histogram2d`` layout: ``arr[i, j]`` ↔ ``(x_i, y_j)``.
    Empty pixels (``dens <= min_mass``) are excluded — never averaged as ``v=0``.

    Returns
    -------
    r_mid, mean_vphi, mass_per_bin
    """
    dens = np.asarray(dens, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    n_pix = dens.shape[0]
    edges_xy = np.linspace(-float(r_max), float(r_max), n_pix + 1)
    xc = 0.5 * (edges_xy[:-1] + edges_xy[1:])
    # dens[i, j] at (x=xc[i], y=xc[j])
    X = xc[:, None]
    Y = xc[None, :]
    R = np.sqrt(X * X + Y * Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        vphi = np.where(R > 1e-8, (-Y * vx + X * vy) / R, np.nan)
    r_outer = float(r_max if r_max_prof is None else r_max_prof)
    edges_r = np.linspace(0.0, r_outer, int(n_bins) + 1)
    r_mid = 0.5 * (edges_r[:-1] + edges_r[1:])
    mean = np.full(int(n_bins), np.nan, dtype=np.float64)
    mass_bin = np.zeros(int(n_bins), dtype=np.float64)
    occupied = dens > float(min_mass)
    for i in range(int(n_bins)):
        mask = occupied & (R >= edges_r[i]) & (R < edges_r[i + 1])
        if not np.any(mask):
            continue
        w = dens[mask]
        mass_bin[i] = float(w.sum())
        mean[i] = float(np.average(vphi[mask], weights=w))
    return r_mid, mean, mass_bin


__all__ = [
    "MOMENT_KEYS",
    "MOMENT_KEYS_BASE",
    "MOMENT_KEYS_DISP",
    "MOMENT_KEYS_FULL",
    "MOMENT_KINDS",
    "DEFAULT_COMPONENTS",
    "resolve_moment_keys",
    "moment_kind",
    "ComponentSliceGrid",
    "ComponentVoxelGrid",
    "MultiScaleSliceConfig",
    "MultiScaleVoxelConfig",
    "SliceMapConfig",
    "VoxelMapConfig",
    "z_edges_for_grid",
    "channel_names_slices",
    "channel_names_voxels",
    "channel_names_for_component_slices",
    "channel_names_for_component_voxels",
    "bin_multiscale_slice_stacks",
    "bin_multiscale_voxel_stacks",
    "fuse_slice_maps_to_common",
    "bin_vertical_slice_stack",
    "bin_voxel_stack",
    "dens_channel_indices",
    "dens_channel_indices_component",
    "moment_channel_indices_component",
    "component_label_for_channel",
    "scale_summary",
    "radial_vphi_from_deposit_slab",
    "ID_TO_COMPONENT",
]
