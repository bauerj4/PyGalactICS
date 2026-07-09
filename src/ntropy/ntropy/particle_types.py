"""Particle type registry for multi-component simulations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParticleTypeSpec:
    """
    Definition of one particle type (component).

    Types carry default softening and **allowed timestep bin ranges**.
    Actual bins are assigned dynamically from local acceleration (GADGET-style);
    types do not fix the timestep, only bound it.

    Attributes
    ----------
    id : int
        Positive integer type identifier stored on each particle.
    label : str
        Human-readable name (e.g. ``'disk'``).
    eps : float
        Default gravitational softening length [kpc] for this type.
    min_timestep_bin : int
        Finest allowed bin (``0`` = base timestep ``dt_base``).
    max_timestep_bin : int or None
        Coarsest allowed bin; ``None`` uses the global integrator maximum.
    """

    id: int
    label: str
    eps: float = 0.01
    min_timestep_bin: int = 0
    max_timestep_bin: int | None = None

    def __post_init__(self) -> None:
        if self.id < 1:
            raise ValueError(f"particle type id must be >= 1, got {self.id}")
        if self.min_timestep_bin < 0:
            raise ValueError(
                f"min_timestep_bin must be >= 0, got {self.min_timestep_bin}"
            )
        if self.max_timestep_bin is not None and self.max_timestep_bin < self.min_timestep_bin:
            raise ValueError(
                f"max_timestep_bin ({self.max_timestep_bin}) < "
                f"min_timestep_bin ({self.min_timestep_bin})"
            )


@dataclass
class TypeRegistry:
    """
    Bidirectional map between integer type ids and string labels.

    Extensible to an arbitrary number of types.  Used for softening defaults
    and per-type timestep bin limits in the tiered integrator.
    """

    types: dict[str, ParticleTypeSpec] = field(default_factory=dict)

    @classmethod
    def default_galaxy(cls) -> TypeRegistry:
        """
        Standard disc / bulge / halo types for Milky-Way-like runs.

        Returns
        -------
        TypeRegistry
            Halo may use coarser bins (``max_timestep_bin=5``), disk is capped
            at finer bins (``max_timestep_bin=3``) but actual bins are set
            dynamically from :math:`|a|` and :math:`\\varepsilon`.
        """
        return cls.from_specs(
            [
                ParticleTypeSpec(
                    id=1, label="halo", eps=0.05, min_timestep_bin=0, max_timestep_bin=5
                ),
                ParticleTypeSpec(
                    id=2, label="bulge", eps=0.02, min_timestep_bin=0, max_timestep_bin=4
                ),
                ParticleTypeSpec(
                    id=3, label="disk", eps=0.01, min_timestep_bin=0, max_timestep_bin=3
                ),
            ]
        )

    @classmethod
    def from_specs(cls, specs: list[ParticleTypeSpec]) -> TypeRegistry:
        """
        Build a registry from an explicit list of type specs.

        Parameters
        ----------
        specs : list of ParticleTypeSpec
            Type definitions to register.

        Returns
        -------
        TypeRegistry
            Populated registry.
        """
        reg = cls()
        for spec in specs:
            reg.register(spec)
        return reg

    @classmethod
    def from_config_dict(cls, raw: dict) -> TypeRegistry:
        """
        Parse ``particle_types`` section from a JSON run config.

        Parameters
        ----------
        raw : dict
            Mapping ``label -> {id, eps, min_timestep_bin?, max_timestep_bin?}``.

        Returns
        -------
        TypeRegistry
            Parsed registry.

        Raises
        ------
        ValueError
            If an entry is not a dict or ids/labels are duplicated.
        """
        specs: list[ParticleTypeSpec] = []
        for label, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(f"particle_types.{label} must be an object")
            max_bin = entry.get("max_timestep_bin")
            specs.append(
                ParticleTypeSpec(
                    id=int(entry["id"]),
                    label=label,
                    eps=float(entry.get("eps", 0.01)),
                    min_timestep_bin=int(entry.get("min_timestep_bin", 0)),
                    max_timestep_bin=int(max_bin) if max_bin is not None else None,
                )
            )
        return cls.from_specs(specs)

    def register(self, spec: ParticleTypeSpec) -> None:
        """
        Add a type to the registry.

        Parameters
        ----------
        spec : ParticleTypeSpec
            Type to register.

        Raises
        ------
        ValueError
            On duplicate label or id.
        """
        if spec.label in self.types:
            raise ValueError(f"duplicate type label {spec.label!r}")
        for existing in self.types.values():
            if existing.id == spec.id:
                raise ValueError(f"duplicate type id {spec.id}")
        self.types[spec.label] = spec

    def id_for(self, label: str) -> int:
        """Return integer id for ``label``."""
        if label not in self.types:
            raise KeyError(f"unknown particle type label {label!r}")
        return self.types[label].id

    def label_for(self, type_id: int) -> str:
        """Return label for integer ``type_id``."""
        for label, spec in self.types.items():
            if spec.id == type_id:
                return label
        raise KeyError(f"unknown particle type id {type_id}")

    def eps_for(self, label: str) -> float:
        """Default softening [kpc] for ``label``."""
        return self.types[label].eps

    def to_config_dict(self) -> dict[str, dict]:
        """
        Serialize to JSON-compatible dict for run configs.

        Returns
        -------
        dict
            ``{label: {id, eps, min_timestep_bin, max_timestep_bin?}}``.
        """
        out: dict[str, dict] = {}
        for label, spec in self.types.items():
            entry: dict = {
                "id": spec.id,
                "eps": spec.eps,
                "min_timestep_bin": spec.min_timestep_bin,
            }
            if spec.max_timestep_bin is not None:
                entry["max_timestep_bin"] = spec.max_timestep_bin
            out[label] = entry
        return out
