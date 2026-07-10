"""Route solve / diskdf / sampling to the selected physics backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from galacticsics.models import GalaxyModel
from galacticsics.physics.backend import PhysicsBackendKind, resolve_physics_backend

def backend_solve_potential(
    model: GalaxyModel,
    work_dir: Path,
    *,
    backend: PhysicsBackendKind | str | None = None,
    npsi: int = 1000,
    nint: int = 20,
    max_iter: int = 100,
    n_workers: int | None = None,
    timeout: float | None = 3600.0,
    stream_output: bool = False,
) -> None:
    kind = resolve_physics_backend(backend)
    if kind == PhysicsBackendKind.LEGACY:
        from galacticsics.physics import legacy_backend

        legacy_backend.legacy_solve_potential(
            model,
            work_dir,
            npsi=npsi,
            nint=nint,
            timeout=timeout,
            stream_output=stream_output,
        )
    else:
        from galacticsics.physics import python_backend

        python_backend.python_solve_potential(
            model,
            work_dir,
            npsi=npsi,
            nint=nint,
            max_iter=max_iter,
            n_workers=n_workers,
            timeout=timeout,
            stream_output=stream_output,
        )


def backend_ensure_disk_df(
    model: GalaxyModel,
    work_dir: Path,
    *,
    backend: PhysicsBackendKind | str | None = None,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
    diskdf_backend: str = "python",
) -> GalaxyModel:
    kind = resolve_physics_backend(backend)
    if kind == PhysicsBackendKind.LEGACY:
        from galacticsics.physics import legacy_backend

        legacy_backend.legacy_ensure_disk_df(model, work_dir, stream_output=stream_output)
        return model
    from galacticsics.physics import python_backend

    return python_backend.python_ensure_disk_df(
        model,
        work_dir,
        stream_output=stream_output,
        progress_log=progress_log,
        diskdf_backend=diskdf_backend,
    )


def backend_sample_disk(
    work_dir: Path,
    config,
    *,
    backend: PhysicsBackendKind | str | None = None,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    kind = resolve_physics_backend(backend)
    if kind == PhysicsBackendKind.LEGACY:
        from galacticsics.physics import legacy_backend

        return legacy_backend.legacy_sample_disk(work_dir, config, stream_output=stream_output)
    from galacticsics.physics import python_backend

    return python_backend.python_sample_disk(
        work_dir, config, stream_output=stream_output, progress_log=progress_log
    )


def backend_sample_halo(
    work_dir: Path,
    config,
    *,
    backend: PhysicsBackendKind | str | None = None,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    kind = resolve_physics_backend(backend)
    if kind == PhysicsBackendKind.LEGACY:
        from galacticsics.physics import legacy_backend

        return legacy_backend.legacy_sample_halo(work_dir, config, stream_output=stream_output)
    from galacticsics.physics import python_backend

    return python_backend.python_sample_halo(
        work_dir, config, stream_output=stream_output, progress_log=progress_log
    )


def backend_sample_bulge(
    work_dir: Path,
    config,
    *,
    backend: PhysicsBackendKind | str | None = None,
    stream_output: bool = False,
    progress_log: Callable[[str], None] | None = None,
):
    kind = resolve_physics_backend(backend)
    if kind == PhysicsBackendKind.LEGACY:
        from galacticsics.physics import legacy_backend

        return legacy_backend.legacy_sample_bulge(work_dir, config, stream_output=stream_output)
    from galacticsics.physics import python_backend

    return python_backend.python_sample_bulge(
        work_dir, config, stream_output=stream_output, progress_log=progress_log
    )
