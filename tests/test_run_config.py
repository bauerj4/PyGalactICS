"""Tests for walkthrough JSON config loading."""

from __future__ import annotations

from pathlib import Path

from galacticsics.campaign.run_config import (
    WalkthroughConfig,
    build_type_registry,
    default_walkthrough_config,
    load_walkthrough_config,
    particles_by_component,
    sample_config_kwargs,
    sample_openmp_summary,
    softening_eps_by_component,
)
from galacticsics.campaign.spec import expand_grid


def test_default_softening_is_10x_legacy():
    eps = softening_eps_by_component(default_walkthrough_config())
    assert eps["disk"] == 0.1
    assert eps["halo"] == 0.5
    assert eps["bulge"] == 0.2


def test_build_type_registry_uses_config_eps():
    reg = build_type_registry(default_walkthrough_config())
    assert reg.eps_for("disk") == 0.1
    assert reg.eps_for("halo") == 0.5


def test_arbitrary_component_softening_and_types():
    raw = {
        "particles": {"disk": 1000, "gas": 500},
        "softening": {"disk": 0.12, "gas": 0.3},
    }
    eps = softening_eps_by_component(raw)
    assert eps["disk"] == 0.12
    assert eps["gas"] == 0.3
    assert eps["halo"] == 0.5  # default galaxy component retained

    reg = build_type_registry(raw)
    assert reg.eps_for("gas") == 0.3
    assert "gas" in reg.types
    assert reg.id_for("gas") == 4


def test_legacy_particle_and_softening_keys():
    raw = {
        "particles": {"n_disk": 50, "n_halo": 80},
        "softening": {"disk_kpc": 0.15, "halo_kpc": 0.55},
    }
    assert particles_by_component(raw) == {"disk": 50, "halo": 80, "bulge": 0}
    assert softening_eps_by_component(raw)["disk"] == 0.15


def test_model_patch_defaults_mw():
    from galacticsics.campaign.spec import expand_grid, model_patch_defaults

    patch = model_patch_defaults("milky_way_disk_halo")
    assert patch["disk.mass"] == 17.0
    assert patch["halo.v0"] == 3.7

    spec = load_walkthrough_config(
        Path(__file__).resolve().parents[1] / "notebooks" / "campaigns" / "mw_walkthrough.json"
    ).base_grid
    spec.patch = {**patch, "disk.mass": 15.0}
    _, model = expand_grid(spec)[0]
    assert model.disk.mass == 15.0


def test_sample_openmp_summary_disabled():
    line = sample_openmp_summary({"sample": {"use_openmp": False}})
    assert "Python fallback" in line
    assert "use_openmp=false" in line


def test_openmp_sampler_status_respects_config():
    from galacticsics.sampling.openmp import openmp_sampler_status
    from galacticsics.sampling.sampler import SampleConfig

    off = openmp_sampler_status(SampleConfig(use_openmp=False))
    assert not off.is_openmp
    assert "use_openmp=false" in off.reason


def test_python_disk_sampler_warns_without_openmp():
    from galacticsics.sampling.openmp import warn_python_sampler_fallback

    logs: list[str] = []
    warn_python_sampler_fallback("gendisk", "sample.use_openmp=false", progress_log=logs.append)
    assert logs and "WARNING: gendisk using slow Python sampler" in logs[0]


def test_sample_openmp_summary_default_config():
    raw = default_walkthrough_config()
    line = sample_openmp_summary(raw)
    assert line.startswith("Sample:")


def test_summary_lines_includes_openmp(tmp_path):
    cfg = WalkthroughConfig.from_raw(
        default_walkthrough_config(),
        config_path=tmp_path / "mw_walkthrough.json",
    )
    lines = cfg.summary_lines()
    assert any("Sample:" in line for line in lines)


def test_load_mw_walkthrough_config():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "campaigns" / "mw_walkthrough.json"
    cfg = load_walkthrough_config(path)
    sk = sample_config_kwargs(cfg.raw)
    assert sk["use_openmp"] is True
    assert sk["n_openmp_threads"] == 0
    kw = cfg.run_campaign_kwargs()
    assert kw["particles_by_component"]["disk"] == 100_000
    assert kw["n_disk"] == 100_000
    assert kw["eps_by_component"]["disk"] == 0.1
    assert kw["force_active_subset"] is True
    label, model = cfg.base_model()
    assert model.disk_kinematics.toomre_q_target == 1.5
    _, expanded = expand_grid(cfg.base_grid)[0]
    assert expanded.disk_kinematics.toomre_q_target == 1.5
