"""Tests for JSON configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ntropy.config import load_config


def test_load_minimal_config(tmp_path):
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "particles": {"file": "parts.dat"},
                "integrator": {"dt": 0.01, "n_steps": 10},
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.seed == 7
    assert cfg.integrator.n_steps == 10
    assert cfg.force.method == "bh"


def test_invalid_force_method(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(
        json.dumps({"particles": {"file": "p.dat"}, "force": {"method": "fft"}})
    )
    with pytest.raises(ValueError, match="force.method"):
        load_config(cfg_path)


def test_missing_particles_file(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"seed": 1}))
    with pytest.raises(ValueError, match="particles"):
        load_config(cfg_path)
