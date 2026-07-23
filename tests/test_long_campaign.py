"""Optional long tests: force benchmarks and evolved density-profile sanity.

Run with::

    pytest tests/test_long_campaign.py -m slow -v

Set ``GALACTICSICS_CAMPAIGN_ARTIFACTS`` to override the notebook artifact root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from galacticsics.campaign.benchmarks import (
    DEFAULT_BENCHMARK_N,
    assert_density_sanity,
    check_density_evolution,
    diagnose_ic_stability,
    find_evolved_campaign_dirs,
    format_force_benchmark,
    ic_looks_stable,
    load_ic_state,
    run_force_benchmark,
    subsample_state,
)
from ntropy.forces.bhtree_c import extension_available

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "notebooks" / "artifacts" / "campaign_walkthrough" / "base_mw"


def _artifact_root() -> Path:
    return Path(os.environ.get("GALACTICSICS_CAMPAIGN_ARTIFACTS", DEFAULT_ARTIFACTS))


@pytest.fixture(scope="module")
def ic_work_dir() -> Path:
    root = _artifact_root()
    if not root.is_dir():
        pytest.skip(f"campaign artifacts missing: {root}")
    for child in sorted(root.iterdir()):
        if (child / "ic_state.npz").is_file():
            return child
    pytest.skip(f"no ic_state.npz under {root}")


@pytest.mark.skipif(not extension_available(), reason="bh_c not built")
def test_bh_c_force_benchmark(ic_work_dir: Path):
    """bh_c optimized preset should beat legacy; active subset should help."""
    state = subsample_state(
        load_ic_state(ic_work_dir / "ic_state.npz"), n=DEFAULT_BENCHMARK_N, seed=1
    )
    result = run_force_benchmark(state, n_repeat=3)

    legacy = result.ms("legacy_full_rebuild1")
    optimized = result.ms("optimized_full_rebuild1")
    active = result.ms("optimized_active_rebuild10")

    assert optimized < legacy * 0.95, format_force_benchmark(result)
    if result.n_active_step1 and result.n_active_step1 < state.n * 0.9:
        assert active < optimized * 0.95, format_force_benchmark(result)


def test_unstable_ic_diagnosis_flags_hot_disk():
    """5bee run had disk v_max~13 (outliers) — should fail ic_looks_stable."""
    root = _artifact_root()
    bad = root / "5bee258b9b75" / "ic_state.npz"
    if not bad.is_file():
        pytest.skip("5bee artifact not present")
    ic = load_ic_state(bad)
    diag = diagnose_ic_stability(ic)
    assert diag["disk_v_max"] > 10.0
    assert not ic_looks_stable(ic)


def test_stable_ic_diagnosis():
    """Good IC (d9e560) should pass velocity sanity."""
    root = _artifact_root()
    good = root / "d9e560b95174" / "ic_state.npz"
    if not good.is_file():
        pytest.skip("d9e560 artifact not present")
    ic = load_ic_state(good)
    assert ic_looks_stable(ic)


def pytest_generate_tests(metafunc):
    if metafunc.definition.name == "test_density_profiles_evolve_sensibly":
        dirs = find_evolved_campaign_dirs(_artifact_root())[:2]
        if not dirs:
            metafunc.parametrize("work_dir", [])
        else:
            metafunc.parametrize("work_dir", dirs, ids=[d.name for d in dirs])


def test_density_profiles_evolve_sensibly(work_dir: Path):
    """IC vs final profiles stay physically plausible for stable campaign runs."""
    if not work_dir:
        pytest.skip("no stable evolved campaign work dirs")
    result = check_density_evolution(work_dir)
    assert_density_sanity(result)
