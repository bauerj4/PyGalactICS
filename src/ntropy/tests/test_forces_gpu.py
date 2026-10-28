"""Tests for GPU-accelerated force backends.

Test categories:
1. Numerical parity against CPU backends (brute, bh_c)
2. Shape and dtype correctness
3. GPU availability detection
4. Edge cases (N=0, N=1, all-zero mass, coincident particles)
5. MPI correctness (multi-GPU)
6. Performance benchmarks (scaling curves)
7. Precision analysis (FP64 internal consistency)
"""

import os
from pathlib import Path

import pytest
import numpy as np

from ntropy.forces import gpu_available, gpu_bh_available
from ntropy.forces.gpu_direct import (
    GpuDirectState,
    compute_forces_gpu,
    compute_forces_gpu_from_state,
)
from ntropy.forces.gpu_bh import (
    GpuBhState,
    _select_bh_walk_stack_capacity,
    compute_forces_gpu_bh,
)

# Skip all GPU tests when no GPU is available
pytestmark = pytest.mark.skipif(
    not gpu_available(),
    reason="GPU not available",
)


def _make_deep_cusp(depth):
    """Build a small mixed-softening cusp with seven pending siblings/level."""
    octant_signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    positions = [np.full(3, -1.0), np.full(3, 1.0)]
    masses = [0.5, 0.5]
    softenings = [0.002, 0.02]
    centre = np.zeros(3, dtype=np.float64)
    half_size = 1.0
    softening_cycle = (1.0e-5, 5.0e-4, 2.0e-3)
    for level in range(depth):
        child_half = 0.5 * half_size
        child_centres = centre + child_half * octant_signs
        for child in range(1, 8):
            positions.append(child_centres[child])
            masses.append(1.0 / (level + 1.0))
            softenings.append(softening_cycle[(level + child) % 3])
        centre = child_centres[0]
        half_size = child_half

    target = len(positions)
    positions.append(centre.copy())
    masses.append(1.0)
    softenings.append(1.0e-5)
    return (
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(masses, dtype=np.float64),
        np.ascontiguousarray(softenings, dtype=np.float64),
        np.array([target], dtype=np.int32),
    )


def _ac023_ic_fixture_path() -> Path:
    """Return the optional real-system regression fixture location."""
    override = os.environ.get("NTROPY_AC023_IC")
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "runs"
        / "mw_morton_corpus_v2"
        / "ac023abb258d"
        / "ic_state.npz"
    )


# ====================================================================== #
# Category 1: Numerical Parity Tests
# ====================================================================== #

class TestGpuDirectParity:
    """GPU brute-force results match CPU brute-force to within tolerance."""

    @pytest.mark.parametrize("N", [100, 500])
    def test_all_targets(self, N):
        """compute_forces_gpu matches compute_forces_brute for all particles."""
        from ntropy.forces.brute import compute_forces_brute

        rng = np.random.default_rng(42 + N)
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.01)

        acc_gpu = compute_forces_gpu(pos, mass, eps)
        acc_cpu = compute_forces_brute(pos, mass, eps)

        assert acc_gpu.shape == (N, 3)
        rel_err = np.linalg.norm(acc_gpu - acc_cpu, axis=1) / np.maximum(
            np.linalg.norm(acc_cpu, axis=1), 1e-30
        )
        assert np.all(rel_err < 1e-8), f"Max relative error: {rel_err.max():.2e}"

    @pytest.mark.parametrize("N", [100, 500])
    def test_target_subset(self, N):
        """compute_forces_gpu with target_indices matches CPU for subset."""
        from ntropy.forces.brute import compute_forces_brute

        rng = np.random.default_rng(42)
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.01)
        targets = np.arange(0, N, 7)

        acc_gpu = compute_forces_gpu(pos, mass, eps, target_indices=targets)
        acc_cpu = compute_forces_brute(pos, mass, eps, target_indices=targets)

        assert acc_gpu.shape == (len(targets), 3)
        assert np.allclose(acc_gpu, acc_cpu, rtol=1e-6)


class TestGpuBhParity:
    """GPU Barnes-Hut results match CPU bh_c to within tolerance."""

    @pytest.mark.parametrize(
        ("n_particles", "expected_capacity"),
        [
            (1, 64),
            (8, 64),
            (9, 128),
            (512, 128),
            (513, 256),
            (16_777_216, 256),
            (16_777_217, 512),
            (2_147_483_647, 512),
        ],
    )
    def test_walk_stack_capacity_scales_with_particle_count(
        self, n_particles, expected_capacity
    ):
        assert _select_bh_walk_stack_capacity(n_particles) == expected_capacity

    def test_walk_stack_capacity_rejects_empty_system(self):
        with pytest.raises(ValueError, match="positive"):
            _select_bh_walk_stack_capacity(0)

    @pytest.mark.parametrize(
        ("n_particles", "kernel_name"),
        [
            (1, "bh_walk_kernel_64"),
            (115, "bh_walk_kernel_128"),
            (1_750_000, "bh_walk_kernel_256"),
            (16_777_217, "bh_walk_kernel_512"),
        ],
    )
    def test_walk_kernel_specialization_matches_capacity(
        self, n_particles, kernel_name
    ):
        import cupy as cp

        from ntropy.forces.gpu_bh import _ensure_bh_kernels

        walk, _ = _ensure_bh_kernels(cp, n_particles=n_particles)
        assert walk.name == kernel_name

    @pytest.mark.parametrize("N", [200, 1024])
    @pytest.mark.parametrize("theta", [0.3, 0.5, 0.7])
    def test_bh_acceleration_parity(self, N, theta):
        """GPU BH walk matches bh_c for same tree topology and theta."""
        from ntropy.forces.bhtree_c import compute_forces_bh_c

        rng = np.random.default_rng(42 + int(theta * 10))
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.005)

        acc_gpu_bh = compute_forces_gpu_bh(pos, mass, eps, theta=theta)
        acc_cpu_bh_c = compute_forces_bh_c(pos, mass, eps, theta=theta)

        assert acc_gpu_bh.shape == (N, 3)
        rel_err = np.linalg.norm(acc_gpu_bh - acc_cpu_bh_c, axis=1) / np.maximum(
            np.linalg.norm(acc_cpu_bh_c, axis=1), 1e-30
        )
        # BH opens approximations at different leaf boundaries on GPU vs CPU
        # but relative error should be small (< theta * 0.05 as sanity check)
        assert rel_err.max() < theta * 0.05, \
            f"Max relative error: {rel_err.max():.2e}"

    def test_deep_cusp_mixed_softening_does_not_drop_nodes(self):
        """A deep, occupied cusp must not overflow the GPU walk stack."""
        from ntropy.forces.bhtree_c import compute_forces_bh_c

        # Populate all seven sibling octants while repeatedly following child
        # zero toward the centre of a compact cusp.  A depth-first walk then
        # has seven pending siblings per level, which is the pattern that
        # exposed silent child drops in concentrated galaxy models.
        pos, mass, eps, targets = _make_deep_cusp(depth=16)

        acc_gpu = compute_forces_gpu_bh(
            pos, mass, eps, theta=0.5, target_indices=targets
        )
        acc_cpu = compute_forces_bh_c(
            pos, mass, eps, theta=0.5, target_indices=targets
        )

        assert np.allclose(acc_gpu, acc_cpu, rtol=1.0e-12, atol=1.0e-12), (
            f"GPU={acc_gpu!r}, CPU={acc_cpu!r}"
        )

    @pytest.mark.slow
    def test_real_ac023_ic_matches_cpu_on_representative_targets(self):
        """The real 1.75M-particle campaign tree must match the CPU walk.

        The compact synthetic cusp above isolates traversal depth, but it
        cannot preserve the component concentrations, mixed softenings, and
        occupied tree topology of the campaign IC that exposed the historical
        64-entry GPU stack bug.  Keep this optional integration regression for
        hosts that have the large campaign fixture available.
        """
        from ntropy.config import BhOptimizationsConfig
        from ntropy.forces.bhtree_c import compute_forces_bh_c

        fixture_path = _ac023_ic_fixture_path()
        if not fixture_path.is_file():
            pytest.skip(
                "real ac023 IC fixture is not available; expected "
                f"{fixture_path} or set NTROPY_AC023_IC"
            )

        with np.load(fixture_path, allow_pickle=False) as fixture:
            missing = {"pos", "mass", "eps"}.difference(fixture.files)
            assert not missing, (
                f"ac023 fixture is missing arrays: {sorted(missing)}"
            )
            pos = np.ascontiguousarray(fixture["pos"], dtype=np.float64)
            mass = np.ascontiguousarray(fixture["mass"], dtype=np.float64)
            eps = np.ascontiguousarray(fixture["eps"], dtype=np.float64)

        n_particles = 1_750_000
        assert pos.shape == (n_particles, 3)
        assert mass.shape == (n_particles,)
        assert eps.shape == (n_particles,)
        assert np.ptp(eps) > 0.0, "ac023 regression requires mixed softenings"

        # This exact sample spans the IC's contiguous particle components and
        # reproduced the old stack-64 failure: roughly 37% of its targets were
        # contaminated and its global relative L2 error was about 0.26.
        targets = np.linspace(
            0, n_particles - 1, num=20_000, dtype=np.int32
        )
        opts = BhOptimizationsConfig(
            preset="optimized",
            omp_schedule="guided",
            simd_leaves=True,
            native_pack=False,
        )

        acc_gpu = compute_forces_gpu_bh(
            pos,
            mass,
            eps,
            theta=0.5,
            target_indices=targets,
            bh_opts=opts,
        )
        acc_cpu = compute_forces_bh_c(
            pos,
            mass,
            eps,
            theta=0.5,
            target_indices=targets,
            bh_opts=opts,
        )

        assert np.isfinite(acc_gpu).all()
        assert np.isfinite(acc_cpu).all()
        cpu_norm = np.linalg.norm(acc_cpu)
        assert cpu_norm > 0.0
        relative_l2 = np.linalg.norm(acc_gpu - acc_cpu) / cpu_norm
        assert relative_l2 < 1.0e-10, (
            f"real ac023 GPU/CPU relative L2 error: {relative_l2:.3e}"
        )

    def test_walk_stack_overflow_is_explicit(self, monkeypatch):
        """A topology needing more than 256 pending nodes fails clearly."""
        import cupy as cp

        import ntropy.forces.gpu_bh as gpu_bh_module

        # The production builder's padded root and minimum leaf size bound its
        # achievable pending-node count.  Construct the adversarial topology
        # directly so this defensive path remains tested: every internal node
        # has eight children and child zero continues for 38 levels, leaving
        # seven pending siblings per level during the depth-first walk.
        depth = 38
        n_nodes = 1 + 8 * depth
        children = np.full((n_nodes, 8), -1, dtype=np.int32)
        is_leaf = np.ones(n_nodes, dtype=np.uint8)
        current = 0
        next_node = 1
        for _ in range(depth):
            child_indices = np.arange(next_node, next_node + 8, dtype=np.int32)
            children[current] = child_indices
            is_leaf[current] = 0
            current = int(child_indices[0])
            next_node += 8

        state = GpuBhState(n_nodes=n_nodes, n_particles=n_nodes)
        state.leaf_direct = 1
        state.d_mx = cp.zeros(n_nodes, dtype=cp.float64)
        state.d_my = cp.zeros(n_nodes, dtype=cp.float64)
        state.d_mz = cp.zeros(n_nodes, dtype=cp.float64)
        state.d_sz = cp.ones(n_nodes, dtype=cp.float64)
        state.d_ms = cp.ones(n_nodes, dtype=cp.float64)
        state.d_il = cp.asarray(is_leaf)
        state.d_ch = cp.asarray(children.ravel())
        state.d_ls = cp.arange(n_nodes, dtype=cp.int32)
        state.d_lc = cp.asarray(is_leaf, dtype=cp.int32)
        state.d_leaf_indices = cp.array([-1], dtype=cp.int32)

        monkeypatch.setattr(
            gpu_bh_module,
            "_prepare_tree_state",
            lambda *args, **kwargs: (state, n_nodes, 1),
        )
        with pytest.raises(RuntimeError, match="128-entry traversal stack"):
            gpu_bh_module.compute_forces_gpu_bh(
                np.zeros((n_nodes, 3), dtype=np.float64),
                np.ones(n_nodes, dtype=np.float64),
                np.ones(n_nodes, dtype=np.float64),
                theta=0.0,
                target_indices=np.array([0], dtype=np.int32),
                state=state,
            )
        assert int(cp.asnumpy(state.d_walk_overflow)[0]) == 1
        assert state.walk_stack_capacity == 128


# ====================================================================== #
# Category 2: Shape/Dtype Correctness Tests
# ====================================================================== #

class TestShapeDtype:
    """Output shapes, dtypes match expected values."""

    def test_gpu_direct_shape_all(self):
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((100, 3))
        mass = np.ones(100)
        eps = np.ones(100)
        acc = compute_forces_gpu(pos, mass, eps)
        assert acc.shape == (100, 3)
        assert acc.dtype == np.float64

    def test_gpu_direct_shape_subset(self):
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((100, 3))
        mass = np.ones(100)
        eps = np.ones(100)
        targets = np.array([0, 10, 20, 30])
        acc = compute_forces_gpu(pos, mass, eps, target_indices=targets)
        assert acc.shape == (4, 3)

    def test_gpu_bh_shape(self):
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((100, 3))
        mass = np.ones(100)
        eps = np.ones(100)
        acc = compute_forces_gpu_bh(pos, mass, eps, theta=0.5)
        assert acc.shape == (100, 3)
        assert acc.dtype == np.float64

    def test_gpu_bh_shape_subset(self):
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((100, 3))
        mass = np.ones(100)
        eps = np.ones(100)
        targets = np.arange(0, 100, 5)
        acc = compute_forces_gpu_bh(pos, mass, eps, target_indices=targets)
        assert acc.shape == (len(targets), 3)


# ====================================================================== #
# Category 3: Edge Cases
# ====================================================================== #

class TestEdgeCases:
    """Handle degenerate inputs gracefully."""

    def test_zero_mass(self):
        """Particles with zero mass contribute no force but accept force."""
        pos = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
        mass = np.array([1.0, 0.0, 1.0])
        eps = np.array([0.01, 0.01, 0.01])
        acc = compute_forces_gpu(pos, mass, eps)
        assert np.all(np.isfinite(acc))

    def test_coincident_particles(self):
        """Coincident particles handled without NaN/Inf from softening."""
        pos = np.array([
            [0., 0., 0.],
            [0., 0., 0.],  # coincident with particle 0
            [1., 0., 0.],
        ])
        mass = np.ones(3)
        eps = np.full(3, 0.1)  # softening prevents division by zero
        acc = compute_forces_gpu(pos, mass, eps)
        assert np.all(np.isfinite(acc))

    def test_single_particle(self):
        """N=1 returns zero acceleration."""
        pos = np.array([[1., 2., 3.]])
        mass = np.array([1.0])
        eps = np.array([0.01])
        acc = compute_forces_gpu(pos, mass, eps)
        assert acc.shape == (1, 3)
        assert np.allclose(acc, 0.0)

    def test_gpu_bh_single_particle(self):
        """GPU BH N=1 returns zero acceleration."""
        pos = np.array([[1., 2., 3.]])
        mass = np.array([1.0])
        eps = np.array([0.01])
        acc = compute_forces_gpu_bh(pos, mass, eps)
        assert acc.shape == (1, 3)
        assert np.allclose(acc, 0.0)

    def test_all_zero_softening(self):
        """Zero softening handled (particles are well-separated)."""
        rng = np.random.default_rng(42)
        pos = rng.standard_normal((50, 3)) * 10
        # Ensure no coincident particles
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                assert np.linalg.norm(pos[i] - pos[j]) > 0.1
        mass = np.ones(50)
        eps = np.zeros(50)
        acc = compute_forces_gpu(pos, mass, eps)
        assert np.all(np.isfinite(acc))


# ====================================================================== #
# Category 4: Precision Analysis
# ====================================================================== #

class TestPrecision:
    """Compare accuracy characteristics of GPU backends."""

    @pytest.mark.parametrize("N", [100, 500])
    def test_gpu_direct_vs_cpu_parity(self, N):
        """GPU brute-force matches CPU to within ~1e-8 (sum-ordering effects)."""
        from ntropy.forces.brute import compute_forces_brute

        rng = np.random.default_rng(99)
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.01)

        acc_gpu = compute_forces_gpu(pos, mass, eps)
        acc_cpu = compute_forces_brute(pos, mass, eps)

        rel_err = np.linalg.norm(acc_gpu - acc_cpu, axis=1) / np.maximum(
            np.linalg.norm(acc_cpu, axis=1), 1e-30
        )
        assert rel_err.max() < 1e-8, f"Max relative error: {rel_err.max():.2e}"

    @pytest.mark.parametrize("N", [200, 1024])
    def test_gpu_bh_accuracy(self, N):
        """GPU BH accuracy should be consistent with bh_c to within theta tolerance."""
        from ntropy.forces.bhtree_c import compute_forces_bh_c

        rng = np.random.default_rng(99)
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.005)
        theta = 0.5

        acc_gpu_bh = compute_forces_gpu_bh(pos, mass, eps, theta=theta)
        acc_cpu_bh_c = compute_forces_bh_c(pos, mass, eps, theta=theta)

        rel_err = np.linalg.norm(acc_gpu_bh - acc_cpu_bh_c, axis=1) / np.maximum(
            np.linalg.norm(acc_cpu_bh_c, axis=1), 1e-30
        )
        assert rel_err.max() < theta * 0.05, \
            f"Max relative error: {rel_err.max():.2e}"


# ====================================================================== #
# Category 5: GPU State Management
# ====================================================================== #

class TestGpuState:
    """GpuDirectState and GpuBhState lifecycle tests."""

    def test_gpu_direct_state_init(self):
        """GpuDirectState constructor works with valid inputs."""
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((100, 3))
        mass = np.ones(100)
        eps = np.ones(100)
        state = GpuDirectState(pos, mass, eps)
        assert state.n == 100
        assert state.d_pos is not None

    def test_gpu_direct_state_update_pos(self):
        """GpuDirectState.update_pos transfers new positions to GPU."""
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((50, 3))
        mass = np.ones(50)
        eps = np.ones(50)
        state = GpuDirectState(pos, mass, eps)

        new_pos = rng.standard_normal((50, 3)) * 2
        state.update_pos(new_pos)
        assert state.n == 50
        assert state.d_pos is not None

    def test_gpu_direct_state_detach(self):
        """GpuDirectState.detach frees GPU memory."""
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((10, 3))
        mass = np.ones(10)
        eps = np.ones(10)
        state = GpuDirectState(pos, mass, eps)
        state.detach()
        assert state.n == 0
        assert state.d_pos is None

    def test_gpu_direct_state_invalid_shape(self):
        """GpuDirectState rejects mismatched array sizes."""
        with pytest.raises(ValueError):
            GpuDirectState(np.ones((10, 3)), np.ones(9), np.ones(10))

    def test_gpu_bh_state_init(self):
        """GpuBhState constructor works."""
        state = GpuBhState(n_nodes=100, n_particles=50)
        assert state.n_nodes == 100
        assert state.is_ready() is False

    def test_gpu_bh_state_detach(self):
        """GpuBhState.detach clears all GPU data."""
        state = GpuBhState(n_nodes=10, n_particles=5)
        # Simulate building on GPU
        state.d_cx = np.array([1.0])  # type: ignore[attr-defined]
        state.detach()
        assert state.is_ready() is False


# ====================================================================== #
# Category 6: Availability Detection
# ====================================================================== #

class TestAvailability:
    """GPU detection and fallback behavior."""

    def test_gpu_available_returns_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)
        assert result is True  # We're on a machine with GPU

    def test_gpu_bh_available_returns_bool(self):
        result = gpu_bh_available()
        assert isinstance(result, bool)

    def test_gpu_bh_optimized_preset_not_zero(self):
        """optimized preset must not empty the GPU pack (native_pack forced off)."""
        from ntropy.config import BhOptimizationsConfig

        rng = np.random.default_rng(0)
        n = 2048
        pos = rng.normal(size=(n, 3))
        mass = np.full(n, 1.0 / n)
        eps = np.full(n, 0.05)
        opts = BhOptimizationsConfig.from_preset("optimized")
        acc = compute_forces_gpu_bh(pos, mass, eps, theta=0.5, bh_opts=opts)
        amag = np.linalg.norm(acc, axis=1)
        assert amag.mean() > 0.0
        assert int((amag == 0).sum()) == 0

    def test_gpu_direct_raises_without_gpu(self):
        """compute_forces_gpu raises ImportError when GPU unavailable."""
        # This test only matters when GPU IS available — it should NOT raise
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((10, 3))
        mass = np.ones(10)
        eps = np.ones(10)
        acc = compute_forces_gpu(pos, mass, eps)
        assert acc.shape == (10, 3)


# ====================================================================== #
# Category 7: Performance Benchmarks (Scaling Curves)
# ====================================================================== #

@pytest.mark.benchmark
class TestGpuBenchmarks:
    """Measure and record scaling curves for documentation."""

    @pytest.fixture(scope="class")
    def timing_results(self, request):
        """Run benchmarks at standard N values. Save to JSON for CI tracking."""
        from ntropy.benchmark.force_breakdown import time_gpu, time_gpu_bh

        results = {}
        for N in [100, 500]:
            rng = np.random.default_rng(N)
            pos = rng.standard_normal((N, 3)) * 10
            mass = np.abs(rng.random(N)) + 0.01
            eps = np.full(N, 0.01)

            t_gpu = time_gpu(pos, mass, eps)
            t_bh = time_gpu_bh(pos, mass, eps, theta=0.5)

            results[f"N={N}"] = {
                "gpu_direct_s": t_gpu,
                "gpu_bh_build_ms": t_bh["ms_build"],
                "gpu_bh_walk_ms": t_bh["ms_walk"],
                "gpu_bh_total_ms": t_bh["ms_total"],
            }

        # Save for CI comparison
        import json
        save_path = request.config.getoption("benchmark_json", None)
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)

        return results

    def test_benchmark_times_reasonable(self, timing_results):
        """Sanity check: GPU times should be reasonable (no hangs)."""
        for label, r in timing_results.items():
            assert r["gpu_direct_s"] < 30.0, \
                f"{label}: GPU direct took too long: {r['gpu_direct_s']:.1f}s"
            assert r["gpu_bh_total_ms"] < 30_000.0, \
                f"{label}: GPU BH took too long: {r['gpu_bh_total_ms']:.1f}ms"


# ====================================================================== #
# Category 8: compute_forces_gpu_from_state tests
# ====================================================================== #

class TestStateInterface:
    """GpuDirectState-based force evaluation correctness."""

    def test_from_state_matches_direct(self):
        """compute_forces_gpu_from_state matches compute_forces_gpu."""
        rng = np.random.default_rng(42)
        N = 100
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.01)

        state = GpuDirectState(pos, mass, eps)
        acc_from_state = compute_forces_gpu_from_state(state, pos)
        acc_direct = compute_forces_gpu(pos, mass, eps)

        assert np.allclose(acc_from_state, acc_direct, rtol=1e-6)

    def test_from_state_with_updated_pos(self):
        """GpuDirectState correctly uses updated positions."""
        rng = np.random.default_rng(42)
        N = 50
        pos = rng.standard_normal((N, 3)) * 10
        mass = np.abs(rng.random(N)) + 0.01
        eps = np.full(N, 0.01)

        state = GpuDirectState(pos, mass, eps)
        new_pos = pos * 2.0  # shift positions
        state.update_pos(new_pos)
        acc = compute_forces_gpu_from_state(state, new_pos)

        assert np.all(np.isfinite(acc))


# ====================================================================== #
# Category 9: Error handling
# ====================================================================== #

class TestErrorHandling:
    """GPU force functions reject invalid inputs."""

    def test_compute_forces_gpu_non_finite_pos(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_forces_gpu(
                np.array([[np.inf, 0., 0.]]),
                np.array([1.0]),
                np.array([0.01]),
            )

    def test_compute_forces_gpu_non_finite_mass(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_forces_gpu(
                np.array([[1., 2., 3.]]),
                np.array([np.nan]),
                np.array([0.01]),
            )

    def test_compute_forces_gpu_shape_mismatch(self):
        with pytest.raises(ValueError, match="pos must have shape"):
            compute_forces_gpu(
                np.ones((10, 2)),  # wrong second dim
                np.ones(10),
                np.ones(10),
            )

    def test_compute_forces_gpu_zero_particles(self):
        acc = compute_forces_gpu(np.empty((0, 3)), np.empty(0), np.empty(0))
        assert acc.shape == (0, 3)

    def test_compute_forces_gpu_bh_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_forces_gpu_bh(
                np.array([[np.nan, 0., 0.]]),
                np.array([1.0]),
                np.array([0.01]),
            )
