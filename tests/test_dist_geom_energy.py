"""Unit tests for mlxmolkit.forcefields.dist_geom energy and gradient functions.

Tests include:
- Known-value energy checks for each term type
- Gradient vs mx.grad (autograd) cross-check (should match exactly)
- Gradient vs finite differences (eps=1e-3, tolerance ~1e-2 for float32)
- Combined DG energy/gradient
"""

import mlx.core as mx
import numpy as np
import numpy.testing as npt
import pytest

from mlxmolkit.forcefields.dist_geom import (
    chiral_violation_energy,
    chiral_violation_grad,
    dg_energy,
    dg_energy_and_grad,
    dist_violation_energy,
    dist_violation_grad_v2,
    fourth_dim_energy,
    fourth_dim_grad,
)
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params
from mlxmolkit.preprocessing.batching import batch_dg_params


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def _finite_diff_grad(energy_fn, pos, eps=1e-3):
    """Compute gradient via central finite differences."""
    pos_np = np.array(pos, dtype=np.float32)
    grad = np.zeros_like(pos_np)
    for i in range(len(pos_np)):
        pos_plus = pos_np.copy()
        pos_minus = pos_np.copy()
        pos_plus[i] += eps
        pos_minus[i] -= eps
        e_plus = float(energy_fn(mx.array(pos_plus)))
        e_minus = float(energy_fn(mx.array(pos_minus)))
        grad[i] = (e_plus - e_minus) / (2 * eps)
    return grad


# --------------------------------------------------
# Distance Violation Tests
# --------------------------------------------------


class TestDistViolationEnergy:
    def test_within_bounds_zero_energy(self):
        """No energy when distance is within bounds."""
        # 2 atoms at distance 2.0, bounds [1.0, 3.0]
        pos = mx.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)  # lb=1.0
        ub2 = mx.array([9.0], dtype=mx.float32)  # ub=3.0
        weight = mx.array([1.0], dtype=mx.float32)

        e = dist_violation_energy(pos, idx1, idx2, lb2, ub2, weight, dim=4)
        assert float(e[0]) == pytest.approx(0.0, abs=1e-7)

    def test_upper_bound_violation(self):
        """Energy for upper bound violation: d^2 > ub^2."""
        # 2 atoms at distance 3.0, ub=2.0 → d^2=9, ub^2=4
        # val = 9/4 - 1 = 1.25
        # E = 1.0 * 1.25^2 = 1.5625
        pos = mx.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        e = dist_violation_energy(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        assert float(e[0]) == pytest.approx(1.5625, rel=1e-5)

    def test_lower_bound_violation(self):
        """Energy for lower bound violation: d^2 < lb^2."""
        # 2 atoms at distance 0.5, lb=1.0 → d^2=0.25, lb^2=1.0
        # val = 2*1.0/(1.0+0.25) - 1 = 2/1.25 - 1 = 0.6
        # E = 1.0 * 0.6^2 = 0.36
        pos = mx.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([9.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        e = dist_violation_energy(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        assert float(e[0]) == pytest.approx(0.36, rel=1e-4)

    def test_weight_scaling(self):
        """Energy scales linearly with weight."""
        pos = mx.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)

        e1 = dist_violation_energy(
            pos, idx1, idx2, lb2, ub2, mx.array([1.0]), dim=3
        )
        e2 = dist_violation_energy(
            pos, idx1, idx2, lb2, ub2, mx.array([2.0]), dim=3
        )
        assert float(e2[0]) == pytest.approx(2.0 * float(e1[0]), rel=1e-5)

    def test_4d_includes_fourth_component(self):
        """Distance in 4D should include 4th component."""
        # Atoms at same xyz but different w: d^2 = w_diff^2
        pos = mx.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=mx.float32
        )
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        e = dist_violation_energy(pos, idx1, idx2, lb2, ub2, weight, dim=4)
        # d^2 = 9, ub^2 = 4, val = 9/4-1=1.25, E = 1.5625
        assert float(e[0]) == pytest.approx(1.5625, rel=1e-5)

    def test_multiple_terms(self):
        """Multiple terms computed correctly."""
        # 3 atoms: A at 0, B at 3, C at 0.5
        pos = mx.array(
            [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=mx.float32
        )
        idx1 = mx.array([1, 2], dtype=mx.int32)
        idx2 = mx.array([0, 0], dtype=mx.int32)
        lb2 = mx.array([1.0, 1.0], dtype=mx.float32)
        ub2 = mx.array([4.0, 9.0], dtype=mx.float32)
        weight = mx.array([1.0, 1.0], dtype=mx.float32)

        e = dist_violation_energy(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        # Term 0: d^2=9, ub^2=4, val=1.25, E=1.5625
        # Term 1: d^2=0.25, lb^2=1, val=2/(1.25)-1=0.6, E=0.36
        assert float(e[0]) == pytest.approx(1.5625, rel=1e-5)
        assert float(e[1]) == pytest.approx(0.36, rel=1e-4)


class TestDistViolationGrad:
    def test_within_bounds_zero_grad(self):
        """No gradient when distance is within bounds."""
        pos = mx.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([9.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        g = dist_violation_grad_v2(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        npt.assert_allclose(np.array(g), 0.0, atol=1e-7)

    def test_grad_vs_finite_diff_upper(self):
        """Gradient matches finite differences for upper bound violation."""
        pos = mx.array([0.0, 0.0, 0.0, 3.0, 0.5, 0.2], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        def energy_fn(p):
            return mx.sum(
                dist_violation_energy(p, idx1, idx2, lb2, ub2, weight, dim=3)
            )

        g_analytic = np.array(
            dist_violation_grad_v2(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        )
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        npt.assert_allclose(g_analytic, g_fd, atol=1e-2, rtol=1e-2)

    def test_grad_vs_finite_diff_lower(self):
        """Gradient matches finite differences for lower bound violation."""
        pos = mx.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.1], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([9.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        def energy_fn(p):
            return mx.sum(
                dist_violation_energy(p, idx1, idx2, lb2, ub2, weight, dim=3)
            )

        g_analytic = np.array(
            dist_violation_grad_v2(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        )
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        npt.assert_allclose(g_analytic, g_fd, atol=1e-2, rtol=1e-2)

    def test_grad_vs_autograd(self):
        """Manual gradient matches mx.grad exactly."""
        pos = mx.array(
            [0.1, 0.2, 0.3, 2.5, 0.8, 0.1], dtype=mx.float32
        )
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        def energy_fn(p):
            return mx.sum(
                dist_violation_energy(p, idx1, idx2, lb2, ub2, weight, dim=3)
            )

        g_analytic = np.array(
            dist_violation_grad_v2(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        )
        g_auto = np.array(mx.grad(energy_fn)(pos))

        npt.assert_allclose(g_analytic, g_auto, atol=1e-5)

    def test_grad_newton_third_law(self):
        """Gradient on atom 1 equals negative gradient on atom 2."""
        pos = mx.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=mx.float32)
        idx1 = mx.array([1], dtype=mx.int32)
        idx2 = mx.array([0], dtype=mx.int32)
        lb2 = mx.array([1.0], dtype=mx.float32)
        ub2 = mx.array([4.0], dtype=mx.float32)
        weight = mx.array([1.0], dtype=mx.float32)

        g = np.array(
            dist_violation_grad_v2(pos, idx1, idx2, lb2, ub2, weight, dim=3)
        ).reshape(-1, 3)

        # g[0] should be -g[1]
        npt.assert_allclose(g[0], -g[1], atol=1e-6)


# --------------------------------------------------
# Chiral Violation Tests
# --------------------------------------------------


class TestChiralViolationEnergy:
    def _make_tetrahedron(self, scale=1.0):
        """Create a regular tetrahedron centered at origin (3D)."""
        # Vertices of a regular tetrahedron
        coords = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float32,
        ) * scale
        return mx.array(coords.flatten())

    def test_within_bounds_zero_energy(self):
        pos = self._make_tetrahedron()
        idx1 = mx.array([0], dtype=mx.int32)
        idx2 = mx.array([1], dtype=mx.int32)
        idx3 = mx.array([2], dtype=mx.int32)
        idx4 = mx.array([3], dtype=mx.int32)

        # Compute actual volume to set loose bounds
        from mlxmolkit.forcefields.dist_geom import _calc_chiral_volume

        vol, _, _, _ = _calc_chiral_volume(pos, idx1, idx2, idx3, idx4, dim=3)
        v = float(vol[0])

        vol_lower = mx.array([v - 10.0], dtype=mx.float32)
        vol_upper = mx.array([v + 10.0], dtype=mx.float32)

        e = chiral_violation_energy(
            pos, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
        )
        assert float(e[0]) == pytest.approx(0.0, abs=1e-7)

    def test_lower_violation(self):
        pos = self._make_tetrahedron()
        idx1 = mx.array([0], dtype=mx.int32)
        idx2 = mx.array([1], dtype=mx.int32)
        idx3 = mx.array([2], dtype=mx.int32)
        idx4 = mx.array([3], dtype=mx.int32)

        from mlxmolkit.forcefields.dist_geom import _calc_chiral_volume

        vol, _, _, _ = _calc_chiral_volume(pos, idx1, idx2, idx3, idx4, dim=3)
        v = float(vol[0])

        # Set lower bound above actual volume
        vol_lower = mx.array([v + 5.0], dtype=mx.float32)
        vol_upper = mx.array([v + 10.0], dtype=mx.float32)

        e = chiral_violation_energy(
            pos, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
        )
        expected = (v - (v + 5.0)) ** 2  # (vol - lower)^2
        assert float(e[0]) == pytest.approx(expected, rel=1e-4)


class TestChiralViolationGrad:
    def test_grad_vs_finite_diff(self):
        """Chiral gradient matches finite differences."""
        np.random.seed(42)
        pos = mx.array(np.random.randn(12).astype(np.float32))
        idx1 = mx.array([0], dtype=mx.int32)
        idx2 = mx.array([1], dtype=mx.int32)
        idx3 = mx.array([2], dtype=mx.int32)
        idx4 = mx.array([3], dtype=mx.int32)
        vol_lower = mx.array([5.0], dtype=mx.float32)
        vol_upper = mx.array([10.0], dtype=mx.float32)

        def energy_fn(p):
            return mx.sum(
                chiral_violation_energy(
                    p, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
                )
            )

        g_analytic = np.array(
            chiral_violation_grad(
                pos, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
            )
        )
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        # Float32 tolerance
        npt.assert_allclose(g_analytic, g_fd, atol=5e-2, rtol=5e-2)

    def test_grad_vs_autograd(self):
        """Manual gradient matches mx.grad."""
        np.random.seed(42)
        pos = mx.array(np.random.randn(12).astype(np.float32))
        idx1 = mx.array([0], dtype=mx.int32)
        idx2 = mx.array([1], dtype=mx.int32)
        idx3 = mx.array([2], dtype=mx.int32)
        idx4 = mx.array([3], dtype=mx.int32)
        vol_lower = mx.array([5.0], dtype=mx.float32)
        vol_upper = mx.array([10.0], dtype=mx.float32)

        def energy_fn(p):
            return mx.sum(
                chiral_violation_energy(
                    p, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
                )
            )

        g_analytic = np.array(
            chiral_violation_grad(
                pos, idx1, idx2, idx3, idx4, vol_lower, vol_upper, 1.0, dim=3
            )
        )
        g_auto = np.array(mx.grad(energy_fn)(pos))

        npt.assert_allclose(g_analytic, g_auto, atol=1e-4)


# --------------------------------------------------
# Fourth Dimension Tests
# --------------------------------------------------


class TestFourthDimEnergy:
    def test_zero_fourth_coord(self):
        """Zero energy when 4th coordinate is zero."""
        pos = mx.array([0.0, 0.0, 0.0, 0.0], dtype=mx.float32)
        idx = mx.array([0], dtype=mx.int32)
        e = fourth_dim_energy(pos, idx, weight=1.0, dim=4)
        assert float(e[0]) == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_fourth_coord(self):
        """Energy is weight * w^2."""
        pos = mx.array([0.0, 0.0, 0.0, 2.0], dtype=mx.float32)
        idx = mx.array([0], dtype=mx.int32)
        e = fourth_dim_energy(pos, idx, weight=0.5, dim=4)
        assert float(e[0]) == pytest.approx(0.5 * 4.0, rel=1e-5)

    def test_3d_returns_zero(self):
        """No fourth dim energy in 3D."""
        pos = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=mx.float32)
        idx = mx.array([0, 1], dtype=mx.int32)
        e = fourth_dim_energy(pos, idx, weight=1.0, dim=3)
        npt.assert_allclose(np.array(e), 0.0, atol=1e-7)


class TestFourthDimGrad:
    def test_grad_vs_finite_diff(self):
        """Fourth dim gradient matches finite differences."""
        pos = mx.array([1.0, 2.0, 3.0, 1.5, 0.5, 0.5, 0.5, -1.0], dtype=mx.float32)
        idx = mx.array([0, 1], dtype=mx.int32)

        def energy_fn(p):
            return mx.sum(fourth_dim_energy(p, idx, weight=0.5, dim=4))

        g_analytic = np.array(fourth_dim_grad(pos, idx, weight=0.5, dim=4))
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        npt.assert_allclose(g_analytic, g_fd, atol=1e-2, rtol=1e-2)


# --------------------------------------------------
# Combined DG Energy/Gradient Tests
# --------------------------------------------------


class TestCombinedDGEnergy:
    def test_with_real_molecule(self, ethanol_mol):
        """Combined energy computes without error for a real molecule."""
        params = extract_dg_params(ethanol_mol, dim=4)
        system = batch_dg_params([params], dim=4)

        # Random 4D positions
        np.random.seed(42)
        n_atoms = params.num_atoms
        pos = mx.array(np.random.randn(n_atoms * 4).astype(np.float32) * 0.5)

        energies = dg_energy(pos, system, chiral_weight=1.0, fourth_dim_weight=0.1)
        assert energies.shape == (1,)
        assert float(energies[0]) >= 0.0

    def test_energy_and_grad_consistency(self, ethanol_mol):
        """Energy from dg_energy matches dg_energy_and_grad."""
        params = extract_dg_params(ethanol_mol, dim=4)
        system = batch_dg_params([params], dim=4)

        np.random.seed(42)
        n_atoms = params.num_atoms
        pos = mx.array(np.random.randn(n_atoms * 4).astype(np.float32) * 0.5)

        e1 = dg_energy(pos, system, chiral_weight=1.0, fourth_dim_weight=0.1)
        e2, g = dg_energy_and_grad(
            pos, system, chiral_weight=1.0, fourth_dim_weight=0.1
        )

        assert float(e1[0]) == pytest.approx(float(e2[0]), rel=1e-5)
        assert g.shape == pos.shape

    def test_combined_grad_vs_finite_diff(self, ethanol_mol):
        """Combined gradient matches finite differences."""
        params = extract_dg_params(ethanol_mol, dim=4)
        system = batch_dg_params([params], dim=4)

        np.random.seed(42)
        n_atoms = params.num_atoms
        pos = mx.array(np.random.randn(n_atoms * 4).astype(np.float32) * 0.5)

        def energy_fn(p):
            return mx.sum(
                dg_energy(p, system, chiral_weight=1.0, fourth_dim_weight=0.1)
            )

        _, g_analytic = dg_energy_and_grad(
            pos, system, chiral_weight=1.0, fourth_dim_weight=0.1
        )
        g_analytic = np.array(g_analytic)
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        npt.assert_allclose(g_analytic, g_fd, atol=0.1, rtol=0.1)

    def test_batched_two_molecules(self, ethanol_mol, benzene_mol):
        """Batched energy computation for 2 molecules."""
        p1 = extract_dg_params(ethanol_mol, dim=4)
        p2 = extract_dg_params(benzene_mol, dim=4)
        system = batch_dg_params([p1, p2], dim=4)

        np.random.seed(42)
        pos = mx.array(
            np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 0.5
        )

        energies = dg_energy(pos, system, chiral_weight=1.0, fourth_dim_weight=0.1)
        assert energies.shape == (2,)
        assert float(energies[0]) >= 0.0
        assert float(energies[1]) >= 0.0
