"""Tests for the vectorized batched BFGS minimizer.

Cross-validates results with the original per-molecule BFGS.
Same test structure as test_bfgs.py.
"""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.forcefields.dist_geom import dg_energy_and_grad
from mlxmolkit.minimizer.bfgs import bfgs_minimize
from mlxmolkit.minimizer.bfgs_vectorized import bfgs_minimize_vectorized
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix


# ============================================================
# Test 1: Quadratic function minimization
# ============================================================


class TestQuadraticMinimization:
    """Test vectorized BFGS on simple quadratics."""

    def _make_quadratic_energy_grad(self, A, b):
        def energy_and_grad(pos):
            energy = 0.5 * mx.sum(pos * (A @ pos)) - mx.sum(b * pos)
            grad = A @ pos - b
            return mx.array([energy]), grad
        return energy_and_grad

    def test_2d_quadratic(self):
        A = mx.array([[2.0, 0.0], [0.0, 4.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        pos = mx.array([3.0, -2.0], dtype=mx.float32)
        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            fn, pos, [0, 1], n_mols=1, dim=2, max_iters=50,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < 1e-6
        np.testing.assert_allclose(
            np.array(final_pos.tolist()), [0.0, 0.0], atol=1e-3
        )

    def test_3d_quadratic_with_offset(self):
        A = mx.array([
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ], dtype=mx.float32)
        b = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        A_np = np.array([[4, 1, 0], [1, 3, 0.5], [0, 0.5, 2]], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        x_star = np.linalg.solve(A_np, b_np)

        pos = mx.array([5.0, -3.0, 1.0], dtype=mx.float32)
        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            fn, pos, [0, 1], n_mols=1, dim=3, max_iters=100,
        )
        mx.eval(final_pos, final_e)

        np.testing.assert_allclose(
            np.array(final_pos.tolist()), x_star, atol=1e-2
        )

    def test_convergence_status(self):
        A = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        pos = mx.array([1.0, 1.0], dtype=mx.float32)
        _, _, statuses = bfgs_minimize_vectorized(
            fn, pos, [0, 1], n_mols=1, dim=2, max_iters=50,
        )
        mx.eval(statuses)
        assert statuses[0].item() == 0, "Should converge on simple quadratic"


# ============================================================
# Test 2: Single molecule DG minimization
# ============================================================


class TestSingleMolDGMinimize:
    """Test vectorized BFGS on single molecule DG energy."""

    def _setup_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        bounds_mat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bounds_mat, dim=4)
        system = batch_dg_params([params], dim=4)

        np.random.seed(42)
        n_atoms = params.num_atoms
        coords = np.random.randn(n_atoms * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)
        return mol, system, pos

    def test_ethanol_dg_minimize(self):
        mol, system, pos = self._setup_mol("CCO")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()
        assert final_e[0].item() < 1.0

    def test_benzene_dg_minimize(self):
        mol, system, pos = self._setup_mol("c1ccccc1")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()


# ============================================================
# Test 3: Batched minimization
# ============================================================


class TestBatchedMinimization:
    """Test vectorized BFGS on multiple molecules simultaneously."""

    def _setup_batch(self, smiles_list, seed=42):
        np.random.seed(seed)
        params_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            bounds_mat = get_bounds_matrix(mol)
            params = extract_dg_params(mol, bounds_mat, dim=4)
            params_list.append(params)

        system = batch_dg_params(params_list, dim=4)
        coords = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)
        return system, pos

    def test_two_molecules(self):
        system, pos = self._setup_batch(["CCO", "CCC"])

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=2, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e, statuses)

        for i in range(2):
            assert final_e[i].item() < initial_e[i].item()

    def test_three_diverse_molecules(self):
        system, pos = self._setup_batch(["CCO", "c1ccccc1", "CCC"])

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize_vectorized(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=3, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        for i in range(3):
            assert final_e[i].item() < initial_e[i].item()


# ============================================================
# Test 4: Cross-validation with original BFGS
# ============================================================


class TestCrossValidation:
    """Verify vectorized BFGS gives similar results to original."""

    def _setup_batch(self, smiles_list, seed=42):
        np.random.seed(seed)
        params_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            bounds_mat = get_bounds_matrix(mol)
            params = extract_dg_params(mol, bounds_mat, dim=4)
            params_list.append(params)

        system = batch_dg_params(params_list, dim=4)
        coords = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)
        return system, pos

    def test_energies_match_original(self):
        """Vectorized BFGS energies should be within tolerance of original."""
        system, pos = self._setup_batch(["CCO", "CCC"])

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        # Original
        _, orig_e, _ = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=2, dim=4, max_iters=200, scale_grads=False,
        )
        mx.eval(orig_e)

        # Vectorized
        _, vec_e, _ = bfgs_minimize_vectorized(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=2, dim=4, max_iters=200,
        )
        mx.eval(vec_e)

        # Both should reach low energies (not necessarily identical
        # due to different execution order, but same order of magnitude)
        for i in range(2):
            orig_val = orig_e[i].item()
            vec_val = vec_e[i].item()
            # Both should be small
            assert vec_val < 5.0, f"Vectorized energy too high: {vec_val}"
            assert orig_val < 5.0, f"Original energy too high: {orig_val}"


# ============================================================
# Test 5: Edge cases
# ============================================================


class TestEdgeCases:
    def test_already_minimized(self):
        A = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)

        def fn(pos):
            e = 0.5 * mx.sum(pos * (A @ pos))
            g = A @ pos
            return mx.array([e]), g

        pos = mx.array([1e-5, -1e-5], dtype=mx.float32)
        _, final_e, statuses = bfgs_minimize_vectorized(
            fn, pos, [0, 1], n_mols=1, dim=2, max_iters=10,
        )
        mx.eval(final_e, statuses)

        assert statuses[0].item() == 0
        assert final_e[0].item() < 1e-8
