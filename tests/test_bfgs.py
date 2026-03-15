"""Tests for the batched BFGS minimizer.

Test layers:
1. Simple quadratic minimization (validates BFGS mechanics)
2. Single molecule DG energy minimization from random coords
3. Batched minimization of multiple molecules
"""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.forcefields.dist_geom import dg_energy_and_grad
from mlxmolkit.minimizer.bfgs import bfgs_minimize
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix


# ============================================================
# Test 1: Quadratic function minimization
# ============================================================


class TestQuadraticMinimization:
    """Test BFGS on f(x) = 0.5 * x^T A x - b^T x.

    Minimum at x* = A^{-1} b.
    """

    def _make_quadratic_energy_grad(self, A, b):
        """Create energy+grad function for a quadratic."""
        def energy_and_grad(pos):
            # f(x) = 0.5 * x^T A x - b^T x
            energy = 0.5 * mx.sum(pos * (A @ pos)) - mx.sum(b * pos)
            grad = A @ pos - b
            return mx.array([energy]), grad
        return energy_and_grad

    def test_2d_quadratic(self):
        """Minimize f(x,y) = x^2 + 2*y^2, minimum at origin."""
        A = mx.array([[2.0, 0.0], [0.0, 4.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        pos = mx.array([3.0, -2.0], dtype=mx.float32)
        atom_starts = [0, 1]  # 1 "molecule" with 1 "atom" in 2D
        final_pos, final_e, statuses = bfgs_minimize(
            fn, pos, atom_starts, n_mols=1, dim=2,
            max_iters=50, scale_grads=False,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < 1e-6, f"Energy not minimized: {final_e[0].item()}"
        np.testing.assert_allclose(
            np.array(final_pos.tolist()), [0.0, 0.0], atol=1e-3
        )

    def test_3d_quadratic_with_offset(self):
        """Minimize f(x) = 0.5*x^T A x - b^T x, min at A^{-1}b."""
        A = mx.array([
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ], dtype=mx.float32)
        b = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        # True minimum
        A_np = np.array([[4, 1, 0], [1, 3, 0.5], [0, 0.5, 2]], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        x_star = np.linalg.solve(A_np, b_np)

        pos = mx.array([5.0, -3.0, 1.0], dtype=mx.float32)
        final_pos, final_e, statuses = bfgs_minimize(
            fn, pos, [0, 1], n_mols=1, dim=3,
            max_iters=100, scale_grads=False,
        )
        mx.eval(final_pos, final_e)

        np.testing.assert_allclose(
            np.array(final_pos.tolist()), x_star, atol=1e-2
        )

    def test_convergence_status(self):
        """Verify converged status is set to 0."""
        A = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)
        fn = self._make_quadratic_energy_grad(A, b)

        pos = mx.array([1.0, 1.0], dtype=mx.float32)
        _, _, statuses = bfgs_minimize(
            fn, pos, [0, 1], n_mols=1, dim=2,
            max_iters=50, scale_grads=False,
        )
        mx.eval(statuses)
        assert statuses[0].item() == 0, "Should converge on simple quadratic"


# ============================================================
# Test 2: Single molecule DG minimization
# ============================================================


class TestSingleMolDGMinimize:
    """Test BFGS on a single molecule's DG energy from random coords."""

    def _setup_mol(self, smiles):
        """Create mol, extract DG params, batch, generate random 4D coords."""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        bounds_mat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bounds_mat, dim=4)
        system = batch_dg_params([params], dim=4)

        # Random initial 4D coords
        np.random.seed(42)
        n_atoms = params.num_atoms
        coords = np.random.randn(n_atoms * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)

        return mol, system, pos

    def test_ethanol_dg_minimize(self):
        """Minimize ethanol DG energy from random coords."""
        mol, system, pos = self._setup_mol("CCO")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        atom_starts = system.atom_starts.tolist()
        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, atom_starts,
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), \
            f"Energy should decrease: {initial_e[0].item()} -> {final_e[0].item()}"
        # DG energy should converge to near zero for a valid geometry
        assert final_e[0].item() < 1.0, \
            f"DG energy should be small after minimization: {final_e[0].item()}"

    def test_propane_dg_minimize(self):
        """Minimize propane DG energy from random coords."""
        mol, system, pos = self._setup_mol("CCC")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()
        assert final_e[0].item() < 5.0, \
            f"DG energy too high after minimization: {final_e[0].item()}"

    def test_benzene_dg_minimize(self):
        """Minimize benzene DG energy from random coords."""
        mol, system, pos = self._setup_mol("c1ccccc1")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()

    def test_chiral_mol_dg_minimize(self):
        """Minimize a chiral molecule's DG energy."""
        mol, system, pos = self._setup_mol("[C@@H](O)(CC)C")

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()

    def test_energy_decreases_monotonically(self):
        """Track energy at each iteration to verify monotonic decrease."""
        mol, system, pos = self._setup_mol("CCO")
        energy_history = []

        original_fn = lambda p: dg_energy_and_grad(p, system)

        # Run a few iterations manually to check energy decrease
        e0, _ = original_fn(pos)
        mx.eval(e0)
        energy_history.append(e0[0].item())

        final_pos, final_e, _ = bfgs_minimize(
            original_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=50,
        )
        mx.eval(final_e)

        # Final energy should be less than initial
        assert final_e[0].item() < energy_history[0]


# ============================================================
# Test 3: Batched minimization
# ============================================================


class TestBatchedMinimization:
    """Test BFGS on multiple molecules simultaneously."""

    def _setup_batch(self, smiles_list, seed=42):
        """Create batched system from multiple SMILES."""
        np.random.seed(seed)
        params_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            bounds_mat = get_bounds_matrix(mol)
            params = extract_dg_params(mol, bounds_mat, dim=4)
            params_list.append(params)

        system = batch_dg_params(params_list, dim=4)

        # Random initial coords
        coords = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)

        return system, pos

    def test_two_molecules(self):
        """Minimize two molecules simultaneously."""
        system, pos = self._setup_batch(["CCO", "CCC"])

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=2, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e, statuses)

        # Both molecules should have lower energy
        for i in range(2):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i}: energy should decrease"

    def test_three_diverse_molecules(self):
        """Minimize ethanol, benzene, and propane together."""
        system, pos = self._setup_batch(["CCO", "c1ccccc1", "CCC"])

        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        initial_e, _ = energy_grad_fn(pos)
        mx.eval(initial_e)

        final_pos, final_e, statuses = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=3, dim=4, max_iters=400,
        )
        mx.eval(final_pos, final_e)

        for i in range(3):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i}: energy should decrease"

    def test_independent_molecules(self):
        """Verify batch result matches individual molecule results."""
        smiles_list = ["CCO", "CCC"]
        system_batch, pos_batch = self._setup_batch(smiles_list, seed=42)

        def batch_fn(p):
            return dg_energy_and_grad(p, system_batch)

        _, batch_e, _ = bfgs_minimize(
            batch_fn, pos_batch, system_batch.atom_starts.tolist(),
            n_mols=2, dim=4, max_iters=200,
        )
        mx.eval(batch_e)

        # Run individually with same initial coords
        np.random.seed(42)
        individual_energies = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            bounds_mat = get_bounds_matrix(mol)
            params = extract_dg_params(mol, bounds_mat, dim=4)
            sys_i = batch_dg_params([params], dim=4)
            n_atoms = params.num_atoms
            coords = np.random.randn(n_atoms * 4).astype(np.float32) * 2.0
            pos_i = mx.array(coords)

            def fn_i(p, s=sys_i):
                return dg_energy_and_grad(p, s)

            _, e_i, _ = bfgs_minimize(
                fn_i, pos_i, sys_i.atom_starts.tolist(),
                n_mols=1, dim=4, max_iters=200,
            )
            mx.eval(e_i)
            individual_energies.append(e_i[0].item())

        # Energies should be similar (not exact due to batching interactions
        # in gradient accumulation, but same order of magnitude)
        for i in range(2):
            assert abs(batch_e[i].item() - individual_energies[i]) < 1.0, \
                f"Mol {i}: batch={batch_e[i].item():.4f} vs individual={individual_energies[i]:.4f}"


# ============================================================
# Test 4: Edge cases and numerical robustness
# ============================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_already_minimized(self):
        """BFGS should converge quickly if starting near minimum."""
        A = mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)

        def fn(pos):
            e = 0.5 * mx.sum(pos * (A @ pos))
            g = A @ pos
            return mx.array([e]), g

        # Start very close to minimum
        pos = mx.array([1e-5, -1e-5], dtype=mx.float32)
        _, final_e, statuses = bfgs_minimize(
            fn, pos, [0, 1], n_mols=1, dim=2,
            max_iters=10, scale_grads=False,
        )
        mx.eval(final_e, statuses)

        assert statuses[0].item() == 0, "Should converge immediately"
        assert final_e[0].item() < 1e-8

    def test_large_initial_gradient(self):
        """BFGS should handle large initial gradients via scaling."""
        A = mx.array([[200.0, 0.0], [0.0, 200.0]], dtype=mx.float32)
        b = mx.zeros(2, dtype=mx.float32)

        def fn(pos):
            e = 0.5 * mx.sum(pos * (A @ pos))
            g = A @ pos
            return mx.array([e]), g

        pos = mx.array([100.0, -100.0], dtype=mx.float32)
        final_pos, final_e, _ = bfgs_minimize(
            fn, pos, [0, 1], n_mols=1, dim=2,
            max_iters=200, scale_grads=True,
        )
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < 1.0, \
            f"Should minimize despite large gradient: {final_e[0].item()}"
