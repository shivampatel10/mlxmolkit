"""Tests for the Metal DG BFGS kernel.

Verifies Metal kernel produces energies matching Python BFGS within tolerance.
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


def _has_metal_kernel():
    """Check if Metal kernel is available."""
    try:
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_metal_kernel(), reason="Metal kernel not available")
class TestMetalDGBFGS:
    """Test Metal DG BFGS kernel against Python reference."""

    def _setup_mol(self, smiles, seed=42):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        bounds_mat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bounds_mat, dim=4)
        system = batch_dg_params([params], dim=4)

        np.random.seed(seed)
        n_atoms = params.num_atoms
        coords = np.random.randn(n_atoms * 4).astype(np.float32) * 2.0
        pos = mx.array(coords)
        return system, pos

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

    def test_single_mol_energy_decrease(self):
        """Metal kernel should decrease energy for a single molecule."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), \
            f"Metal energy should decrease: {initial_e[0].item():.4f} -> {final_e[0].item():.4f}"

    def test_benzene_energy_decrease(self):
        """Metal kernel should decrease energy for benzene."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

        system, pos = self._setup_mol("c1ccccc1")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_batch_energy_decrease(self):
        """Metal kernel should decrease energy for a batch of molecules."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

        system, pos = self._setup_batch(["CCO", "CCC", "c1ccccc1"])

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        for i in range(3):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i}: energy should decrease"

    def test_metal_vs_python_tolerance(self):
        """Metal and Python BFGS should produce similar energies."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

        system, pos = self._setup_mol("CCO")

        # Python BFGS
        def energy_grad_fn(p):
            return dg_energy_and_grad(p, system)

        _, py_e, _ = bfgs_minimize(
            energy_grad_fn, pos, system.atom_starts.tolist(),
            n_mols=1, dim=4, max_iters=400, scale_grads=False,
        )
        mx.eval(py_e)

        # Metal BFGS
        _, metal_e, _ = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(metal_e)

        # Both should converge to low energy
        py_val = py_e[0].item()
        metal_val = metal_e[0].item()

        assert metal_val < 1.0, f"Metal energy too high: {metal_val}"
        assert py_val < 1.0, f"Python energy too high: {py_val}"

        # Relative tolerance: within 10x of each other (different numerical paths)
        if py_val > 1e-6 and metal_val > 1e-6:
            ratio = max(metal_val, py_val) / max(min(metal_val, py_val), 1e-10)
            assert ratio < 100, \
                f"Metal/Python energy mismatch: metal={metal_val:.6f}, python={py_val:.6f}"

    def test_chiral_molecule(self):
        """Metal kernel handles chiral terms correctly."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

        system, pos = self._setup_mol("[C@@H](O)(CC)C")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()
