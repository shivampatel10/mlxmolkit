"""Tests for the Metal DG L-BFGS kernel.

Verifies threadgroup-parallel L-BFGS kernel produces correct results:
- Energy decreases from random starting coordinates
- Results comparable to dense BFGS within tolerance
- Works with different TPM and history depth values
- Handles chiral terms correctly
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


def _has_metal_lbfgs():
    """Check if Metal L-BFGS kernel is available."""
    try:
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs
        return True
    except ImportError:
        return False


def _has_metal_bfgs():
    """Check if Metal dense BFGS kernel is available."""
    try:
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_metal_lbfgs(), reason="Metal L-BFGS kernel not available")
class TestMetalDGLBFGS:
    """Test Metal DG L-BFGS kernel."""

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
        """L-BFGS should decrease energy for a single molecule."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), \
            f"L-BFGS energy should decrease: {initial_e[0].item():.4f} -> {final_e[0].item():.4f}"

    def test_benzene_energy_decrease(self):
        """L-BFGS should decrease energy for benzene."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("c1ccccc1")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_batch_energy_decrease(self):
        """L-BFGS should decrease energy for a batch of molecules."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_batch(["CCO", "CCC", "c1ccccc1"])

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        for i in range(3):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i}: energy should decrease"

    def test_chiral_molecule(self):
        """L-BFGS handles chiral terms correctly."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("[C@@H](O)(CC)C")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    @pytest.mark.skipif(not _has_metal_bfgs(), reason="Dense BFGS not available")
    def test_lbfgs_vs_dense_bfgs(self):
        """L-BFGS and dense BFGS should produce similar final energies."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        # Dense BFGS
        _, bfgs_e, _ = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(bfgs_e)

        # L-BFGS
        _, lbfgs_e, _ = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(lbfgs_e)

        bfgs_val = bfgs_e[0].item()
        lbfgs_val = lbfgs_e[0].item()

        # Both should converge to low energy
        assert lbfgs_val < 1.0, f"L-BFGS energy too high: {lbfgs_val}"
        assert bfgs_val < 1.0, f"Dense BFGS energy too high: {bfgs_val}"

        # Within 100x of each other (different numerical paths, but same ballpark)
        if bfgs_val > 1e-6 and lbfgs_val > 1e-6:
            ratio = max(lbfgs_val, bfgs_val) / max(min(lbfgs_val, bfgs_val), 1e-10)
            assert ratio < 100, \
                f"L-BFGS/BFGS energy mismatch: lbfgs={lbfgs_val:.6f}, bfgs={bfgs_val:.6f}"

    def test_tpm_1(self):
        """L-BFGS works with TPM=1 (sequential, no threadgroup parallelism)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(
            pos, system, max_iters=400, tpm=1,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_tpm_8(self):
        """L-BFGS works with TPM=8."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(
            pos, system, max_iters=400, tpm=8,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_tpm_32(self):
        """L-BFGS works with TPM=32 (default)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(
            pos, system, max_iters=400, tpm=32,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_history_depth_4(self):
        """L-BFGS works with m=4 (shallow history)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(
            pos, system, max_iters=400, lbfgs_m=4,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_history_depth_16(self):
        """L-BFGS works with m=16 (deep history)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(
            pos, system, max_iters=400, lbfgs_m=16,
        )
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_larger_molecule(self):
        """L-BFGS works on a larger molecule (aspirin, 21 atoms)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        system, pos = self._setup_mol("CC(=O)Oc1ccccc1C(=O)O")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    def test_large_batch(self):
        """L-BFGS handles a larger batch (10 molecules)."""
        from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

        smiles = [
            "CCO", "CCC", "c1ccccc1", "CC(=O)O", "CC(C)O",
            "c1ccc(cc1)O", "CC(=O)NC", "C1CCCCC1", "CC(C)(C)O", "c1ccncc1",
        ]
        system, pos = self._setup_batch(smiles)

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        for i in range(len(smiles)):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i} ({smiles[i]}): energy should decrease"
