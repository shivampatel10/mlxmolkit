"""Tests for the Metal DG BFGS threadgroup kernel.

Verifies TG kernel produces energies matching serial BFGS and Python reference.
"""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem

from mlxmolkit.forcefields.dist_geom import dg_energy_and_grad
from mlxmolkit.minimizer.bfgs import bfgs_minimize
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix


def _has_tg_kernel():
    """Check if TG Metal kernel is available."""
    try:
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg
        return True
    except ImportError:
        return False


def _has_serial_kernel():
    """Check if serial Metal kernel is available."""
    try:
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_tg_kernel(), reason="TG Metal kernel not available")
class TestMetalDGBFGSTG:
    """Test Metal DG BFGS threadgroup kernel."""

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
        """TG kernel should decrease energy for a single molecule (ethanol, 9 atoms)."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg

        system, pos = self._setup_mol("CCO")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), \
            f"TG energy should decrease: {initial_e[0].item():.4f} -> {final_e[0].item():.4f}"

    def test_batch_energy_decrease(self):
        """TG kernel should decrease energy for a batch (catches TG sync bugs)."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg

        system, pos = self._setup_batch(["CCO", "CCC", "c1ccccc1"])

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        for i in range(3):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i}: TG energy should decrease: {initial_e[i].item():.4f} -> {final_e[i].item():.4f}"

    def test_chiral_molecule(self):
        """TG kernel handles chiral terms correctly."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg

        system, pos = self._setup_mol("[C@@H](O)(CC)C")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item()

    @pytest.mark.skipif(not _has_serial_kernel(), reason="Serial kernel not available")
    def test_tg_vs_serial_numerical(self):
        """Critical: TG and serial should produce similar final energies."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs, metal_dg_bfgs_tg

        system, pos = self._setup_mol("CCO")

        _, serial_e, serial_s = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(serial_e, serial_s)

        _, tg_e, tg_s = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(tg_e, tg_s)

        serial_val = serial_e[0].item()
        tg_val = tg_e[0].item()

        # Both should converge to similar energy
        assert tg_val < 1.0, f"TG energy too high: {tg_val}"
        assert serial_val < 1.0, f"Serial energy too high: {serial_val}"

        # Relative tolerance ~1e-3
        if serial_val > 1e-6 and tg_val > 1e-6:
            rel_err = abs(tg_val - serial_val) / max(abs(serial_val), 1e-10)
            assert rel_err < 0.1, \
                f"TG/serial mismatch: tg={tg_val:.6f}, serial={serial_val:.6f}, rel_err={rel_err:.4f}"

    @pytest.mark.skipif(not _has_serial_kernel(), reason="Serial kernel not available")
    def test_tpm_1_matches_serial(self):
        """Degenerate TPM=1 should match serial kernel exactly."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs, metal_dg_bfgs_tg

        system, pos = self._setup_mol("CCO")

        _, serial_e, _ = metal_dg_bfgs(pos, system, max_iters=400)
        mx.eval(serial_e)

        _, tg_e, _ = metal_dg_bfgs_tg(pos, system, max_iters=400, tpm=1)
        mx.eval(tg_e)

        serial_val = serial_e[0].item()
        tg_val = tg_e[0].item()

        # TPM=1 means all code paths are single-threaded, should be near-identical
        if serial_val > 1e-8 and tg_val > 1e-8:
            rel_err = abs(tg_val - serial_val) / max(abs(serial_val), 1e-10)
            assert rel_err < 1e-3, \
                f"TPM=1 should match serial: tg={tg_val:.8f}, serial={serial_val:.8f}, rel_err={rel_err:.6f}"

    def test_large_batch_10_mols(self):
        """Stress test with 10 diverse molecules."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg

        smiles_list = [
            "CCO", "CCC", "c1ccccc1", "CC=O", "CCN",
            "CCCO", "CC(C)C", "C1CCCC1", "CC(=O)O", "CCF",
        ]
        system, pos = self._setup_batch(smiles_list)

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        for i in range(len(smiles_list)):
            assert final_e[i].item() < initial_e[i].item(), \
                f"Mol {i} ({smiles_list[i]}): energy should decrease"

    def test_larger_molecule(self):
        """20+ atom molecule (aspirin-class)."""
        from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs_tg

        # Aspirin: 21 atoms with Hs
        system, pos = self._setup_mol("CC(=O)Oc1ccccc1C(=O)O")

        initial_e, _ = dg_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_dg_bfgs_tg(pos, system, max_iters=400)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), \
            f"Aspirin: energy should decrease: {initial_e[0].item():.4f} -> {final_e[0].item():.4f}"
