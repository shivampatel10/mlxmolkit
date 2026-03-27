"""Tests for the Metal MMFF BFGS kernel.

Verifies Metal kernel produces energies matching pure-MLX MMFF within tolerance.
"""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.forcefields.mmff import mmff_energy_and_grad
from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
from mlxmolkit.preprocessing.mmff_extract import extract_mmff_params


def _has_metal_kernel():
    """Check if Metal MMFF kernel is available."""
    try:
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs
        return True
    except ImportError:
        return False


def _make_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _setup_single(smiles):
    mol = _make_mol(smiles)
    params = extract_mmff_params(mol)
    assert params is not None
    system = batch_mmff_params([params])
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    ).flatten()
    pos = mx.array(coords)
    return system, pos


def _setup_batch(smiles_list):
    params_list = []
    all_pos = []
    for smi in smiles_list:
        mol = _make_mol(smi)
        params = extract_mmff_params(mol)
        assert params is not None
        params_list.append(params)
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        ).flatten()
        all_pos.append(coords)
    system = batch_mmff_params(params_list)
    pos = mx.array(np.concatenate(all_pos))
    return system, pos


@pytest.mark.skipif(not _has_metal_kernel(), reason="Metal MMFF kernel not available")
class TestMetalMMFFBFGS:

    def test_single_mol_energy_decrease(self):
        """Metal kernel should decrease energy for ethanol."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        system, pos = _setup_single("CCO")
        initial_e, _ = mmff_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e, statuses)

        assert final_e[0].item() < initial_e[0].item(), (
            f"Metal energy should decrease: {initial_e[0].item():.4f} -> {final_e[0].item():.4f}"
        )

    def test_propane_energy_decrease(self):
        """Metal kernel should decrease energy for propane."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        system, pos = _setup_single("CCC")
        initial_e, _ = mmff_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, _ = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()

    def test_benzene_energy_decrease(self):
        """Metal kernel should decrease energy for benzene."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        system, pos = _setup_single("c1ccccc1")
        initial_e, _ = mmff_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, _ = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e)

        assert final_e[0].item() < initial_e[0].item()

    def test_metal_vs_rdkit_energy(self):
        """Metal-optimized energy should be close to RDKit-optimized energy."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        mol = _make_mol("CCO")
        mol_rdkit = Chem.Mol(mol)

        # Metal optimization
        system, pos = _setup_single("CCO")
        _, final_e, _ = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_e)
        e_metal = final_e[0].item()

        # RDKit optimization
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol_rdkit)
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_rdkit)
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_rdkit, props)
        e_rdkit = ff.CalcEnergy()

        assert abs(e_metal - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 1.0), (
            f"Metal {e_metal:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_batch_5_molecules(self):
        """Metal kernel should work for a batch of 5 molecules."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        smiles = ["CCO", "CCC", "CCCC", "CC(C)C", "c1ccccc1"]
        system, pos = _setup_batch(smiles)

        initial_e, _ = mmff_energy_and_grad(pos, system)
        mx.eval(initial_e)

        final_pos, final_e, statuses = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e, statuses)

        assert final_e.shape == (5,)
        for i in range(5):
            assert final_e[i].item() <= initial_e[i].item() + 0.1

    def test_batch_20_molecules(self):
        """Metal kernel should handle 20 molecules."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        smiles = ["CCO", "CCC", "CCCC", "CC(C)C", "c1ccccc1",
                   "CCN", "CC=O", "CC(O)C", "CCOCC", "CCCCCC",
                   "C1CCCCC1", "CC(=O)O", "CCF", "CCCl", "CCOC",
                   "CC(C)(C)C", "C1CCC1", "C1CCCC1", "CCCCC", "CC#N"]
        system, pos = _setup_batch(smiles)

        final_pos, final_e, statuses = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e, statuses)

        assert final_e.shape == (20,)
        for i in range(20):
            assert np.isfinite(final_e[i].item())

    def test_sdf_molecules(self, sdf_mols):
        """Metal kernel should work with SDF molecules."""
        from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs

        params_list = []
        all_pos = []
        for mol in sdf_mols:
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            if props is None:
                continue
            params = extract_mmff_params(mol)
            if params is None:
                continue
            conf = mol.GetConformer()
            coords = np.array(
                [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                dtype=np.float32,
            ).flatten()
            params_list.append(params)
            all_pos.append(coords)

        if not params_list:
            pytest.skip("No valid SDF molecules")

        system = batch_mmff_params(params_list)
        pos = mx.array(np.concatenate(all_pos))

        final_pos, final_e, _ = metal_mmff_bfgs(pos, system, max_iters=200)
        mx.eval(final_pos, final_e)

        for i in range(len(params_list)):
            assert np.isfinite(final_e[i].item())
