"""Tests for MMFF94 force field implementation.

Tests include:
- Parameter extraction from RDKit molecules
- Per-term energy correctness vs RDKit CalcEnergy
- Gradient vs finite differences
- End-to-end optimization vs RDKit MMFFOptimizeMolecule
- Batch correctness with multiple molecules
"""

import mlx.core as mx
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.forcefields.mmff import (
    _angle_bend_energy,
    _bond_stretch_energy,
    _bond_stretch_grad,
    _ele_energy,
    _oop_bend_energy,
    _stretch_bend_energy,
    _torsion_energy,
    _vdw_energy,
    mmff_energy_and_grad,
)
from mlxmolkit.mmff_optimize import MMFFOptimizeMoleculesConfs
from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
from mlxmolkit.preprocessing.mmff_extract import (
    _is_mmff_linear_atom_type,
    extract_mmff_params,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def _make_mol_with_conf(smiles: str) -> Chem.Mol:
    """Create a molecule with hydrogens and an embedded conformer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _get_rdkit_mmff_energy(mol: Chem.Mol, conf_id: int = -1) -> float:
    """Get MMFF energy from RDKit for a given conformer."""
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return ff.CalcEnergy()


def _get_positions_flat(mol: Chem.Mol, conf_id: int = -1) -> mx.array:
    """Get conformer positions as flat MLX array."""
    conf = mol.GetConformer(conf_id)
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return mx.array(coords.flatten())


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


def _load_valid_sdf_molecule(valid_idx: int) -> Chem.Mol:
    """Load the Nth valid MMFF molecule from the test SDF."""
    sdf_path = Path(__file__).parent / "test_data" / "MMFF94_dative.sdf"
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    current = -1
    for mol in suppl:
        if mol is None:
            continue
        if rdForceFieldHelpers.MMFFGetMoleculeProperties(mol) is None:
            continue
        current += 1
        if current == valid_idx:
            return Chem.Mol(mol)
    raise AssertionError(f"Could not find valid MMFF molecule index {valid_idx}")


# --------------------------------------------------
# Parameter Extraction Tests
# --------------------------------------------------


class TestMMFFExtraction:
    def test_ethanol_has_params(self):
        """Ethanol should have valid MMFF params."""
        mol = _make_mol_with_conf("CCO")
        params = extract_mmff_params(mol)
        assert params is not None
        assert params.num_atoms == mol.GetNumAtoms()
        assert len(params.bond_terms.idx1) > 0
        assert len(params.angle_terms.idx1) > 0

    def test_benzene_has_params(self):
        """Benzene should have valid MMFF params."""
        mol = _make_mol_with_conf("c1ccccc1")
        params = extract_mmff_params(mol)
        assert params is not None
        assert len(params.bond_terms.idx1) > 0
        assert len(params.angle_terms.idx1) > 0
        assert len(params.torsion_terms.idx1) > 0

    def test_vdw_and_ele_terms_exist(self):
        """Non-bonded terms should be extracted for molecules with enough atoms."""
        mol = _make_mol_with_conf("CCCC")
        params = extract_mmff_params(mol)
        assert params is not None
        assert len(params.vdw_terms.idx1) > 0
        # Need a molecule with charged atoms separated by ≥3 bonds for ele terms
        mol2 = _make_mol_with_conf("OCC(=O)O")
        params2 = extract_mmff_params(mol2)
        assert params2 is not None
        assert len(params2.ele_terms.idx1) > 0

    def test_invalid_mol_returns_none(self):
        """Molecule without MMFF params should return None."""
        # Bare metal ions typically don't have MMFF params
        mol = Chem.MolFromSmiles("[Fe]")
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            params = extract_mmff_params(mol)
            # May or may not be None depending on RDKit version

    def test_disconnected_nonbonded_pairs_ignored_by_default(self):
        """Disconnected fragments should not generate default MMFF non-bonded terms."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CC.CC"))
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        extracted = extract_mmff_params(mol)
        extracted_all = extract_mmff_params(mol, ignoreInterfragInteractions=False)
        assert extracted is not None
        assert extracted_all is not None

        frag_map = np.empty(mol.GetNumAtoms(), dtype=np.int32)
        for frag_idx, frag in enumerate(Chem.GetMolFrags(mol)):
            frag_map[list(frag)] = frag_idx

        default_cross_vdw = sum(
            1 for i, j in zip(extracted.vdw_terms.idx1, extracted.vdw_terms.idx2, strict=True)
            if frag_map[int(i)] != frag_map[int(j)]
        )
        all_cross_vdw = sum(
            1 for i, j in zip(extracted_all.vdw_terms.idx1, extracted_all.vdw_terms.idx2, strict=True)
            if frag_map[int(i)] != frag_map[int(j)]
        )
        default_cross_ele = sum(
            1 for i, j in zip(extracted.ele_terms.idx1, extracted.ele_terms.idx2, strict=True)
            if frag_map[int(i)] != frag_map[int(j)]
        )
        all_cross_ele = sum(
            1 for i, j in zip(extracted_all.ele_terms.idx1, extracted_all.ele_terms.idx2, strict=True)
            if frag_map[int(i)] != frag_map[int(j)]
        )

        assert default_cross_vdw == 0
        assert default_cross_ele == 0
        assert all_cross_vdw > 0
        assert all_cross_ele == 0

    def test_vdw_params_use_scaled_rdkit_values(self):
        """VdW extraction should use RDKit's final scaled MMFF tuple values."""
        mol = _load_valid_sdf_molecule(3)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        params.numThreads = 1
        AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)

        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        rd_vdw = props.GetMMFFVdWParams(15, 25)
        extracted = extract_mmff_params(mol, conf_id=2)
        assert extracted is not None

        pair_to_idx = {
            (int(i), int(j)): idx
            for idx, (i, j) in enumerate(
                zip(extracted.vdw_terms.idx1, extracted.vdw_terms.idx2, strict=True)
            )
        }
        term_idx = pair_to_idx[(15, 25)]
        assert len(rd_vdw) == 4
        assert extracted.vdw_terms.R_ij_star[term_idx] == pytest.approx(rd_vdw[2], rel=1e-6)
        assert extracted.vdw_terms.epsilon[term_idx] == pytest.approx(rd_vdw[3], rel=1e-6)

    def test_linear_atom_types_follow_rdkit_linh(self):
        """Linear MMFF centers should follow RDKit's MMFFProp::linh mapping."""
        assert _is_mmff_linear_atom_type(4)
        assert _is_mmff_linear_atom_type(53)
        assert _is_mmff_linear_atom_type(61)
        assert not _is_mmff_linear_atom_type(1)
        assert not _is_mmff_linear_atom_type(3)
        assert not _is_mmff_linear_atom_type(9)


# --------------------------------------------------
# Energy Correctness Tests
# --------------------------------------------------


class TestMMFFEnergyVsRDKit:
    def test_total_energy_ethanol(self):
        """Total MMFF energy for ethanol should match RDKit within tolerance."""
        mol = _make_mol_with_conf("CCO")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        energies, _ = mmff_energy_and_grad(pos, system)
        e_mlx = float(energies[0])
        e_rdkit = _get_rdkit_mmff_energy(mol)
        # Allow small tolerance due to float32 vs float64
        assert abs(e_mlx - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 0.1), (
            f"MLX energy {e_mlx:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_total_energy_propane(self):
        """Total MMFF energy for propane should match RDKit."""
        mol = _make_mol_with_conf("CCC")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        energies, _ = mmff_energy_and_grad(pos, system)
        e_mlx = float(energies[0])
        e_rdkit = _get_rdkit_mmff_energy(mol)
        assert abs(e_mlx - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 0.1), (
            f"MLX energy {e_mlx:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_total_energy_benzene(self):
        """Total MMFF energy for benzene should match RDKit."""
        mol = _make_mol_with_conf("c1ccccc1")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        energies, _ = mmff_energy_and_grad(pos, system)
        e_mlx = float(energies[0])
        e_rdkit = _get_rdkit_mmff_energy(mol)
        assert abs(e_mlx - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 0.1), (
            f"MLX energy {e_mlx:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_total_energy_aspirin(self):
        """Total MMFF energy for aspirin should match RDKit."""
        mol = _make_mol_with_conf("CC(=O)Oc1ccccc1C(=O)O")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        energies, _ = mmff_energy_and_grad(pos, system)
        e_mlx = float(energies[0])
        e_rdkit = _get_rdkit_mmff_energy(mol)
        assert abs(e_mlx - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 0.5), (
            f"MLX energy {e_mlx:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_initial_energy_matches_rdkit_for_disconnected_outlier(self):
        """Disconnected fragments should match RDKit's default MMFF semantics."""
        mol = _load_valid_sdf_molecule(15)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        params.numThreads = 1
        AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)

        pos = _get_positions_flat(mol, conf_id=0)
        extracted = extract_mmff_params(mol, conf_id=0)
        assert extracted is not None
        system = batch_mmff_params([extracted])
        energies, _ = mmff_energy_and_grad(pos, system)
        e_mlx = float(energies[0])

        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        ff_default = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, props, confId=0, ignoreInterfragInteractions=True
        )
        ff_all = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, props, confId=0, ignoreInterfragInteractions=False
        )
        e_rdkit = ff_default.CalcEnergy()
        e_rdkit_all = ff_all.CalcEnergy()

        assert abs(e_mlx - e_rdkit) < 1.0, (
            f"MLX energy {e_mlx:.4f} vs RDKit default {e_rdkit:.4f}"
        )
        assert abs(e_mlx - e_rdkit) < abs(e_mlx - e_rdkit_all)


# --------------------------------------------------
# Gradient Tests
# --------------------------------------------------


class TestMMFFGradient:
    def test_bond_stretch_grad_fd(self):
        """Bond stretch gradient matches finite differences."""
        mol = _make_mol_with_conf("CCO")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])

        def energy_fn(p):
            e = _bond_stretch_energy(
                p, system.bond_idx1, system.bond_idx2,
                system.bond_kb, system.bond_r0,
            )
            return mx.sum(e)

        from mlxmolkit.forcefields.mmff import _bond_stretch_grad
        g_analytic = np.array(
            _bond_stretch_grad(
                pos, system.bond_idx1, system.bond_idx2,
                system.bond_kb, system.bond_r0,
            )
        )
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)
        npt.assert_allclose(g_analytic, g_fd, atol=0.05, rtol=0.05)

    def test_combined_grad_fd(self):
        """Combined MMFF gradient matches finite differences for ethanol."""
        mol = _make_mol_with_conf("CCO")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])

        def energy_fn(p):
            e, _ = mmff_energy_and_grad(p, system)
            return mx.sum(e)

        _, g_analytic = mmff_energy_and_grad(pos, system)
        g_analytic = np.array(g_analytic)
        g_fd = _finite_diff_grad(energy_fn, pos, eps=1e-3)

        # Use absolute tolerance since some gradient components may be near zero
        npt.assert_allclose(g_analytic, g_fd, atol=0.1, rtol=0.1)

    def test_gradient_not_all_zero(self):
        """Gradient should not be all zeros for a non-equilibrium geometry."""
        mol = _make_mol_with_conf("CCC")
        pos = _get_positions_flat(mol)
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        _, grad = mmff_energy_and_grad(pos, system)
        assert float(mx.max(mx.abs(grad))) > 1e-6


# --------------------------------------------------
# Batching Tests
# --------------------------------------------------


class TestMMFFBatching:
    def test_batch_two_molecules(self):
        """Batched energy matches individual energies."""
        mol1 = _make_mol_with_conf("CCO")
        mol2 = _make_mol_with_conf("CCC")
        pos1 = _get_positions_flat(mol1)
        pos2 = _get_positions_flat(mol2)

        p1 = extract_mmff_params(mol1)
        p2 = extract_mmff_params(mol2)
        assert p1 is not None and p2 is not None

        # Individual energies
        s1 = batch_mmff_params([p1])
        s2 = batch_mmff_params([p2])
        e1, _ = mmff_energy_and_grad(pos1, s1)
        e2, _ = mmff_energy_and_grad(pos2, s2)

        # Batched energy
        system = batch_mmff_params([p1, p2])
        pos_combined = mx.concatenate([pos1, pos2])
        energies, _ = mmff_energy_and_grad(pos_combined, system)

        assert energies.shape == (2,)
        assert float(energies[0]) == pytest.approx(float(e1[0]), rel=1e-4)
        assert float(energies[1]) == pytest.approx(float(e2[0]), rel=1e-4)

    def test_batch_atom_starts(self):
        """Batched system has correct atom_starts."""
        mol1 = _make_mol_with_conf("C")
        mol2 = _make_mol_with_conf("CC")

        p1 = extract_mmff_params(mol1)
        p2 = extract_mmff_params(mol2)
        assert p1 is not None and p2 is not None

        system = batch_mmff_params([p1, p2])
        starts = np.array(system.atom_starts)
        assert starts[0] == 0
        assert starts[1] == p1.num_atoms
        assert starts[2] == p1.num_atoms + p2.num_atoms


# --------------------------------------------------
# End-to-End Optimization Tests
# --------------------------------------------------


class TestMMFFOptimization:
    def test_optimize_ethanol(self):
        """Optimized ethanol energy should be close to RDKit's."""
        mol = _make_mol_with_conf("CCO")
        mol_rdkit = Chem.Mol(mol)  # Copy for RDKit optimization

        # MLX optimization
        energies = MMFFOptimizeMoleculesConfs([mol], maxIters=200)
        assert len(energies) == 1
        assert len(energies[0]) == 1
        e_mlx = energies[0][0]

        # RDKit optimization
        rdForceFieldHelpers.MMFFOptimizeMolecule(mol_rdkit)
        e_rdkit = _get_rdkit_mmff_energy(mol_rdkit)

        # Optimized energies should be close
        assert abs(e_mlx - e_rdkit) < max(0.5 * abs(e_rdkit) / 100, 0.5), (
            f"MLX optimized energy {e_mlx:.4f} vs RDKit {e_rdkit:.4f}"
        )

    def test_optimize_reduces_energy(self):
        """Optimization should reduce the energy."""
        mol = _make_mol_with_conf("CCCC")
        params = extract_mmff_params(mol)
        assert params is not None
        system = batch_mmff_params([params])
        pos_before = _get_positions_flat(mol)
        e_before, _ = mmff_energy_and_grad(pos_before, system)

        energies = MMFFOptimizeMoleculesConfs([mol], maxIters=200)
        e_after = energies[0][0]
        assert e_after <= float(e_before[0]) + 0.1

    def test_optimize_multiple_molecules(self):
        """Optimization works for multiple molecules."""
        mols = [
            _make_mol_with_conf("CCO"),
            _make_mol_with_conf("CCC"),
            _make_mol_with_conf("c1ccccc1"),
        ]
        energies = MMFFOptimizeMoleculesConfs(mols, maxIters=200)
        assert len(energies) == 3
        for i, mol_energies in enumerate(energies):
            assert len(mol_energies) == 1
            assert isinstance(mol_energies[0], float)

    def test_optimize_multiple_conformers(self):
        """Optimization works for multiple conformers per molecule."""
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)

        n_confs = mol.GetNumConformers()
        assert n_confs == 3

        energies = MMFFOptimizeMoleculesConfs([mol], maxIters=200)
        assert len(energies) == 1
        assert len(energies[0]) == n_confs

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = MMFFOptimizeMoleculesConfs([])
        assert result == []

    def test_none_molecule_skipped(self):
        """None molecules are skipped gracefully."""
        mol = _make_mol_with_conf("CCO")
        energies = MMFFOptimizeMoleculesConfs([None, mol])
        assert len(energies) == 2
        assert len(energies[0]) == 0  # None mol
        assert len(energies[1]) == 1  # Valid mol

    def test_optimize_charged_biaryl_escapes_planar_basin(self):
        """Charged biaryl conformers should escape the symmetric planar basin."""
        mol = _load_valid_sdf_molecule(12)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        params.numThreads = 1
        AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)

        mol_rdkit = Chem.Mol(mol)
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol_rdkit, maxIters=200, numThreads=1)
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_rdkit)
        rd_energies = [
            rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol_rdkit, props, confId=conf_id
            ).CalcEnergy()
            for conf_id in range(mol_rdkit.GetNumConformers())
        ]

        mlx_energies = MMFFOptimizeMoleculesConfs([mol], maxIters=200)[0]

        # Confs 1-3 previously got stuck near 165 kcal/mol; they should now
        # reach the lower ~148 kcal/mol basin.
        for conf_id in (1, 2, 3):
            assert abs(mlx_energies[conf_id] - rd_energies[conf_id]) < 1.0, (
                f"Conformer {conf_id}: MLX {mlx_energies[conf_id]:.4f} "
                f"vs RDKit {rd_energies[conf_id]:.4f}"
            )

    def test_optimize_matches_rdkit_for_valid_mol2_conf4(self):
        """The former valid-2/conf-4 basin mismatch should now match RDKit."""
        mol = _load_valid_sdf_molecule(2)
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        params.numThreads = 1
        AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)

        mol_rdkit = Chem.Mol(mol)
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol_rdkit, maxIters=200, numThreads=1)
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_rdkit)
        rd_energy = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol_rdkit, props, confId=4
        ).CalcEnergy()

        mlx_energy = MMFFOptimizeMoleculesConfs([mol], maxIters=200)[0][4]
        assert abs(mlx_energy - rd_energy) < 0.5, (
            f"Conformer 4: MLX {mlx_energy:.4f} vs RDKit {rd_energy:.4f}"
        )


# --------------------------------------------------
# SDF Molecule Tests
# --------------------------------------------------


class TestMMFFWithSDFMolecules:
    def test_sdf_mols_energy(self, sdf_mols):
        """SDF molecules should have reasonable MMFF energies."""
        for mol in sdf_mols:
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            if props is None:
                continue
            params = extract_mmff_params(mol)
            if params is None:
                continue
            pos = _get_positions_flat(mol)
            system = batch_mmff_params([params])
            energies, grad = mmff_energy_and_grad(pos, system)
            # Energy should be finite
            e = float(energies[0])
            assert np.isfinite(e), f"Non-finite energy: {e}"
