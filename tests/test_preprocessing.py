"""Unit tests for mlxmolkit.preprocessing."""

import mlx.core as mx
import numpy as np
import numpy.testing as npt
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.preprocessing.rdkit_extract import (
    DGParams,
    extract_dg_params,
    get_bounds_matrix,
)
from mlxmolkit.preprocessing.batching import batch_dg_params


class TestGetBoundsMatrix:
    def test_ethanol(self, ethanol_mol):
        bm = get_bounds_matrix(ethanol_mol)
        n = ethanol_mol.GetNumAtoms()
        assert bm.shape == (n, n)
        # Diagonal should be zero
        npt.assert_allclose(np.diag(bm), 0.0, atol=1e-10)
        # Upper triangle values should be >= lower triangle values
        for i in range(n):
            for j in range(i + 1, n):
                ub = bm[i, j]
                lb = bm[j, i]
                assert ub >= lb, f"Upper bound < lower bound for ({i},{j})"

    def test_benzene(self, benzene_mol):
        bm = get_bounds_matrix(benzene_mol)
        n = benzene_mol.GetNumAtoms()
        assert bm.shape == (n, n)


class TestExtractDGParams:
    def test_ethanol_distance_terms(self, ethanol_mol):
        params = extract_dg_params(ethanol_mol, dim=4)

        assert params.num_atoms == ethanol_mol.GetNumAtoms()
        assert len(params.dist_terms.idx1) > 0
        assert len(params.dist_terms.idx1) == len(params.dist_terms.idx2)
        assert len(params.dist_terms.idx1) == len(params.dist_terms.lb2)
        assert len(params.dist_terms.idx1) == len(params.dist_terms.ub2)
        assert len(params.dist_terms.idx1) == len(params.dist_terms.weight)

        # All indices should be valid atom indices
        assert np.all(params.dist_terms.idx1 >= 0)
        assert np.all(params.dist_terms.idx1 < params.num_atoms)
        assert np.all(params.dist_terms.idx2 >= 0)
        assert np.all(params.dist_terms.idx2 < params.num_atoms)

        # idx1 > idx2 (upper triangle convention)
        assert np.all(params.dist_terms.idx1 > params.dist_terms.idx2)

        # Bounds should be positive
        assert np.all(params.dist_terms.lb2 >= 0)
        assert np.all(params.dist_terms.ub2 > 0)

        # Upper bound squared >= lower bound squared
        assert np.all(params.dist_terms.ub2 >= params.dist_terms.lb2)

        # Weights should be positive
        assert np.all(params.dist_terms.weight > 0)

    def test_ethanol_fourth_dim_terms(self, ethanol_mol):
        params = extract_dg_params(ethanol_mol, dim=4)
        assert len(params.fourth_dim_terms.idx) == params.num_atoms
        npt.assert_array_equal(
            params.fourth_dim_terms.idx, np.arange(params.num_atoms)
        )

    def test_ethanol_no_fourth_dim_in_3d(self, ethanol_mol):
        params = extract_dg_params(ethanol_mol, dim=3)
        assert len(params.fourth_dim_terms.idx) == 0

    def test_benzene_has_distance_terms(self, benzene_mol):
        params = extract_dg_params(benzene_mol, dim=4)
        assert len(params.dist_terms.idx1) > 0

    def test_chiral_mol_has_chiral_terms(self, chiral_mol):
        params = extract_dg_params(chiral_mol, dim=4)
        # Should have at least one chiral term
        assert len(params.chiral_terms.idx1) > 0
        # Volume bounds should be set
        assert len(params.chiral_terms.vol_lower) == len(params.chiral_terms.idx1)
        assert len(params.chiral_terms.vol_upper) == len(params.chiral_terms.idx1)

    def test_basin_size_filter(self, ethanol_mol):
        # Very small basin size should include fewer terms
        params_strict = extract_dg_params(ethanol_mol, dim=4, basin_size_tol=0.5)
        params_loose = extract_dg_params(ethanol_mol, dim=4, basin_size_tol=10.0)
        assert len(params_strict.dist_terms.idx1) <= len(params_loose.dist_terms.idx1)


class TestBatching:
    def test_single_molecule(self, ethanol_mol):
        params = extract_dg_params(ethanol_mol, dim=4)
        batched = batch_dg_params([params], dim=4)

        assert batched.n_mols == 1
        assert batched.dim == 4
        assert batched.n_atoms_total == params.num_atoms
        assert np.array(batched.atom_starts).tolist() == [0, params.num_atoms]

        # Distance terms should match
        assert batched.dist_idx1.size == len(params.dist_terms.idx1)

    def test_two_molecules(self, ethanol_mol, benzene_mol):
        p1 = extract_dg_params(ethanol_mol, dim=4)
        p2 = extract_dg_params(benzene_mol, dim=4)
        batched = batch_dg_params([p1, p2], dim=4)

        assert batched.n_mols == 2
        assert batched.n_atoms_total == p1.num_atoms + p2.num_atoms

        # Atom starts should reflect cumulative sum
        atom_starts = np.array(batched.atom_starts)
        assert atom_starts[0] == 0
        assert atom_starts[1] == p1.num_atoms
        assert atom_starts[2] == p1.num_atoms + p2.num_atoms

        # Total distance terms should be sum
        total_dist = len(p1.dist_terms.idx1) + len(p2.dist_terms.idx1)
        assert batched.dist_idx1.size == total_dist

        # Second molecule's indices should be offset
        n1_dist = len(p1.dist_terms.idx1)
        if n1_dist > 0:
            # First molecule: indices should be in [0, p1.num_atoms)
            first_max_idx = int(mx.max(batched.dist_idx1[:n1_dist]))
            assert first_max_idx < p1.num_atoms

        if len(p2.dist_terms.idx1) > 0:
            # Second molecule: indices should be in [p1.num_atoms, total)
            second_min_idx = int(mx.min(batched.dist_idx1[n1_dist:]))
            assert second_min_idx >= p1.num_atoms

    def test_term_starts_are_csr(self, ethanol_mol, benzene_mol):
        p1 = extract_dg_params(ethanol_mol, dim=4)
        p2 = extract_dg_params(benzene_mol, dim=4)
        batched = batch_dg_params([p1, p2], dim=4)

        dist_starts = np.array(batched.dist_term_starts)
        assert dist_starts[0] == 0
        assert dist_starts[1] == len(p1.dist_terms.idx1)
        assert dist_starts[2] == len(p1.dist_terms.idx1) + len(p2.dist_terms.idx1)

    def test_mol_indices_are_correct(self, ethanol_mol, benzene_mol):
        p1 = extract_dg_params(ethanol_mol, dim=4)
        p2 = extract_dg_params(benzene_mol, dim=4)
        batched = batch_dg_params([p1, p2], dim=4)

        mol_idx = np.array(batched.dist_mol_indices)
        n1 = len(p1.dist_terms.idx1)
        n2 = len(p2.dist_terms.idx1)

        if n1 > 0:
            assert np.all(mol_idx[:n1] == 0)
        if n2 > 0:
            assert np.all(mol_idx[n1 : n1 + n2] == 1)
