"""Tests for stereochemistry check stages and tetrahedral atom extraction."""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem

from mlxmolkit.pipeline.context import create_pipeline_context
from mlxmolkit.pipeline.stage_coordgen import stage_coordgen
from mlxmolkit.pipeline.stage_distgeom_minimize import stage_distgeom_minimize
from mlxmolkit.pipeline.stage_stereochem_checks import (
    _same_side,
    stage_first_chiral_check,
    stage_tetrahedral_check,
)
from mlxmolkit.preprocessing.rdkit_extract import extract_tetrahedral_atoms


# ---------------------
# Tetrahedral Atom Extraction
# ---------------------


class TestExtractTetrahedralAtoms:
    def test_ethanol_two_carbons(self, ethanol_mol):
        """Ethanol has 2 sp3 carbons, each with 4 neighbors after H addition."""
        terms = extract_tetrahedral_atoms(ethanol_mol)
        assert len(terms.idx0) == 2
        # Both should have 4 neighbors (idx4 != idx0)
        for i in range(2):
            assert terms.idx0[i] != terms.idx4[i]

    def test_benzene_no_tetrahedral(self, benzene_mol):
        """Benzene has no sp3 atoms — all carbons are sp2."""
        terms = extract_tetrahedral_atoms(benzene_mol)
        assert len(terms.idx0) == 0

    def test_propane_three_carbons(self, propane_mol):
        """Propane has 3 sp3 carbons after H addition."""
        terms = extract_tetrahedral_atoms(propane_mol)
        assert len(terms.idx0) == 3

    def test_chiral_mol_includes_center(self, chiral_mol):
        """Chiral molecule should have tetrahedral atoms including the chiral center."""
        terms = extract_tetrahedral_atoms(chiral_mol)
        assert len(terms.idx0) >= 1

    def test_methane(self):
        """Methane (CH4): single carbon with 4 H neighbors."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 1
        assert terms.idx0[0] == 0  # Carbon is atom 0

    def test_ethylene_no_tetrahedral(self):
        """Ethylene (C=C) has sp2 carbons — no tetrahedral atoms."""
        mol = Chem.MolFromSmiles("C=C")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        # sp2 carbons have 3 neighbors, not 4
        assert len(terms.idx0) == 0

    def test_neighbor_indices_sorted(self, ethanol_mol):
        """Neighbor indices should be sorted for each term."""
        terms = extract_tetrahedral_atoms(ethanol_mol)
        for i in range(len(terms.idx0)):
            neighbors = [terms.idx1[i], terms.idx2[i], terms.idx3[i]]
            assert neighbors == sorted(neighbors)

    def test_fused_ring_detection(self):
        """Atoms in 3-membered rings should have in_fused_small_rings=True."""
        # Cyclopropane: 3-membered ring
        mol = Chem.MolFromSmiles("C1CC1")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 3
        # All carbons are in a 3-membered ring
        assert all(terms.in_fused_small_rings)

    def test_cyclohexane_not_fused(self):
        """Cyclohexane carbons should not be in fused small rings."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 6
        assert not any(terms.in_fused_small_rings)


# ---------------------
# _sameSide Helper
# ---------------------


class TestSameSide:
    def test_same_side(self):
        """Points on same side of a plane should return True."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([1, 0, 0], dtype=np.float64)
        v3 = np.array([0, 1, 0], dtype=np.float64)
        v4 = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        p0 = np.array([0.5, 0.5, 2.0], dtype=np.float64)
        assert _same_side(v1, v2, v3, v4, p0, 0.3)

    def test_opposite_side(self):
        """Points on opposite sides should return False."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([1, 0, 0], dtype=np.float64)
        v3 = np.array([0, 1, 0], dtype=np.float64)
        v4 = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        p0 = np.array([0.5, 0.5, -2.0], dtype=np.float64)
        assert not _same_side(v1, v2, v3, v4, p0, 0.3)

    def test_on_plane_fails(self):
        """Point exactly on the plane should return False (within tolerance)."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([1, 0, 0], dtype=np.float64)
        v3 = np.array([0, 1, 0], dtype=np.float64)
        v4 = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        p0 = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        assert not _same_side(v1, v2, v3, v4, p0, 0.3)

    def test_near_plane_within_tolerance(self):
        """Point very close to the plane (< tol) should return False."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([1, 0, 0], dtype=np.float64)
        v3 = np.array([0, 1, 0], dtype=np.float64)
        v4 = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        p0 = np.array([0.5, 0.5, 0.1], dtype=np.float64)  # Close to plane
        assert not _same_side(v1, v2, v3, v4, p0, 0.3)

    def test_reference_point_on_plane(self):
        """If reference point (v4) is on the plane, should return False."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([1, 0, 0], dtype=np.float64)
        v3 = np.array([0, 1, 0], dtype=np.float64)
        v4 = np.array([0.5, 0.5, 0.0], dtype=np.float64)  # On plane
        p0 = np.array([0.5, 0.5, 2.0], dtype=np.float64)
        assert not _same_side(v1, v2, v3, v4, p0, 0.3)


# ---------------------
# Tetrahedral Check Stage
# ---------------------


class TestTetrahedralCheckStage:
    def test_perfect_tetrahedron_passes(self):
        """A perfect tetrahedral geometry should pass."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])

        # Regular tetrahedron: center at origin, vertices at unit distance
        # Use coordinates that form a regular tetrahedron
        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        # Carbon at origin
        positions[0:4] = [0, 0, 0, 0]
        # H1
        positions[4:8] = [1.0, 1.0, 1.0, 0]
        # H2
        positions[8:12] = [1.0, -1.0, -1.0, 0]
        # H3
        positions[12:16] = [-1.0, 1.0, -1.0, 0]
        # H4
        positions[16:20] = [-1.0, -1.0, 1.0, 0]

        ctx.positions = mx.array(positions)
        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]

    def test_planar_geometry_fails(self):
        """A planar arrangement of neighbors should fail the volume test."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])

        # All atoms in the xy-plane (flat, non-tetrahedral)
        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        positions[0:4] = [0, 0, 0, 0]
        positions[4:8] = [1, 0, 0, 0]
        positions[8:12] = [0, 1, 0, 0]
        positions[12:16] = [-1, 0, 0, 0]
        positions[16:20] = [0, -1, 0, 0]

        ctx.positions = mx.array(positions)
        stage_tetrahedral_check(ctx, tol=0.3)
        assert ctx.failed[0]

    def test_minimized_ethanol_passes(self, ethanol_mol):
        """DG-minimized ethanol should pass the tetrahedral check."""
        ctx = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx, seed=42)
        stage_distgeom_minimize(ctx, 1.0, 0.1, 400, False)
        mx.eval(ctx.positions)

        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]

    def test_benzene_no_check_needed(self, benzene_mol):
        """Benzene has no tetrahedral atoms — check should trivially pass."""
        ctx = create_pipeline_context([benzene_mol])
        stage_coordgen(ctx, seed=42)
        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]

    def test_inactive_molecule_skipped(self):
        """Inactive molecules should not be checked."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])

        # Planar (would fail) but inactive
        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        positions[0:4] = [0, 0, 0, 0]
        positions[4:8] = [1, 0, 0, 0]
        positions[8:12] = [0, 1, 0, 0]
        positions[12:16] = [-1, 0, 0, 0]
        positions[16:20] = [0, -1, 0, 0]

        ctx.positions = mx.array(positions)
        ctx.active[0] = False

        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]  # Not checked because inactive


# ---------------------
# First Chiral Check Stage
# ---------------------


class TestFirstChiralCheckStage:
    def test_no_chiral_centers_passes(self, ethanol_mol):
        """Ethanol has no chiral centers — check should trivially pass."""
        ctx = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx, seed=42)
        stage_first_chiral_check(ctx)
        assert not ctx.failed[0]

    def test_chiral_mol_after_minimize(self, chiral_mol):
        """A minimized chiral molecule should be checked without crashing."""
        ctx = create_pipeline_context([chiral_mol])
        stage_coordgen(ctx, seed=42)
        stage_distgeom_minimize(ctx, 1.0, 0.1, 400, False)
        mx.eval(ctx.positions)

        # This should not crash — the result depends on the minimization
        stage_first_chiral_check(ctx)

    def test_inactive_molecule_skipped(self, chiral_mol):
        """Inactive molecules should not be checked."""
        ctx = create_pipeline_context([chiral_mol])
        stage_coordgen(ctx, seed=42)
        ctx.active[0] = False

        stage_first_chiral_check(ctx)
        assert not ctx.failed[0]

    def test_batched_chiral_check(self, ethanol_mol, chiral_mol):
        """Batch of molecules with and without chiral centers."""
        ctx = create_pipeline_context([ethanol_mol, chiral_mol])
        stage_coordgen(ctx, seed=42)
        stage_distgeom_minimize(ctx, 1.0, 0.1, 400, False)
        mx.eval(ctx.positions)

        stage_first_chiral_check(ctx)
        # Ethanol should pass (no chiral centers)
        assert not ctx.failed[0]
