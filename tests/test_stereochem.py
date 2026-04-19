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
    stage_chiral_volume_check,
    stage_first_chiral_check,
    stage_tetrahedral_check,
)
from mlxmolkit.preprocessing.rdkit_extract import (
    extract_chiral_center_terms,
    extract_dg_params,
    extract_tetrahedral_atoms,
)


# ---------------------
# Tetrahedral Atom Extraction
# ---------------------


class TestExtractTetrahedralAtoms:
    def test_ethanol_no_tetrahedral_terms(self, ethanol_mol):
        """Acyclic sp3 atoms are not nvMolKit tetrahedral check terms."""
        terms = extract_tetrahedral_atoms(ethanol_mol)
        assert len(terms.idx0) == 0

    def test_benzene_no_tetrahedral(self, benzene_mol):
        """Benzene has no sp3 atoms — all carbons are sp2."""
        terms = extract_tetrahedral_atoms(benzene_mol)
        assert len(terms.idx0) == 0

    def test_propane_no_tetrahedral_terms(self, propane_mol):
        """Acyclic sp3 carbons are not checked as tetrahedralCarbons."""
        terms = extract_tetrahedral_atoms(propane_mol)
        assert len(terms.idx0) == 0

    def test_explicit_chiral_mol_not_tetrahedral_data(self, chiral_mol):
        """Explicit chiral centers are stored in separate chiral-center data."""
        terms = extract_tetrahedral_atoms(chiral_mol)
        assert len(terms.idx0) == 0

    def test_methane_no_tetrahedral_terms(self):
        """Methane is acyclic and should not produce tetrahedral terms."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 0

    def test_ethylene_no_tetrahedral(self):
        """Ethylene (C=C) has sp2 carbons — no tetrahedral atoms."""
        mol = Chem.MolFromSmiles("C=C")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        # sp2 carbons have 3 neighbors, not 4
        assert len(terms.idx0) == 0

    def test_cyclopropane_excluded(self):
        """Atoms in 3-membered rings are excluded from tetrahedral data."""
        mol = Chem.MolFromSmiles("C1CC1")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 0

    def test_cyclohexane_no_tetrahedral_terms(self):
        """Simple single-ring atoms are not tetrahedralCarbons."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) == 0

    def test_fused_ring_positive(self):
        """Fused non-small-ring bridgehead atoms match nvMolKit extraction."""
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) > 0
        assert not any(terms.in_fused_small_rings)

    def test_fused_small_ring_flag(self):
        """Atoms in more than one ring smaller than 5 are flagged."""
        mol = Chem.MolFromSmiles("C1C2CC1C2")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) > 0
        assert any(terms.in_fused_small_rings)

    def test_preserves_rdkit_neighbor_order(self):
        """Neighbor order is RDKit iteration order, not sorted."""
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        mol = Chem.AddHs(mol)
        terms = extract_tetrahedral_atoms(mol)
        assert len(terms.idx0) > 0
        center = int(terms.idx0[0])
        rdkit_neighbors = [
            n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
        ]
        extracted = [
            int(terms.idx1[0]),
            int(terms.idx2[0]),
            int(terms.idx3[0]),
            int(terms.idx4[0]),
        ]
        assert extracted == rdkit_neighbors


class TestExtractChiralCenterTerms:
    def test_four_neighbor_center_uses_all_neighbors(self, chiral_mol):
        """4-neighbor chiral centers keep center in idx0 and all neighbors in idx1-4."""
        terms = extract_chiral_center_terms(chiral_mol)
        assert len(terms.idx0) == 1

        center = int(terms.idx0[0])
        expected_neighbors = {
            n.GetIdx() for n in chiral_mol.GetAtomWithIdx(center).GetNeighbors()
        }
        actual_neighbors = {
            int(terms.idx1[0]),
            int(terms.idx2[0]),
            int(terms.idx3[0]),
            int(terms.idx4[0]),
        }
        assert actual_neighbors == expected_neighbors
        assert center not in actual_neighbors

    def test_dg_chiral_bounds_are_fixed_constants(self, chiral_mol):
        """DG chiral bounds use nvMolKit/RDKit constants and signs."""
        params = extract_dg_params(chiral_mol)
        assert len(params.chiral_terms.idx1) == 1

        lb = float(params.chiral_terms.vol_lower[0])
        ub = float(params.chiral_terms.vol_upper[0])
        if lb > 0:
            assert (lb, ub) == (5.0, 100.0)
        else:
            assert (lb, ub) == (-100.0, -5.0)


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
    def test_methane_without_tetrahedral_terms_noops(self):
        """Methane has no nvMolKit tetrahedral terms, so the stage no-ops."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])

        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        positions[0:4] = [0, 0, 0, 0]
        positions[4:8] = [1, 0, 0, 0]
        positions[8:12] = [0, 1, 0, 0]
        positions[12:16] = [-1, 0, 0, 0]
        positions[16:20] = [0, -1, 0, 0]

        ctx.positions = mx.array(positions)
        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]

    def test_fused_ring_planar_geometry_fails(self):
        """A planar fused-ring tetrahedral term should fail the volume test."""
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])
        assert ctx.tet_data is not None

        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        for atom_idx in range(ctx.n_atoms_total):
            positions[atom_idx * ctx.dim:(atom_idx + 1) * ctx.dim] = [
                float(atom_idx),
                float(atom_idx % 3),
                0.0,
                0.0,
            ]

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
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        mol = Chem.AddHs(mol)
        ctx = create_pipeline_context([mol])

        # Planar (would fail) but inactive
        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        for atom_idx in range(ctx.n_atoms_total):
            positions[atom_idx * ctx.dim:(atom_idx + 1) * ctx.dim] = [
                float(atom_idx),
                float(atom_idx % 3),
                0.0,
                0.0,
            ]

        ctx.positions = mx.array(positions)
        ctx.active[0] = False

        stage_tetrahedral_check(ctx, tol=0.3)
        assert not ctx.failed[0]  # Not checked because inactive


class TestChiralVolumeCheckStage:
    def test_uses_explicit_chiral_center_data(self, chiral_mol):
        """Final volume check is driven by explicit chiral center terms only."""
        ctx = create_pipeline_context([chiral_mol])
        assert ctx.tet_data is None
        assert ctx.chiral_center_data is not None

        positions = np.zeros(ctx.n_atoms_total * ctx.dim, dtype=np.float32)
        for atom_idx in range(ctx.n_atoms_total):
            positions[atom_idx * ctx.dim:(atom_idx + 1) * ctx.dim] = [
                float(atom_idx),
                float(atom_idx % 3),
                0.0,
                0.0,
            ]

        ctx.positions = mx.array(positions)
        stage_chiral_volume_check(ctx)
        assert ctx.failed[0]


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
