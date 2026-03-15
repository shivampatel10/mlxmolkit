"""Integration tests for pipeline stages 1-4.

Milestone: stages 1-4 produce stereochemically valid 3D coords for ethanol.
"""

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem

from mlxmolkit.pipeline.context import create_pipeline_context
from mlxmolkit.pipeline.driver import run_dg_pipeline
from mlxmolkit.pipeline.stage_coordgen import stage_coordgen
from mlxmolkit.pipeline.stage_distgeom_minimize import stage_distgeom_minimize
from mlxmolkit.pipeline.stage_stereochem_checks import (
    stage_first_chiral_check,
    stage_tetrahedral_check,
)


# ---------------------
# Pipeline Context Creation
# ---------------------


class TestCreatePipelineContext:
    def test_ethanol_context(self, ethanol_mol):
        """Context for ethanol should have correct dimensions."""
        ctx = create_pipeline_context([ethanol_mol])
        assert ctx.n_mols == 1
        assert ctx.dim == 4
        assert ctx.n_atoms_total == ethanol_mol.GetNumAtoms()
        assert len(ctx.atom_starts) == 2
        assert ctx.atom_starts[0] == 0
        assert ctx.atom_starts[1] == ethanol_mol.GetNumAtoms()
        assert ctx.positions.shape == (ctx.n_atoms_total * ctx.dim,)
        assert all(ctx.active)
        assert not any(ctx.failed)

    def test_multi_mol_context(self, ethanol_mol, propane_mol):
        """Context for multiple molecules has correct batching."""
        ctx = create_pipeline_context([ethanol_mol, propane_mol])
        assert ctx.n_mols == 2
        assert ctx.n_atoms_total == ethanol_mol.GetNumAtoms() + propane_mol.GetNumAtoms()
        assert len(ctx.atom_starts) == 3
        assert ctx.atom_starts[0] == 0
        assert ctx.atom_starts[1] == ethanol_mol.GetNumAtoms()
        assert ctx.atom_starts[2] == ctx.n_atoms_total

    def test_tetrahedral_data_present(self, ethanol_mol):
        """Ethanol should have tetrahedral check data."""
        ctx = create_pipeline_context([ethanol_mol])
        assert ctx.tet_data is not None
        # Ethanol has 2 sp3 carbons
        assert ctx.tet_data.idx0.shape[0] == 2

    def test_benzene_no_tetrahedral_data(self, benzene_mol):
        """Benzene should have no tetrahedral check data."""
        ctx = create_pipeline_context([benzene_mol])
        assert ctx.tet_data is None

    def test_collect_failures(self, ethanol_mol):
        """collect_failures should deactivate failed molecules."""
        ctx = create_pipeline_context([ethanol_mol])
        assert ctx.active[0]
        ctx.failed[0] = True
        ctx.collect_failures()
        assert not ctx.active[0]
        assert not ctx.failed[0]


# ---------------------
# Coordinate Generation Stage
# ---------------------


class TestCoordgenStage:
    def test_generates_nonzero_positions(self, ethanol_mol):
        """Coordgen should produce non-zero positions."""
        ctx = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx, seed=42)

        pos = np.array(ctx.positions)
        assert not np.allclose(pos, 0)

    def test_reproducible_with_seed(self, ethanol_mol):
        """Same seed should produce same positions."""
        ctx1 = create_pipeline_context([ethanol_mol])
        ctx2 = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx1, seed=42)
        stage_coordgen(ctx2, seed=42)

        np.testing.assert_array_equal(np.array(ctx1.positions), np.array(ctx2.positions))

    def test_different_seeds_differ(self, ethanol_mol):
        """Different seeds should produce different positions."""
        ctx1 = create_pipeline_context([ethanol_mol])
        ctx2 = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx1, seed=42)
        stage_coordgen(ctx2, seed=123)

        assert not np.allclose(np.array(ctx1.positions), np.array(ctx2.positions))


# ---------------------
# DG Minimization Stage
# ---------------------


class TestDistgeomMinimizeStage:
    def test_reduces_energy(self, ethanol_mol):
        """DG minimization should reduce energy."""
        ctx = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx, seed=42)

        # Compute initial energy
        from mlxmolkit.forcefields.dist_geom import dg_energy

        e_before = dg_energy(ctx.positions, ctx.dg_system, 1.0, 0.1)
        mx.eval(e_before)

        stage_distgeom_minimize(ctx, 1.0, 0.1, 400, False)

        e_after = dg_energy(ctx.positions, ctx.dg_system, 1.0, 0.1)
        mx.eval(e_after)

        assert e_after[0].item() < e_before[0].item()

    def test_energy_check_fails_bad_coords(self, ethanol_mol):
        """Extreme coordinates should fail the energy-per-atom check."""
        ctx = create_pipeline_context([ethanol_mol])
        # Use extreme random coords that won't minimize well in 1 iter
        stage_coordgen(ctx, seed=42, box_size_mult=100.0)
        stage_distgeom_minimize(ctx, 1.0, 0.1, max_iters=1, check_energy=True)
        # With only 1 BFGS iteration and extreme coords, likely to fail
        # (but not guaranteed — just check it doesn't crash)

    def test_fourth_dim_minimize_reduces_w(self, ethanol_mol):
        """Fourth dim minimization should push 4th coords toward zero."""
        ctx = create_pipeline_context([ethanol_mol])
        stage_coordgen(ctx, seed=42)
        stage_distgeom_minimize(ctx, 1.0, 0.1, 400, False)

        pos_4d_before = np.array(ctx.positions).reshape(-1, 4)[:, 3].copy()

        stage_distgeom_minimize(ctx, 0.2, 1.0, 200, False)

        pos_4d_after = np.array(ctx.positions).reshape(-1, 4)[:, 3]

        # 4th dimension should be smaller after fourth-dim-focused minimization
        assert np.mean(np.abs(pos_4d_after)) < np.mean(np.abs(pos_4d_before))


# ---------------------
# Full Pipeline Integration
# ---------------------


class TestFullPipeline:
    def test_ethanol_passes_pipeline(self, ethanol_mol):
        """Ethanol should pass through all stages 1-4."""
        ctx = create_pipeline_context([ethanol_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)

        # Molecule should still be active (passed all checks)
        assert ctx.active[0], "Ethanol failed pipeline stages 1-4"

        # Check 3D coordinates are reasonable
        pos_3d = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]
        assert np.all(np.isfinite(pos_3d))

        # 4th dimension should be small after fourth-dim minimization
        pos_4d = np.array(ctx.positions).reshape(-1, ctx.dim)[:, 3]
        assert np.max(np.abs(pos_4d)) < 2.0

    def test_propane_passes_pipeline(self, propane_mol):
        """Propane should pass through all stages 1-4."""
        ctx = create_pipeline_context([propane_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)
        assert ctx.active[0], "Propane failed pipeline stages 1-4"

    def test_benzene_passes_pipeline(self, benzene_mol):
        """Benzene should pass (no tetrahedral checks needed)."""
        ctx = create_pipeline_context([benzene_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)
        assert ctx.active[0], "Benzene failed pipeline stages 1-4"

    def test_batch_pipeline(self, ethanol_mol, propane_mol, benzene_mol):
        """Batch of diverse molecules through the pipeline."""
        ctx = create_pipeline_context([ethanol_mol, propane_mol, benzene_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)

        # At least some should pass
        n_passed = sum(ctx.active)
        assert n_passed >= 1, f"Only {n_passed}/3 molecules passed"

    def test_chiral_molecule_pipeline(self, chiral_mol):
        """Chiral molecule through the pipeline (may or may not pass)."""
        ctx = create_pipeline_context([chiral_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)
        # We don't assert pass — just check it doesn't crash

    def test_pipeline_without_chirality_enforcement(self, ethanol_mol):
        """Pipeline should work with chirality enforcement disabled."""
        ctx = create_pipeline_context([ethanol_mol])
        run_dg_pipeline(ctx, enforce_chirality=False, seed=42)
        assert ctx.active[0]

    def test_multiple_seeds_ethanol(self, ethanol_mol):
        """Ethanol should pass with multiple random seeds."""
        n_pass = 0
        for seed in range(10):
            ctx = create_pipeline_context([ethanol_mol])
            run_dg_pipeline(ctx, enforce_chirality=True, seed=seed)
            if ctx.active[0]:
                n_pass += 1

        # At least 50% should pass
        assert n_pass >= 5, f"Only {n_pass}/10 seeds passed for ethanol"

    def test_3d_distances_reasonable(self, ethanol_mol):
        """After pipeline, bonded atom distances should be reasonable."""
        ctx = create_pipeline_context([ethanol_mol])
        run_dg_pipeline(ctx, enforce_chirality=True, seed=42)

        if not ctx.active[0]:
            pytest.skip("Ethanol failed pipeline with this seed")

        pos_3d = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

        # Check bond lengths are in reasonable range (0.5-3.0 Angstroms)
        mol = ethanol_mol
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(pos_3d[i] - pos_3d[j])
            assert 0.5 < dist < 3.0, f"Bond {i}-{j} distance {dist:.2f} out of range"
