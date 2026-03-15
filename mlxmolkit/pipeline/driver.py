"""Pipeline driver for full ETKDG pipeline.

Orchestrates all 7 stages with retry loop for failed conformers.
"""

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from .context import PipelineContext, create_pipeline_context
from .stage_coordgen import stage_coordgen
from .stage_distgeom_minimize import stage_distgeom_minimize
from .stage_etk_minimize import stage_etk_minimize
from .stage_stereochem_checks import (
    stage_chiral_dist_matrix_check,
    stage_chiral_volume_check,
    stage_double_bond_geometry_check,
    stage_double_bond_stereo_check,
    stage_first_chiral_check,
    stage_tetrahedral_check,
)


def run_dg_pipeline(
    ctx: PipelineContext,
    enforce_chirality: bool = True,
    seed: int | None = None,
    box_size_mult: float = 2.0,
) -> None:
    """Run ETKDG pipeline stages 1-4.

    Args:
        ctx: Pipeline context (modified in place).
        enforce_chirality: Whether to enforce chirality checks.
        seed: Random seed for coordinate generation.
        box_size_mult: Box size multiplier for random coords.
    """
    # Stage 1: Coordinate generation
    stage_coordgen(ctx, seed=seed, box_size_mult=box_size_mult)
    ctx.collect_failures()

    if ctx.n_active() == 0:
        return

    # Stage 2: First DG minimization
    stage_distgeom_minimize(
        ctx,
        chiral_weight=1.0,
        fourth_dim_weight=0.1,
        max_iters=400,
        check_energy=True,
    )
    ctx.collect_failures()

    if ctx.n_active() == 0:
        return

    # Stage 3a: Tetrahedral check
    stage_tetrahedral_check(ctx, tol=0.3)
    ctx.collect_failures()

    if ctx.n_active() == 0:
        return

    # Stage 3b: First chiral check
    if enforce_chirality:
        stage_first_chiral_check(ctx)
        ctx.collect_failures()

        if ctx.n_active() == 0:
            return

    # Stage 4: Fourth dimension minimization
    stage_distgeom_minimize(
        ctx,
        chiral_weight=0.2,
        fourth_dim_weight=1.0,
        max_iters=200,
        check_energy=False,
    )
    ctx.collect_failures()


def run_full_pipeline(
    ctx: PipelineContext,
    enforce_chirality: bool = True,
    use_exp_torsion: bool = True,
    use_basic_knowledge: bool = True,
    force_tol: float | None = None,
    seed: int | None = None,
    box_size_mult: float = 2.0,
) -> None:
    """Run full ETKDG pipeline stages 1-7.

    Args:
        ctx: Pipeline context (modified in place).
        enforce_chirality: Whether to enforce chirality checks.
        use_exp_torsion: Include CSD torsion angle preferences.
        use_basic_knowledge: Include bond/angle knowledge.
        force_tol: Force tolerance for ETK BFGS.
        seed: Random seed for coordinate generation.
        box_size_mult: Box size multiplier.
    """
    # Stages 1-4
    run_dg_pipeline(ctx, enforce_chirality=enforce_chirality,
                    seed=seed, box_size_mult=box_size_mult)

    if ctx.n_active() == 0:
        return

    # Stage 5: ETK minimization
    if (use_exp_torsion or use_basic_knowledge) and ctx.etk_system is not None:
        stage_etk_minimize(
            ctx, ctx.etk_system,
            use_basic_knowledge=use_basic_knowledge,
            max_iters=300,
            force_tol=force_tol,
        )
        ctx.collect_failures()

        if ctx.n_active() == 0:
            return

    # Stage 6: Double bond geometry check
    stage_double_bond_geometry_check(ctx, ctx.double_bond_data)
    ctx.collect_failures()

    if ctx.n_active() == 0:
        return

    # Stage 6b-6d: Final chirality checks
    if enforce_chirality:
        # Final chiral check (same as first — pass-through in nvMolKit)
        stage_first_chiral_check(ctx)
        ctx.collect_failures()
        if ctx.n_active() == 0:
            return

        # Chiral distance matrix check
        stage_chiral_dist_matrix_check(ctx, ctx.chiral_dist_data)
        ctx.collect_failures()
        if ctx.n_active() == 0:
            return

        # Chiral center volume check (center-in-volume only)
        stage_chiral_volume_check(ctx)
        ctx.collect_failures()
        if ctx.n_active() == 0:
            return

        # Double bond stereo check
        stage_double_bond_stereo_check(ctx, ctx.stereo_bond_data)
        ctx.collect_failures()


def _write_conformers(
    mol: Chem.Mol,
    positions: mx.array,
    atom_start: int,
    n_atoms: int,
    dim: int,
) -> int:
    """Write 3D coordinates back to RDKit molecule as a new conformer.

    Args:
        mol: RDKit molecule to add the conformer to.
        positions: Flat MLX positions array.
        atom_start: Start index of this molecule's atoms.
        n_atoms: Number of atoms in this molecule.
        dim: Coordinate dimension.

    Returns:
        Conformer ID of the added conformer.
    """
    mx.eval(positions)
    pos = np.array(positions).reshape(-1, dim)

    conf = Chem.Conformer(n_atoms)
    for j in range(n_atoms):
        x = float(pos[atom_start + j, 0])
        y = float(pos[atom_start + j, 1])
        z = float(pos[atom_start + j, 2])
        conf.SetAtomPosition(j, (x, y, z))

    conf.SetId(mol.GetNumConformers())
    return mol.AddConformer(conf, assignId=True)


def embed_molecules_pipeline(
    mols: list[Chem.Mol],
    params: rdDistGeom.EmbedParameters,
    confs_per_mol: int = 1,
    max_iterations: int = -1,
) -> None:
    """Full ETKDG embedding pipeline with retry loop.

    Generates conformers for each molecule using the ETKDG algorithm,
    retrying failed conformers up to ``max_iterations`` times.

    Args:
        mols: RDKit molecules with hydrogens added.
        params: RDKit EmbedParameters controlling algorithm behavior.
        confs_per_mol: Number of conformers to generate per molecule.
        max_iterations: Max retry iterations (-1 = 10 * max_atoms).
    """
    if not mols:
        return

    n_mols = len(mols)

    # Determine max iterations
    max_atoms = max(mol.GetNumAtoms() for mol in mols)
    if max_iterations <= 0:
        max_iterations = 10 * max_atoms

    # Determine pipeline flags
    use_exp_torsion = params.useExpTorsionAnglePrefs
    use_basic_knowledge = params.useBasicKnowledge
    use_etk = use_exp_torsion or use_basic_knowledge
    enforce_chirality = params.enforceChirality
    force_tol = getattr(params, 'optimizerForceTol', None)
    box_size_mult = getattr(params, 'boxSizeMult', 2.0)

    # Seed handling
    base_seed = params.randomSeed if params.randomSeed >= 0 else None

    # Track conformers generated per molecule
    confs_generated = [0] * n_mols

    # Retry loop
    for iteration in range(max_iterations):
        # Check which molecules still need conformers
        need_more = [i for i in range(n_mols) if confs_generated[i] < confs_per_mol]
        if not need_more:
            break

        # Create context for molecules needing conformers
        batch_mols = [mols[i] for i in need_more]

        ctx = create_pipeline_context(
            batch_mols,
            dim=4,
            basin_size_tol=1e8,
            params=params if use_etk else None,
            use_etk=use_etk,
            enforce_chirality=enforce_chirality,
        )

        # Compute seed for this iteration
        seed = None
        if base_seed is not None:
            seed = base_seed + iteration * n_mols

        # Run full pipeline
        run_full_pipeline(
            ctx,
            enforce_chirality=enforce_chirality,
            use_exp_torsion=use_exp_torsion,
            use_basic_knowledge=use_basic_knowledge,
            force_tol=force_tol,
            seed=seed,
            box_size_mult=box_size_mult,
        )

        # Write back successful conformers
        for batch_idx in range(len(batch_mols)):
            mol_idx = need_more[batch_idx]

            if ctx.active[batch_idx] and not ctx.failed[batch_idx]:
                n_atoms = ctx.atom_starts[batch_idx + 1] - ctx.atom_starts[batch_idx]
                conf_id = _write_conformers(
                    mols[mol_idx],
                    ctx.positions,
                    ctx.atom_starts[batch_idx],
                    n_atoms,
                    ctx.dim,
                )
                if conf_id >= 0:
                    confs_generated[mol_idx] += 1
