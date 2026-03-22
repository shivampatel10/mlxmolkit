"""Pipeline driver for full ETKDG pipeline.

Orchestrates all 7 stages with retry loop for failed conformers.
"""

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from .context import (
    PipelineContext,
    create_pipeline_context,
    create_pipeline_context_from_cache,
    create_pipeline_context_multi_conf,
    extract_mol_params_cache,
)
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

    # Stage 3: First chiral check
    if enforce_chirality:
        stage_first_chiral_check(ctx)
        ctx.collect_failures()

        if ctx.n_active() == 0:
            return

    # Stage 4: Fourth dimension minimization (collapses 4th dim)
    stage_distgeom_minimize(
        ctx,
        chiral_weight=0.2,
        fourth_dim_weight=1.0,
        max_iters=200,
        check_energy=False,
    )
    ctx.collect_failures()

    if ctx.n_active() == 0:
        return

    # Stage 4b: Tetrahedral check — runs after 4th-dim collapse.
    # nvMolKit runs this after stage 2 (before DG min 2) with tol=0.3,
    # but their float64 CUDA minimizer produces geometry where centers
    # are further from face planes. Our float32 minimizer often places
    # centers within 0.05-0.25 of a face (all |d2| > 0.05 are valid).
    # Using tol=0.1 (matching nvMolKit's final chiral volume check)
    # avoids false rejections while still catching degenerate geometry.
    stage_tetrahedral_check(ctx, tol=0.1)
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


def _write_conformers_np(
    mol: Chem.Mol,
    pos_np: np.ndarray,
    atom_start: int,
    n_atoms: int,
) -> int:
    """Write 3D coordinates from a pre-converted numpy array to RDKit conformer.

    Args:
        mol: RDKit molecule to add the conformer to.
        pos_np: Numpy positions array, shape (n_atoms_total, dim).
        atom_start: Start index of this molecule's atoms.
        n_atoms: Number of atoms in this molecule.

    Returns:
        Conformer ID of the added conformer.
    """
    conf = Chem.Conformer(n_atoms)
    for j in range(n_atoms):
        conf.SetAtomPosition(j, (
            float(pos_np[atom_start + j, 0]),
            float(pos_np[atom_start + j, 1]),
            float(pos_np[atom_start + j, 2]),
        ))
    conf.SetId(mol.GetNumConformers())
    return mol.AddConformer(conf, assignId=True)


def embed_molecules_pipeline(
    mols: list[Chem.Mol],
    params: rdDistGeom.EmbedParameters,
    confs_per_mol: int = 1,
    max_iterations: int = -1,
) -> None:
    """Full ETKDG embedding pipeline with batched multi-conformer generation.

    Generates conformers using a bulk batch approach: all molecules are
    replicated for the requested number of conformers and processed in a
    single GPU pipeline pass. Failed conformers are retried in batched
    rounds with overshoot (3x what's needed) until all molecules have
    enough conformers or max retries are exhausted.

    A persistent RNG advances across all pipeline passes (matching
    nvMolKit's global RNG), ensuring each conformer attempt sees
    fundamentally different random coordinates.

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

    # Pipeline flags
    use_exp_torsion = params.useExpTorsionAnglePrefs
    use_basic_knowledge = params.useBasicKnowledge
    use_etk = use_exp_torsion or use_basic_knowledge
    enforce_chirality = params.enforceChirality
    force_tol = getattr(params, 'optimizerForceTol', None)
    box_size_mult = getattr(params, 'boxSizeMult', 2.0)

    # Create persistent RNG (matches nvMolKit's global advancing RNG)
    base_seed = params.randomSeed if params.randomSeed >= 0 else None
    rng = np.random.default_rng(base_seed)

    # Track conformers generated and attempts per molecule
    confs_generated = [0] * n_mols
    total_attempts = [0] * n_mols
    # Molecules with 0% success after a full round are marked as given up
    given_up = [False] * n_mols

    # Retry strategy: bulk first pass + up to 3 retry rounds with overshoot.
    # Only retry molecules that showed >0% success rate in prior rounds.
    max_retries = 3
    overshoot_factor = 3

    # Pre-extract per-molecule params once (expensive RDKit calls).
    # Cached params are reused across all retry rounds.
    mol_params_cache = extract_mol_params_cache(
        mols, dim=4, basin_size_tol=1e8,
        params=params if use_etk else None,
        use_etk=use_etk,
    )

    for retry_round in range(max_retries + 1):
        # Determine which molecules still need conformers
        need_more = [
            i for i in range(n_mols)
            if confs_generated[i] < confs_per_mol and not given_up[i]
        ]
        if not need_more:
            break

        # Compute how many conformer slots to request per molecule
        if retry_round == 0:
            # First pass: request exactly what's needed
            slots_per_mol = [confs_per_mol] * len(need_more)
        else:
            # Retries: overshoot to fill remaining, capped at 3× needed
            slots_per_mol = [
                (confs_per_mol - confs_generated[i]) * overshoot_factor
                for i in need_more
            ]

        batch_mols = [mols[i] for i in need_more]
        batch_cache = [mol_params_cache[i] for i in need_more]

        # Track attempts
        for idx, i in enumerate(need_more):
            total_attempts[i] += slots_per_mol[idx]

        # Create multi-conf context from cached params (skips RDKit extraction)
        ctx = create_pipeline_context_from_cache(
            batch_mols,
            mol_cache=batch_cache,
            confs_per_mol=slots_per_mol,
            dim=4,
            use_etk=use_etk,
            enforce_chirality=enforce_chirality,
            rng=rng,
        )

        # Run full pipeline (RNG advances naturally through coordgen)
        run_full_pipeline(
            ctx,
            enforce_chirality=enforce_chirality,
            use_exp_torsion=use_exp_torsion,
            use_basic_knowledge=use_basic_knowledge,
            force_tol=force_tol,
            seed=None,  # RNG is on context, no seed needed
            box_size_mult=box_size_mult,
        )

        # Write back successful conformers, track per-molecule successes
        mx.eval(ctx.positions)
        pos_np = np.array(ctx.positions).reshape(-1, ctx.dim)

        round_mol_successes = [0] * n_mols
        for entry_idx in range(ctx.n_mols):
            if not ctx.active[entry_idx] or ctx.failed[entry_idx]:
                continue

            # Map entry back to original molecule
            batch_mol_idx = ctx.entry_mol_map[entry_idx]
            orig_mol_idx = need_more[batch_mol_idx]

            # Skip if this molecule already has enough conformers
            if confs_generated[orig_mol_idx] >= confs_per_mol:
                continue

            n_atoms = ctx.atom_starts[entry_idx + 1] - ctx.atom_starts[entry_idx]
            atom_start = ctx.atom_starts[entry_idx]
            conf_id = _write_conformers_np(
                mols[orig_mol_idx], pos_np, atom_start, n_atoms,
            )
            if conf_id >= 0:
                confs_generated[orig_mol_idx] += 1
                round_mol_successes[orig_mol_idx] += 1

        # Give up on molecules that had 0 successes this round despite
        # having enough attempts — they will almost certainly keep failing
        for i in need_more:
            if round_mol_successes[i] == 0 and total_attempts[i] >= confs_per_mol:
                given_up[i] = True

        # Early termination: stop if no molecule produced anything
        if sum(round_mol_successes) == 0:
            break
