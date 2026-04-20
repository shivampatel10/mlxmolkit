"""Pipeline driver for full ETKDG pipeline.

Orchestrates all 7 stages with retry scheduling for failed conformers.
"""

from dataclasses import dataclass, field
import time
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


@dataclass
class EmbedPassStats:
    """Timing and count data for one embedding dispatch pass."""

    pass_index: int
    attempts: int
    unique_mols: int
    context_seconds: float
    pipeline_seconds: float
    writeback_seconds: float
    successes: int
    conformers_written: int


@dataclass
class EmbedPipelineStats:
    """Retry statistics from the most recent embedding pipeline call."""

    n_mols: int
    confs_per_mol: int
    max_iterations: int
    passes: list[EmbedPassStats] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return sum(pass_stats.attempts for pass_stats in self.passes)

    @property
    def total_successes(self) -> int:
        return sum(pass_stats.successes for pass_stats in self.passes)

    @property
    def total_conformers_written(self) -> int:
        return sum(pass_stats.conformers_written for pass_stats in self.passes)


_LAST_EMBED_STATS: EmbedPipelineStats | None = None


def get_last_embed_stats() -> EmbedPipelineStats | None:
    """Return retry statistics from the most recent embedding pipeline call."""
    return _LAST_EMBED_STATS


class _RoundRobinRetryScheduler:
    """Dispatch conformer attempts in round-robin waves.

    Mirrors nvMolKit's scheduler structure: each round permits one additional
    `confs_per_mol` wave of attempts for molecules still short of the target.
    """

    def __init__(
        self,
        n_mols: int,
        confs_per_mol: int,
        max_iterations: int,
    ) -> None:
        if n_mols <= 0 or confs_per_mol <= 0 or max_iterations <= 0:
            raise ValueError("Scheduler parameters must be greater than 0")

        self.n_mols = n_mols
        self.confs_per_mol = confs_per_mol
        self.max_attempts_per_mol = max_iterations * confs_per_mol
        self.completed_confs = [0] * n_mols
        self.total_attempts = [0] * n_mols
        self.round_robin_iter = 1

    def _round_is_exhausted(self, max_attempts_this_round: int) -> bool:
        return all(
            completed >= self.confs_per_mol or attempts >= max_attempts_this_round
            for completed, attempts in zip(
                self.completed_confs,
                self.total_attempts,
                strict=True,
            )
        )

    def dispatch(
        self,
        batch_size: int,
        min_batch_size: int = 0,
        lookahead_rounds: int = 1,
    ) -> list[int]:
        """Return a batch of molecule ids to attempt next."""
        if batch_size <= 0:
            return []

        mol_ids: list[int] = []
        rounds_dispatched = 0
        coalesce = min_batch_size > 0 and lookahead_rounds > 1

        while len(mol_ids) < batch_size:
            before_round = len(mol_ids)
            max_attempts_this_round = min(
                self.max_attempts_per_mol,
                self.confs_per_mol * self.round_robin_iter,
            )
            for mol_idx in range(self.n_mols):
                while (
                    self.completed_confs[mol_idx] < self.confs_per_mol
                    and self.total_attempts[mol_idx] < max_attempts_this_round
                ):
                    if len(mol_ids) >= batch_size:
                        break
                    mol_ids.append(mol_idx)
                    self.total_attempts[mol_idx] += 1
                if len(mol_ids) >= batch_size:
                    break

            if self._round_is_exhausted(max_attempts_this_round):
                self.round_robin_iter += 1

            rounds_dispatched += 1
            made_progress = len(mol_ids) > before_round
            if not coalesce:
                break
            if len(mol_ids) >= min_batch_size:
                break
            if rounds_dispatched >= lookahead_rounds:
                break
            if not made_progress:
                break

        return mol_ids

    def record(self, mol_ids: list[int], completed: list[bool]) -> None:
        """Record which attempts succeeded."""
        if len(mol_ids) != len(completed):
            raise ValueError("mol_ids and completed must have the same size")

        for mol_idx, did_complete in zip(mol_ids, completed, strict=True):
            if did_complete:
                self.completed_confs[mol_idx] += 1


def _resolve_max_iterations(
    mols: list[Chem.Mol],
    params: rdDistGeom.EmbedParameters,
    max_iterations: int,
) -> int:
    """Resolve retry iterations from explicit arg, params, or auto fallback."""
    if max_iterations > 0:
        return max_iterations

    params_max_iterations = getattr(params, "maxIterations", 0)
    if params_max_iterations > 0:
        return params_max_iterations

    max_atoms = max(mol.GetNumAtoms() for mol in mols)
    return 10 * max_atoms


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

    # Stage 3: Tetrahedral check
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

    Generates conformers using repeated GPU pipeline passes. Each pass uses a
    round-robin retry scheduler so molecules that still need conformers get one
    additional wave of attempts, instead of a fixed overshoot multiplier.

    A persistent RNG advances across all pipeline passes (matching
    nvMolKit's global RNG), ensuring each conformer attempt sees
    fundamentally different random coordinates.

    Args:
        mols: RDKit molecules with hydrogens added.
        params: RDKit EmbedParameters controlling algorithm behavior.
        confs_per_mol: Number of conformers to generate per molecule.
        max_iterations: Max retry iterations. Values <= 0 defer to
            params.maxIterations, then 10 * max_atoms.
    """
    global _LAST_EMBED_STATS

    if not mols:
        _LAST_EMBED_STATS = EmbedPipelineStats(
            n_mols=0,
            confs_per_mol=confs_per_mol,
            max_iterations=0,
        )
        return

    n_mols = len(mols)

    max_iterations = _resolve_max_iterations(mols, params, max_iterations)
    stats = EmbedPipelineStats(
        n_mols=n_mols,
        confs_per_mol=confs_per_mol,
        max_iterations=max_iterations,
    )
    _LAST_EMBED_STATS = stats

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

    # Track how many conformers were written back per molecule.
    confs_written = [0] * n_mols
    scheduler = _RoundRobinRetryScheduler(n_mols, confs_per_mol, max_iterations)
    dispatch_batch_size = n_mols * confs_per_mol
    retry_min_batch_size = max(confs_per_mol, dispatch_batch_size // 2)
    retry_lookahead_rounds = 8

    # Pre-extract per-molecule params once (expensive RDKit calls).
    # Cached params are reused across all retry rounds.
    mol_params_cache = extract_mol_params_cache(
        mols, dim=4, basin_size_tol=1e8,
        params=params if use_etk else None,
        use_etk=use_etk,
    )

    pass_index = 0
    while True:
        if pass_index == 0:
            mol_ids = scheduler.dispatch(dispatch_batch_size)
        else:
            mol_ids = scheduler.dispatch(
                dispatch_batch_size,
                min_batch_size=retry_min_batch_size,
                lookahead_rounds=retry_lookahead_rounds,
            )
        if not mol_ids:
            break
        pass_index += 1

        slots_by_mol = np.bincount(
            np.array(mol_ids, dtype=np.int32),
            minlength=n_mols,
        ).tolist()
        batch_mol_ids = [i for i, slots in enumerate(slots_by_mol) if slots > 0]
        slots_per_mol = [slots_by_mol[i] for i in batch_mol_ids]
        batch_mols = [mols[i] for i in batch_mol_ids]
        batch_cache = [mol_params_cache[i] for i in batch_mol_ids]
        entry_mol_ids = [
            mol_idx
            for mol_idx, slots in zip(batch_mol_ids, slots_per_mol, strict=True)
            for _ in range(slots)
        ]

        # Create multi-conf context from cached params (skips RDKit extraction)
        t_context = time.perf_counter()
        ctx = create_pipeline_context_from_cache(
            batch_mols,
            mol_cache=batch_cache,
            confs_per_mol=slots_per_mol,
            dim=4,
            use_etk=use_etk,
            enforce_chirality=enforce_chirality,
            rng=rng,
        )
        context_seconds = time.perf_counter() - t_context

        # Run full pipeline (RNG advances naturally through coordgen)
        t_pipeline = time.perf_counter()
        run_full_pipeline(
            ctx,
            enforce_chirality=enforce_chirality,
            use_exp_torsion=use_exp_torsion,
            use_basic_knowledge=use_basic_knowledge,
            force_tol=force_tol,
            seed=None,  # RNG is on context, no seed needed
            box_size_mult=box_size_mult,
        )
        pipeline_seconds = time.perf_counter() - t_pipeline

        # Write back successful conformers, track per-molecule successes
        t_writeback = time.perf_counter()
        mx.eval(ctx.positions)
        pos_np = np.array(ctx.positions).reshape(-1, ctx.dim)

        completed = [False] * ctx.n_mols
        conformers_written_this_pass = 0
        for entry_idx in range(ctx.n_mols):
            if not ctx.active[entry_idx] or ctx.failed[entry_idx]:
                continue
            completed[entry_idx] = True

            # Map entry back to original molecule
            batch_mol_idx = ctx.entry_mol_map[entry_idx]
            orig_mol_idx = batch_mol_ids[batch_mol_idx]

            # Skip if this molecule already has enough conformers
            if confs_written[orig_mol_idx] >= confs_per_mol:
                continue

            n_atoms = ctx.atom_starts[entry_idx + 1] - ctx.atom_starts[entry_idx]
            atom_start = ctx.atom_starts[entry_idx]
            conf_id = _write_conformers_np(
                mols[orig_mol_idx], pos_np, atom_start, n_atoms,
            )
            if conf_id >= 0:
                confs_written[orig_mol_idx] += 1
                conformers_written_this_pass += 1

        scheduler.record(entry_mol_ids, completed)
        writeback_seconds = time.perf_counter() - t_writeback
        stats.passes.append(
            EmbedPassStats(
                pass_index=pass_index,
                attempts=len(mol_ids),
                unique_mols=len(batch_mol_ids),
                context_seconds=context_seconds,
                pipeline_seconds=pipeline_seconds,
                writeback_seconds=writeback_seconds,
                successes=sum(completed),
                conformers_written=conformers_written_this_pass,
            )
        )

        # Early termination: stop if no molecule produced anything
        if not any(completed):
            break
