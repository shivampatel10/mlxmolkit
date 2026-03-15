"""Stage 5: ETK 3D BFGS minimization.

Runs BFGS optimization with the ETK force field (torsion preferences,
improper torsions, distance/angle constraints) to refine 3D molecular
geometries after the 4D distance geometry stages.

Default cascade: Metal kernel -> vectorized BFGS -> original BFGS.

Port of nvMolKit's ETKMinimizationStage.
"""

import logging
import math

import mlx.core as mx
import numpy as np

from ..forcefields.dist_geom_3d import compute_planar_energy, etk_energy_and_grad
from ..minimizer.bfgs import bfgs_minimize
from ..preprocessing.etk_batching import BatchedETKSystem

log = logging.getLogger(__name__)

PLANAR_TOLERANCE_FACTOR = 0.7

_METAL_MAX_ATOMS = 64


def _update_reference_positions(
    etk_system: BatchedETKSystem, positions: mx.array, dim: int
) -> None:
    """Update dist12 and dist13 bounds from current 3D positions.

    Computes current distances for non-improper-constrained terms and
    re-centers bounds around them (preserving the half-width).

    Args:
        etk_system: Batched ETK system whose bounds are updated in place.
        positions: Flat positions array of shape ``(n_atoms_total * dim,)``.
        dim: Coordinate dimension.
    """
    pos_r = positions.reshape(-1, dim)

    # Update dist12 terms
    if etk_system.dist12_idx1.size > 0:
        p1 = pos_r[etk_system.dist12_idx1][:, :3]
        p2 = pos_r[etk_system.dist12_idx2][:, :3]
        diff = p1 - p2
        dist = mx.sqrt(mx.sum(diff * diff, axis=1))

        old_min = etk_system.dist12_min
        old_max = etk_system.dist12_max
        half_width = (old_max - old_min) / 2.0

        etk_system.dist12_min = dist - half_width
        etk_system.dist12_max = dist + half_width

    # Update dist13 terms (skip improper-constrained)
    if etk_system.dist13_idx1.size > 0:
        p1 = pos_r[etk_system.dist13_idx1][:, :3]
        p2 = pos_r[etk_system.dist13_idx2][:, :3]
        diff = p1 - p2
        dist = mx.sqrt(mx.sum(diff * diff, axis=1))

        old_min = etk_system.dist13_min
        old_max = etk_system.dist13_max
        half_width = (old_max - old_min) / 2.0

        new_min = dist - half_width
        new_max = dist + half_width

        # Keep original bounds for improper-constrained terms
        is_imp = etk_system.dist13_is_improper
        etk_system.dist13_min = mx.where(is_imp, old_min, new_min)
        etk_system.dist13_max = mx.where(is_imp, old_max, new_max)

    # Update long-range terms
    if etk_system.long_range_idx1.size > 0:
        p1 = pos_r[etk_system.long_range_idx1][:, :3]
        p2 = pos_r[etk_system.long_range_idx2][:, :3]
        diff = p1 - p2
        dist = mx.sqrt(mx.sum(diff * diff, axis=1))

        old_min = etk_system.long_range_min
        old_max = etk_system.long_range_max
        half_width = (old_max - old_min) / 2.0

        etk_system.long_range_min = dist - half_width
        etk_system.long_range_max = dist + half_width


def _try_metal_etk_lbfgs(
    ctx: "PipelineContext",
    etk_system: BatchedETKSystem,
    use_basic_knowledge: bool,
    max_iters: int,
    grad_tol: float | None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Try Metal ETK L-BFGS kernel (threadgroup-parallel).

    Args:
        ctx: Pipeline context with positions and atom layout.
        etk_system: Batched ETK force field system.
        use_basic_knowledge: Include improper torsion terms.
        max_iters: Maximum L-BFGS iterations.
        grad_tol: Gradient tolerance, or None for default.

    Returns:
        Tuple of (positions, energies, statuses) arrays, or None on failure.
    """
    try:
        from ..metal_kernels.etk_lbfgs import metal_etk_lbfgs

        max_atoms = max(
            ctx.atom_starts[i + 1] - ctx.atom_starts[i]
            for i in range(ctx.n_mols)
        )
        if max_atoms > _METAL_MAX_ATOMS:
            return None

        result = metal_etk_lbfgs(
            ctx.positions, etk_system, use_basic_knowledge,
            max_iters=max_iters, grad_tol=grad_tol,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Metal ETK L-BFGS unavailable, falling back: %s", e)
        return None


def _try_metal_etk_bfgs(
    ctx: "PipelineContext",
    etk_system: BatchedETKSystem,
    use_basic_knowledge: bool,
    max_iters: int,
    grad_tol: float | None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Try Metal ETK BFGS kernel.

    Args:
        ctx: Pipeline context with positions and atom layout.
        etk_system: Batched ETK force field system.
        use_basic_knowledge: Include improper torsion terms.
        max_iters: Maximum BFGS iterations.
        grad_tol: Gradient tolerance, or None for default.

    Returns:
        Tuple of (positions, energies, statuses) arrays, or None on failure.
    """
    try:
        from ..metal_kernels.etk_bfgs import metal_etk_bfgs

        max_atoms = max(
            ctx.atom_starts[i + 1] - ctx.atom_starts[i]
            for i in range(ctx.n_mols)
        )
        if max_atoms > _METAL_MAX_ATOMS:
            return None

        result = metal_etk_bfgs(
            ctx.positions, etk_system, use_basic_knowledge,
            max_iters=max_iters, grad_tol=grad_tol,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Metal ETK BFGS unavailable, falling back: %s", e)
        return None


def _try_vectorized_etk_bfgs(
    ctx: "PipelineContext",
    etk_system: BatchedETKSystem,
    use_basic_knowledge: bool,
    max_iters: int,
    grad_tol: float | None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Try vectorized BFGS with ETK energy.

    Args:
        ctx: Pipeline context with positions and atom layout.
        etk_system: Batched ETK force field system.
        use_basic_knowledge: Include improper torsion terms.
        max_iters: Maximum BFGS iterations.
        grad_tol: Gradient tolerance, or None for default.

    Returns:
        Tuple of (positions, energies, statuses) arrays, or None on failure.
    """
    try:
        from ..minimizer.bfgs_vectorized import bfgs_minimize_vectorized

        def energy_and_grad(pos):
            return etk_energy_and_grad(pos, etk_system, use_basic_knowledge)

        kwargs = dict(max_iters=max_iters)
        if grad_tol is not None:
            kwargs['grad_tol'] = grad_tol

        result = bfgs_minimize_vectorized(
            energy_and_grad,
            ctx.positions,
            ctx.atom_starts,
            ctx.n_mols,
            ctx.dim,
            **kwargs,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Vectorized ETK BFGS failed, falling back: %s", e)
        return None


def stage_etk_minimize(
    ctx: "PipelineContext",
    etk_system: BatchedETKSystem,
    use_basic_knowledge: bool = True,
    max_iters: int = 300,
    force_tol: float | None = None,
    minimizer: str = "metal",
) -> None:
    """Run BFGS minimization with ETK force field.

    Args:
        ctx: Pipeline context (modified in place).
        etk_system: BatchedETKSystem with ETK terms.
        use_basic_knowledge: Include improper torsion terms.
        max_iters: Maximum BFGS iterations.
        force_tol: Force tolerance for BFGS (None = default).
        minimizer: Which minimizer to use.
            'metal' (default) -- Metal kernel, fastest.
            'vectorized' -- vectorized Python BFGS.
            'original' -- original per-molecule BFGS (slowest, for debugging).

    Returns:
        None. Mutates ``ctx.positions`` and ``ctx.failed`` in place.
    """
    # 1. Update reference positions from current 3D coords
    _update_reference_positions(etk_system, ctx.positions, ctx.dim)
    mx.eval(etk_system.dist12_min, etk_system.dist12_max)
    if etk_system.dist13_idx1.size > 0:
        mx.eval(etk_system.dist13_min, etk_system.dist13_max)
    if etk_system.long_range_idx1.size > 0:
        mx.eval(etk_system.long_range_min, etk_system.long_range_max)

    # 2. Minimization — cascade: L-BFGS -> dense BFGS -> vectorized -> original
    result = None

    if minimizer == "metal":
        result = _try_metal_etk_lbfgs(ctx, etk_system, use_basic_knowledge, max_iters, force_tol)

    if result is None and minimizer == "metal":
        result = _try_metal_etk_bfgs(ctx, etk_system, use_basic_knowledge, max_iters, force_tol)

    if result is None and minimizer in ("metal", "vectorized"):
        result = _try_vectorized_etk_bfgs(ctx, etk_system, use_basic_knowledge, max_iters, force_tol)

    if result is None:
        def energy_and_grad(pos):
            return etk_energy_and_grad(pos, etk_system, use_basic_knowledge)

        kwargs = dict(
            max_iters=max_iters,
            scale_grads=False,
        )
        if force_tol is not None:
            kwargs['grad_tol'] = force_tol

        result = bfgs_minimize(
            energy_and_grad,
            ctx.positions,
            ctx.atom_starts,
            ctx.n_mols,
            ctx.dim,
            **kwargs,
        )

    final_pos, final_energies, statuses = result
    ctx.positions = final_pos
    mx.eval(final_energies)

    # 3. Planar tolerance check (if useBasicKnowledge)
    if use_basic_knowledge:
        planar_e = compute_planar_energy(ctx.positions, etk_system, ctx.dim)
        mx.eval(planar_e)

        for i in range(ctx.n_mols):
            if not ctx.active[i]:
                continue
            n_impropers = etk_system.num_impropers_per_mol[i]
            if n_impropers > 0:
                tolerance = PLANAR_TOLERANCE_FACTOR * n_impropers
                if planar_e[i].item() > tolerance:
                    ctx.failed[i] = True
