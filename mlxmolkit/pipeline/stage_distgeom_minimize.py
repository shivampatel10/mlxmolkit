"""Stage 2/4: Distance geometry BFGS minimization.

Runs BFGS optimization with the DG force field to refine random 4D
coordinates into physically valid molecular geometries.

Default cascade: Metal kernel -> vectorized BFGS -> original BFGS.
Metal is ~115x faster than original at batch sizes >= 100.
"""

import logging

import mlx.core as mx

from ..forcefields.dist_geom import dg_energy_and_grad
from ..minimizer.bfgs import bfgs_minimize
from .context import PipelineContext

log = logging.getLogger(__name__)

# Max energy per atom threshold (from nvMolKit)
MAX_MINIMIZED_E_PER_ATOM = 0.05

# Max atoms per molecule for Metal kernel (Hessian memory limit)
_METAL_MAX_ATOMS = 64


def _try_metal_dg_lbfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters):
    """Try Metal DG L-BFGS kernel (threadgroup-parallel). Returns result or None."""
    try:
        from ..metal_kernels.dg_lbfgs import metal_dg_lbfgs

        max_atoms = max(
            ctx.atom_starts[i + 1] - ctx.atom_starts[i]
            for i in range(ctx.n_mols)
        )
        if max_atoms > _METAL_MAX_ATOMS:
            return None

        result = metal_dg_lbfgs(
            ctx.positions, system, chiral_weight, fourth_dim_weight,
            max_iters=max_iters,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Metal DG L-BFGS unavailable, falling back: %s", e)
        return None


def _try_metal_dg_bfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters):
    """Try Metal DG BFGS kernel. Returns result or None on failure."""
    try:
        from ..metal_kernels.dg_bfgs import metal_dg_bfgs

        # Check max atoms — fall back for large molecules
        max_atoms = max(
            ctx.atom_starts[i + 1] - ctx.atom_starts[i]
            for i in range(ctx.n_mols)
        )
        if max_atoms > _METAL_MAX_ATOMS:
            return None

        result = metal_dg_bfgs(
            ctx.positions, system, chiral_weight, fourth_dim_weight,
            max_iters=max_iters,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Metal DG BFGS unavailable, falling back: %s", e)
        return None


def _try_vectorized_bfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters):
    """Try vectorized BFGS. Returns result or None on failure."""
    try:
        from ..minimizer.bfgs_vectorized import bfgs_minimize_vectorized

        def energy_and_grad(pos):
            return dg_energy_and_grad(pos, system, chiral_weight, fourth_dim_weight)

        result = bfgs_minimize_vectorized(
            energy_and_grad,
            ctx.positions,
            ctx.atom_starts,
            ctx.n_mols,
            ctx.dim,
            max_iters=max_iters,
        )
        mx.eval(*result)
        return result
    except Exception as e:
        log.debug("Vectorized BFGS failed, falling back: %s", e)
        return None


def stage_distgeom_minimize(
    ctx: PipelineContext,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    max_iters: int = 400,
    check_energy: bool = True,
    minimizer: str = "metal",
):
    """Run BFGS minimization with DG force field.

    Args:
        ctx: Pipeline context (modified in place).
        chiral_weight: Weight for chiral violation terms.
        fourth_dim_weight: Weight for fourth dimension penalty.
        max_iters: Maximum BFGS iterations.
        check_energy: If True, fail molecules with energy per atom >= 0.05.
        minimizer: Which minimizer to use.
            'metal' (default) — Metal L-BFGS -> dense BFGS -> vectorized -> original.
            'vectorized' — vectorized Python BFGS (~58x speedup).
            'original' — original per-molecule BFGS (slowest, for debugging).
    """
    system = ctx.dg_system
    result = None

    # Cascade: L-BFGS -> dense BFGS -> vectorized -> original
    if minimizer == "metal":
        result = _try_metal_dg_lbfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters)

    if result is None and minimizer == "metal":
        result = _try_metal_dg_bfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters)

    if result is None and minimizer in ("metal", "vectorized"):
        result = _try_vectorized_bfgs(ctx, system, chiral_weight, fourth_dim_weight, max_iters)

    if result is None:
        def energy_and_grad(pos):
            return dg_energy_and_grad(pos, system, chiral_weight, fourth_dim_weight)

        result = bfgs_minimize(
            energy_and_grad,
            ctx.positions,
            ctx.atom_starts,
            ctx.n_mols,
            ctx.dim,
            max_iters=max_iters,
            scale_grads=False,
        )

    final_pos, final_energies, statuses = result
    ctx.positions = final_pos
    mx.eval(final_energies)

    if check_energy:
        for i in range(ctx.n_mols):
            if not ctx.active[i]:
                continue
            n_atoms = ctx.atom_starts[i + 1] - ctx.atom_starts[i]
            e_per_atom = final_energies[i].item() / n_atoms
            if e_per_atom >= MAX_MINIMIZED_E_PER_ATOM:
                ctx.failed[i] = True
