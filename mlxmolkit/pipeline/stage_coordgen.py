"""Stage 1: Random 4D coordinate generation.

Generates random initial coordinates in a box, matching nvMolKit's
random coordinate generation for ETKDG.
"""

import numpy as np
import mlx.core as mx

from .context import PipelineContext


def stage_coordgen(
    ctx: PipelineContext,
    seed: int | None = None,
    box_size_mult: float = 2.0,
) -> None:
    """Generate random 4D coordinates for all molecules.

    Each coordinate is sampled uniformly from ``[-box_size/2, box_size/2]``.
    Positive ``box_size_mult`` values scale the default box size, while
    non-positive values are interpreted as an absolute box size following
    RDKit/nvMolKit semantics.

    Args:
        ctx: Pipeline context (modified in place).
        seed: Random seed for reproducibility.
        box_size_mult: Box size multiplier (default 2.0, so box_size = 10.0).
            Non-positive values use ``box_size = -box_size_mult``.
    """
    box_size = 5.0 * box_size_mult if box_size_mult > 0 else -box_size_mult

    # Use persistent RNG from context if available (advances across calls),
    # otherwise create a fresh one from seed (legacy behavior).
    rng = ctx.rng if ctx.rng is not None else np.random.default_rng(seed)

    n_total = ctx.n_atoms_total * ctx.dim
    rand_coords = (rng.random(n_total).astype(np.float32) - 0.5) * box_size

    ctx.positions = mx.array(rand_coords)
