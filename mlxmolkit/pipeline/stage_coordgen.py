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
):
    """Generate random 4D coordinates for all molecules.

    Matches nvMolKit's random coordinate generation:
      pos[atom, dim] = (random() - 0.5) * box_size
    where box_size = 5.0 * box_size_mult.

    Args:
        ctx: Pipeline context (modified in place).
        seed: Random seed for reproducibility.
        box_size_mult: Box size multiplier (default 2.0, so box_size = 10.0).
    """
    box_size = 5.0 * box_size_mult
    rng = np.random.default_rng(seed)

    n_total = ctx.n_atoms_total * ctx.dim
    rand_coords = (rng.random(n_total).astype(np.float32) - 0.5) * box_size

    ctx.positions = mx.array(rand_coords)
