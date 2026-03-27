"""Metal kernel for DG L-BFGS minimization with threadgroup parallelism.

Multiple threads per molecule (TPM) parallelize energy/gradient computation
and line search energy trials. L-BFGS replaces dense O(n^2) Hessian with
O(m*n) two-loop recursion using m=8 history vectors.

Combined optimizations:
  - Threadgroup parallelism: 85% of work (energy + line search) parallelized
  - L-BFGS: eliminates 24% Hessian cost, 5x less memory per molecule
  - Higher GPU occupancy from reduced memory footprint

Uses mx.fast.metal_kernel() with threadgroup=(TPM,1,1).
Gradient computation is serial (thread 0) since non-atomic helper functions
are reused. Energy, line search, and L-BFGS two-loop are parallelized.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from ..minimizer.bfgs import (
    DEFAULT_GRAD_TOL,
    EPS,
    FUNCTOL,
    MAX_STEP_FACTOR,
    MOVETOL,
    TOLX,
)
from ..preprocessing.batching import BatchedDGSystem

# Default threads per molecule (must be power of 2 for tree reduction)
DEFAULT_TPM = 32

# L-BFGS history depth
DEFAULT_LBFGS_M = 8

# Maximum atoms for Metal kernel
MAX_ATOMS_METAL = 64

# Load MSL header and kernel source from the .metal file
_KERNEL_DIR = Path(__file__).parent
_metal_src = (_KERNEL_DIR / "dg_lbfgs.metal").read_text()
_MSL_HEADER, _MSL_SOURCE = _metal_src.split("// ---- DG_LBFGS_SPLIT ----\n")


def _pack_kernel_inputs(
    system: BatchedDGSystem,
    chiral_weight: float,
    fourth_dim_weight: float,
    max_iters: int,
    grad_tol: float,
    lbfgs_m: int,
) -> dict[str, mx.array | int]:
    """Pack BatchedDGSystem fields into flat arrays for the Metal kernel.

    Args:
        system: Batched DG system containing distance, chiral, and
            fourth-dimension terms.
        chiral_weight: Weight applied to chiral violation energy terms.
        fourth_dim_weight: Weight applied to fourth-dimension penalty terms.
        max_iters: Maximum number of L-BFGS iterations.
        grad_tol: Gradient convergence tolerance.
        lbfgs_m: L-BFGS history depth (number of stored vector pairs).

    Returns:
        Dictionary of kernel input arrays and scalar sizes.
    """
    dim = system.dim
    n_mols = system.n_mols
    atom_starts_np = np.array(system.atom_starts.tolist(), dtype=np.int32)

    # Compute L-BFGS history starts (2 * m * n_vars per molecule)
    lbfgs_history_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i in range(n_mols):
        n_atoms = atom_starts_np[i + 1] - atom_starts_np[i]
        n_vars = n_atoms * dim
        # s and y vectors: m * n_vars each = 2 * m * n_vars total
        lbfgs_history_starts_np[i + 1] = (
            lbfgs_history_starts_np[i] + 2 * lbfgs_m * n_vars
        )

    total_pos_size = int(atom_starts_np[-1]) * dim
    total_lbfgs_size = int(lbfgs_history_starts_np[-1])

    # Pack distance terms: pairs (n_dist, 2) and bounds (n_dist, 3)
    n_dist = system.dist_idx1.size
    if n_dist > 0:
        dist_pairs_np = np.stack([
            np.array(system.dist_idx1.tolist(), dtype=np.int32),
            np.array(system.dist_idx2.tolist(), dtype=np.int32),
        ], axis=1).flatten()
        dist_bounds_np = np.stack([
            np.array(system.dist_lb2.tolist(), dtype=np.float32),
            np.array(system.dist_ub2.tolist(), dtype=np.float32),
            np.array(system.dist_weight.tolist(), dtype=np.float32),
        ], axis=1).flatten()
    else:
        dist_pairs_np = np.zeros(2, dtype=np.int32)
        dist_bounds_np = np.zeros(3, dtype=np.float32)

    # Pack chiral terms
    n_chiral = system.chiral_idx1.size
    if n_chiral > 0:
        chiral_quads_np = np.stack([
            np.array(system.chiral_idx1.tolist(), dtype=np.int32),
            np.array(system.chiral_idx2.tolist(), dtype=np.int32),
            np.array(system.chiral_idx3.tolist(), dtype=np.int32),
            np.array(system.chiral_idx4.tolist(), dtype=np.int32),
        ], axis=1).flatten()
        chiral_bounds_np = np.stack([
            np.array(system.chiral_vol_lower.tolist(), dtype=np.float32),
            np.array(system.chiral_vol_upper.tolist(), dtype=np.float32),
        ], axis=1).flatten()
    else:
        chiral_quads_np = np.zeros(4, dtype=np.int32)
        chiral_bounds_np = np.zeros(2, dtype=np.float32)

    # Fourth dimension indices
    n_fourth = system.fourth_idx.size
    fourth_idx_np = (
        np.array(system.fourth_idx.tolist(), dtype=np.int32)
        if n_fourth > 0 else np.zeros(1, dtype=np.int32)
    )

    # Config array
    config_np = np.array([
        n_mols, max_iters, grad_tol, chiral_weight, fourth_dim_weight, dim
    ], dtype=np.float32)

    # Term starts
    dist_term_starts_np = np.array(system.dist_term_starts.tolist(), dtype=np.int32)
    chiral_term_starts_np = np.array(system.chiral_term_starts.tolist(), dtype=np.int32)
    fourth_term_starts_np = np.array(system.fourth_term_starts.tolist(), dtype=np.int32)

    return {
        'atom_starts': mx.array(atom_starts_np),
        'lbfgs_history_starts': mx.array(lbfgs_history_starts_np),
        'dist_pairs': mx.array(dist_pairs_np),
        'dist_bounds': mx.array(dist_bounds_np),
        'dist_term_starts': mx.array(dist_term_starts_np),
        'chiral_quads': mx.array(chiral_quads_np),
        'chiral_bounds': mx.array(chiral_bounds_np),
        'chiral_term_starts': mx.array(chiral_term_starts_np),
        'fourth_idx_arr': mx.array(fourth_idx_np),
        'fourth_term_starts_arr': mx.array(fourth_term_starts_np),
        'config': mx.array(config_np),
        'total_pos_size': total_pos_size,
        'total_lbfgs_size': total_lbfgs_size,
    }


def metal_dg_lbfgs(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    max_iters: int = 400,
    grad_tol: float | None = None,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run DG L-BFGS minimization on-device with threadgroup parallelism.

    Args:
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        system: Batched DG system with all energy terms pre-packed.
        chiral_weight: Weight for chiral violation energy terms.
        fourth_dim_weight: Weight for fourth-dimension penalty terms.
        max_iters: Maximum number of L-BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance. Defaults to
            ``DEFAULT_GRAD_TOL``.
        tpm: Threads per molecule (must be a power of 2, e.g. 1, 8, 32).
        lbfgs_m: L-BFGS history depth (number of stored vector pairs).

    Returns:
        Tuple of ``(final_pos, final_energies, statuses)`` where
        *final_pos* has the same shape as *pos*, *final_energies* is
        shape ``(n_mols,)`` float32, and *statuses* is shape ``(n_mols,)``
        int32 (0 = converged, 1 = active/not converged).
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    dim = system.dim

    # Pack inputs
    inputs = _pack_kernel_inputs(
        system, chiral_weight, fourth_dim_weight,
        max_iters, grad_tol, lbfgs_m,
    )

    total_pos_size = inputs['total_pos_size']
    total_lbfgs_size = inputs['total_lbfgs_size']

    # Build kernel with template parameters
    kernel = mx.fast.metal_kernel(
        name="dg_lbfgs",
        input_names=[
            "pos", "atom_starts", "lbfgs_history_starts",
            "dist_pairs", "dist_bounds", "dist_term_starts",
            "chiral_quads", "chiral_bounds", "chiral_term_starts",
            "fourth_idx_arr", "fourth_term_starts_arr", "config",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch",
            "work_lbfgs", "work_rho", "work_alpha",
        ],
        header=_MSL_HEADER,
        source=_MSL_SOURCE,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'],
            inputs['lbfgs_history_starts'],
            inputs['dist_pairs'],
            inputs['dist_bounds'],
            inputs['dist_term_starts'],
            inputs['chiral_quads'],
            inputs['chiral_bounds'],
            inputs['chiral_term_starts'],
            inputs['fourth_idx_arr'],
            inputs['fourth_term_starts_arr'],
            inputs['config'],
        ],
        output_shapes=[
            (total_pos_size,),                  # out_pos
            (n_mols,),                          # out_energies
            (n_mols,),                          # out_statuses
            (total_pos_size,),                  # work_grad
            (total_pos_size,),                  # work_dir
            (total_pos_size * 3,),              # work_scratch (old_pos, old_grad, q)
            (max(total_lbfgs_size, 1),),        # work_lbfgs (S and Y history)
            (max(n_mols * lbfgs_m, 1),),        # work_rho
            (max(n_mols * lbfgs_m, 1),),        # work_alpha
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols * tpm, 1, 1),
        threadgroup=(tpm, 1, 1),
        template=[
            ("TPM", tpm),
            ("LBFGS_M", lbfgs_m),
            ("total_pos_size", total_pos_size),
        ],
    )

    return outputs[0], outputs[1], outputs[2]
