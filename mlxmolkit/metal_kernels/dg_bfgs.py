"""Metal kernel for DG BFGS minimization — entire optimization on-device.

One thread per molecule runs the complete BFGS loop (up to max_iters)
with DG energy+gradient computed inline. Zero Python round-trips.

Uses mx.fast.metal_kernel() to launch MSL code.
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


# Maximum atoms per molecule for Metal kernel (Hessian fits in threadgroup memory)
# 64 atoms * 4 dim = 256 terms -> 256*256*4 = 256KB per Hessian
MAX_ATOMS_METAL = 64

# Load MSL source from external .metal file
_KERNEL_DIR = Path(__file__).parent
__metal_source = (_KERNEL_DIR / "dg_bfgs.metal").read_text()
_MSL_HEADER, _MSL_SOURCE = __metal_source.split(
    "// ---- DG_BFGS_SPLIT ----\n"
)
del __metal_source


def _pack_kernel_inputs(
    system: BatchedDGSystem,
    chiral_weight: float,
    fourth_dim_weight: float,
    max_iters: int,
    grad_tol: float,
) -> dict[str, mx.array | int]:
    """Pack BatchedDGSystem fields into flat arrays for the Metal kernel.

    Args:
        system: Batched DG system containing distance, chiral, and
            fourth-dimension terms.
        chiral_weight: Weight applied to chiral violation energy terms.
        fourth_dim_weight: Weight applied to fourth-dimension penalty terms.
        max_iters: Maximum number of BFGS iterations.
        grad_tol: Gradient convergence tolerance.

    Returns:
        Dictionary of kernel input arrays and scalar sizes.
    """
    dim = system.dim
    n_mols = system.n_mols
    atom_starts_np = np.array(system.atom_starts.tolist(), dtype=np.int32)

    # Compute hessian_starts from mol_dims
    hessian_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i in range(n_mols):
        n_atoms = atom_starts_np[i + 1] - atom_starts_np[i]
        n_terms = n_atoms * dim
        hessian_starts_np[i + 1] = hessian_starts_np[i] + n_terms * n_terms

    total_pos_size = int(atom_starts_np[-1]) * dim
    total_hessian_size = int(hessian_starts_np[-1])

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

    # Pack chiral terms: quads (n_chiral, 4) and bounds (n_chiral, 2)
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
    fourth_idx_np = np.array(system.fourth_idx.tolist(), dtype=np.int32) if n_fourth > 0 else np.zeros(1, dtype=np.int32)

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
        'hessian_starts': mx.array(hessian_starts_np),
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
        'total_hessian_size': total_hessian_size,
    }


def metal_dg_bfgs(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    max_iters: int = 400,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run DG BFGS minimization entirely on-device via a Metal kernel.

    Args:
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        system: Batched DG system with all energy terms pre-packed.
        chiral_weight: Weight for chiral violation energy terms.
        fourth_dim_weight: Weight for fourth-dimension penalty terms.
        max_iters: Maximum number of BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance. Defaults to
            ``DEFAULT_GRAD_TOL``.

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
    inputs = _pack_kernel_inputs(system, chiral_weight, fourth_dim_weight,
                                  max_iters, grad_tol)

    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    # Build kernel
    kernel = mx.fast.metal_kernel(
        name="dg_bfgs",
        input_names=[
            "pos", "atom_starts", "hessian_starts",
            "dist_pairs", "dist_bounds", "dist_term_starts",
            "chiral_quads", "chiral_bounds", "chiral_term_starts",
            "fourth_idx_arr", "fourth_term_starts_arr", "config",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=_MSL_HEADER,
        source=_MSL_SOURCE,
    )

    # Launch kernel — inputs must be a list in same order as input_names
    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'],
            inputs['hessian_starts'],
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
            (total_pos_size,),       # out_pos
            (n_mols,),               # out_energies
            (n_mols,),               # out_statuses
            (total_pos_size,),       # work_grad
            (total_pos_size,),       # work_dir
            (total_pos_size * 3,),   # work_scratch
            (max(total_hessian_size, 1),),  # work_hessian
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols, 1, 1),
        threadgroup=(1, 1, 1),
        template=[("total_pos_size", total_pos_size)],
    )

    out_pos = outputs[0]
    out_energies = outputs[1]
    out_statuses = outputs[2]

    return out_pos, out_energies, out_statuses


# ---- Threadgroup kernel (TG) ----

# Default threads per molecule (must be power of 2 for tree reduction)
DEFAULT_TPM = 32

# Load TG MSL source from external .metal file
_tg_metal_source = (_KERNEL_DIR / "dg_bfgs_tg.metal").read_text()
_TG_MSL_HEADER, _TG_MSL_SOURCE = _tg_metal_source.split(
    "// ---- DG_BFGS_TG_SPLIT ----\n"
)
del _tg_metal_source


def metal_dg_bfgs_tg(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    max_iters: int = 400,
    grad_tol: float | None = None,
    tpm: int = DEFAULT_TPM,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run DG BFGS minimization on-device with threadgroup parallelism.

    One threadgroup per molecule. Energy evaluation, line search, Hessian
    ops, and convergence checks are parallelized across TPM threads.
    Gradient computation remains serial (thread 0) to avoid atomics.

    Args:
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        system: Batched DG system with all energy terms pre-packed.
        chiral_weight: Weight for chiral violation energy terms.
        fourth_dim_weight: Weight for fourth-dimension penalty terms.
        max_iters: Maximum number of BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance. Defaults to
            ``DEFAULT_GRAD_TOL``.
        tpm: Threads per molecule (must be a power of 2, e.g. 1, 8, 32).

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

    # Pack inputs — same preprocessing as serial kernel
    inputs = _pack_kernel_inputs(system, chiral_weight, fourth_dim_weight,
                                  max_iters, grad_tol)

    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    # Build kernel with template parameters
    kernel = mx.fast.metal_kernel(
        name="dg_bfgs_tg",
        input_names=[
            "pos", "atom_starts", "hessian_starts",
            "dist_pairs", "dist_bounds", "dist_term_starts",
            "chiral_quads", "chiral_bounds", "chiral_term_starts",
            "fourth_idx_arr", "fourth_term_starts_arr", "config",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=_TG_MSL_HEADER,
        source=_TG_MSL_SOURCE,
    )

    # Launch: one threadgroup per molecule, TPM threads per threadgroup
    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'],
            inputs['hessian_starts'],
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
            (total_pos_size,),       # out_pos
            (n_mols,),               # out_energies
            (n_mols,),               # out_statuses
            (total_pos_size,),       # work_grad
            (total_pos_size,),       # work_dir
            (total_pos_size * 3,),   # work_scratch
            (max(total_hessian_size, 1),),  # work_hessian
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols * tpm, 1, 1),
        threadgroup=(tpm, 1, 1),
        template=[("TPM", tpm), ("total_pos_size", total_pos_size)],
    )

    return outputs[0], outputs[1], outputs[2]


def metal_dg_bfgs_binned(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    max_iters: int = 400,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run DG BFGS with size-based binning for efficiency.

    Groups molecules by atom count and dispatches separate kernel calls
    per bin. Falls back to vectorized BFGS for molecules exceeding
    ``MAX_ATOMS_METAL``.

    Args:
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        system: Batched DG system with all energy terms pre-packed.
        chiral_weight: Weight for chiral violation energy terms.
        fourth_dim_weight: Weight for fourth-dimension penalty terms.
        max_iters: Maximum number of BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance. Defaults to
            ``DEFAULT_GRAD_TOL``.

    Returns:
        Tuple of ``(final_pos, final_energies, statuses)`` with the same
        semantics as :func:`metal_dg_bfgs`.
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    # For now, just dispatch to metal_dg_bfgs directly
    # Size binning is a future optimization
    atom_starts = system.atom_starts.tolist()
    max_atoms = max(atom_starts[i + 1] - atom_starts[i] for i in range(system.n_mols))

    if max_atoms > MAX_ATOMS_METAL:
        # Fall back to vectorized BFGS for large molecules
        from .dg_bfgs_fallback import _fallback_large_molecules
        return _fallback_large_molecules(
            pos, system, chiral_weight, fourth_dim_weight, max_iters, grad_tol
        )

    return metal_dg_bfgs(
        pos, system, chiral_weight, fourth_dim_weight, max_iters, grad_tol
    )
