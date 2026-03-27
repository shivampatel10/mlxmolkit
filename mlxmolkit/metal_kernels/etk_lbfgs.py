"""Metal kernel for ETK L-BFGS minimization with threadgroup parallelism.

Same threadgroup-parallel L-BFGS architecture as dg_lbfgs.py, but with ETK
energy terms (torsion, improper, dist12/13, angle, long-range).

The BFGS loop body is identical — only energy/gradient evaluation differs.
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
from ..preprocessing.etk_batching import BatchedETKSystem

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8
MAX_ATOMS_METAL = 64

_KERNEL_DIR = Path(__file__).parent
_METAL_CONTENT = (_KERNEL_DIR / "etk_lbfgs.metal").read_text()
_MSL_HEADER, _MSL_SOURCE = _METAL_CONTENT.split(
    "// ---- ETK_LBFGS_SPLIT ----\n"
)


def _pack_etk_inputs(
    system: BatchedETKSystem,
    use_basic_knowledge: bool,
    max_iters: int,
    grad_tol: float,
    lbfgs_m: int,
) -> dict[str, mx.array | int]:
    """Pack a BatchedETKSystem into flat arrays for the Metal kernel.

    Args:
        system: Batched ETK system containing all energy terms.
        use_basic_knowledge: Whether to include improper torsion terms.
        max_iters: Maximum number of L-BFGS iterations.
        grad_tol: Gradient convergence tolerance.
        lbfgs_m: L-BFGS history depth (number of stored vector pairs).

    Returns:
        Dictionary of kernel input arrays and scalar sizes.
    """
    dim = system.dim
    n_mols = system.n_mols
    atom_starts = system.atom_starts

    # L-BFGS history starts
    lbfgs_history_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i in range(n_mols):
        n_atoms = atom_starts[i + 1] - atom_starts[i]
        n_vars = n_atoms * dim
        lbfgs_history_starts_np[i + 1] = (
            lbfgs_history_starts_np[i] + 2 * lbfgs_m * n_vars
        )

    total_pos_size = atom_starts[-1] * dim
    total_lbfgs_size = int(lbfgs_history_starts_np[-1])

    config_np = np.array([n_mols, max_iters, grad_tol, dim,
                          1 if use_basic_knowledge else 0], dtype=np.float32)

    def _pack_quads(idx1, idx2, idx3, idx4):
        if idx1.size > 0:
            return np.stack([np.array(idx1.tolist()), np.array(idx2.tolist()),
                           np.array(idx3.tolist()), np.array(idx4.tolist())], axis=1).flatten().astype(np.int32)
        return np.zeros(4, dtype=np.int32)

    def _pack_pairs(idx1, idx2):
        if idx1.size > 0:
            return np.stack([np.array(idx1.tolist()), np.array(idx2.tolist())], axis=1).flatten().astype(np.int32)
        return np.zeros(2, dtype=np.int32)

    torsion_quads_np = _pack_quads(system.torsion_idx1, system.torsion_idx2,
                                    system.torsion_idx3, system.torsion_idx4)
    n_tor = system.torsion_idx1.size
    if n_tor > 0:
        torsion_fc_np = np.array(system.torsion_fc.tolist(), dtype=np.float32).flatten()
        torsion_signs_np = np.array(system.torsion_signs.tolist(), dtype=np.float32).flatten()
    else:
        torsion_fc_np = np.zeros(6, dtype=np.float32)
        torsion_signs_np = np.zeros(6, dtype=np.float32)

    improper_quads_np = _pack_quads(system.improper_idx1, system.improper_idx2,
                                     system.improper_idx3, system.improper_idx4)
    n_imp = system.improper_idx1.size
    if n_imp > 0:
        improper_coeffs_np = np.stack([
            np.array(system.improper_C0.tolist()), np.array(system.improper_C1.tolist()),
            np.array(system.improper_C2.tolist()), np.array(system.improper_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        improper_coeffs_np = np.zeros(4, dtype=np.float32)

    dist12_pairs_np = _pack_pairs(system.dist12_idx1, system.dist12_idx2)
    n_d12 = system.dist12_idx1.size
    if n_d12 > 0:
        dist12_bounds_np = np.stack([
            np.array(system.dist12_min.tolist()), np.array(system.dist12_max.tolist()),
            np.array(system.dist12_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        dist12_bounds_np = np.zeros(3, dtype=np.float32)

    dist13_pairs_np = _pack_pairs(system.dist13_idx1, system.dist13_idx2)
    n_d13 = system.dist13_idx1.size
    if n_d13 > 0:
        dist13_bounds_np = np.stack([
            np.array(system.dist13_min.tolist()), np.array(system.dist13_max.tolist()),
            np.array(system.dist13_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        dist13_bounds_np = np.zeros(3, dtype=np.float32)

    n_ang = system.angle13_idx1.size
    if n_ang > 0:
        angle_triples_np = np.stack([
            np.array(system.angle13_idx1.tolist()), np.array(system.angle13_idx2.tolist()),
            np.array(system.angle13_idx3.tolist()),
        ], axis=1).flatten().astype(np.int32)
        angle_bounds_np = np.stack([
            np.array(system.angle13_min_angle.tolist()), np.array(system.angle13_max_angle.tolist()),
            np.array(system.angle13_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        angle_triples_np = np.zeros(3, dtype=np.int32)
        angle_bounds_np = np.zeros(3, dtype=np.float32)

    lr_pairs_np = _pack_pairs(system.long_range_idx1, system.long_range_idx2)
    n_lr = system.long_range_idx1.size
    if n_lr > 0:
        lr_bounds_np = np.stack([
            np.array(system.long_range_min.tolist()), np.array(system.long_range_max.tolist()),
            np.array(system.long_range_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        lr_bounds_np = np.zeros(3, dtype=np.float32)

    def _build_term_starts(mol_indices, n_terms_total):
        starts = np.zeros(n_mols + 1, dtype=np.int32)
        if n_terms_total > 0:
            mi = np.array(mol_indices.tolist(), dtype=np.int32)
            for m in mi:
                starts[m + 1] += 1
            np.cumsum(starts, out=starts)
        return starts

    torsion_starts_np = _build_term_starts(system.torsion_mol_indices, n_tor)
    improper_starts_np = _build_term_starts(system.improper_mol_indices, n_imp)
    dist12_starts_np = _build_term_starts(system.dist12_mol_indices, n_d12)
    dist13_starts_np = _build_term_starts(system.dist13_mol_indices, n_d13)
    angle_starts_np = _build_term_starts(system.angle13_mol_indices, n_ang)
    lr_starts_np = _build_term_starts(system.long_range_mol_indices, n_lr)

    return {
        'atom_starts': mx.array(np.array(atom_starts, dtype=np.int32)),
        'lbfgs_history_starts': mx.array(lbfgs_history_starts_np),
        'config': mx.array(config_np),
        'torsion_quads': mx.array(torsion_quads_np),
        'torsion_fc': mx.array(torsion_fc_np),
        'torsion_signs_arr': mx.array(torsion_signs_np),
        'torsion_starts': mx.array(torsion_starts_np),
        'improper_quads': mx.array(improper_quads_np),
        'improper_coeffs': mx.array(improper_coeffs_np),
        'improper_starts': mx.array(improper_starts_np),
        'dist12_pairs': mx.array(dist12_pairs_np),
        'dist12_bounds': mx.array(dist12_bounds_np),
        'dist12_starts': mx.array(dist12_starts_np),
        'dist13_pairs': mx.array(dist13_pairs_np),
        'dist13_bounds': mx.array(dist13_bounds_np),
        'dist13_starts': mx.array(dist13_starts_np),
        'angle_triples': mx.array(angle_triples_np),
        'angle_bounds': mx.array(angle_bounds_np),
        'angle_starts': mx.array(angle_starts_np),
        'lr_pairs': mx.array(lr_pairs_np),
        'lr_bounds': mx.array(lr_bounds_np),
        'lr_starts': mx.array(lr_starts_np),
        'total_pos_size': total_pos_size,
        'total_lbfgs_size': total_lbfgs_size,
    }


def metal_etk_lbfgs(
    pos: mx.array,
    system: BatchedETKSystem,
    use_basic_knowledge: bool = True,
    max_iters: int = 300,
    grad_tol: float | None = None,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run ETK L-BFGS minimization on-device with threadgroup parallelism.

    Args:
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        system: Batched ETK system with all energy terms pre-packed.
        use_basic_knowledge: Whether to include improper torsion terms.
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
    inputs = _pack_etk_inputs(system, use_basic_knowledge, max_iters, grad_tol, lbfgs_m)
    total_pos_size = inputs['total_pos_size']
    total_lbfgs_size = inputs['total_lbfgs_size']

    kernel = mx.fast.metal_kernel(
        name="etk_lbfgs",
        input_names=[
            "pos", "atom_starts", "lbfgs_history_starts", "config",
            "torsion_quads", "torsion_fc", "torsion_signs_arr", "torsion_starts",
            "improper_quads", "improper_coeffs", "improper_starts",
            "dist12_pairs", "dist12_bounds", "dist12_starts",
            "dist13_pairs", "dist13_bounds", "dist13_starts",
            "angle_triples", "angle_bounds", "angle_starts",
            "lr_pairs", "lr_bounds", "lr_starts",
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
            inputs['config'],
            inputs['torsion_quads'],
            inputs['torsion_fc'],
            inputs['torsion_signs_arr'],
            inputs['torsion_starts'],
            inputs['improper_quads'],
            inputs['improper_coeffs'],
            inputs['improper_starts'],
            inputs['dist12_pairs'],
            inputs['dist12_bounds'],
            inputs['dist12_starts'],
            inputs['dist13_pairs'],
            inputs['dist13_bounds'],
            inputs['dist13_starts'],
            inputs['angle_triples'],
            inputs['angle_bounds'],
            inputs['angle_starts'],
            inputs['lr_pairs'],
            inputs['lr_bounds'],
            inputs['lr_starts'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_lbfgs_size, 1),),
            (max(n_mols * lbfgs_m, 1),),
            (max(n_mols * lbfgs_m, 1),),
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
