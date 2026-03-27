"""Metal kernel for MMFF94 BFGS minimization — entire optimization on-device.

One thread per molecule runs the complete BFGS loop (up to max_iters)
with all 7 MMFF energy+gradient terms computed inline. Zero Python round-trips.

Uses mx.fast.metal_kernel() to launch MSL code.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from ..minimizer.bfgs import DEFAULT_GRAD_TOL
from ..preprocessing.mmff_batching import BatchedMMFFSystem

_KERNEL_DIR = Path(__file__).parent

# Maximum atoms per molecule for Metal kernel (Hessian fits in device memory)
# 64 atoms * 3 dim = 192 terms -> 192*192*4 = ~144KB per Hessian
MAX_ATOMS_METAL = 64

# MSL helper functions for all 7 MMFF energy+gradient terms
_MSL_HEADER = (_KERNEL_DIR / "mmff_bfgs_header.metal").read_text()

# MSL kernel body — one thread per molecule
_MSL_SOURCE = (_KERNEL_DIR / "mmff_bfgs_source.metal").read_text()


def _pack_kernel_inputs(
    system: BatchedMMFFSystem,
    max_iters: int,
    grad_tol: float,
) -> dict[str, mx.array | int]:
    """Pack BatchedMMFFSystem fields into flat arrays for the Metal kernel."""
    n_mols = system.n_mols
    atom_starts_np = np.array(system.atom_starts, copy=False).astype(np.int32)

    # Compute hessian_starts from atom counts
    n_atoms_per_mol = np.diff(atom_starts_np)
    n_terms_per_mol = n_atoms_per_mol * 3
    hessian_sizes = n_terms_per_mol * n_terms_per_mol
    hessian_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    np.cumsum(hessian_sizes, out=hessian_starts_np[1:])

    total_pos_size = int(atom_starts_np[-1]) * 3
    total_hessian_size = int(hessian_starts_np[-1])

    def _to_np_i32(arr):
        return np.array(arr, copy=False).astype(np.int32) if arr.size > 0 else np.zeros(0, dtype=np.int32)

    def _to_np_f32(arr):
        return np.array(arr, copy=False).astype(np.float32) if arr.size > 0 else np.zeros(0, dtype=np.float32)

    def build_term_starts(mol_indices_arr):
        starts = np.zeros(n_mols + 1, dtype=np.int32)
        if mol_indices_arr.size > 0:
            mi = _to_np_i32(mol_indices_arr)
            counts = np.bincount(mi, minlength=n_mols)
            np.cumsum(counts, out=starts[1:])
        return starts

    bond_ts = build_term_starts(system.bond_mol_indices)
    angle_ts = build_term_starts(system.angle_mol_indices)
    sb_ts = build_term_starts(system.sb_mol_indices)
    oop_ts = build_term_starts(system.oop_mol_indices)
    tor_ts = build_term_starts(system.tor_mol_indices)
    vdw_ts = build_term_starts(system.vdw_mol_indices)
    ele_ts = build_term_starts(system.ele_mol_indices)

    # --- Pack term data using fast numpy conversion ---
    def _pack_pairs(idx1, idx2):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2)], axis=1).flatten()
        return np.zeros(2, dtype=np.int32)

    def _pack_trips(idx1, idx2, idx3):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2), _to_np_i32(idx3)], axis=1).flatten()
        return np.zeros(3, dtype=np.int32)

    def _pack_quads(idx1, idx2, idx3, idx4):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2), _to_np_i32(idx3), _to_np_i32(idx4)], axis=1).flatten()
        return np.zeros(4, dtype=np.int32)

    def _pack_params(*arrs, fallback_stride=1):
        if arrs[0].size > 0:
            return np.stack([_to_np_f32(a) for a in arrs], axis=1).flatten()
        return np.zeros(fallback_stride, dtype=np.float32)

    bond_pairs_np = _pack_pairs(system.bond_idx1, system.bond_idx2)
    bond_params_np = _pack_params(system.bond_kb, system.bond_r0, fallback_stride=2)

    angle_trips_np = _pack_trips(system.angle_idx1, system.angle_idx2, system.angle_idx3)
    angle_params_np = _pack_params(system.angle_ka, system.angle_theta0, system.angle_is_linear, fallback_stride=3)

    sb_trips_np = _pack_trips(system.sb_idx1, system.sb_idx2, system.sb_idx3)
    sb_params_np = _pack_params(system.sb_r0_ij, system.sb_r0_kj, system.sb_theta0, system.sb_kba_ij, system.sb_kba_kj, fallback_stride=5)

    oop_quads_np = _pack_quads(system.oop_idx1, system.oop_idx2, system.oop_idx3, system.oop_idx4)
    oop_params_np = _to_np_f32(system.oop_koop) if system.oop_koop.size > 0 else np.zeros(1, dtype=np.float32)

    tor_quads_np = _pack_quads(system.tor_idx1, system.tor_idx2, system.tor_idx3, system.tor_idx4)
    tor_params_np = _pack_params(system.tor_V1, system.tor_V2, system.tor_V3, fallback_stride=3)

    vdw_pairs_np = _pack_pairs(system.vdw_idx1, system.vdw_idx2)
    vdw_params_np = _pack_params(system.vdw_R_star, system.vdw_epsilon, fallback_stride=2)

    ele_pairs_np = _pack_pairs(system.ele_idx1, system.ele_idx2)
    ele_params_np = _pack_params(system.ele_charge_term, system.ele_diel_model, system.ele_is_1_4, fallback_stride=3)

    # Config array
    config_np = np.array([n_mols, max_iters, grad_tol], dtype=np.float32)

    # Combine all 7 term_starts into a single flat array
    all_term_starts_np = np.concatenate([
        bond_ts, angle_ts, sb_ts, oop_ts, tor_ts, vdw_ts, ele_ts
    ]).astype(np.int32)

    return {
        'atom_starts': mx.array(atom_starts_np),
        'hessian_starts': mx.array(hessian_starts_np),
        'config': mx.array(config_np),
        'all_term_starts': mx.array(all_term_starts_np),
        # Term data
        'bond_pairs': mx.array(bond_pairs_np),
        'bond_params': mx.array(bond_params_np),
        'angle_trips': mx.array(angle_trips_np),
        'angle_params': mx.array(angle_params_np),
        'sb_trips': mx.array(sb_trips_np),
        'sb_params': mx.array(sb_params_np),
        'oop_quads': mx.array(oop_quads_np),
        'oop_params': mx.array(oop_params_np),
        'tor_quads': mx.array(tor_quads_np),
        'tor_params': mx.array(tor_params_np),
        'vdw_pairs': mx.array(vdw_pairs_np),
        'vdw_params': mx.array(vdw_params_np),
        'ele_pairs': mx.array(ele_pairs_np),
        'ele_params': mx.array(ele_params_np),
        'total_pos_size': total_pos_size,
        'total_hessian_size': total_hessian_size,
    }


def metal_mmff_bfgs(
    pos: mx.array,
    system: BatchedMMFFSystem,
    max_iters: int = 200,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run MMFF BFGS minimization entirely on-device via a Metal kernel.

    Args:
        pos: Initial flat positions, shape (n_atoms_total * 3,), float32.
        system: Batched MMFF system with all energy terms pre-packed.
        max_iters: Maximum number of BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance.

    Returns:
        Tuple of (final_pos, final_energies, statuses).
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    inputs = _pack_kernel_inputs(system, max_iters, grad_tol)
    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    kernel = mx.fast.metal_kernel(
        name="mmff_bfgs",
        input_names=[
            "pos", "atom_starts", "hessian_starts", "config",
            "all_term_starts",
            "bond_pairs", "bond_params",
            "angle_trips", "angle_params",
            "sb_trips", "sb_params",
            "oop_quads", "oop_params",
            "tor_quads", "tor_params",
            "vdw_pairs", "vdw_params",
            "ele_pairs", "ele_params",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=_MSL_HEADER,
        source=_MSL_SOURCE,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'], inputs['hessian_starts'], inputs['config'],
            inputs['all_term_starts'],
            inputs['bond_pairs'], inputs['bond_params'],
            inputs['angle_trips'], inputs['angle_params'],
            inputs['sb_trips'], inputs['sb_params'],
            inputs['oop_quads'], inputs['oop_params'],
            inputs['tor_quads'], inputs['tor_params'],
            inputs['vdw_pairs'], inputs['vdw_params'],
            inputs['ele_pairs'], inputs['ele_params'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_hessian_size, 1),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols, 1, 1),
        threadgroup=(1, 1, 1),
        template=[("total_pos_size", total_pos_size)],
    )

    return outputs[0], outputs[1], outputs[2]


# Threadgroup size for the parallel kernel (Hessian/direction parallelism)
TG_SIZE = 32

# Threadgroup-parallel MSL kernel — TG_SIZE threads per molecule.
# Energy+gradient: thread 0 only (gradient functions use non-atomic +=).
# Hessian update + direction: all threads (embarrassingly parallel O(n²) ops).
_MSL_SOURCE_TG = (_KERNEL_DIR / "mmff_bfgs_source_tg.metal").read_text()


def metal_mmff_bfgs_tg(
    pos: mx.array,
    system: BatchedMMFFSystem,
    max_iters: int = 200,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run MMFF BFGS with threadgroup parallelism (TG_SIZE threads per molecule)."""
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    inputs = _pack_kernel_inputs(system, max_iters, grad_tol)
    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    tg_header = _MSL_HEADER + f"\nconstant int TG_SIZE_VAL = {TG_SIZE};\n"

    kernel = mx.fast.metal_kernel(
        name="mmff_bfgs_tg",
        input_names=[
            "pos", "atom_starts", "hessian_starts", "config",
            "all_term_starts",
            "bond_pairs", "bond_params",
            "angle_trips", "angle_params",
            "sb_trips", "sb_params",
            "oop_quads", "oop_params",
            "tor_quads", "tor_params",
            "vdw_pairs", "vdw_params",
            "ele_pairs", "ele_params",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=tg_header,
        source=_MSL_SOURCE_TG,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'], inputs['hessian_starts'], inputs['config'],
            inputs['all_term_starts'],
            inputs['bond_pairs'], inputs['bond_params'],
            inputs['angle_trips'], inputs['angle_params'],
            inputs['sb_trips'], inputs['sb_params'],
            inputs['oop_quads'], inputs['oop_params'],
            inputs['tor_quads'], inputs['tor_params'],
            inputs['vdw_pairs'], inputs['vdw_params'],
            inputs['ele_pairs'], inputs['ele_params'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_hessian_size, 1),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols * TG_SIZE, 1, 1),
        threadgroup=(TG_SIZE, 1, 1),
        template=[("total_pos_size", total_pos_size)],
    )

    return outputs[0], outputs[1], outputs[2]
