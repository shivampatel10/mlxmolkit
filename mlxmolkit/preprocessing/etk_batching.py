"""CSR batch assembly for ETK 3D force field terms.

Assembles per-molecule ETK3DParams into batched arrays with global indices.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .torsion_prefs import ETK3DParams


@dataclass
class BatchedETKSystem:
    """Batched ETK 3D force field system for multiple molecules."""

    n_mols: int
    dim: int  # Always 3 for ETK stage (but positions may be in dim=4 array)
    atom_starts: list[int]  # Python list for indexing

    # Experimental torsion terms
    torsion_idx1: mx.array
    torsion_idx2: mx.array
    torsion_idx3: mx.array
    torsion_idx4: mx.array
    torsion_fc: mx.array      # float32 (n_torsion, 6)
    torsion_signs: mx.array   # int32 (n_torsion, 6)
    torsion_mol_indices: mx.array

    # Improper torsion terms
    improper_idx1: mx.array
    improper_idx2: mx.array
    improper_idx3: mx.array
    improper_idx4: mx.array
    improper_C0: mx.array
    improper_C1: mx.array
    improper_C2: mx.array
    improper_fc: mx.array
    improper_mol_indices: mx.array
    num_impropers_per_mol: list[int]  # For planar tolerance check

    # 1-2 distance constraints
    dist12_idx1: mx.array
    dist12_idx2: mx.array
    dist12_min: mx.array   # Mutable — updated with reference positions
    dist12_max: mx.array
    dist12_fc: mx.array
    dist12_mol_indices: mx.array
    dist12_is_improper: mx.array  # Always False for dist12

    # 1-3 distance constraints
    dist13_idx1: mx.array
    dist13_idx2: mx.array
    dist13_min: mx.array   # Mutable — updated with reference positions
    dist13_max: mx.array
    dist13_fc: mx.array
    dist13_mol_indices: mx.array
    dist13_is_improper: mx.array  # bool — skip ref update for improper-constrained

    # Angle constraints (triple bonds)
    angle13_idx1: mx.array
    angle13_idx2: mx.array
    angle13_idx3: mx.array
    angle13_min_angle: mx.array
    angle13_max_angle: mx.array
    angle13_fc: mx.array
    angle13_mol_indices: mx.array

    # Long-range distance constraints
    long_range_idx1: mx.array
    long_range_idx2: mx.array
    long_range_min: mx.array  # Mutable — updated with reference positions
    long_range_max: mx.array
    long_range_fc: mx.array
    long_range_mol_indices: mx.array


KNOWN_DIST_FORCE_CONSTANT = 100.0
ANGLE_FORCE_CONSTANT = 100.0


def batch_etk_params(
    params_list: list[ETK3DParams],
    atom_starts: list[int],
    dim: int = 4,
) -> BatchedETKSystem:
    """Assemble per-molecule ETK3DParams into a batched system.

    Args:
        params_list: List of per-molecule ETK3DParams.
        atom_starts: Atom boundary list (n_mols + 1).
        dim: Coordinate dimension (typically 4, ETK uses 3D subset).

    Returns:
        BatchedETKSystem with global atom indices.
    """
    n_mols = len(params_list)

    def _batch_arrays(get_fn, idx_names, val_names):
        """Generic batching helper."""
        idx_parts = {name: [] for name in idx_names}
        val_parts = {name: [] for name in val_names}
        mol_parts = []

        for i, p in enumerate(params_list):
            offset = atom_starts[i]
            first_idx = get_fn(p, idx_names[0])
            n_terms = len(first_idx)
            if n_terms == 0:
                continue

            for name in idx_names:
                arr = get_fn(p, name)
                idx_parts[name].append(arr + offset)

            for name in val_names:
                val_parts[name].append(get_fn(p, name))

            mol_parts.append(np.full(n_terms, i, dtype=np.int32))

        return idx_parts, val_parts, mol_parts

    def _get_attr(p, name):
        return getattr(p, name)

    def _concat(parts):
        if parts:
            return mx.array(np.concatenate(parts))
        return mx.array(np.array([], dtype=np.float32))

    def _concat_i32(parts):
        if parts:
            return mx.array(np.concatenate(parts).astype(np.int32))
        return mx.array(np.array([], dtype=np.int32))

    def _concat_2d(parts, cols):
        if parts:
            return mx.array(np.concatenate(parts).reshape(-1, cols))
        return mx.array(np.zeros((0, cols), dtype=np.float32))

    # --- Torsion terms ---
    t_idx1, t_idx2, t_idx3, t_idx4 = [], [], [], []
    t_fc, t_signs, t_mol = [], [], []

    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.torsion_idx1)
        if n == 0:
            continue
        t_idx1.append(p.torsion_idx1 + offset)
        t_idx2.append(p.torsion_idx2 + offset)
        t_idx3.append(p.torsion_idx3 + offset)
        t_idx4.append(p.torsion_idx4 + offset)
        t_fc.append(p.torsion_fc)
        t_signs.append(p.torsion_signs)
        t_mol.append(np.full(n, i, dtype=np.int32))

    # --- Improper terms ---
    im_idx1, im_idx2, im_idx3, im_idx4 = [], [], [], []
    im_C0, im_C1, im_C2, im_fc, im_mol = [], [], [], [], []
    num_impropers_per_mol = []

    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.improper_idx1)
        num_impropers_per_mol.append(p.num_improper_atoms)
        if n == 0:
            continue
        im_idx1.append(p.improper_idx1 + offset)
        im_idx2.append(p.improper_idx2 + offset)
        im_idx3.append(p.improper_idx3 + offset)
        im_idx4.append(p.improper_idx4 + offset)
        im_C0.append(p.improper_C0)
        im_C1.append(p.improper_C1)
        im_C2.append(p.improper_C2)
        im_fc.append(p.improper_fc)
        im_mol.append(np.full(n, i, dtype=np.int32))

    # --- 1-2 distance terms ---
    d12_idx1, d12_idx2, d12_min, d12_max, d12_mol = [], [], [], [], []
    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.dist12_idx1)
        if n == 0:
            continue
        d12_idx1.append(p.dist12_idx1 + offset)
        d12_idx2.append(p.dist12_idx2 + offset)
        d12_min.append(p.dist12_min)
        d12_max.append(p.dist12_max)
        d12_mol.append(np.full(n, i, dtype=np.int32))

    # --- 1-3 distance terms ---
    d13_idx1, d13_idx2, d13_min, d13_max, d13_mol, d13_imp = [], [], [], [], [], []
    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.dist13_idx1)
        if n == 0:
            continue
        d13_idx1.append(p.dist13_idx1 + offset)
        d13_idx2.append(p.dist13_idx2 + offset)
        d13_min.append(p.dist13_min)
        d13_max.append(p.dist13_max)
        d13_mol.append(np.full(n, i, dtype=np.int32))
        d13_imp.append(p.dist13_is_improper)

    # --- Angle terms ---
    a_idx1, a_idx2, a_idx3, a_min, a_max, a_mol = [], [], [], [], [], []
    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.angle13_idx1)
        if n == 0:
            continue
        a_idx1.append(p.angle13_idx1 + offset)
        a_idx2.append(p.angle13_idx2 + offset)
        a_idx3.append(p.angle13_idx3 + offset)
        a_min.append(p.angle13_min)
        a_max.append(p.angle13_max)
        a_mol.append(np.full(n, i, dtype=np.int32))

    # --- Long-range distance terms ---
    lr_idx1, lr_idx2, lr_min, lr_max, lr_fc, lr_mol = [], [], [], [], [], []
    for i, p in enumerate(params_list):
        offset = atom_starts[i]
        n = len(p.long_range_idx1)
        if n == 0:
            continue
        lr_idx1.append(p.long_range_idx1 + offset)
        lr_idx2.append(p.long_range_idx2 + offset)
        lr_min.append(p.long_range_min)
        lr_max.append(p.long_range_max)
        lr_fc.append(p.long_range_fc)
        lr_mol.append(np.full(n, i, dtype=np.int32))

    n_d12 = sum(len(x) for x in d12_idx1) if d12_idx1 else 0

    return BatchedETKSystem(
        n_mols=n_mols,
        dim=dim,
        atom_starts=atom_starts,
        # Torsion
        torsion_idx1=_concat_i32(t_idx1),
        torsion_idx2=_concat_i32(t_idx2),
        torsion_idx3=_concat_i32(t_idx3),
        torsion_idx4=_concat_i32(t_idx4),
        torsion_fc=_concat_2d(t_fc, 6),
        torsion_signs=mx.array(np.concatenate(t_signs).reshape(-1, 6).astype(np.int32)) if t_signs else mx.array(np.zeros((0, 6), dtype=np.int32)),
        torsion_mol_indices=_concat_i32(t_mol),
        # Improper
        improper_idx1=_concat_i32(im_idx1),
        improper_idx2=_concat_i32(im_idx2),
        improper_idx3=_concat_i32(im_idx3),
        improper_idx4=_concat_i32(im_idx4),
        improper_C0=_concat(im_C0),
        improper_C1=_concat(im_C1),
        improper_C2=_concat(im_C2),
        improper_fc=_concat(im_fc),
        improper_mol_indices=_concat_i32(im_mol),
        num_impropers_per_mol=num_impropers_per_mol,
        # Dist12
        dist12_idx1=_concat_i32(d12_idx1),
        dist12_idx2=_concat_i32(d12_idx2),
        dist12_min=_concat(d12_min),
        dist12_max=_concat(d12_max),
        dist12_fc=mx.full((n_d12,), KNOWN_DIST_FORCE_CONSTANT, dtype=mx.float32) if n_d12 > 0 else mx.array(np.array([], dtype=np.float32)),
        dist12_mol_indices=_concat_i32(d12_mol),
        dist12_is_improper=mx.zeros(n_d12, dtype=mx.bool_) if n_d12 > 0 else mx.array(np.array([], dtype=np.bool_)),
        # Dist13
        dist13_idx1=_concat_i32(d13_idx1),
        dist13_idx2=_concat_i32(d13_idx2),
        dist13_min=_concat(d13_min),
        dist13_max=_concat(d13_max),
        dist13_fc=mx.full((sum(len(x) for x in d13_idx1),), KNOWN_DIST_FORCE_CONSTANT, dtype=mx.float32) if d13_idx1 else mx.array(np.array([], dtype=np.float32)),
        dist13_mol_indices=_concat_i32(d13_mol),
        dist13_is_improper=mx.array(np.concatenate(d13_imp)) if d13_imp else mx.array(np.array([], dtype=np.bool_)),
        # Angle
        angle13_idx1=_concat_i32(a_idx1),
        angle13_idx2=_concat_i32(a_idx2),
        angle13_idx3=_concat_i32(a_idx3),
        angle13_min_angle=_concat(a_min),
        angle13_max_angle=_concat(a_max),
        angle13_fc=mx.full((sum(len(x) for x in a_idx1),), ANGLE_FORCE_CONSTANT, dtype=mx.float32) if a_idx1 else mx.array(np.array([], dtype=np.float32)),
        angle13_mol_indices=_concat_i32(a_mol),
        # Long-range
        long_range_idx1=_concat_i32(lr_idx1),
        long_range_idx2=_concat_i32(lr_idx2),
        long_range_min=_concat(lr_min),
        long_range_max=_concat(lr_max),
        long_range_fc=_concat(lr_fc),
        long_range_mol_indices=_concat_i32(lr_mol),
    )
