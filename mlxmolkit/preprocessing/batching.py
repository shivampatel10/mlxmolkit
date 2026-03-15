"""CSR batch assembly for multiple molecules.

Assembles per-molecule DG parameters into batched arrays with CSR indexing,
suitable for efficient GPU processing on MLX.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .rdkit_extract import DGParams


@dataclass
class BatchedDGSystem:
    """Batched distance geometry system for multiple molecules.

    All arrays use global (batch-level) atom/term indices.
    CSR-style *_starts arrays map molecule boundaries.
    """

    n_mols: int
    dim: int

    # Molecule boundaries
    atom_starts: mx.array  # int32 (n_mols + 1,) cumulative atom counts
    n_atoms_total: int

    # Distance violation terms
    dist_idx1: mx.array  # int32 (n_dist_terms,) global atom index
    dist_idx2: mx.array  # int32 (n_dist_terms,)
    dist_lb2: mx.array  # float32 (n_dist_terms,) squared lower bounds
    dist_ub2: mx.array  # float32 (n_dist_terms,) squared upper bounds
    dist_weight: mx.array  # float32 (n_dist_terms,)
    dist_term_starts: mx.array  # int32 (n_mols + 1,)
    dist_mol_indices: mx.array  # int32 (n_dist_terms,) molecule owning each term

    # Chiral violation terms
    chiral_idx1: mx.array  # int32 (n_chiral_terms,) global atom index
    chiral_idx2: mx.array  # int32 (n_chiral_terms,)
    chiral_idx3: mx.array  # int32 (n_chiral_terms,)
    chiral_idx4: mx.array  # int32 (n_chiral_terms,)
    chiral_vol_lower: mx.array  # float32 (n_chiral_terms,)
    chiral_vol_upper: mx.array  # float32 (n_chiral_terms,)
    chiral_term_starts: mx.array  # int32 (n_mols + 1,)
    chiral_mol_indices: mx.array  # int32 (n_chiral_terms,)

    # Fourth dimension terms
    fourth_idx: mx.array  # int32 (n_fourth_terms,) global atom index
    fourth_term_starts: mx.array  # int32 (n_mols + 1,)
    fourth_mol_indices: mx.array  # int32 (n_fourth_terms,)


def batch_dg_params(params_list: list[DGParams], dim: int = 4) -> BatchedDGSystem:
    """Assemble per-molecule DG parameters into a batched system.

    Applies atom index offsets so all indices are global (batch-level).
    Builds CSR term_starts arrays for per-molecule access.

    Args:
        params_list: List of per-molecule DGParams.
        dim: Coordinate dimension (3 or 4).

    Returns:
        BatchedDGSystem with all terms batched.
    """
    n_mols = len(params_list)

    # Compute atom starts (CSR boundaries)
    atom_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i, p in enumerate(params_list):
        atom_starts_np[i + 1] = atom_starts_np[i] + p.num_atoms
    n_atoms_total = int(atom_starts_np[-1])

    # --- Distance terms ---
    dist_idx1_parts = []
    dist_idx2_parts = []
    dist_lb2_parts = []
    dist_ub2_parts = []
    dist_weight_parts = []
    dist_mol_indices_parts = []
    dist_term_starts_np = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n_terms = len(p.dist_terms.idx1)
        dist_term_starts_np[i + 1] = dist_term_starts_np[i] + n_terms

        if n_terms > 0:
            dist_idx1_parts.append(p.dist_terms.idx1 + offset)
            dist_idx2_parts.append(p.dist_terms.idx2 + offset)
            dist_lb2_parts.append(p.dist_terms.lb2)
            dist_ub2_parts.append(p.dist_terms.ub2)
            dist_weight_parts.append(p.dist_terms.weight)
            dist_mol_indices_parts.append(np.full(n_terms, i, dtype=np.int32))

    # --- Chiral terms ---
    chiral_idx1_parts = []
    chiral_idx2_parts = []
    chiral_idx3_parts = []
    chiral_idx4_parts = []
    chiral_vol_lower_parts = []
    chiral_vol_upper_parts = []
    chiral_mol_indices_parts = []
    chiral_term_starts_np = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n_terms = len(p.chiral_terms.idx1)
        chiral_term_starts_np[i + 1] = chiral_term_starts_np[i] + n_terms

        if n_terms > 0:
            chiral_idx1_parts.append(p.chiral_terms.idx1 + offset)
            chiral_idx2_parts.append(p.chiral_terms.idx2 + offset)
            chiral_idx3_parts.append(p.chiral_terms.idx3 + offset)
            chiral_idx4_parts.append(p.chiral_terms.idx4 + offset)
            chiral_vol_lower_parts.append(p.chiral_terms.vol_lower)
            chiral_vol_upper_parts.append(p.chiral_terms.vol_upper)
            chiral_mol_indices_parts.append(np.full(n_terms, i, dtype=np.int32))

    # --- Fourth dim terms ---
    fourth_idx_parts = []
    fourth_mol_indices_parts = []
    fourth_term_starts_np = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n_terms = len(p.fourth_dim_terms.idx)
        fourth_term_starts_np[i + 1] = fourth_term_starts_np[i] + n_terms

        if n_terms > 0:
            fourth_idx_parts.append(p.fourth_dim_terms.idx + offset)
            fourth_mol_indices_parts.append(np.full(n_terms, i, dtype=np.int32))

    def _concat_or_empty(parts, dtype):
        if parts:
            return mx.array(np.concatenate(parts).astype(dtype))
        return mx.array(np.array([], dtype=dtype))

    return BatchedDGSystem(
        n_mols=n_mols,
        dim=dim,
        atom_starts=mx.array(atom_starts_np),
        n_atoms_total=n_atoms_total,
        # Distance terms
        dist_idx1=_concat_or_empty(dist_idx1_parts, np.int32),
        dist_idx2=_concat_or_empty(dist_idx2_parts, np.int32),
        dist_lb2=_concat_or_empty(dist_lb2_parts, np.float32),
        dist_ub2=_concat_or_empty(dist_ub2_parts, np.float32),
        dist_weight=_concat_or_empty(dist_weight_parts, np.float32),
        dist_term_starts=mx.array(dist_term_starts_np),
        dist_mol_indices=_concat_or_empty(dist_mol_indices_parts, np.int32),
        # Chiral terms
        chiral_idx1=_concat_or_empty(chiral_idx1_parts, np.int32),
        chiral_idx2=_concat_or_empty(chiral_idx2_parts, np.int32),
        chiral_idx3=_concat_or_empty(chiral_idx3_parts, np.int32),
        chiral_idx4=_concat_or_empty(chiral_idx4_parts, np.int32),
        chiral_vol_lower=_concat_or_empty(chiral_vol_lower_parts, np.float32),
        chiral_vol_upper=_concat_or_empty(chiral_vol_upper_parts, np.float32),
        chiral_term_starts=mx.array(chiral_term_starts_np),
        chiral_mol_indices=_concat_or_empty(chiral_mol_indices_parts, np.int32),
        # Fourth dim terms
        fourth_idx=_concat_or_empty(fourth_idx_parts, np.int32),
        fourth_term_starts=mx.array(fourth_term_starts_np),
        fourth_mol_indices=_concat_or_empty(fourth_mol_indices_parts, np.int32),
    )
