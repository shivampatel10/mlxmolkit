"""CSR batch assembly for MMFF94 force field parameters.

Assembles per-molecule MMFFParams into batched arrays with CSR indexing,
suitable for efficient GPU processing on MLX. Follows the pattern in batching.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .mmff_extract import MMFFParams


@dataclass
class BatchedMMFFSystem:
    """Batched MMFF94 system for multiple molecules.

    All arrays use global (batch-level) atom/term indices.
    """

    n_mols: int
    atom_starts: mx.array  # int32 (n_mols + 1,)
    n_atoms_total: int

    # Bond stretch (n_bond_terms,)
    bond_idx1: mx.array  # int32
    bond_idx2: mx.array  # int32
    bond_kb: mx.array  # float32
    bond_r0: mx.array  # float32
    bond_mol_indices: mx.array  # int32

    # Angle bend (n_angle_terms,)
    angle_idx1: mx.array  # int32
    angle_idx2: mx.array  # int32 central atom
    angle_idx3: mx.array  # int32
    angle_ka: mx.array  # float32
    angle_theta0: mx.array  # float32 degrees
    angle_is_linear: mx.array  # bool
    angle_mol_indices: mx.array  # int32

    # Stretch-bend (n_sb_terms,)
    sb_idx1: mx.array  # int32
    sb_idx2: mx.array  # int32 central atom
    sb_idx3: mx.array  # int32
    sb_r0_ij: mx.array  # float32
    sb_r0_kj: mx.array  # float32
    sb_theta0: mx.array  # float32 degrees
    sb_kba_ij: mx.array  # float32
    sb_kba_kj: mx.array  # float32
    sb_mol_indices: mx.array  # int32

    # Out-of-plane (n_oop_terms,)
    oop_idx1: mx.array  # int32
    oop_idx2: mx.array  # int32 central atom
    oop_idx3: mx.array  # int32
    oop_idx4: mx.array  # int32
    oop_koop: mx.array  # float32
    oop_mol_indices: mx.array  # int32

    # Torsion (n_torsion_terms,)
    tor_idx1: mx.array  # int32
    tor_idx2: mx.array  # int32
    tor_idx3: mx.array  # int32
    tor_idx4: mx.array  # int32
    tor_V1: mx.array  # float32
    tor_V2: mx.array  # float32
    tor_V3: mx.array  # float32
    tor_mol_indices: mx.array  # int32

    # Van der Waals (n_vdw_terms,)
    vdw_idx1: mx.array  # int32
    vdw_idx2: mx.array  # int32
    vdw_R_star: mx.array  # float32
    vdw_epsilon: mx.array  # float32
    vdw_mol_indices: mx.array  # int32

    # Electrostatic (n_ele_terms,)
    ele_idx1: mx.array  # int32
    ele_idx2: mx.array  # int32
    ele_charge_term: mx.array  # float32
    ele_diel_model: mx.array  # int32
    ele_is_1_4: mx.array  # bool
    ele_mol_indices: mx.array  # int32


def _concat_or_empty(parts: list[np.ndarray], dtype: np.dtype) -> mx.array:
    """Concatenate numpy arrays into an MLX array, or return an empty array."""
    if parts:
        return mx.array(np.concatenate(parts).astype(dtype))
    return mx.array(np.array([], dtype=dtype))


def batch_mmff_params(params_list: list[MMFFParams]) -> BatchedMMFFSystem:
    """Assemble per-molecule MMFF parameters into a batched system.

    Applies atom index offsets so all indices are global (batch-level).

    Args:
        params_list: List of per-molecule MMFFParams.

    Returns:
        BatchedMMFFSystem with all terms batched.
    """
    n_mols = len(params_list)

    # Compute atom starts (CSR boundaries)
    atom_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i, p in enumerate(params_list):
        atom_starts_np[i + 1] = atom_starts_np[i] + p.num_atoms
    n_atoms_total = int(atom_starts_np[-1])

    # --- Bond terms ---
    bond_idx1_parts, bond_idx2_parts = [], []
    bond_kb_parts, bond_r0_parts = [], []
    bond_mol_parts = []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.bond_terms.idx1)
        if n > 0:
            bond_idx1_parts.append(p.bond_terms.idx1 + offset)
            bond_idx2_parts.append(p.bond_terms.idx2 + offset)
            bond_kb_parts.append(p.bond_terms.kb)
            bond_r0_parts.append(p.bond_terms.r0)
            bond_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- Angle terms ---
    ang_idx1_parts, ang_idx2_parts, ang_idx3_parts = [], [], []
    ang_ka_parts, ang_theta0_parts = [], []
    ang_linear_parts, ang_mol_parts = [], []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.angle_terms.idx1)
        if n > 0:
            ang_idx1_parts.append(p.angle_terms.idx1 + offset)
            ang_idx2_parts.append(p.angle_terms.idx2 + offset)
            ang_idx3_parts.append(p.angle_terms.idx3 + offset)
            ang_ka_parts.append(p.angle_terms.ka)
            ang_theta0_parts.append(p.angle_terms.theta0)
            ang_linear_parts.append(p.angle_terms.is_linear)
            ang_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- Stretch-bend terms ---
    sb_idx1_parts, sb_idx2_parts, sb_idx3_parts = [], [], []
    sb_r0_ij_parts, sb_r0_kj_parts, sb_theta0_parts = [], [], []
    sb_kba_ij_parts, sb_kba_kj_parts = [], []
    sb_mol_parts = []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.stretch_bend_terms.idx1)
        if n > 0:
            sb_idx1_parts.append(p.stretch_bend_terms.idx1 + offset)
            sb_idx2_parts.append(p.stretch_bend_terms.idx2 + offset)
            sb_idx3_parts.append(p.stretch_bend_terms.idx3 + offset)
            sb_r0_ij_parts.append(p.stretch_bend_terms.r0_ij)
            sb_r0_kj_parts.append(p.stretch_bend_terms.r0_kj)
            sb_theta0_parts.append(p.stretch_bend_terms.theta0)
            sb_kba_ij_parts.append(p.stretch_bend_terms.kba_ij)
            sb_kba_kj_parts.append(p.stretch_bend_terms.kba_kj)
            sb_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- OOP terms ---
    oop_idx1_parts, oop_idx2_parts = [], []
    oop_idx3_parts, oop_idx4_parts = [], []
    oop_koop_parts, oop_mol_parts = [], []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.oop_terms.idx1)
        if n > 0:
            oop_idx1_parts.append(p.oop_terms.idx1 + offset)
            oop_idx2_parts.append(p.oop_terms.idx2 + offset)
            oop_idx3_parts.append(p.oop_terms.idx3 + offset)
            oop_idx4_parts.append(p.oop_terms.idx4 + offset)
            oop_koop_parts.append(p.oop_terms.koop)
            oop_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- Torsion terms ---
    tor_idx1_parts, tor_idx2_parts = [], []
    tor_idx3_parts, tor_idx4_parts = [], []
    tor_V1_parts, tor_V2_parts, tor_V3_parts = [], [], []
    tor_mol_parts = []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.torsion_terms.idx1)
        if n > 0:
            tor_idx1_parts.append(p.torsion_terms.idx1 + offset)
            tor_idx2_parts.append(p.torsion_terms.idx2 + offset)
            tor_idx3_parts.append(p.torsion_terms.idx3 + offset)
            tor_idx4_parts.append(p.torsion_terms.idx4 + offset)
            tor_V1_parts.append(p.torsion_terms.V1)
            tor_V2_parts.append(p.torsion_terms.V2)
            tor_V3_parts.append(p.torsion_terms.V3)
            tor_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- VdW terms ---
    vdw_idx1_parts, vdw_idx2_parts = [], []
    vdw_R_star_parts, vdw_eps_parts = [], []
    vdw_mol_parts = []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.vdw_terms.idx1)
        if n > 0:
            vdw_idx1_parts.append(p.vdw_terms.idx1 + offset)
            vdw_idx2_parts.append(p.vdw_terms.idx2 + offset)
            vdw_R_star_parts.append(p.vdw_terms.R_ij_star)
            vdw_eps_parts.append(p.vdw_terms.epsilon)
            vdw_mol_parts.append(np.full(n, i, dtype=np.int32))

    # --- Electrostatic terms ---
    ele_idx1_parts, ele_idx2_parts = [], []
    ele_ct_parts, ele_dm_parts, ele_14_parts = [], [], []
    ele_mol_parts = []
    for i, p in enumerate(params_list):
        offset = atom_starts_np[i]
        n = len(p.ele_terms.idx1)
        if n > 0:
            ele_idx1_parts.append(p.ele_terms.idx1 + offset)
            ele_idx2_parts.append(p.ele_terms.idx2 + offset)
            ele_ct_parts.append(p.ele_terms.charge_term)
            ele_dm_parts.append(p.ele_terms.diel_model)
            ele_14_parts.append(p.ele_terms.is_1_4)
            ele_mol_parts.append(np.full(n, i, dtype=np.int32))

    return BatchedMMFFSystem(
        n_mols=n_mols,
        atom_starts=mx.array(atom_starts_np),
        n_atoms_total=n_atoms_total,
        # Bond
        bond_idx1=_concat_or_empty(bond_idx1_parts, np.int32),
        bond_idx2=_concat_or_empty(bond_idx2_parts, np.int32),
        bond_kb=_concat_or_empty(bond_kb_parts, np.float32),
        bond_r0=_concat_or_empty(bond_r0_parts, np.float32),
        bond_mol_indices=_concat_or_empty(bond_mol_parts, np.int32),
        # Angle
        angle_idx1=_concat_or_empty(ang_idx1_parts, np.int32),
        angle_idx2=_concat_or_empty(ang_idx2_parts, np.int32),
        angle_idx3=_concat_or_empty(ang_idx3_parts, np.int32),
        angle_ka=_concat_or_empty(ang_ka_parts, np.float32),
        angle_theta0=_concat_or_empty(ang_theta0_parts, np.float32),
        angle_is_linear=_concat_or_empty(ang_linear_parts, bool),
        angle_mol_indices=_concat_or_empty(ang_mol_parts, np.int32),
        # Stretch-bend
        sb_idx1=_concat_or_empty(sb_idx1_parts, np.int32),
        sb_idx2=_concat_or_empty(sb_idx2_parts, np.int32),
        sb_idx3=_concat_or_empty(sb_idx3_parts, np.int32),
        sb_r0_ij=_concat_or_empty(sb_r0_ij_parts, np.float32),
        sb_r0_kj=_concat_or_empty(sb_r0_kj_parts, np.float32),
        sb_theta0=_concat_or_empty(sb_theta0_parts, np.float32),
        sb_kba_ij=_concat_or_empty(sb_kba_ij_parts, np.float32),
        sb_kba_kj=_concat_or_empty(sb_kba_kj_parts, np.float32),
        sb_mol_indices=_concat_or_empty(sb_mol_parts, np.int32),
        # OOP
        oop_idx1=_concat_or_empty(oop_idx1_parts, np.int32),
        oop_idx2=_concat_or_empty(oop_idx2_parts, np.int32),
        oop_idx3=_concat_or_empty(oop_idx3_parts, np.int32),
        oop_idx4=_concat_or_empty(oop_idx4_parts, np.int32),
        oop_koop=_concat_or_empty(oop_koop_parts, np.float32),
        oop_mol_indices=_concat_or_empty(oop_mol_parts, np.int32),
        # Torsion
        tor_idx1=_concat_or_empty(tor_idx1_parts, np.int32),
        tor_idx2=_concat_or_empty(tor_idx2_parts, np.int32),
        tor_idx3=_concat_or_empty(tor_idx3_parts, np.int32),
        tor_idx4=_concat_or_empty(tor_idx4_parts, np.int32),
        tor_V1=_concat_or_empty(tor_V1_parts, np.float32),
        tor_V2=_concat_or_empty(tor_V2_parts, np.float32),
        tor_V3=_concat_or_empty(tor_V3_parts, np.float32),
        tor_mol_indices=_concat_or_empty(tor_mol_parts, np.int32),
        # VdW
        vdw_idx1=_concat_or_empty(vdw_idx1_parts, np.int32),
        vdw_idx2=_concat_or_empty(vdw_idx2_parts, np.int32),
        vdw_R_star=_concat_or_empty(vdw_R_star_parts, np.float32),
        vdw_epsilon=_concat_or_empty(vdw_eps_parts, np.float32),
        vdw_mol_indices=_concat_or_empty(vdw_mol_parts, np.int32),
        # Electrostatic
        ele_idx1=_concat_or_empty(ele_idx1_parts, np.int32),
        ele_idx2=_concat_or_empty(ele_idx2_parts, np.int32),
        ele_charge_term=_concat_or_empty(ele_ct_parts, np.float32),
        ele_diel_model=_concat_or_empty(ele_dm_parts, np.int32),
        ele_is_1_4=_concat_or_empty(ele_14_parts, bool),
        ele_mol_indices=_concat_or_empty(ele_mol_parts, np.int32),
    )
