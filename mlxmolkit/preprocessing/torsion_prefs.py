"""Extract ETK 3D force field parameters from RDKit molecules.

Extracts experimental torsion preferences, improper torsion atoms,
bond/angle constraints, and long-range distance constraints needed
for the ETK minimization stage.

Port of nvMolKit's construct3DForceFieldContribs (dist_geom_flattened_builder.cpp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

if TYPE_CHECKING:
    from rdkit.Chem.rdDistGeom import EmbedParameters

# Constants matching nvMolKit's dist_geom_flattened_builder.cpp
KNOWN_DIST_FORCE_CONSTANT = 100.0
KNOWN_DIST_TOL = 0.01
TRIPLE_BOND_MIN_ANGLE = 179.0
TRIPLE_BOND_MAX_ANGLE = 180.0
IMPROPER_TORSION_FORCE_SCALING = 10.0


@dataclass
class ETK3DParams:
    """ETK 3D force field parameters for a single molecule."""

    num_atoms: int

    # Experimental torsion terms
    torsion_idx1: np.ndarray  # int32 (n_torsion,)
    torsion_idx2: np.ndarray  # int32
    torsion_idx3: np.ndarray  # int32
    torsion_idx4: np.ndarray  # int32
    torsion_fc: np.ndarray    # float32 (n_torsion, 6) force constants
    torsion_signs: np.ndarray # int32 (n_torsion, 6) signs

    # Improper torsion terms (3 permutations per improper atom)
    improper_idx1: np.ndarray  # int32 (n_improper,)
    improper_idx2: np.ndarray  # int32 — central atom
    improper_idx3: np.ndarray  # int32
    improper_idx4: np.ndarray  # int32
    improper_C0: np.ndarray    # float32 (n_improper,)
    improper_C1: np.ndarray    # float32
    improper_C2: np.ndarray    # float32
    improper_fc: np.ndarray    # float32
    num_improper_atoms: int    # Number of original improper centers (before permutations)

    # 1-2 distance constraints (bonds)
    dist12_idx1: np.ndarray     # int32 (n_dist12,)
    dist12_idx2: np.ndarray     # int32
    dist12_min: np.ndarray      # float32 — will be updated with reference positions
    dist12_max: np.ndarray      # float32

    # 1-3 distance constraints (angles)
    dist13_idx1: np.ndarray     # int32 (n_dist13,)
    dist13_idx2: np.ndarray     # int32
    dist13_min: np.ndarray      # float32
    dist13_max: np.ndarray      # float32
    dist13_is_improper: np.ndarray  # bool (n_dist13,)

    # Angle constraints (triple bonds)
    angle13_idx1: np.ndarray    # int32 (n_angle,)
    angle13_idx2: np.ndarray    # int32 — central atom
    angle13_idx3: np.ndarray    # int32
    angle13_min: np.ndarray     # float32 — in degrees
    angle13_max: np.ndarray     # float32

    # Long-range distance constraints
    long_range_idx1: np.ndarray  # int32 (n_long,)
    long_range_idx2: np.ndarray  # int32
    long_range_min: np.ndarray   # float32
    long_range_max: np.ndarray   # float32
    long_range_fc: np.ndarray    # float32

    # Atom pairs bitset tracking (for long-range exclusion)
    # Not stored — computed during extraction


def _calc_inversion_coefficients(
    atomic_num: int, is_c_bound_to_o: bool
) -> tuple[float, float, float, float]:
    """Compute inversion coefficients and force constant.

    Port of nvMolKit's calcInversionCoefficientsAndForceConstant.

    Args:
        atomic_num: Atomic number of the central atom.
        is_c_bound_to_o: Whether a carbon center is bonded to oxygen.

    Returns:
        Tuple of (force_constant, C0, C1, C2).
    """
    if atomic_num in (6, 7, 8):  # C, N, O — sp2
        C0 = 1.0
        C1 = -1.0
        C2 = 0.0
        fc = 50.0 if (atomic_num == 6 and is_c_bound_to_o) else 6.0
    else:
        # Group 5 elements
        inversion_angles = {15: 84.4339, 33: 86.9735, 51: 87.7047, 83: 90.0}
        angle_deg = inversion_angles.get(atomic_num, 1.0)
        angle_rad = math.radians(angle_deg)

        C2 = 1.0
        C1 = -4.0 * math.cos(angle_rad)
        C0 = -(C1 * math.cos(angle_rad) + C2 * math.cos(2.0 * angle_rad))
        fc = 22.0 / (C0 + C1 + C2)

    fc /= 3.0
    return fc, C0, C1, C2


def _extract_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Extract bonded atom pairs from molecular topology.

    Args:
        mol: RDKit molecule.

    Returns:
        List of (begin_atom_idx, end_atom_idx) tuples.
    """
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bonds


def _extract_angles(mol: Chem.Mol) -> list[tuple[int, int, int, bool]]:
    """Extract 1-3 atom triples from molecular topology.

    Args:
        mol: RDKit molecule.

    Returns:
        List of (idx1, central_idx, idx3, is_triple_bond) tuples.
    """
    angles = []
    for atom in mol.GetAtoms():
        central = atom.GetIdx()
        neighbors = sorted([n.GetIdx() for n in atom.GetNeighbors()])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a1, a3 = neighbors[i], neighbors[j]
                # Check if either bond is triple
                bond1 = mol.GetBondBetweenAtoms(central, a1)
                bond2 = mol.GetBondBetweenAtoms(central, a3)
                is_triple = (
                    (bond1 and bond1.GetBondType() == Chem.BondType.TRIPLE)
                    or (bond2 and bond2.GetBondType() == Chem.BondType.TRIPLE)
                )
                angles.append((a1, central, a3, is_triple))
    return angles


def _extract_improper_atoms(
    mol: Chem.Mol,
) -> list[tuple[int, int, int, int, int, bool]]:
    """Extract improper torsion atoms (sp2 centers with 3 neighbors).

    Args:
        mol: RDKit molecule.

    Returns:
        List of (neighbor0, central, neighbor1, neighbor2, atomic_num,
        is_c_bound_to_o) tuples.
    """
    impropers = []
    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        neighbors = sorted([n.GetIdx() for n in atom.GetNeighbors()])
        if len(neighbors) != 3:
            continue

        atomic_num = atom.GetAtomicNum()
        # Check if carbon bound to oxygen
        is_c_bound_to_o = False
        if atomic_num == 6:
            for n in atom.GetNeighbors():
                if n.GetAtomicNum() == 8:
                    is_c_bound_to_o = True
                    break

        impropers.append((
            neighbors[0], atom.GetIdx(), neighbors[1], neighbors[2],
            atomic_num, is_c_bound_to_o
        ))
    return impropers


def extract_etk_params(
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    params: EmbedParameters | None = None,
    use_exp_torsion: bool = True,
    use_basic_knowledge: bool = True,
    use_small_ring_torsions: bool = False,
    use_macrocycle_torsions: bool = True,
    et_version: int = 2,
    bounds_mat_force_scaling: float = 1.0,
) -> ETK3DParams:
    """Extract all ETK 3D force field parameters from an RDKit molecule.

    This mirrors nvMolKit's construct3DForceFieldContribs().

    Note: positions are not yet available at construction time. The
    dist12/dist13 min/max bounds will be updated with reference positions
    in the ETK minimization stage (setReferenceValues).

    Args:
        mol: RDKit molecule with hydrogens.
        bounds_mat: Distance bounds matrix.
        params: Optional EmbedParameters (overrides individual flags).
        use_exp_torsion: Include CSD torsion preferences.
        use_basic_knowledge: Include bond/angle knowledge and improper torsions.
        use_small_ring_torsions: Include small ring torsion preferences.
        use_macrocycle_torsions: Include macrocycle torsion preferences.
        et_version: CSD version (1 or 2).
        bounds_mat_force_scaling: Force scaling for bounds matrix constraints.

    Returns:
        ETK3DParams with all force field terms.
    """
    if params is not None:
        use_exp_torsion = params.useExpTorsionAnglePrefs
        use_basic_knowledge = params.useBasicKnowledge
        use_small_ring_torsions = params.useSmallRingTorsions
        use_macrocycle_torsions = params.useMacrocycleTorsions
        et_version = params.ETversion
        bounds_mat_force_scaling = params.boundsMatForceScaling

    num_atoms = mol.GetNumAtoms()

    # Track atom pairs for long-range exclusion
    atom_pairs = set()

    # --- 1. Experimental torsion terms ---
    t_idx1, t_idx2, t_idx3, t_idx4 = [], [], [], []
    t_fc_list, t_signs_list = [], []

    if use_exp_torsion or use_basic_knowledge:
        torsion_data = rdDistGeom.GetExperimentalTorsions(
            mol,
            useExpTorsionAnglePrefs=use_exp_torsion,
            useSmallRingTorsions=use_small_ring_torsions,
            useMacrocycleTorsions=use_macrocycle_torsions,
            useBasicKnowledge=use_basic_knowledge,
            ETversion=et_version,
        )

        for item in torsion_data:
            atoms = list(item['atomIndices'])
            if len(atoms) != 4:
                continue
            a1, a2, a3, a4 = atoms
            t_idx1.append(a1)
            t_idx2.append(a2)
            t_idx3.append(a3)
            t_idx4.append(a4)

            fc_vals = list(item['V'])
            sign_vals = list(item['signs'])
            # Pad to 6 if needed
            while len(fc_vals) < 6:
                fc_vals.append(0.0)
            while len(sign_vals) < 6:
                sign_vals.append(0)
            t_fc_list.append(fc_vals[:6])
            t_signs_list.append(sign_vals[:6])

            # Track pair
            lo, hi = min(a1, a4), max(a1, a4)
            atom_pairs.add((lo, hi))

    # --- 2. Improper torsion terms ---
    imp_idx1, imp_idx2, imp_idx3, imp_idx4 = [], [], [], []
    imp_C0, imp_C1, imp_C2, imp_fc_list = [], [], [], []
    is_improper_constrained = set()
    num_improper_atoms = 0

    if use_basic_knowledge:
        improper_atoms = _extract_improper_atoms(mol)
        num_improper_atoms = len(improper_atoms)

        for n0, central, n1, n2, atomic_num, is_c_bound_to_o in improper_atoms:
            base_fc, c0, c1, c2 = _calc_inversion_coefficients(
                atomic_num, is_c_bound_to_o
            )
            fc = base_fc * IMPROPER_TORSION_FORCE_SCALING

            # 3 permutations matching nvMolKit exactly
            permutations = [
                (n0, central, n1, n2),  # 0, 1, 2, 3
                (n0, central, n2, n1),  # 0, 1, 3, 2
                (n1, central, n2, n0),  # 2, 1, 3, 0
            ]
            for i1, i2, i3, i4 in permutations:
                imp_idx1.append(i1)
                imp_idx2.append(i2)
                imp_idx3.append(i3)
                imp_idx4.append(i4)
                imp_C0.append(c0)
                imp_C1.append(c1)
                imp_C2.append(c2)
                imp_fc_list.append(fc)

            is_improper_constrained.add(central)

    # --- 3. 1-2 distance terms (bonds) ---
    d12_idx1, d12_idx2, d12_min_list, d12_max_list = [], [], [], []
    bonds = _extract_bonds(mol)

    for a1, a2 in bonds:
        lo, hi = min(a1, a2), max(a1, a2)
        atom_pairs.add((lo, hi))

        # Initial bounds will be updated with reference positions later
        # Use bounds matrix as initial estimate
        lb = bounds_mat[max(a1, a2), min(a1, a2)]
        ub = bounds_mat[min(a1, a2), max(a1, a2)]
        mid = (lb + ub) / 2.0
        d12_idx1.append(a1)
        d12_idx2.append(a2)
        d12_min_list.append(mid - KNOWN_DIST_TOL)
        d12_max_list.append(mid + KNOWN_DIST_TOL)

    # --- 4. 1-3 distance/angle terms ---
    d13_idx1, d13_idx2, d13_min_list, d13_max_list = [], [], [], []
    d13_is_imp = []
    a13_idx1, a13_idx2, a13_idx3, a13_min_list, a13_max_list = [], [], [], [], []
    angles = _extract_angles(mol)

    for a1, central, a3, is_triple in angles:
        lo, hi = min(a1, a3), max(a1, a3)
        atom_pairs.add((lo, hi))

        if use_basic_knowledge and is_triple:
            # Angle constraint for triple bonds
            a13_idx1.append(a1)
            a13_idx2.append(central)
            a13_idx3.append(a3)
            a13_min_list.append(TRIPLE_BOND_MIN_ANGLE)
            a13_max_list.append(TRIPLE_BOND_MAX_ANGLE)
        elif central in is_improper_constrained:
            # Use bounds matrix for improper-constrained central atoms
            lb = bounds_mat[max(a1, a3), min(a1, a3)]
            ub = bounds_mat[min(a1, a3), max(a1, a3)]
            d13_idx1.append(a1)
            d13_idx2.append(a3)
            d13_min_list.append(lb)
            d13_max_list.append(ub)
            d13_is_imp.append(True)
        else:
            # Normal: use bounds matrix as initial estimate
            lb = bounds_mat[max(a1, a3), min(a1, a3)]
            ub = bounds_mat[min(a1, a3), max(a1, a3)]
            mid = (lb + ub) / 2.0
            d13_idx1.append(a1)
            d13_idx2.append(a3)
            d13_min_list.append(mid - KNOWN_DIST_TOL)
            d13_max_list.append(mid + KNOWN_DIST_TOL)
            d13_is_imp.append(False)

    # --- 5. Long-range distance constraints ---
    lr_idx1, lr_idx2, lr_min_list, lr_max_list, lr_fc_list = [], [], [], [], []
    fdist = bounds_mat_force_scaling * 10.0

    for i in range(1, num_atoms):
        for j in range(i):
            if (j, i) not in atom_pairs:
                lb = bounds_mat[i, j]
                ub = bounds_mat[j, i]
                lr_idx1.append(i)
                lr_idx2.append(j)
                lr_min_list.append(lb)
                lr_max_list.append(ub)
                lr_fc_list.append(fdist)

    def _arr_i32(lst: list[int]) -> np.ndarray:
        """Convert an int list to a numpy int32 array.

        Args:
            lst: List of integers.

        Returns:
            Numpy int32 array, or empty array if lst is empty.
        """
        return np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32)

    def _arr_f32(lst: list[float]) -> np.ndarray:
        """Convert a float list to a numpy float32 array.

        Args:
            lst: List of floats.

        Returns:
            Numpy float32 array, or empty array if lst is empty.
        """
        return np.array(lst, dtype=np.float32) if lst else np.array([], dtype=np.float32)

    torsion_fc = np.array(t_fc_list, dtype=np.float32).reshape(-1, 6) if t_fc_list else np.zeros((0, 6), dtype=np.float32)
    torsion_signs = np.array(t_signs_list, dtype=np.int32).reshape(-1, 6) if t_signs_list else np.zeros((0, 6), dtype=np.int32)

    return ETK3DParams(
        num_atoms=num_atoms,
        torsion_idx1=_arr_i32(t_idx1),
        torsion_idx2=_arr_i32(t_idx2),
        torsion_idx3=_arr_i32(t_idx3),
        torsion_idx4=_arr_i32(t_idx4),
        torsion_fc=torsion_fc,
        torsion_signs=torsion_signs,
        improper_idx1=_arr_i32(imp_idx1),
        improper_idx2=_arr_i32(imp_idx2),
        improper_idx3=_arr_i32(imp_idx3),
        improper_idx4=_arr_i32(imp_idx4),
        improper_C0=_arr_f32(imp_C0),
        improper_C1=_arr_f32(imp_C1),
        improper_C2=_arr_f32(imp_C2),
        improper_fc=_arr_f32(imp_fc_list),
        num_improper_atoms=num_improper_atoms,
        dist12_idx1=_arr_i32(d12_idx1),
        dist12_idx2=_arr_i32(d12_idx2),
        dist12_min=_arr_f32(d12_min_list),
        dist12_max=_arr_f32(d12_max_list),
        dist13_idx1=_arr_i32(d13_idx1),
        dist13_idx2=_arr_i32(d13_idx2),
        dist13_min=_arr_f32(d13_min_list),
        dist13_max=_arr_f32(d13_max_list),
        dist13_is_improper=np.array(d13_is_imp, dtype=bool) if d13_is_imp else np.array([], dtype=bool),
        angle13_idx1=_arr_i32(a13_idx1),
        angle13_idx2=_arr_i32(a13_idx2),
        angle13_idx3=_arr_i32(a13_idx3),
        angle13_min=_arr_f32(a13_min_list),
        angle13_max=_arr_f32(a13_max_list),
        long_range_idx1=_arr_i32(lr_idx1),
        long_range_idx2=_arr_i32(lr_idx2),
        long_range_min=_arr_f32(lr_min_list),
        long_range_max=_arr_f32(lr_max_list),
        long_range_fc=_arr_f32(lr_fc_list),
    )
