"""Extract MMFF94 force field parameters from RDKit molecules.

Converts RDKit molecular data into numpy arrays suitable for MLX computation.
Follows the same pattern as rdkit_extract.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers


# RDKit MMFFProp::linh is set for atom types 4, 53, and 61 in Params.cpp.
_MMFF_LINEAR_ATOM_TYPES = frozenset({4, 53, 61})


@dataclass
class BondStretchTerms:
    """MMFF bond stretch terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    kb: np.ndarray  # float32 (n_terms,)
    r0: np.ndarray  # float32 (n_terms,)


@dataclass
class AngleBendTerms:
    """MMFF angle bend terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,) central atom
    idx3: np.ndarray  # int32 (n_terms,)
    ka: np.ndarray  # float32 (n_terms,)
    theta0: np.ndarray  # float32 (n_terms,) degrees
    is_linear: np.ndarray  # bool (n_terms,)


@dataclass
class StretchBendTerms:
    """MMFF stretch-bend cross terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,) central atom
    idx3: np.ndarray  # int32 (n_terms,)
    r0_ij: np.ndarray  # float32 (n_terms,)
    r0_kj: np.ndarray  # float32 (n_terms,)
    theta0: np.ndarray  # float32 (n_terms,) degrees
    kba_ij: np.ndarray  # float32 (n_terms,)
    kba_kj: np.ndarray  # float32 (n_terms,)


@dataclass
class OutOfPlaneTerms:
    """MMFF out-of-plane bend terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,) central atom
    idx3: np.ndarray  # int32 (n_terms,)
    idx4: np.ndarray  # int32 (n_terms,)
    koop: np.ndarray  # float32 (n_terms,)


@dataclass
class TorsionTerms:
    """MMFF torsion terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    idx3: np.ndarray  # int32 (n_terms,)
    idx4: np.ndarray  # int32 (n_terms,)
    V1: np.ndarray  # float32 (n_terms,)
    V2: np.ndarray  # float32 (n_terms,)
    V3: np.ndarray  # float32 (n_terms,)


@dataclass
class VdwTerms:
    """MMFF Van der Waals terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    R_ij_star: np.ndarray  # float32 (n_terms,)
    epsilon: np.ndarray  # float32 (n_terms,)


@dataclass
class ElectrostaticTerms:
    """MMFF electrostatic terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    charge_term: np.ndarray  # float32 (n_terms,) qi*qj/dielConst
    diel_model: np.ndarray  # int32 (n_terms,) 1 or 2
    is_1_4: np.ndarray  # bool (n_terms,)


@dataclass
class MMFFParams:
    """MMFF94 force field parameters for a single molecule."""

    num_atoms: int
    bond_terms: BondStretchTerms
    angle_terms: AngleBendTerms
    stretch_bend_terms: StretchBendTerms
    oop_terms: OutOfPlaneTerms
    torsion_terms: TorsionTerms
    vdw_terms: VdwTerms
    ele_terms: ElectrostaticTerms


def _empty_int(n=0):
    return np.array([], dtype=np.int32) if n == 0 else np.zeros(n, dtype=np.int32)


def _empty_float(n=0):
    return np.array([], dtype=np.float32) if n == 0 else np.zeros(n, dtype=np.float32)


def _empty_bool(n=0):
    return np.array([], dtype=bool) if n == 0 else np.zeros(n, dtype=bool)


def _is_mmff_linear_atom_type(atom_type: int) -> bool:
    """Return RDKit MMFFProp::linh semantics for the given atom type."""
    return atom_type in _MMFF_LINEAR_ATOM_TYPES


def extract_mmff_params(
    mol: Chem.Mol,
    conf_id: int = -1,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
) -> MMFFParams | None:
    """Extract MMFF94 parameters from an RDKit molecule.

    Args:
        mol: RDKit molecule with hydrogens and at least one conformer.
        conf_id: Conformer ID for non-bonded distance filtering.
        nonBondedThreshold: Distance cutoff for non-bonded terms (Angstroms).
        ignoreInterfragInteractions: Whether to exclude non-bonded interactions
            between disconnected fragments. Matches RDKit MMFF defaults.

    Returns:
        MMFFParams or None if molecule lacks valid MMFF atom types.
    """
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None

    num_atoms = mol.GetNumAtoms()

    # --- Bond Stretch Terms ---
    bond_idx1, bond_idx2, bond_kb, bond_r0 = [], [], [], []
    # Also build a lookup dict for rest lengths (used by stretch-bend)
    bond_r0_dict: dict[tuple[int, int], float] = {}

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        res = props.GetMMFFBondStretchParams(mol, i, j)
        if res is not None and res:
            # res = (bondType, kb, r0)
            kb_val = float(res[1])
            r0_val = float(res[2])
            bond_idx1.append(i)
            bond_idx2.append(j)
            bond_kb.append(kb_val)
            bond_r0.append(r0_val)
            key = (min(i, j), max(i, j))
            bond_r0_dict[key] = r0_val

    bond_terms = BondStretchTerms(
        idx1=np.array(bond_idx1, dtype=np.int32) if bond_idx1 else _empty_int(),
        idx2=np.array(bond_idx2, dtype=np.int32) if bond_idx1 else _empty_int(),
        kb=np.array(bond_kb, dtype=np.float32) if bond_idx1 else _empty_float(),
        r0=np.array(bond_r0, dtype=np.float32) if bond_idx1 else _empty_float(),
    )

    # --- Angle Bend Terms ---
    ang_idx1, ang_idx2, ang_idx3 = [], [], []
    ang_ka, ang_theta0 = [], []
    ang_is_linear = []
    # Also build lookup for theta0 (used by stretch-bend)
    angle_theta0_dict: dict[tuple[int, int, int], float] = {}

    for j in range(num_atoms):
        atom_j = mol.GetAtomWithIdx(j)
        if atom_j.GetDegree() < 2:
            continue
        neighbors = [n.GetIdx() for n in atom_j.GetNeighbors()]
        for ni in range(len(neighbors)):
            for nk in range(ni + 1, len(neighbors)):
                i = neighbors[ni]
                k = neighbors[nk]
                res = props.GetMMFFAngleBendParams(mol, i, j, k)
                if res is not None and res:
                    # res = (angleType, ka, theta0)
                    ka_val = float(res[1])
                    theta0_val = float(res[2])
                    is_lin = _is_mmff_linear_atom_type(props.GetMMFFAtomType(j))
                    ang_idx1.append(i)
                    ang_idx2.append(j)
                    ang_idx3.append(k)
                    ang_ka.append(ka_val)
                    ang_theta0.append(theta0_val)
                    ang_is_linear.append(is_lin)
                    angle_theta0_dict[(i, j, k)] = theta0_val

    angle_terms = AngleBendTerms(
        idx1=np.array(ang_idx1, dtype=np.int32) if ang_idx1 else _empty_int(),
        idx2=np.array(ang_idx2, dtype=np.int32) if ang_idx1 else _empty_int(),
        idx3=np.array(ang_idx3, dtype=np.int32) if ang_idx1 else _empty_int(),
        ka=np.array(ang_ka, dtype=np.float32) if ang_idx1 else _empty_float(),
        theta0=np.array(ang_theta0, dtype=np.float32) if ang_idx1 else _empty_float(),
        is_linear=np.array(ang_is_linear, dtype=bool) if ang_idx1 else _empty_bool(),
    )

    # --- Stretch-Bend Terms ---
    sb_idx1, sb_idx2, sb_idx3 = [], [], []
    sb_r0_ij, sb_r0_kj, sb_theta0, sb_kba_ij, sb_kba_kj = [], [], [], [], []

    for j in range(num_atoms):
        atom_j = mol.GetAtomWithIdx(j)
        if atom_j.GetDegree() < 2:
            continue
        is_linear_center = _is_mmff_linear_atom_type(props.GetMMFFAtomType(j))
        neighbors = [n.GetIdx() for n in atom_j.GetNeighbors()]
        for ni in range(len(neighbors)):
            for nk in range(ni + 1, len(neighbors)):
                i = neighbors[ni]
                k = neighbors[nk]
                # RDKit skips stretch-bend terms for MMFF linear central atoms.
                if is_linear_center:
                    continue
                theta0_val = angle_theta0_dict.get((i, j, k))
                res = props.GetMMFFStretchBendParams(mol, i, j, k)
                if res is not None and res:
                    # res format varies; try to extract kba values
                    # Common format: (sbType, kbaIJK, kbaKJI, ...)
                    # or (sbType, stbnParams, bond1, bond2, angle)
                    kba_ij_val, kba_kj_val = _parse_stretch_bend_result(res)
                    if kba_ij_val is None:
                        continue
                    r0_ij_val = bond_r0_dict.get((min(i, j), max(i, j)))
                    r0_kj_val = bond_r0_dict.get((min(k, j), max(k, j)))
                    if r0_ij_val is None or r0_kj_val is None:
                        continue
                    if theta0_val is None:
                        continue
                    sb_idx1.append(i)
                    sb_idx2.append(j)
                    sb_idx3.append(k)
                    sb_r0_ij.append(r0_ij_val)
                    sb_r0_kj.append(r0_kj_val)
                    sb_theta0.append(theta0_val)
                    sb_kba_ij.append(kba_ij_val)
                    sb_kba_kj.append(kba_kj_val)

    stretch_bend_terms = StretchBendTerms(
        idx1=np.array(sb_idx1, dtype=np.int32) if sb_idx1 else _empty_int(),
        idx2=np.array(sb_idx2, dtype=np.int32) if sb_idx1 else _empty_int(),
        idx3=np.array(sb_idx3, dtype=np.int32) if sb_idx1 else _empty_int(),
        r0_ij=np.array(sb_r0_ij, dtype=np.float32) if sb_idx1 else _empty_float(),
        r0_kj=np.array(sb_r0_kj, dtype=np.float32) if sb_idx1 else _empty_float(),
        theta0=np.array(sb_theta0, dtype=np.float32) if sb_idx1 else _empty_float(),
        kba_ij=np.array(sb_kba_ij, dtype=np.float32) if sb_idx1 else _empty_float(),
        kba_kj=np.array(sb_kba_kj, dtype=np.float32) if sb_idx1 else _empty_float(),
    )

    # --- Out-of-Plane Bend Terms ---
    oop_idx1, oop_idx2, oop_idx3, oop_idx4, oop_koop = [], [], [], [], []

    for j in range(num_atoms):
        atom_j = mol.GetAtomWithIdx(j)
        if atom_j.GetDegree() != 3:
            continue
        neighbors = sorted([n.GetIdx() for n in atom_j.GetNeighbors()])
        i, k, l = neighbors[0], neighbors[1], neighbors[2]
        # Check if OOP params exist for this center
        res = props.GetMMFFOopBendParams(mol, i, j, k, l)
        if res is None or res == 0:
            continue
        koop_val = float(res) if isinstance(res, (int, float)) else float(res[-1])
        # Add 3 permutations (matching nvMolKit convention)
        perms = [
            (i, j, k, l),
            (i, j, l, k),
            (k, j, l, i),
        ]
        for p1, p2, p3, p4 in perms:
            oop_idx1.append(p1)
            oop_idx2.append(p2)
            oop_idx3.append(p3)
            oop_idx4.append(p4)
            oop_koop.append(koop_val)

    oop_terms = OutOfPlaneTerms(
        idx1=np.array(oop_idx1, dtype=np.int32) if oop_idx1 else _empty_int(),
        idx2=np.array(oop_idx2, dtype=np.int32) if oop_idx1 else _empty_int(),
        idx3=np.array(oop_idx3, dtype=np.int32) if oop_idx1 else _empty_int(),
        idx4=np.array(oop_idx4, dtype=np.int32) if oop_idx1 else _empty_int(),
        koop=np.array(oop_koop, dtype=np.float32) if oop_idx1 else _empty_float(),
    )

    # --- Torsion Terms ---
    tor_idx1, tor_idx2, tor_idx3, tor_idx4 = [], [], [], []
    tor_V1, tor_V2, tor_V3 = [], [], []

    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        j_atom = mol.GetAtomWithIdx(j)
        k_atom = mol.GetAtomWithIdx(k)
        # Only SP2/SP3 bonds have torsions
        j_hyb = j_atom.GetHybridization()
        k_hyb = k_atom.GetHybridization()
        if j_hyb not in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP3):
            continue
        if k_hyb not in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP3):
            continue
        for i_atom in j_atom.GetNeighbors():
            i = i_atom.GetIdx()
            if i == k:
                continue
            for l_atom in k_atom.GetNeighbors():
                l = l_atom.GetIdx()
                if l == j or l == i:
                    continue
                res = props.GetMMFFTorsionParams(mol, i, j, k, l)
                if res is not None and res:
                    # res = (torType, V1, V2, V3)
                    tor_idx1.append(i)
                    tor_idx2.append(j)
                    tor_idx3.append(k)
                    tor_idx4.append(l)
                    tor_V1.append(float(res[1]))
                    tor_V2.append(float(res[2]))
                    tor_V3.append(float(res[3]))

    torsion_terms = TorsionTerms(
        idx1=np.array(tor_idx1, dtype=np.int32) if tor_idx1 else _empty_int(),
        idx2=np.array(tor_idx2, dtype=np.int32) if tor_idx1 else _empty_int(),
        idx3=np.array(tor_idx3, dtype=np.int32) if tor_idx1 else _empty_int(),
        idx4=np.array(tor_idx4, dtype=np.int32) if tor_idx1 else _empty_int(),
        V1=np.array(tor_V1, dtype=np.float32) if tor_idx1 else _empty_float(),
        V2=np.array(tor_V2, dtype=np.float32) if tor_idx1 else _empty_float(),
        V3=np.array(tor_V3, dtype=np.float32) if tor_idx1 else _empty_float(),
    )

    # --- Non-bonded Terms (VdW and Electrostatic) ---
    # Build graph distance matrix to classify atom pairs
    dist_matrix = Chem.GetDistanceMatrix(mol)
    # RDKit MMFFMolProperties only has setters for dielectric; defaults are 1.0 / model 1
    diel_const = 1.0
    diel_model = 1

    frag_mapping: np.ndarray | None = None
    if ignoreInterfragInteractions:
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            frag_mapping = np.empty(num_atoms, dtype=np.int32)
            for frag_idx, frag in enumerate(frags):
                frag_mapping[list(frag)] = frag_idx

    # Get conformer for distance filtering
    conf = mol.GetConformer(conf_id) if mol.GetNumConformers() > 0 else None

    vdw_idx1, vdw_idx2, vdw_R_star, vdw_eps = [], [], [], []
    ele_idx1, ele_idx2, ele_ct, ele_dm, ele_14 = [], [], [], [], []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if frag_mapping is not None and frag_mapping[i] != frag_mapping[j]:
                continue

            graph_dist = int(dist_matrix[i, j])
            if graph_dist < 3:
                # 1,2 or 1,3 relationship — skip
                continue
            is_1_4 = graph_dist == 3

            # Distance filter using conformer coordinates
            if conf is not None:
                pi = conf.GetAtomPosition(i)
                pj = conf.GetAtomPosition(j)
                d = pi.Distance(pj)
                if d > nonBondedThreshold:
                    continue

            # VdW terms
            vdw_res = props.GetMMFFVdWParams(i, j)
            if vdw_res is not None and vdw_res:
                # RDKit returns (R_ij_starUnscaled, epsilonUnscaled, R_ij_star, epsilon).
                # Use the final scaled MMFF values when present.
                if hasattr(vdw_res, "__len__") and len(vdw_res) >= 4:
                    r_star_val = float(vdw_res[2])
                    epsilon_val = float(vdw_res[3])
                else:
                    r_star_val = float(vdw_res[0])
                    epsilon_val = float(vdw_res[1])
                vdw_idx1.append(i)
                vdw_idx2.append(j)
                vdw_R_star.append(r_star_val)
                vdw_eps.append(epsilon_val)

            # Electrostatic terms
            qi = props.GetMMFFPartialCharge(i)
            qj = props.GetMMFFPartialCharge(j)
            if abs(qi) > 1e-10 and abs(qj) > 1e-10:
                charge_term = qi * qj / diel_const
                ele_idx1.append(i)
                ele_idx2.append(j)
                ele_ct.append(charge_term)
                ele_dm.append(diel_model)
                ele_14.append(is_1_4)

    vdw_terms = VdwTerms(
        idx1=np.array(vdw_idx1, dtype=np.int32) if vdw_idx1 else _empty_int(),
        idx2=np.array(vdw_idx2, dtype=np.int32) if vdw_idx1 else _empty_int(),
        R_ij_star=np.array(vdw_R_star, dtype=np.float32) if vdw_idx1 else _empty_float(),
        epsilon=np.array(vdw_eps, dtype=np.float32) if vdw_idx1 else _empty_float(),
    )

    ele_terms = ElectrostaticTerms(
        idx1=np.array(ele_idx1, dtype=np.int32) if ele_idx1 else _empty_int(),
        idx2=np.array(ele_idx2, dtype=np.int32) if ele_idx1 else _empty_int(),
        charge_term=np.array(ele_ct, dtype=np.float32) if ele_idx1 else _empty_float(),
        diel_model=np.array(ele_dm, dtype=np.int32) if ele_idx1 else _empty_int(),
        is_1_4=np.array(ele_14, dtype=bool) if ele_idx1 else _empty_bool(),
    )

    return MMFFParams(
        num_atoms=num_atoms,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        stretch_bend_terms=stretch_bend_terms,
        oop_terms=oop_terms,
        torsion_terms=torsion_terms,
        vdw_terms=vdw_terms,
        ele_terms=ele_terms,
    )


def _parse_stretch_bend_result(res) -> tuple[float | None, float | None]:
    """Parse the result of GetMMFFStretchBendParams.

    Handles different RDKit API return formats.

    Returns:
        (kba_ij, kba_kj) or (None, None) if parsing fails.
    """
    if isinstance(res, (int, float)):
        return None, None
    if not hasattr(res, "__len__"):
        return None, None
    if len(res) >= 3:
        # Try flat format: (sbType, kbaIJK, kbaKJI, ...)
        try:
            sub = res[1]
            if isinstance(sub, (int, float)):
                return float(res[1]), float(res[2])
            # Object format: res[1] has .kbaIJK, .kbaKJI
            if hasattr(sub, "kbaIJK"):
                return float(sub.kbaIJK), float(sub.kbaKJI)
        except (IndexError, TypeError, AttributeError):
            pass
    return None, None
