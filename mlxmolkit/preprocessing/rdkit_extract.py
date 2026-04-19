"""Extract distance geometry force field parameters from RDKit molecules.

Converts RDKit molecular data into numpy arrays suitable for MLX computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


@dataclass
class DistViolationTerms:
    """Distance violation force field terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    lb2: np.ndarray  # float32 (n_terms,) squared lower bounds
    ub2: np.ndarray  # float32 (n_terms,) squared upper bounds
    weight: np.ndarray  # float32 (n_terms,)


@dataclass
class ChiralViolationTerms:
    """Chiral violation force field terms."""

    idx1: np.ndarray  # int32 (n_terms,)
    idx2: np.ndarray  # int32 (n_terms,)
    idx3: np.ndarray  # int32 (n_terms,)
    idx4: np.ndarray  # int32 (n_terms,)
    vol_lower: np.ndarray  # float32 (n_terms,)
    vol_upper: np.ndarray  # float32 (n_terms,)


@dataclass
class FourthDimTerms:
    """Fourth dimension penalty terms."""

    idx: np.ndarray  # int32 (n_terms,)


@dataclass
class TetrahedralCheckTerms:
    """Tetrahedral geometry check terms for a single molecule.

    Used by the tetrahedral check stage to validate that sp3 centers
    have proper tetrahedral (non-planar) geometry.
    """

    idx0: np.ndarray  # int32 (n_terms,) center atom
    idx1: np.ndarray  # int32 (n_terms,) neighbor 1
    idx2: np.ndarray  # int32 (n_terms,) neighbor 2
    idx3: np.ndarray  # int32 (n_terms,) neighbor 3
    idx4: np.ndarray  # int32 (n_terms,) neighbor 4 (or idx0 for 3-coord centers)
    in_fused_small_rings: np.ndarray  # bool (n_terms,)


@dataclass
class DGParams:
    """Distance geometry force field parameters for a single molecule."""

    num_atoms: int
    dist_terms: DistViolationTerms
    chiral_terms: ChiralViolationTerms
    fourth_dim_terms: FourthDimTerms


def get_bounds_matrix(mol: Chem.Mol) -> np.ndarray:
    """Get the distance bounds matrix from an RDKit molecule.

    Args:
        mol: RDKit molecule with hydrogens added.

    Returns:
        2D numpy array where upper triangle contains upper bounds
        and lower triangle contains lower bounds.
    """
    return rdDistGeom.GetMoleculeBoundsMatrix(mol)


def _is_explicit_tetrahedral_chiral(atom: Chem.Atom) -> bool:
    """Return True for explicitly assigned tetrahedral CW/CCW atoms."""
    return atom.GetChiralTag() in (
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    )


def _get_explicit_chiral_neighbors(mol: Chem.Mol, atom_idx: int) -> list[int] | None:
    """Return nvMolKit/RDKit-ordered chiral neighbors.

    Four-coordinate chiral centers use all four neighbors. Three-coordinate
    centers use the center atom as the fourth point.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

    if len(neighbors) == 3:
        return [neighbors[0], neighbors[1], neighbors[2], atom_idx]
    if len(neighbors) == 4:
        return neighbors
    return None


def _explicit_chiral_volume_bounds(
    chiral_tag: Chem.ChiralType, n_neighbors: int
) -> tuple[float, float]:
    """Return fixed nvMolKit/RDKit chiral volume bounds."""
    min_vol = 2.0 if n_neighbors == 3 else 5.0
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return -100.0, -min_vol
    return min_vol, 100.0


def extract_dg_params(
    mol: Chem.Mol,
    bounds_mat: np.ndarray | None = None,
    dim: int = 4,
    basin_size_tol: float = 5.0,
) -> DGParams:
    """Extract distance geometry force field parameters from an RDKit molecule.

    This mirrors nvMolKit's constructForceFieldContribs() function.

    Args:
        mol: RDKit molecule with hydrogens added.
        bounds_mat: Pre-computed bounds matrix (or None to compute).
        dim: Coordinate dimension (3 or 4).
        basin_size_tol: Maximum basin size for including distance terms (Angstroms).

    Returns:
        DGParams containing all force field terms.
    """
    if bounds_mat is None:
        bounds_mat = get_bounds_matrix(mol)

    num_atoms = mol.GetNumAtoms()

    # --- Distance Violation Terms ---
    # From nvMolKit's addDistViolationContribs:
    # Iterate upper triangle (i > j), include if upperBound - lowerBound <= basinSizeTol
    idx1_list = []
    idx2_list = []
    lb2_list = []
    ub2_list = []
    weight_list = []

    for i in range(1, num_atoms):
        for j in range(i):
            # bounds_mat: upper triangle (i<j) = upper bound, lower triangle (i>j) = lower bound
            lower_bound = bounds_mat[i, j]  # lower triangle = lower bound
            upper_bound = bounds_mat[j, i]  # upper triangle = upper bound

            if upper_bound - lower_bound <= basin_size_tol:
                idx1_list.append(i)
                idx2_list.append(j)
                lb2_list.append(lower_bound * lower_bound)
                ub2_list.append(upper_bound * upper_bound)
                weight_list.append(1.0)

    if idx1_list:
        dist_terms = DistViolationTerms(
            idx1=np.array(idx1_list, dtype=np.int32),
            idx2=np.array(idx2_list, dtype=np.int32),
            lb2=np.array(lb2_list, dtype=np.float32),
            ub2=np.array(ub2_list, dtype=np.float32),
            weight=np.array(weight_list, dtype=np.float32),
        )
    else:
        dist_terms = DistViolationTerms(
            idx1=np.array([], dtype=np.int32),
            idx2=np.array([], dtype=np.int32),
            lb2=np.array([], dtype=np.float32),
            ub2=np.array([], dtype=np.float32),
            weight=np.array([], dtype=np.float32),
        )

    # --- Chiral Violation Terms ---
    chiral_idx1 = []
    chiral_idx2 = []
    chiral_idx3 = []
    chiral_idx4 = []
    vol_lower_list = []
    vol_upper_list = []

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for atom in mol.GetAtoms():
        if not _is_explicit_tetrahedral_chiral(atom):
            continue

        atom_idx = atom.GetIdx()
        n_neighbors = atom.GetDegree()
        neighbors = _get_explicit_chiral_neighbors(mol, atom_idx)
        if neighbors is None:
            continue

        n1, n2, n3, n4 = neighbors
        vol_lower, vol_upper = _explicit_chiral_volume_bounds(
            atom.GetChiralTag(), n_neighbors
        )

        chiral_idx1.append(n1)
        chiral_idx2.append(n2)
        chiral_idx3.append(n3)
        chiral_idx4.append(n4)
        vol_lower_list.append(vol_lower)
        vol_upper_list.append(vol_upper)

    if chiral_idx1:
        chiral_terms = ChiralViolationTerms(
            idx1=np.array(chiral_idx1, dtype=np.int32),
            idx2=np.array(chiral_idx2, dtype=np.int32),
            idx3=np.array(chiral_idx3, dtype=np.int32),
            idx4=np.array(chiral_idx4, dtype=np.int32),
            vol_lower=np.array(vol_lower_list, dtype=np.float32),
            vol_upper=np.array(vol_upper_list, dtype=np.float32),
        )
    else:
        chiral_terms = ChiralViolationTerms(
            idx1=np.array([], dtype=np.int32),
            idx2=np.array([], dtype=np.int32),
            idx3=np.array([], dtype=np.int32),
            idx4=np.array([], dtype=np.int32),
            vol_lower=np.array([], dtype=np.float32),
            vol_upper=np.array([], dtype=np.float32),
        )

    # --- Fourth Dimension Terms ---
    if dim == 4:
        fourth_dim_terms = FourthDimTerms(
            idx=np.arange(num_atoms, dtype=np.int32),
        )
    else:
        fourth_dim_terms = FourthDimTerms(
            idx=np.array([], dtype=np.int32),
        )

    return DGParams(
        num_atoms=num_atoms,
        dist_terms=dist_terms,
        chiral_terms=chiral_terms,
        fourth_dim_terms=fourth_dim_terms,
    )


def extract_tetrahedral_atoms(mol: Chem.Mol) -> TetrahedralCheckTerms:
    """Extract tetrahedral atoms for geometry validation.

    Mirrors nvMolKit's tetrahedralCarbons extraction: non-explicitly-chiral
    C/N atoms with degree 4 in fused ring systems, excluding 3-membered rings.

    Args:
        mol: RDKit molecule with hydrogens added.

    Returns:
        TetrahedralCheckTerms with atom indices and fused ring flags.
    """
    idx0_list = []
    idx1_list = []
    idx2_list = []
    idx3_list = []
    idx4_list = []
    fused_list = []

    ring_info = mol.GetRingInfo()

    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()

        # Only carbon (6) and nitrogen (7)
        if atomic_num not in (6, 7):
            continue

        if _is_explicit_tetrahedral_chiral(atom):
            continue

        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) != 4:
            continue

        if ring_info.NumAtomRings(atom_idx) < 2:
            continue

        if ring_info.IsAtomInRingOfSize(atom_idx, 3):
            continue

        idx0_list.append(atom_idx)
        idx1_list.append(neighbors[0])
        idx2_list.append(neighbors[1])
        idx3_list.append(neighbors[2])
        idx4_list.append(neighbors[3])

        num_small_rings = sum(
            1 for ring_size in ring_info.AtomRingSizes(atom_idx) if ring_size < 5
        )
        fused_list.append(num_small_rings > 1)

    if idx0_list:
        return TetrahedralCheckTerms(
            idx0=np.array(idx0_list, dtype=np.int32),
            idx1=np.array(idx1_list, dtype=np.int32),
            idx2=np.array(idx2_list, dtype=np.int32),
            idx3=np.array(idx3_list, dtype=np.int32),
            idx4=np.array(idx4_list, dtype=np.int32),
            in_fused_small_rings=np.array(fused_list, dtype=bool),
        )
    else:
        return TetrahedralCheckTerms(
            idx0=np.array([], dtype=np.int32),
            idx1=np.array([], dtype=np.int32),
            idx2=np.array([], dtype=np.int32),
            idx3=np.array([], dtype=np.int32),
            idx4=np.array([], dtype=np.int32),
            in_fused_small_rings=np.array([], dtype=bool),
        )


def extract_chiral_center_terms(mol: Chem.Mol) -> TetrahedralCheckTerms:
    """Extract explicit CW/CCW chiral centers for final volume checks."""
    idx0_list = []
    idx1_list = []
    idx2_list = []
    idx3_list = []
    idx4_list = []

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for atom in mol.GetAtoms():
        if not _is_explicit_tetrahedral_chiral(atom):
            continue

        atom_idx = atom.GetIdx()
        neighbors = _get_explicit_chiral_neighbors(mol, atom_idx)
        if neighbors is None:
            continue

        idx0_list.append(atom_idx)
        idx1_list.append(neighbors[0])
        idx2_list.append(neighbors[1])
        idx3_list.append(neighbors[2])
        idx4_list.append(neighbors[3])

    if idx0_list:
        return TetrahedralCheckTerms(
            idx0=np.array(idx0_list, dtype=np.int32),
            idx1=np.array(idx1_list, dtype=np.int32),
            idx2=np.array(idx2_list, dtype=np.int32),
            idx3=np.array(idx3_list, dtype=np.int32),
            idx4=np.array(idx4_list, dtype=np.int32),
            in_fused_small_rings=np.zeros(len(idx0_list), dtype=bool),
        )
    return TetrahedralCheckTerms(
        idx0=np.array([], dtype=np.int32),
        idx1=np.array([], dtype=np.int32),
        idx2=np.array([], dtype=np.int32),
        idx3=np.array([], dtype=np.int32),
        idx4=np.array([], dtype=np.int32),
        in_fused_small_rings=np.array([], dtype=bool),
    )
