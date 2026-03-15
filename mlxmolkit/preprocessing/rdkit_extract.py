"""Extract distance geometry force field parameters from RDKit molecules.

Converts RDKit molecular data into numpy arrays suitable for MLX computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms


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


def _compute_chiral_volume_bounds(
    bounds_mat: np.ndarray, i1: int, i2: int, i3: int, i4: int
) -> tuple[float, float]:
    """Compute lower and upper bounds on chiral volume from distance bounds.

    Uses the Cayley-Menger determinant to compute the range of possible
    signed volumes for a tetrahedron defined by 4 atoms, given distance bounds.

    The signed volume is V = (p1-p4) . ((p2-p4) x (p3-p4)).
    288 * V^2 = det(CM) where CM is the Cayley-Menger matrix.

    We try all 2^6 = 64 combinations of lower/upper bounds for the 6 pairwise
    distances to find the range of V^2, then determine the sign from the
    chirality.

    Args:
        bounds_mat: Distance bounds matrix (upper tri = upper, lower tri = lower).
        i1, i2, i3, i4: Atom indices.

    Returns:
        (vol_lower, vol_upper) bounds on the signed chiral volume.
    """
    # The 6 pairwise distances
    pairs = [(i1, i2), (i1, i3), (i1, i4), (i2, i3), (i2, i4), (i3, i4)]

    # Get lower and upper distance bounds for each pair
    d_lower = np.zeros(6)
    d_upper = np.zeros(6)
    for k, (a, b) in enumerate(pairs):
        lo, hi = min(a, b), max(a, b)
        d_lower[k] = bounds_mat[hi, lo]  # lower triangle = lower bound
        d_upper[k] = bounds_mat[lo, hi]  # upper triangle = upper bound

    # Try all 2^6 combinations to find min/max of V^2
    max_vol_sq = -1.0
    min_vol_sq = float("inf")

    for mask in range(64):
        # Choose lower or upper bound for each distance
        d = np.where(
            np.array([(mask >> k) & 1 for k in range(6)], dtype=bool),
            d_upper,
            d_lower,
        )
        d_sq = d * d

        # Cayley-Menger determinant (5x5)
        # Row/col 0 is the homogeneous coordinate
        cm = np.zeros((5, 5))
        cm[0, 1:] = 1.0
        cm[1:, 0] = 1.0

        # d_sq indices: 0=d12, 1=d13, 2=d14, 3=d23, 4=d24, 5=d34
        cm[1, 2] = cm[2, 1] = d_sq[0]  # d12^2
        cm[1, 3] = cm[3, 1] = d_sq[1]  # d13^2
        cm[1, 4] = cm[4, 1] = d_sq[2]  # d14^2
        cm[2, 3] = cm[3, 2] = d_sq[3]  # d23^2
        cm[2, 4] = cm[4, 2] = d_sq[4]  # d24^2
        cm[3, 4] = cm[4, 3] = d_sq[5]  # d34^2

        det_val = np.linalg.det(cm)
        vol_sq = det_val / 288.0

        if vol_sq > max_vol_sq:
            max_vol_sq = vol_sq
        if vol_sq < min_vol_sq:
            min_vol_sq = vol_sq

    # Convert V^2 bounds to V bounds
    if max_vol_sq <= 0:
        return 0.0, 0.0

    max_vol = np.sqrt(max(max_vol_sq, 0.0))
    if min_vol_sq > 0:
        min_vol = np.sqrt(min_vol_sq)
    else:
        min_vol = 0.0

    # Return full range: volume can be positive or negative
    return -max_vol, max_vol


def _get_chiral_atom_neighbors(
    mol: Chem.Mol, atom_idx: int
) -> tuple[int, int, int] | None:
    """Get the 3 neighbor indices for a tetrahedral chiral center.

    Returns neighbors in canonical order for consistent volume computation.
    Returns None if the atom doesn't have exactly 3 heavy/H neighbors suitable
    for chirality computation.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

    if len(neighbors) < 3:
        return None

    # Sort neighbors for canonical ordering
    neighbors.sort()

    if len(neighbors) == 3:
        return (neighbors[0], neighbors[1], neighbors[2])
    elif len(neighbors) == 4:
        # For 4-coordinated atoms, use first 3 neighbors
        return (neighbors[0], neighbors[1], neighbors[2])
    return None


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

    # Find chiral centers using RDKit
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chiral_centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )

    for atom_idx, chirality in chiral_centers:
        neighbors = _get_chiral_atom_neighbors(mol, atom_idx)
        if neighbors is None:
            continue

        n1, n2, n3 = neighbors
        # Use the chiral center as idx4 (pivot), neighbors as idx1-3
        # This matches nvMolKit's convention: V = (p1-p4) · ((p2-p4) × (p3-p4))
        vol_lower, vol_upper = _compute_chiral_volume_bounds(
            bounds_mat, n1, n2, n3, atom_idx
        )

        atom = mol.GetAtomWithIdx(atom_idx)
        chiral_tag = atom.GetChiralTag()

        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            # CW chirality: volume should be negative
            vol_upper = min(vol_upper, 0.0)
        elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            # CCW chirality: volume should be positive
            vol_lower = max(vol_lower, 0.0)
        # For unspecified chirality, keep full range

        chiral_idx1.append(n1)
        chiral_idx2.append(n2)
        chiral_idx3.append(n3)
        chiral_idx4.append(atom_idx)
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

    Identifies carbon and nitrogen atoms with 4 neighbors that should
    have proper tetrahedral (non-planar) geometry. These are distinct
    from chiral centers — tetrahedral atoms include ALL sp3 centers,
    not just those with explicit stereochemistry.

    Also identifies sp3 nitrogen with 3 bonds (+ lone pair) as
    3-coordinate tetrahedral centers (idx4 == idx0).

    Port of nvMolKit's findChiralSets() tetrahedral carbon extraction.

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

        neighbors = sorted([n.GetIdx() for n in atom.GetNeighbors()])
        degree = len(neighbors)

        if degree == 4:
            idx0_list.append(atom_idx)
            idx1_list.append(neighbors[0])
            idx2_list.append(neighbors[1])
            idx3_list.append(neighbors[2])
            idx4_list.append(neighbors[3])

            # Check if in fused small rings (3 or 4-membered)
            in_fused = False
            if ring_info.NumAtomRings(atom_idx) > 0:
                for ring_size in [3, 4]:
                    if ring_info.IsAtomInRingOfSize(atom_idx, ring_size):
                        in_fused = True
                        break
            fused_list.append(in_fused)

        elif degree == 3 and atomic_num == 7:
            # Nitrogen with 3 bonds + lone pair (3-coordinate tetrahedral)
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                idx0_list.append(atom_idx)
                idx1_list.append(neighbors[0])
                idx2_list.append(neighbors[1])
                idx3_list.append(neighbors[2])
                idx4_list.append(atom_idx)  # idx4 == idx0 for 3-coord
                fused_list.append(False)

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
