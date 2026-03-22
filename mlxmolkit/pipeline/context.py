"""Pipeline context for ETKDG stages.

Provides the PipelineContext dataclass that holds all mutable state for a
pipeline iteration, plus a factory function to create it from RDKit molecules.
"""

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from ..preprocessing.batching import BatchedDGSystem, batch_dg_params
from ..preprocessing.etk_batching import BatchedETKSystem, batch_etk_params
from ..preprocessing.rdkit_extract import (
    TetrahedralCheckTerms,
    extract_dg_params,
    extract_tetrahedral_atoms,
    get_bounds_matrix,
)
from ..preprocessing.torsion_prefs import extract_etk_params


@dataclass
class BatchedTetrahedralData:
    """Batched tetrahedral check data for multiple molecules."""

    idx0: mx.array  # int32 (n_terms,) — center atom (global index)
    idx1: mx.array  # int32 (n_terms,)
    idx2: mx.array  # int32 (n_terms,)
    idx3: mx.array  # int32 (n_terms,)
    idx4: mx.array  # int32 (n_terms,) — neighbor 4 (or idx0 for 3-coord)
    in_fused_small_rings: mx.array  # bool (n_terms,)
    mol_indices: mx.array  # int32 (n_terms,)


@dataclass
class PipelineContext:
    """Mutable context for a pipeline iteration over a batch of molecules.

    Modified in place by pipeline stages. After each stage, call
    collect_failures() to deactivate failed molecules.
    """

    n_mols: int
    dim: int
    atom_starts: list[int]  # Python list, n_mols + 1
    n_atoms_total: int

    # Positions — updated by stages
    positions: mx.array  # float32 (n_atoms_total * dim,)

    # Per-molecule status
    active: list[bool]
    failed: list[bool]

    # Force field data (batched)
    dg_system: BatchedDGSystem

    # Tetrahedral check data (batched)
    tet_data: BatchedTetrahedralData | None

    # ETK 3D force field data (batched) — created if use ETK stages
    etk_system: BatchedETKSystem | None = None

    # Double bond check data — created during context setup
    double_bond_data: dict | None = None
    stereo_bond_data: dict | None = None
    chiral_dist_data: dict | None = None

    # Bounds matrices (per-molecule, for chiral dist matrix check)
    bounds_matrices: list | None = None

    # Persistent RNG for coordinate generation (advances across pipeline calls)
    rng: np.random.Generator | None = None

    # Mapping from batch entry index to (unique_mol_index, conf_slot)
    # Used by multi-conf pipeline to track which molecule each entry belongs to
    entry_mol_map: list[int] | None = None

    def collect_failures(self) -> None:
        """Deactivate failed molecules. Call after each stage."""
        for i in range(self.n_mols):
            if self.failed[i]:
                self.active[i] = False
                self.failed[i] = False

    def n_active(self) -> int:
        """Return the number of active molecules."""
        return sum(self.active)


def _batch_tetrahedral_terms(
    terms_list: list[TetrahedralCheckTerms],
    atom_starts_np: np.ndarray,
) -> BatchedTetrahedralData | None:
    """Batch per-molecule tetrahedral check terms with atom offset.

    Args:
        terms_list: Per-molecule tetrahedral check terms.
        atom_starts_np: Cumulative atom start indices.

    Returns:
        Batched tetrahedral data with global atom indices, or None if empty.
    """
    idx0_parts = []
    idx1_parts = []
    idx2_parts = []
    idx3_parts = []
    idx4_parts = []
    fused_parts = []
    mol_parts = []

    for i, terms in enumerate(terms_list):
        n = len(terms.idx0)
        if n == 0:
            continue
        offset = int(atom_starts_np[i])
        idx0_parts.append(terms.idx0 + offset)
        idx1_parts.append(terms.idx1 + offset)
        idx2_parts.append(terms.idx2 + offset)
        idx3_parts.append(terms.idx3 + offset)
        idx4_parts.append(terms.idx4 + offset)
        fused_parts.append(terms.in_fused_small_rings)
        mol_parts.append(np.full(n, i, dtype=np.int32))

    if not idx0_parts:
        return None

    return BatchedTetrahedralData(
        idx0=mx.array(np.concatenate(idx0_parts).astype(np.int32)),
        idx1=mx.array(np.concatenate(idx1_parts).astype(np.int32)),
        idx2=mx.array(np.concatenate(idx2_parts).astype(np.int32)),
        idx3=mx.array(np.concatenate(idx3_parts).astype(np.int32)),
        idx4=mx.array(np.concatenate(idx4_parts).astype(np.int32)),
        in_fused_small_rings=mx.array(np.concatenate(fused_parts)),
        mol_indices=mx.array(np.concatenate(mol_parts).astype(np.int32)),
    )


def _extract_double_bond_data(
    mols: list[Chem.Mol], atom_starts: list[int]
) -> dict[str, np.ndarray] | None:
    """Extract double bond geometry check data from molecules.

    Args:
        mols: RDKit molecules.
        atom_starts: Cumulative atom start indices.

    Returns:
        Dict with 'idx0', 'idx1', 'idx2', 'mol_indices' arrays, or None
        if no double bonds found.
    """
    idx0_list, idx1_list, idx2_list, mol_list = [], [], [], []

    for i, mol in enumerate(mols):
        offset = atom_starts[i]
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                # Check both ends for geometry
                for center, other in [(a1, a2), (a2, a1)]:
                    atom = mol.GetAtomWithIdx(center)
                    for neighbor in atom.GetNeighbors():
                        nidx = neighbor.GetIdx()
                        if nidx != other:
                            idx0_list.append(nidx + offset)
                            idx1_list.append(center + offset)
                            idx2_list.append(other + offset)
                            mol_list.append(i)

    if not idx0_list:
        return None

    return {
        'idx0': np.array(idx0_list, dtype=np.int32),
        'idx1': np.array(idx1_list, dtype=np.int32),
        'idx2': np.array(idx2_list, dtype=np.int32),
        'mol_indices': np.array(mol_list, dtype=np.int32),
    }


def _extract_stereo_bond_data(
    mols: list[Chem.Mol], atom_starts: list[int]
) -> dict[str, np.ndarray] | None:
    """Extract double bond stereo check data from molecules.

    Args:
        mols: RDKit molecules.
        atom_starts: Cumulative atom start indices.

    Returns:
        Dict with 'idx0'-'idx3', 'signs', 'mol_indices' arrays, or None
        if no stereo bonds found.
    """
    idx0_list, idx1_list, idx2_list, idx3_list = [], [], [], []
    signs_list, mol_list = [], []

    for i, mol in enumerate(mols):
        offset = atom_starts[i]
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        for bond in mol.GetBonds():
            stereo = bond.GetStereo()
            if stereo == Chem.BondStereo.STEREONONE:
                continue
            if bond.GetBondTypeAsDouble() != 2.0:
                continue

            stereo_atoms = list(bond.GetStereoAtoms())
            if len(stereo_atoms) != 2:
                continue

            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            s1 = stereo_atoms[0]
            s2 = stereo_atoms[1]

            # Sign: -1 for Z/cis (same side), +1 for E/trans (opposite side)
            # Matches nvMolKit's convention (embedder_utils.cpp:648-651)
            if stereo == Chem.BondStereo.STEREOZ:
                sign = -1
            elif stereo == Chem.BondStereo.STEREOE:
                sign = 1
            else:
                continue

            idx0_list.append(s1 + offset)
            idx1_list.append(a1 + offset)
            idx2_list.append(a2 + offset)
            idx3_list.append(s2 + offset)
            signs_list.append(sign)
            mol_list.append(i)

    if not idx0_list:
        return None

    return {
        'idx0': np.array(idx0_list, dtype=np.int32),
        'idx1': np.array(idx1_list, dtype=np.int32),
        'idx2': np.array(idx2_list, dtype=np.int32),
        'idx3': np.array(idx3_list, dtype=np.int32),
        'signs': np.array(signs_list, dtype=np.int32),
        'mol_indices': np.array(mol_list, dtype=np.int32),
    }


def _extract_chiral_dist_data(
    mols: list[Chem.Mol],
    bounds_matrices: list[np.ndarray],
    atom_starts: list[int],
    dg_system: BatchedDGSystem,
) -> dict[str, np.ndarray] | None:
    """Extract chiral distance matrix check data.

    Collects all pairwise distances between atoms involved in chirality
    (center + all neighbors) for each CW/CCW chiral center and looks up
    their bounds from the distance bounds matrix.

    Only includes explicitly assigned chiral centers (CW or CCW), matching
    nvMolKit's ``eargs.chiralCenters`` which excludes unassigned centers.
    Includes all neighbors (not just 3), matching nvMolKit's ChiralSet
    which stores center + 4 neighbors.

    Args:
        mols: RDKit molecules.
        bounds_matrices: Per-molecule distance bounds matrices.
        atom_starts: Cumulative atom start indices.
        dg_system: Batched DG system (unused, kept for API compat).

    Returns:
        Dict with 'idx0', 'idx1', 'lower', 'upper', 'mol_indices' arrays,
        or None if no chiral centers found.
    """
    idx0_list, idx1_list = [], []
    lower_list, upper_list = [], []
    mol_list = []

    # Build a set of CW/CCW chiral center indices per molecule.
    # nvMolKit's chiralDistMatrixCheck uses only eargs.chiralCenters
    # (explicitly CW/CCW atoms), not unassigned tetrahedralCenters.
    mol_cw_ccw = []
    for mol in mols:
        cw_ccw = set()
        for atom in mol.GetAtoms():
            tag = atom.GetChiralTag()
            if tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                cw_ccw.add(atom.GetIdx())
        mol_cw_ccw.append(cw_ccw)

    chiral_idx1 = np.array(dg_system.chiral_idx1)
    chiral_idx2 = np.array(dg_system.chiral_idx2)
    chiral_idx3 = np.array(dg_system.chiral_idx3)
    chiral_idx4 = np.array(dg_system.chiral_idx4)
    chiral_mol = np.array(dg_system.chiral_mol_indices)

    for i, mol in enumerate(mols):
        offset = atom_starts[i]
        bmat = bounds_matrices[i]

        # Collect atoms from chiral terms that belong to CW/CCW centers
        chiral_mask = chiral_mol == i
        if not np.any(chiral_mask):
            continue

        cw_ccw = mol_cw_ccw[i]
        if not cw_ccw:
            continue

        chiral_atoms = set()
        for t in np.where(chiral_mask)[0]:
            # idx4 is the center atom — only include this term if
            # the center is a CW/CCW chiral center
            center_local = int(chiral_idx4[t]) - offset
            if center_local not in cw_ccw:
                continue
            for idx_arr in [chiral_idx1, chiral_idx2, chiral_idx3, chiral_idx4]:
                local = int(idx_arr[t]) - offset
                if 0 <= local < mol.GetNumAtoms():
                    chiral_atoms.add(local)

        if not chiral_atoms:
            continue

        chiral_atoms = sorted(chiral_atoms)
        for j in range(len(chiral_atoms)):
            for k in range(j + 1, len(chiral_atoms)):
                a0 = chiral_atoms[j]
                a1 = chiral_atoms[k]
                lb = bmat[max(a0, a1), min(a0, a1)]
                ub = bmat[min(a0, a1), max(a0, a1)]

                idx0_list.append(a0 + offset)
                idx1_list.append(a1 + offset)
                lower_list.append(lb)
                upper_list.append(ub)
                mol_list.append(i)

    if not idx0_list:
        return None

    return {
        'idx0': np.array(idx0_list, dtype=np.int32),
        'idx1': np.array(idx1_list, dtype=np.int32),
        'lower': np.array(lower_list, dtype=np.float64),
        'upper': np.array(upper_list, dtype=np.float64),
        'mol_indices': np.array(mol_list, dtype=np.int32),
    }


def create_pipeline_context(
    mols: list[Chem.Mol],
    dim: int = 4,
    basin_size_tol: float = 1e8,
    params: "rdDistGeom.EmbedParameters | None" = None,
    use_etk: bool = False,
    enforce_chirality: bool = True,
) -> PipelineContext:
    """Create a pipeline context from a list of RDKit molecules.

    Handles all preprocessing: bounds matrix, DG params, tetrahedral atoms,
    and optionally ETK 3D force field params.

    Args:
        mols: RDKit molecules with hydrogens added.
        dim: Coordinate dimension (4 for ETKDG).
        basin_size_tol: Basin size tolerance for distance terms.
            Default 1e8 matches nvMolKit's randomCoordsBasinThresh.
        params: Optional RDKit EmbedParameters for ETK extraction.
        use_etk: Whether to extract ETK parameters.
        enforce_chirality: Whether to set up chirality check data.

    Returns:
        PipelineContext ready for pipeline stages.
    """
    n_mols = len(mols)

    # Extract per-molecule data
    dg_params_list = []
    tet_terms_list = []
    etk_params_list = []
    bounds_matrices = []

    for mol in mols:
        bounds_mat = get_bounds_matrix(mol)
        bounds_matrices.append(bounds_mat)
        dg_params = extract_dg_params(mol, bounds_mat, dim, basin_size_tol)
        dg_params_list.append(dg_params)
        tet_terms = extract_tetrahedral_atoms(mol)
        tet_terms_list.append(tet_terms)

        if use_etk:
            etk = extract_etk_params(mol, bounds_mat, params=params)
            etk_params_list.append(etk)

    # Batch DG params
    dg_system = batch_dg_params(dg_params_list, dim)

    # Batch tetrahedral terms
    atom_starts_np = np.array(dg_system.atom_starts.tolist(), dtype=np.int32)
    tet_data = _batch_tetrahedral_terms(tet_terms_list, atom_starts_np)

    atom_starts = dg_system.atom_starts.tolist()
    n_atoms_total = dg_system.n_atoms_total

    # Initialize positions to zeros (coordgen will fill them)
    positions = mx.zeros(n_atoms_total * dim, dtype=mx.float32)

    # Build ETK system if requested
    etk_system = None
    if use_etk and etk_params_list:
        etk_system = batch_etk_params(etk_params_list, atom_starts, dim)

    # Extract double bond and stereo data
    double_bond_data = _extract_double_bond_data(mols, atom_starts)
    stereo_bond_data = _extract_stereo_bond_data(mols, atom_starts)

    # Chiral distance matrix check data
    chiral_dist_data = None
    if enforce_chirality:
        chiral_dist_data = _extract_chiral_dist_data(
            mols, bounds_matrices, atom_starts, dg_system
        )

    return PipelineContext(
        n_mols=n_mols,
        dim=dim,
        atom_starts=atom_starts,
        n_atoms_total=n_atoms_total,
        positions=positions,
        active=[True] * n_mols,
        failed=[False] * n_mols,
        dg_system=dg_system,
        tet_data=tet_data,
        etk_system=etk_system,
        double_bond_data=double_bond_data,
        stereo_bond_data=stereo_bond_data,
        chiral_dist_data=chiral_dist_data,
        bounds_matrices=bounds_matrices,
    )


def create_pipeline_context_multi_conf(
    mols: list[Chem.Mol],
    confs_per_mol: list[int],
    dim: int = 4,
    basin_size_tol: float = 1e8,
    params: "rdDistGeom.EmbedParameters | None" = None,
    use_etk: bool = False,
    enforce_chirality: bool = True,
    rng: np.random.Generator | None = None,
) -> PipelineContext:
    """Create a pipeline context with multiple conformer slots per molecule.

    Extracts parameters once per unique molecule, then replicates them for
    the requested number of conformer attempts. Each conformer slot is
    treated as an independent entry in the batch.

    Args:
        mols: Unique RDKit molecules with hydrogens added.
        confs_per_mol: Number of conformer slots per molecule.
        dim: Coordinate dimension (4 for ETKDG).
        basin_size_tol: Basin size tolerance for distance terms.
        params: Optional RDKit EmbedParameters for ETK extraction.
        use_etk: Whether to extract ETK parameters.
        enforce_chirality: Whether to set up chirality check data.
        rng: Persistent numpy RNG (stored on context, advances across calls).

    Returns:
        PipelineContext with n_entries = sum(confs_per_mol) batch slots,
        each mapped back to its source molecule via entry_mol_map.
    """
    n_unique = len(mols)

    # Extract per-molecule data ONCE
    per_mol_dg = []
    per_mol_tet = []
    per_mol_etk = []
    per_mol_bounds = []

    for mol in mols:
        bounds_mat = get_bounds_matrix(mol)
        per_mol_bounds.append(bounds_mat)
        dg_params = extract_dg_params(mol, bounds_mat, dim, basin_size_tol)
        per_mol_dg.append(dg_params)
        tet_terms = extract_tetrahedral_atoms(mol)
        per_mol_tet.append(tet_terms)

        if use_etk:
            etk = extract_etk_params(mol, bounds_mat, params=params)
            per_mol_etk.append(etk)

    # Replicate params for each conformer slot
    dg_params_list = []
    tet_terms_list = []
    etk_params_list = []
    bounds_matrices = []
    entry_mol_map: list[int] = []  # entry_idx -> unique mol index
    replicated_mols: list[Chem.Mol] = []  # for double bond / stereo extraction

    for mol_idx in range(n_unique):
        n_confs = confs_per_mol[mol_idx]
        for _ in range(n_confs):
            dg_params_list.append(per_mol_dg[mol_idx])
            tet_terms_list.append(per_mol_tet[mol_idx])
            bounds_matrices.append(per_mol_bounds[mol_idx])
            entry_mol_map.append(mol_idx)
            replicated_mols.append(mols[mol_idx])
            if use_etk:
                etk_params_list.append(per_mol_etk[mol_idx])

    n_entries = len(dg_params_list)

    # Batch using existing functions (they handle atom offsets)
    dg_system = batch_dg_params(dg_params_list, dim)

    atom_starts_np = np.array(dg_system.atom_starts.tolist(), dtype=np.int32)
    tet_data = _batch_tetrahedral_terms(tet_terms_list, atom_starts_np)

    atom_starts = dg_system.atom_starts.tolist()
    n_atoms_total = dg_system.n_atoms_total

    positions = mx.zeros(n_atoms_total * dim, dtype=mx.float32)

    etk_system = None
    if use_etk and etk_params_list:
        etk_system = batch_etk_params(etk_params_list, atom_starts, dim)

    double_bond_data = _extract_double_bond_data(replicated_mols, atom_starts)
    stereo_bond_data = _extract_stereo_bond_data(replicated_mols, atom_starts)

    chiral_dist_data = None
    if enforce_chirality:
        chiral_dist_data = _extract_chiral_dist_data(
            replicated_mols, bounds_matrices, atom_starts, dg_system
        )

    return PipelineContext(
        n_mols=n_entries,
        dim=dim,
        atom_starts=atom_starts,
        n_atoms_total=n_atoms_total,
        positions=positions,
        active=[True] * n_entries,
        failed=[False] * n_entries,
        dg_system=dg_system,
        tet_data=tet_data,
        etk_system=etk_system,
        double_bond_data=double_bond_data,
        stereo_bond_data=stereo_bond_data,
        chiral_dist_data=chiral_dist_data,
        bounds_matrices=bounds_matrices,
        rng=rng,
        entry_mol_map=entry_mol_map,
    )


def extract_mol_params_cache(
    mols: list[Chem.Mol],
    dim: int = 4,
    basin_size_tol: float = 1e8,
    params: "rdDistGeom.EmbedParameters | None" = None,
    use_etk: bool = False,
) -> list[tuple]:
    """Pre-extract per-molecule parameters for caching across retry rounds.

    Returns a list of (bounds_mat, dg_params, tet_terms, etk_params) tuples,
    one per molecule. etk_params is None if use_etk is False.
    """
    cache = []
    for mol in mols:
        bounds_mat = get_bounds_matrix(mol)
        dg_params = extract_dg_params(mol, bounds_mat, dim, basin_size_tol)
        tet_terms = extract_tetrahedral_atoms(mol)
        etk = extract_etk_params(mol, bounds_mat, params=params) if use_etk else None
        cache.append((bounds_mat, dg_params, tet_terms, etk))
    return cache


def create_pipeline_context_from_cache(
    mols: list[Chem.Mol],
    mol_cache: list[tuple],
    confs_per_mol: list[int],
    dim: int = 4,
    use_etk: bool = False,
    enforce_chirality: bool = True,
    rng: np.random.Generator | None = None,
) -> PipelineContext:
    """Create a pipeline context from pre-extracted per-molecule parameters.

    Same as create_pipeline_context_multi_conf but skips the expensive
    RDKit parameter extraction, using cached results instead.

    Args:
        mols: Unique RDKit molecules with hydrogens added.
        mol_cache: Pre-extracted params from extract_mol_params_cache().
        confs_per_mol: Number of conformer slots per molecule.
        dim: Coordinate dimension (4 for ETKDG).
        use_etk: Whether ETK parameters were extracted.
        enforce_chirality: Whether to set up chirality check data.
        rng: Persistent numpy RNG.

    Returns:
        PipelineContext with sum(confs_per_mol) batch slots.
    """
    n_unique = len(mols)

    # Replicate cached params for each conformer slot
    dg_params_list = []
    tet_terms_list = []
    etk_params_list = []
    bounds_matrices = []
    entry_mol_map: list[int] = []
    replicated_mols: list[Chem.Mol] = []

    for mol_idx in range(n_unique):
        bounds_mat, dg_params, tet_terms, etk = mol_cache[mol_idx]
        n_confs = confs_per_mol[mol_idx]
        for _ in range(n_confs):
            dg_params_list.append(dg_params)
            tet_terms_list.append(tet_terms)
            bounds_matrices.append(bounds_mat)
            entry_mol_map.append(mol_idx)
            replicated_mols.append(mols[mol_idx])
            if use_etk and etk is not None:
                etk_params_list.append(etk)

    n_entries = len(dg_params_list)

    # Batch using existing functions
    dg_system = batch_dg_params(dg_params_list, dim)

    atom_starts_np = np.array(dg_system.atom_starts.tolist(), dtype=np.int32)
    tet_data = _batch_tetrahedral_terms(tet_terms_list, atom_starts_np)

    atom_starts = dg_system.atom_starts.tolist()
    n_atoms_total = dg_system.n_atoms_total

    positions = mx.zeros(n_atoms_total * dim, dtype=mx.float32)

    etk_system = None
    if use_etk and etk_params_list:
        etk_system = batch_etk_params(etk_params_list, atom_starts, dim)

    double_bond_data = _extract_double_bond_data(replicated_mols, atom_starts)
    stereo_bond_data = _extract_stereo_bond_data(replicated_mols, atom_starts)

    chiral_dist_data = None
    if enforce_chirality:
        chiral_dist_data = _extract_chiral_dist_data(
            replicated_mols, bounds_matrices, atom_starts, dg_system
        )

    return PipelineContext(
        n_mols=n_entries,
        dim=dim,
        atom_starts=atom_starts,
        n_atoms_total=n_atoms_total,
        positions=positions,
        active=[True] * n_entries,
        failed=[False] * n_entries,
        dg_system=dg_system,
        tet_data=tet_data,
        etk_system=etk_system,
        double_bond_data=double_bond_data,
        stereo_bond_data=stereo_bond_data,
        chiral_dist_data=chiral_dist_data,
        bounds_matrices=bounds_matrices,
        rng=rng,
        entry_mol_map=entry_mol_map,
    )
