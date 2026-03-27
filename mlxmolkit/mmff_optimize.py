"""GPU-accelerated MMFF94 optimization for molecular conformers.

Public API: MMFFOptimizeMoleculesConfs()
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from .forcefields.mmff import mmff_energy_and_grad
from .minimizer.bfgs_vectorized import bfgs_minimize_vectorized
from .preprocessing.mmff_batching import batch_mmff_params
from .preprocessing.mmff_extract import MMFFParams, extract_mmff_params

if TYPE_CHECKING:
    from rdkit.Chem import Mol


RESTART_MAX_INITIAL_DROP = 10.0
RESTART_PLANAR_DEG = 5.0
RESTART_TORSION_DELTAS_DEG = (0.5, -0.5)
RESTART_ACCEPT_TOL = 1e-3


def _has_formal_charge(mol: Mol) -> bool:
    """Return whether the molecule has any formal charge."""
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _is_pi_bond_atom(atom: Chem.Atom) -> bool:
    """Return whether an atom participates in a pi system."""
    return atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2


def _dihedral_deg(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float:
    """Compute the signed dihedral angle in degrees."""
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3

    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-12:
        return 0.0
    b1_unit = b1 / b1_norm

    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < 1e-12 or w_norm < 1e-12:
        return 0.0

    v_unit = v / v_norm
    w_unit = w / w_norm
    x = np.dot(v_unit, w_unit)
    y = np.dot(np.cross(b1_unit, v_unit), w_unit)
    return math.degrees(math.atan2(y, x))


def _collect_component_atoms(mol: Mol, start: int, blocked: int) -> list[int]:
    """Collect the connected component reached from ``start`` without crossing a bond."""
    seen: set[int] = set()
    stack = [start]
    while stack:
        atom_idx = stack.pop()
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if ((atom_idx == start and nbr_idx == blocked)
                    or (atom_idx == blocked and nbr_idx == start)):
                continue
            stack.append(nbr_idx)
    return sorted(seen)


def _choose_restart_bonds(mol: Mol, coords: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """Find near-planar non-ring pi bonds that are good restart candidates."""
    candidates: list[tuple[int, int, list[int]]] = []
    n_atoms = mol.GetNumAtoms()
    if coords.shape != (n_atoms, 3):
        return candidates

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        atom_j = bond.GetBeginAtom()
        atom_k = bond.GetEndAtom()
        if atom_j.GetAtomicNum() == 1 or atom_k.GetAtomicNum() == 1:
            continue
        if atom_j.GetDegree() < 2 or atom_k.GetDegree() < 2:
            continue
        if not (_is_pi_bond_atom(atom_j) and _is_pi_bond_atom(atom_k)):
            continue

        nbr_i = next(
            (nbr.GetIdx() for nbr in atom_j.GetNeighbors()
             if nbr.GetIdx() != k and nbr.GetAtomicNum() > 1),
            None,
        )
        nbr_l = next(
            (nbr.GetIdx() for nbr in atom_k.GetNeighbors()
             if nbr.GetIdx() != j and nbr.GetAtomicNum() > 1),
            None,
        )
        if nbr_i is None or nbr_l is None:
            continue

        rotate_atoms = _collect_component_atoms(mol, k, j)
        if not rotate_atoms or len(rotate_atoms) == n_atoms:
            continue

        phi = _dihedral_deg(coords[nbr_i], coords[j], coords[k], coords[nbr_l])
        planarity = min(abs(phi), abs(abs(phi) - 180.0))
        if planarity > RESTART_PLANAR_DEG:
            continue

        candidates.append((j, k, rotate_atoms))

    return candidates


def _rotate_atoms_about_bond(
    coords: np.ndarray,
    atom_j: int,
    atom_k: int,
    rotate_atoms: list[int],
    delta_deg: float,
) -> np.ndarray | None:
    """Rotate one side of a bond by ``delta_deg`` around the bond axis."""
    origin = coords[atom_j]
    axis = coords[atom_k] - coords[atom_j]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return None

    axis_unit = axis / axis_norm
    theta = math.radians(delta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = coords.copy()

    rotate_idx = np.array(rotate_atoms, dtype=np.int32)
    rel = rotated[rotate_idx] - origin
    cross = np.cross(axis_unit, rel)
    dot = rel @ axis_unit
    rotated_rel = (
        rel * cos_t
        + cross * sin_t
        + np.outer(dot, axis_unit) * (1.0 - cos_t)
    )
    rotated[rotate_idx] = origin + rotated_rel
    return rotated


def _maybe_restart_with_symmetry_break(
    mol: Mol,
    params: MMFFParams,
    initial_energy: float,
    initial_coords: np.ndarray,
    final_coords: np.ndarray,
    final_energy: float,
    max_iters: int,
) -> tuple[np.ndarray, float]:
    """Retry optimization from a tiny deterministic torsional perturbation.

    This targets charged, near-planar torsional systems where BFGS can settle
    into a higher symmetric stationary basin. Only a lower-energy restart is
    accepted, preserving the current MMFF architecture.
    """
    if not _has_formal_charge(mol):
        return final_coords, final_energy
    if initial_energy - final_energy > RESTART_MAX_INITIAL_DROP:
        return final_coords, final_energy

    single_system = batch_mmff_params([params])
    atom_starts_list = [0, params.num_atoms]

    best_coords = final_coords
    best_energy = final_energy

    # Restarting from the already-converged geometry can stay trapped in the
    # same symmetric basin. Also probe the original conformer when available.
    restart_seeds = [final_coords.reshape(-1, 3)]
    initial_coords_2d = initial_coords.reshape(-1, 3)
    if not np.allclose(initial_coords_2d, restart_seeds[0]):
        restart_seeds.append(initial_coords_2d)

    for seed_coords in restart_seeds:
        restart_bonds = _choose_restart_bonds(mol, seed_coords)
        if not restart_bonds:
            continue
        for atom_j, atom_k, rotate_atoms in restart_bonds:
            for delta_deg in RESTART_TORSION_DELTAS_DEG:
                rotated = _rotate_atoms_about_bond(
                    seed_coords, atom_j, atom_k, rotate_atoms, delta_deg
                )
                if rotated is None:
                    continue
                trial_pos = mx.array(rotated.reshape(-1), dtype=mx.float32)
                trial_pos, trial_energies = _run_bfgs(
                    trial_pos, single_system, atom_starts_list, max_iters
                )
                mx.eval(trial_pos, trial_energies)
                trial_energy = float(np.array(trial_energies, dtype=np.float64)[0])
                if trial_energy + RESTART_ACCEPT_TOL < best_energy:
                    best_coords = np.array(trial_pos, dtype=np.float64)
                    best_energy = trial_energy

    return best_coords, best_energy


def _run_bfgs(
    pos: mx.array,
    system,
    atom_starts_list: list[int],
    max_iters: int,
) -> tuple[mx.array, mx.array]:
    """Run BFGS minimization, preferring Metal kernel with pure-MLX fallback."""
    try:
        from .metal_kernels.mmff_bfgs import metal_mmff_bfgs_tg

        final_pos, final_energies, _ = metal_mmff_bfgs_tg(
            pos, system, max_iters=max_iters
        )
        return final_pos, final_energies
    except Exception:
        pass

    # Fallback: pure-MLX vectorized BFGS
    def energy_and_grad_fn(p):
        return mmff_energy_and_grad(p, system)

    final_pos, final_energies, _ = bfgs_minimize_vectorized(
        energy_and_grad_fn,
        pos,
        atom_starts_list,
        system.n_mols,
        dim=3,
        max_iters=max_iters,
    )
    return final_pos, final_energies


def MMFFOptimizeMoleculesConfs(
    molecules: list[Mol],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    batchSize: int = 250,
) -> list[list[float]]:
    """Optimize conformers using MMFF94 force field with GPU-accelerated BFGS.

    Validates molecules, extracts MMFF parameters, batches conformers in
    chunks, runs vectorized BFGS minimization, and updates conformer
    coordinates in-place.

    Args:
        molecules: List of RDKit molecules with conformers already generated.
        maxIters: Maximum BFGS iterations per conformer.
        nonBondedThreshold: Distance cutoff for non-bonded interactions (Angstroms).
        batchSize: Maximum molecules per GPU batch. Larger batches are
            more efficient but may hit Metal GPU timeout limits. Default
            250 (5,000 conformers at 20 confs/mol).

    Returns:
        List of lists of final energies, grouped by molecule.
        Each inner list has one energy per conformer.
    """
    if not molecules:
        return []

    result: list[list[float]] = [[] for _ in range(len(molecules))]

    for chunk_start in range(0, len(molecules), batchSize):
        chunk = molecules[chunk_start : chunk_start + batchSize]
        _mmff_optimize_chunk(chunk, chunk_start, maxIters, nonBondedThreshold, result)

    return result


def _mmff_optimize_chunk(
    chunk: list[Mol],
    mol_offset: int,
    max_iters: int,
    non_bonded_threshold: float,
    result: list[list[float]],
) -> None:
    """Optimize one chunk of molecules and accumulate results."""
    all_params: list[MMFFParams] = []
    all_positions: list[np.ndarray] = []
    conf_map: list[tuple[int, int]] = []  # (global_mol_idx, conf_idx)

    for local_idx, mol in enumerate(chunk):
        if mol is None:
            continue
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if props is None:
            continue
        n_confs = mol.GetNumConformers()
        if n_confs == 0:
            continue

        params = extract_mmff_params(
            mol, conf_id=-1, nonBondedThreshold=non_bonded_threshold
        )
        if params is None:
            continue

        n_atoms = mol.GetNumAtoms()
        global_idx = mol_offset + local_idx
        for conf_idx in range(n_confs):
            conf = mol.GetConformer(conf_idx)
            positions = np.empty(n_atoms * 3, dtype=np.float32)
            for i in range(n_atoms):
                pt = conf.GetAtomPosition(i)
                positions[i * 3] = pt.x
                positions[i * 3 + 1] = pt.y
                positions[i * 3 + 2] = pt.z
            all_params.append(params)
            all_positions.append(positions)
            conf_map.append((global_idx, conf_idx))

    if not all_params:
        return

    system = batch_mmff_params(all_params)
    pos = mx.array(np.concatenate(all_positions), dtype=mx.float32)
    atom_starts_list = [int(system.atom_starts[i]) for i in range(system.n_mols + 1)]
    initial_energies, _ = mmff_energy_and_grad(pos, system)
    mx.eval(initial_energies)

    final_pos, final_energies = _run_bfgs(pos, system, atom_starts_list, max_iters)

    final_pos_np = np.array(final_pos, dtype=np.float64)
    final_energies_np = np.array(final_energies, dtype=np.float64)
    initial_energies_np = np.array(initial_energies, dtype=np.float64)

    for batch_idx, (global_mol_idx, conf_idx) in enumerate(conf_map):
        mol = chunk[global_mol_idx - mol_offset]
        conf = mol.GetConformer(conf_idx)
        start = atom_starts_list[batch_idx]
        n_atoms = mol.GetNumAtoms()
        initial_coords = all_positions[batch_idx]
        coords = final_pos_np[start * 3 : (start + n_atoms) * 3]
        coords, final_energies_np[batch_idx] = _maybe_restart_with_symmetry_break(
            mol,
            all_params[batch_idx],
            float(initial_energies_np[batch_idx]),
            initial_coords,
            coords,
            float(final_energies_np[batch_idx]),
            max_iters,
        )
        coords = coords.reshape(-1, 3)
        for atom_idx in range(n_atoms):
            conf.SetAtomPosition(
                atom_idx,
                (float(coords[atom_idx, 0]),
                 float(coords[atom_idx, 1]),
                 float(coords[atom_idx, 2])),
            )
        result[global_mol_idx].append(float(final_energies_np[batch_idx]))
