"""GPU-accelerated MMFF94 optimization for molecular conformers.

Public API: MMFFOptimizeMoleculesConfs()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from rdkit.Chem import rdForceFieldHelpers

from .forcefields.mmff import mmff_energy_and_grad
from .minimizer.bfgs_vectorized import bfgs_minimize_vectorized
from .preprocessing.mmff_batching import batch_mmff_params
from .preprocessing.mmff_extract import MMFFParams, extract_mmff_params

if TYPE_CHECKING:
    from rdkit.Chem import Mol


def _run_bfgs(
    pos: mx.array,
    system,
    atom_starts_list: list[int],
    max_iters: int,
) -> tuple[mx.array, mx.array]:
    """Run BFGS minimization, preferring Metal kernel with pure-MLX fallback."""
    try:
        from .metal_kernels.mmff_bfgs import metal_mmff_bfgs

        final_pos, final_energies, _ = metal_mmff_bfgs(
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

    final_pos, final_energies = _run_bfgs(pos, system, atom_starts_list, max_iters)

    final_pos_np = np.array(final_pos, dtype=np.float64)
    final_energies_np = np.array(final_energies, dtype=np.float64)

    for batch_idx, (global_mol_idx, conf_idx) in enumerate(conf_map):
        mol = chunk[global_mol_idx - mol_offset]
        conf = mol.GetConformer(conf_idx)
        start = atom_starts_list[batch_idx]
        n_atoms = mol.GetNumAtoms()
        coords = final_pos_np[start * 3 : (start + n_atoms) * 3].reshape(-1, 3)
        for atom_idx in range(n_atoms):
            conf.SetAtomPosition(
                atom_idx,
                (float(coords[atom_idx, 0]),
                 float(coords[atom_idx, 1]),
                 float(coords[atom_idx, 2])),
            )
        result[global_mol_idx].append(float(final_energies_np[batch_idx]))
