"""Benchmark: MMFF94 optimization performance (pure-MLX path).

Measures wall-clock time for MMFFOptimizeMoleculesConfs using the
pure-MLX vectorized BFGS path. Baseline for Metal kernel comparison.

Usage:
    uv run python benchmarks/bench_mmff.py
"""

import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.mmff_optimize import MMFFOptimizeMoleculesConfs
from mlxmolkit.preprocessing.mmff_extract import extract_mmff_params
from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
from mlxmolkit.forcefields.mmff import mmff_energy_and_grad

SDF_PATH = "tests/test_data/MMFF94_dative.sdf"


def load_molecules(n):
    """Load first n molecules from SDF, with hydrogens."""
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
    mols = []
    for mol in suppl:
        if mol is not None:
            mols.append(mol)
        if len(mols) >= n:
            break
    return mols


def embed_conformers(mols, n_confs=10):
    """Embed multiple conformers per molecule using RDKit."""
    for mol in mols:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=42)
        elif mol.GetNumConformers() < n_confs:
            # Already has conformers from SDF, embed more
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=42)


def benchmark_rdkit_mmff(mols, max_iters=200, n_repeats=3):
    """Benchmark RDKit's MMFFOptimizeMolecule (CPU, sequential)."""
    times = []
    for _ in range(n_repeats):
        # Make copies so we don't re-optimize already-optimized conformers
        copies = [Chem.Mol(m) for m in mols]
        t0 = time.perf_counter()
        for mol in copies:
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            if props is None:
                continue
            for conf_id in range(mol.GetNumConformers()):
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                    mol, props, confId=conf_id
                )
                if ff is not None:
                    ff.Minimize(maxIts=max_iters)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def benchmark_mlx_mmff(mols, max_iters=200, n_repeats=3):
    """Benchmark mlxmolkit's MMFFOptimizeMoleculesConfs (MLX GPU)."""
    times = []
    for _ in range(n_repeats):
        copies = [Chem.Mol(m) for m in mols]
        t0 = time.perf_counter()
        MMFFOptimizeMoleculesConfs(copies, maxIters=max_iters)
        mx.eval()  # Force evaluation
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def count_conformers(mols):
    """Count total conformers across all molecules."""
    return sum(m.GetNumConformers() for m in mols)


def count_atoms(mols):
    """Count total atoms across all molecules."""
    return sum(m.GetNumAtoms() for m in mols)


def print_header():
    print("=" * 78)
    print("MMFF94 Optimization Benchmark — Pure MLX vs RDKit CPU")
    print("=" * 78)
    print()


def print_results(label, mols, n_confs_per_mol, rdkit_times, mlx_times):
    n_mols = len(mols)
    total_confs = count_conformers(mols)
    total_atoms = count_atoms(mols)

    rdkit_med = np.median(rdkit_times) * 1000
    mlx_med = np.median(mlx_times) * 1000
    speedup = rdkit_med / mlx_med if mlx_med > 0 else float("inf")

    print(f"--- {label} ---")
    print(f"  Molecules:       {n_mols}")
    print(f"  Confs/mol:       {n_confs_per_mol}")
    print(f"  Total conformers:{total_confs}")
    print(f"  Total atoms:     {total_atoms}")
    print(f"  RDKit CPU:       {rdkit_med:8.1f} ms (median of {len(rdkit_times)})")
    print(f"  MLX GPU:         {mlx_med:8.1f} ms (median of {len(mlx_times)})")
    print(f"  Speedup:         {speedup:8.2f}x")
    print(f"  MLX confs/sec:   {total_confs / (mlx_med / 1000):8.1f}")
    print(f"  MLX mols/sec:    {n_mols / (mlx_med / 1000):8.1f}")
    print()


def main():
    print_header()

    max_iters = 200
    n_repeats = 3

    # Warmup run
    print("Warmup...")
    warmup_mols = load_molecules(2)
    embed_conformers(warmup_mols, n_confs=2)
    MMFFOptimizeMoleculesConfs([Chem.Mol(m) for m in warmup_mols], maxIters=50)
    mx.eval()
    print("Warmup done.\n")

    # --- Batch size: 1 molecule ---
    mols_1 = load_molecules(1)
    embed_conformers(mols_1, n_confs=10)
    rdkit_t = benchmark_rdkit_mmff(mols_1, max_iters, n_repeats)
    mlx_t = benchmark_mlx_mmff(mols_1, max_iters, n_repeats)
    print_results("1 molecule, 10 conformers", mols_1, 10, rdkit_t, mlx_t)

    # --- Batch size: 10 molecules ---
    mols_10 = load_molecules(10)
    embed_conformers(mols_10, n_confs=10)
    rdkit_t = benchmark_rdkit_mmff(mols_10, max_iters, n_repeats)
    mlx_t = benchmark_mlx_mmff(mols_10, max_iters, n_repeats)
    print_results("10 molecules, 10 conformers each", mols_10, 10, rdkit_t, mlx_t)

    # --- Batch size: 50 molecules ---
    mols_50 = load_molecules(50)
    embed_conformers(mols_50, n_confs=10)
    rdkit_t = benchmark_rdkit_mmff(mols_50, max_iters, n_repeats)
    mlx_t = benchmark_mlx_mmff(mols_50, max_iters, n_repeats)
    print_results("50 molecules, 10 conformers each", mols_50, 10, rdkit_t, mlx_t)

    print("=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
