"""Benchmark: DG energy+gradient computation — MLX GPU vs NumPy CPU.

Also benchmarks RDKit EmbedMultipleConfs for overall context.
"""

import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.forcefields.dist_geom import dg_energy, dg_energy_and_grad
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params

SDF_PATH = "tests/test_data/MMFF94_dative.sdf"


# ---------- NumPy CPU reference implementation ----------


def _np_dist_violation_energy_grad(pos, idx1, idx2, lb2, ub2, weight, dim):
    """Pure numpy CPU implementation of distance violation energy+grad."""
    pos_r = pos.reshape(-1, dim)
    p1 = pos_r[idx1]
    p2 = pos_r[idx2]
    diff = p1 - p2
    d2 = np.sum(diff * diff, axis=1)

    # Energy
    val_ub = (d2 / ub2) - 1.0
    e_ub = np.where(d2 > ub2, weight * val_ub * val_ub, 0.0)
    val_lb = (2.0 * lb2) / (lb2 + d2) - 1.0
    e_lb = np.where(d2 < lb2, weight * val_lb * val_lb, 0.0)
    energy = e_ub + e_lb

    # Gradient
    pf_ub = np.where(d2 > ub2, 4.0 * ((d2 / ub2) - 1.0) / ub2, 0.0)
    l2d2 = d2 + lb2
    pf_lb = np.where(
        d2 < lb2, 8.0 * lb2 * (1.0 - 2.0 * lb2 / l2d2) / (l2d2 * l2d2), 0.0
    )
    prefactor = weight * (pf_ub + pf_lb)
    g = prefactor[:, None] * diff

    n_atoms = pos_r.shape[0]
    grad = np.zeros((n_atoms * dim,), dtype=np.float32)
    for d_idx in range(dim):
        np.add.at(grad, idx1 * dim + d_idx, g[:, d_idx])
        np.add.at(grad, idx2 * dim + d_idx, -g[:, d_idx])

    return energy, grad


def np_dg_energy_and_grad(pos_np, system_np, chiral_weight=1.0, fourth_dim_weight=0.1):
    """NumPy CPU implementation of combined DG energy+gradient."""
    dim = system_np["dim"]
    n_mols = system_np["n_mols"]

    energies = np.zeros(n_mols, dtype=np.float32)
    grad = np.zeros_like(pos_np)

    # Distance violation
    if len(system_np["dist_idx1"]) > 0:
        e_dist, g_dist = _np_dist_violation_energy_grad(
            pos_np,
            system_np["dist_idx1"],
            system_np["dist_idx2"],
            system_np["dist_lb2"],
            system_np["dist_ub2"],
            system_np["dist_weight"],
            dim,
        )
        np.add.at(energies, system_np["dist_mol_indices"], e_dist)
        grad += g_dist

    # Fourth dim
    if dim == 4 and len(system_np["fourth_idx"]) > 0:
        idx = system_np["fourth_idx"]
        w = pos_np.reshape(-1, dim)[idx, 3]
        e_fourth = fourth_dim_weight * w * w
        np.add.at(energies, system_np["fourth_mol_indices"], e_fourth)
        np.add.at(grad, idx * dim + 3, 2.0 * fourth_dim_weight * w)

    return energies, grad


def batched_system_to_numpy(system):
    """Convert BatchedDGSystem to plain numpy dict for CPU benchmark."""
    return {
        "n_mols": system.n_mols,
        "dim": system.dim,
        "dist_idx1": np.array(system.dist_idx1),
        "dist_idx2": np.array(system.dist_idx2),
        "dist_lb2": np.array(system.dist_lb2),
        "dist_ub2": np.array(system.dist_ub2),
        "dist_weight": np.array(system.dist_weight),
        "dist_mol_indices": np.array(system.dist_mol_indices),
        "fourth_idx": np.array(system.fourth_idx),
        "fourth_mol_indices": np.array(system.fourth_mol_indices),
    }


def load_molecules(n):
    """Load first n molecules from SDF."""
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
    mols = []
    for mol in suppl:
        if mol is not None:
            mols.append(mol)
        if len(mols) >= n:
            break
    return mols


def benchmark_mlx_energy_grad(system, pos_mx, n_iters=100):
    """Benchmark MLX GPU energy+gradient computation."""
    # Warmup
    for _ in range(5):
        e, g = dg_energy_and_grad(pos_mx, system)
        mx.eval(e, g)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        e, g = dg_energy_and_grad(pos_mx, system)
        mx.eval(e, g)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def benchmark_numpy_energy_grad(system_np, pos_np, n_iters=100):
    """Benchmark NumPy CPU energy+gradient computation."""
    # Warmup
    for _ in range(5):
        np_dg_energy_and_grad(pos_np, system_np)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        np_dg_energy_and_grad(pos_np, system_np)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def benchmark_rdkit_embed(mols, n_confs=5, n_iters=3):
    """Benchmark RDKit EmbedMultipleConfs for reference."""
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42

    # Warmup
    for mol in mols[:2]:
        m = Chem.RWMol(mol)
        rdDistGeom.EmbedMultipleConfs(m, n_confs, params)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        for mol in mols:
            m = Chem.RWMol(mol)
            rdDistGeom.EmbedMultipleConfs(m, n_confs, params)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def main():
    print("=" * 70)
    print("DG Energy+Gradient Benchmark: MLX GPU vs NumPy CPU")
    print("=" * 70)

    for n_mols in [5, 20, 50, 100]:
        mols = load_molecules(n_mols)
        actual_n = len(mols)
        if actual_n < n_mols:
            print(f"\n[Only {actual_n} molecules available, requested {n_mols}]")

        # Extract and batch DG params
        params_list = [extract_dg_params(mol, dim=4) for mol in mols]
        system = batch_dg_params(params_list, dim=4)
        system_np = batched_system_to_numpy(system)

        # Random positions
        np.random.seed(42)
        pos_np = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 0.5
        pos_mx = mx.array(pos_np)

        n_dist_terms = system.dist_idx1.size
        n_atoms = system.n_atoms_total

        print(f"\n--- {actual_n} molecules, {n_atoms} atoms, {n_dist_terms} dist terms ---")

        # MLX benchmark
        n_iters = 200 if n_mols <= 20 else 100
        t_mlx = benchmark_mlx_energy_grad(system, pos_mx, n_iters=n_iters)
        print(f"  MLX (GPU):   {t_mlx*1000:.3f} ms/iter")

        # NumPy benchmark
        n_iters_np = 50 if n_mols <= 20 else 20
        t_np = benchmark_numpy_energy_grad(system_np, pos_np, n_iters=n_iters_np)
        print(f"  NumPy (CPU): {t_np*1000:.3f} ms/iter")

        speedup = t_np / t_mlx if t_mlx > 0 else float("inf")
        print(f"  Speedup:     {speedup:.1f}x")

    # RDKit full embedding benchmark for context
    print("\n" + "=" * 70)
    print("RDKit EmbedMultipleConfs (full pipeline, CPU) — for context")
    print("=" * 70)

    for n_mols in [5, 20, 50]:
        mols = load_molecules(n_mols)
        actual_n = len(mols)
        t_rdkit = benchmark_rdkit_embed(mols, n_confs=5, n_iters=3)
        print(f"  {actual_n} mols × 5 confs: {t_rdkit*1000:.1f} ms total")


if __name__ == "__main__":
    main()
