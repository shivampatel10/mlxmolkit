"""Benchmark: DG BFGS TG vs serial BFGS vs L-BFGS TG.

Uses molecules from MMFF94_dative.sdf at batch sizes 1,10,20,...100.
Measures wall-clock time for the minimization kernel only (no preprocessing).
"""

import time

import mlx.core as mx
import numpy as np
from rdkit import Chem

from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs, metal_dg_bfgs_tg
from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix


def load_mols(path="tests/test_data/MMFF94_dative.sdf", max_atoms=64):
    """Load molecules, filter to those fitting in Metal kernel."""
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    mols = []
    for m in suppl:
        if m is not None and m.GetNumAtoms() <= max_atoms:
            mols.append(m)
    return mols


def prepare_batch(mols, dim=4, seed=42):
    """Extract DG params, batch, and create random initial positions."""
    params_list = []
    for mol in mols:
        bounds_mat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bounds_mat, dim=dim)
        params_list.append(params)

    system = batch_dg_params(params_list, dim=dim)
    np.random.seed(seed)
    coords = np.random.randn(system.n_atoms_total * dim).astype(np.float32) * 2.0
    pos = mx.array(coords)
    return system, pos


def bench(fn, pos, system, n_warmup=3, n_repeats=10, **kwargs):
    """Benchmark a kernel function, return median time in seconds."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(pos, system, max_iters=400, **kwargs)
        mx.eval(*result)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(pos, system, max_iters=400, **kwargs)
        mx.eval(*result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]  # median


def fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    all_mols = load_mols()
    print(f"Loaded {len(all_mols)} molecules (max 64 atoms each)")
    print()

    batch_sizes = [1, 10, 20, 40, 60, 80, 100]

    # Header
    print(f"{'N':>5s}  {'atoms':>6s}  {'Serial BFGS':>12s}  {'BFGS TG':>12s}  {'L-BFGS TG':>12s}  {'TG speedup':>10s}  {'L-BFGS speedup':>14s}")
    print("-" * 90)

    for n in batch_sizes:
        mols = all_mols[:n]
        n_atoms = sum(m.GetNumAtoms() for m in mols)
        system, pos = prepare_batch(mols)

        t_serial = bench(metal_dg_bfgs, pos, system)
        t_tg = bench(metal_dg_bfgs_tg, pos, system)
        t_lbfgs = bench(metal_dg_lbfgs, pos, system)

        speedup_tg = t_serial / t_tg
        speedup_lb = t_serial / t_lbfgs

        print(
            f"{n:>5d}  {n_atoms:>6d}  {fmt(t_serial):>12s}  {fmt(t_tg):>12s}  "
            f"{fmt(t_lbfgs):>12s}  {speedup_tg:>9.1f}x  {speedup_lb:>13.1f}x"
        )

    print()
    print("TG speedup = Serial BFGS time / BFGS TG time")
    print("L-BFGS speedup = Serial BFGS time / L-BFGS TG time")


if __name__ == "__main__":
    main()
