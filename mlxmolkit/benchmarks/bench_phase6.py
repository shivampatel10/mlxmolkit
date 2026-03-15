"""Phase 6 benchmark: Metal GPU vs RDKit parallel CPU.

Usage:
    uv run python -m mlxmolkit.benchmarks.bench_phase6

Compares wall clock throughput for DG BFGS minimization:
  - Metal L-BFGS kernel (GPU, threadgroup-parallel)
  - Metal BFGS kernel (GPU, one thread per mol)
  - Vectorized Python BFGS (GPU, batched)
  - RDKit CPU parallel (multiprocessing.Pool, 8 workers)
  - RDKit CPU serial (single thread baseline)

Sweeps batch sizes to find GPU saturation point.
"""

import multiprocessing as mp
import time
from functools import partial

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from mlxmolkit.forcefields.dist_geom import dg_energy_and_grad
from mlxmolkit.minimizer.bfgs_vectorized import bfgs_minimize_vectorized
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix

# Drug-like molecules of varying sizes
TEST_SMILES = [
    "CCO",                        # ethanol (9 atoms w/ H)
    "CCC",                        # propane (11)
    "c1ccccc1",                   # benzene (12)
    "CC(=O)O",                    # acetic acid (8)
    "CC(C)O",                     # isopropanol (12)
    "c1ccc(cc1)O",                # phenol (13)
    "CC(=O)NC",                   # N-methylacetamide (12)
    "C1CCCCC1",                   # cyclohexane (18)
    "CC(C)(C)O",                  # tert-butanol (15)
    "c1ccncc1",                   # pyridine (11)
    "c1ccc2ccccc2c1",             # naphthalene (18)
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin (21)
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine (24)
    "c1ccc(cc1)CC(=O)O",         # phenylacetic acid (18)
    "OC(=O)c1ccccc1O",           # salicylic acid (16)
]

N_WORKERS = 8


def _make_mols(n_mols, seed=42):
    """Create n_mols RDKit molecules with Hs."""
    mols = []
    for i in range(n_mols):
        smi = TEST_SMILES[i % len(TEST_SMILES)]
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        mols.append(mol)
    return mols


def _make_batch(n_mols, seed=42):
    """Create batched DG system + random 4D coords."""
    np.random.seed(seed)
    mols = _make_mols(n_mols)
    params_list = []
    for mol in mols:
        bounds_mat = get_bounds_matrix(mol)
        params = extract_dg_params(mol, bounds_mat, dim=4)
        params_list.append(params)

    system = batch_dg_params(params_list, dim=4)
    coords = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 2.0
    pos = mx.array(coords)
    return system, pos, mols


# --- RDKit workers ---

def _rdkit_embed_one(mol_block, max_iters):
    """Worker: embed one molecule from MolBlock. Returns 1 on success."""
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return 0
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    params.maxIterations = max_iters
    cid = AllChem.EmbedMolecule(mol, params)
    return 1 if cid >= 0 else 0


def bench_rdkit_parallel(n_mols, max_iters=200, n_workers=N_WORKERS):
    """RDKit CPU with multiprocessing.Pool."""
    mols = _make_mols(n_mols)
    mol_blocks = [Chem.MolToMolBlock(m) for m in mols]

    worker_fn = partial(_rdkit_embed_one, max_iters=max_iters)

    # Warmup pool
    with mp.Pool(n_workers) as pool:
        list(pool.map(worker_fn, mol_blocks[:min(n_workers, len(mol_blocks))]))

    # Timed run
    start = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        results = list(pool.map(worker_fn, mol_blocks))
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def bench_rdkit_serial(n_mols, max_iters=200):
    """RDKit CPU single-threaded."""
    mols = _make_mols(n_mols)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    params.maxIterations = max_iters
    params.numThreads = 1

    # Warmup
    AllChem.EmbedMolecule(Chem.AddHs(Chem.MolFromSmiles("C")), params)

    start = time.perf_counter()
    ok = 0
    for mol in mols:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid >= 0:
            ok += 1
    elapsed = time.perf_counter() - start
    return elapsed, ok


# --- Metal / Vectorized ---

def bench_metal_lbfgs(n_mols, max_iters=200):
    """Metal DG L-BFGS kernel (GPU, threadgroup-parallel)."""
    from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

    system, pos, _ = _make_batch(n_mols)

    # Warmup (compile + first run)
    metal_dg_lbfgs(pos, system, max_iters=5)
    mx.eval(pos)

    start = time.perf_counter()
    final_pos, final_e, statuses = metal_dg_lbfgs(pos, system, max_iters=max_iters)
    mx.eval(final_pos, final_e, statuses)
    elapsed = time.perf_counter() - start
    return elapsed, final_e


def bench_metal_bfgs(n_mols, max_iters=200):
    """Metal DG BFGS kernel (GPU)."""
    from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

    system, pos, _ = _make_batch(n_mols)

    # Warmup (compile + first run)
    metal_dg_bfgs(pos, system, max_iters=5)
    mx.eval(pos)

    start = time.perf_counter()
    final_pos, final_e, statuses = metal_dg_bfgs(pos, system, max_iters=max_iters)
    mx.eval(final_pos, final_e, statuses)
    elapsed = time.perf_counter() - start
    return elapsed, final_e


def bench_vectorized_bfgs(n_mols, max_iters=200):
    """Vectorized Python BFGS (GPU)."""
    system, pos, _ = _make_batch(n_mols)

    def energy_grad_fn(p):
        return dg_energy_and_grad(p, system)

    # Warmup
    bfgs_minimize_vectorized(energy_grad_fn, pos, system.atom_starts.tolist(),
                             n_mols=n_mols, dim=4, max_iters=5)

    start = time.perf_counter()
    final_pos, final_e, statuses = bfgs_minimize_vectorized(
        energy_grad_fn, pos, system.atom_starts.tolist(),
        n_mols=n_mols, dim=4, max_iters=max_iters,
    )
    mx.eval(final_pos, final_e, statuses)
    elapsed = time.perf_counter() - start
    return elapsed, final_e


def main():
    batch_sizes = [1, 10, 50, 100, 200, 500, 1000]

    print("=" * 95)
    print("Phase 6 Benchmark: Metal GPU vs RDKit CPU (DG BFGS minimization)")
    print(f"  RDKit parallel workers: {N_WORKERS}")
    print(f"  Max BFGS iterations: 200")
    print("=" * 95)
    print()
    print(f"{'N':>6}  {'Metal L-BFGS':>14}  {'Metal BFGS':>14}  {'Vec. BFGS':>14}  {'RDKit 8-CPU':>14}  {'RDKit 1-CPU':>14}  {'L-BFGS/RDKit8':>14}")
    print(f"{'':>6}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'(speedup)':>14}")
    print("-" * 95)

    for n in batch_sizes:
        row = f"{n:>6}"

        # Metal L-BFGS
        try:
            t, _ = bench_metal_lbfgs(n)
            lbfgs_ms = t / n * 1000
            row += f"  {lbfgs_ms:>11.2f} ms"
        except Exception as ex:
            lbfgs_ms = None
            row += f"  {'FAIL':>14}"

        # Metal BFGS
        try:
            t, _ = bench_metal_bfgs(n)
            metal_ms = t / n * 1000
            row += f"  {metal_ms:>11.2f} ms"
        except Exception as ex:
            metal_ms = None
            row += f"  {'FAIL':>14}"

        # Vectorized BFGS
        try:
            t, _ = bench_vectorized_bfgs(n)
            vec_ms = t / n * 1000
            row += f"  {vec_ms:>11.2f} ms"
        except Exception as ex:
            vec_ms = None
            row += f"  {'FAIL':>14}"

        # RDKit parallel (8 CPU)
        try:
            t, _ = bench_rdkit_parallel(n)
            rdkit8_ms = t / n * 1000
            row += f"  {rdkit8_ms:>11.2f} ms"
        except Exception as ex:
            rdkit8_ms = None
            row += f"  {'FAIL':>14}"

        # RDKit serial (1 CPU)
        try:
            t, _ = bench_rdkit_serial(n)
            rdkit1_ms = t / n * 1000
            row += f"  {rdkit1_ms:>11.2f} ms"
        except Exception as ex:
            rdkit1_ms = None
            row += f"  {'FAIL':>14}"

        # Speedup ratio (L-BFGS vs RDKit 8-CPU)
        if lbfgs_ms and rdkit8_ms:
            if lbfgs_ms < rdkit8_ms:
                row += f"  {rdkit8_ms/lbfgs_ms:>11.1f}x  <-- L-BFGS wins"
            else:
                row += f"  {lbfgs_ms/rdkit8_ms:>10.1f}x slower"
        else:
            row += f"  {'N/A':>14}"

        print(row)

    print()
    print("=" * 95)


if __name__ == "__main__":
    main()
