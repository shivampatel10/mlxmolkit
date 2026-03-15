#!/usr/bin/env python3
"""Head-to-head: Metal L-BFGS GPU vs RDKit 8-CPU parallel.

Measures full DG BFGS minimization throughput (the bottleneck stage).
RDKit uses mp.Pool with 8 workers matching the user's confgen.py pattern.
Metal batches all molecules into one GPU dispatch.

Usage:
    cd /Users/shivam.patel/repos/Code/OpenSource/mlxmolkit
    uv run python bench_lbfgs.py
"""

import multiprocessing as mp
import time
from functools import partial

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix

# Drug-like molecules of varying sizes (9–24 heavy atoms w/ H)
SMILES = [
    "CCO",                                    # ethanol
    "CCC",                                    # propane
    "c1ccccc1",                               # benzene
    "CC(=O)O",                                # acetic acid
    "CC(C)O",                                 # isopropanol
    "c1ccc(cc1)O",                            # phenol
    "CC(=O)NC",                               # N-methylacetamide
    "C1CCCCC1",                               # cyclohexane
    "CC(C)(C)O",                              # tert-butanol
    "c1ccncc1",                               # pyridine
    "c1ccc2ccccc2c1",                         # naphthalene
    "CC(=O)Oc1ccccc1C(=O)O",                 # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",          # caffeine
    "c1ccc(cc1)CC(=O)O",                      # phenylacetic acid
    "OC(=O)c1ccccc1O",                        # salicylic acid
]

N_WORKERS = 8
MAX_ITERS = 200


# ---------- RDKit 8-CPU parallel (same pattern as confgen.py) ----------

def _rdkit_worker(mol_block):
    """Embed one molecule from MolBlock. Returns 1 on success."""
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return 0
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useRandomCoords = True
    params.maxIterations = MAX_ITERS
    params.numThreads = 0          # single-thread per worker
    cid = AllChem.EmbedMolecule(mol, params)
    return 1 if cid >= 0 else 0


def bench_rdkit_parallel(n_mols):
    mols = [Chem.AddHs(Chem.MolFromSmiles(SMILES[i % len(SMILES)])) for i in range(n_mols)]
    mol_blocks = [Chem.MolToMolBlock(m) for m in mols]

    # warm up pool
    with mp.Pool(N_WORKERS) as pool:
        list(pool.map(_rdkit_worker, mol_blocks[:N_WORKERS]))

    start = time.perf_counter()
    with mp.Pool(N_WORKERS) as pool:
        results = list(pool.map(_rdkit_worker, mol_blocks))
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


# ---------- Metal L-BFGS GPU ----------

def _make_batch(n_mols, seed=42):
    np.random.seed(seed)
    params_list = []
    for i in range(n_mols):
        smi = SMILES[i % len(SMILES)]
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        bm = get_bounds_matrix(mol)
        params_list.append(extract_dg_params(mol, bm, dim=4))
    system = batch_dg_params(params_list, dim=4)
    coords = np.random.randn(system.n_atoms_total * 4).astype(np.float32) * 2.0
    pos = mx.array(coords)
    return system, pos


def bench_metal_lbfgs(n_mols):
    from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

    system, pos = _make_batch(n_mols)

    # warm up (JIT compile)
    metal_dg_lbfgs(pos, system, max_iters=5)
    mx.eval(pos)

    start = time.perf_counter()
    fp, fe, fs = metal_dg_lbfgs(pos, system, max_iters=MAX_ITERS)
    mx.eval(fp, fe, fs)
    elapsed = time.perf_counter() - start
    return elapsed, fe


def bench_metal_bfgs(n_mols):
    from mlxmolkit.metal_kernels.dg_bfgs import metal_dg_bfgs

    system, pos = _make_batch(n_mols)

    metal_dg_bfgs(pos, system, max_iters=5)
    mx.eval(pos)

    start = time.perf_counter()
    fp, fe, fs = metal_dg_bfgs(pos, system, max_iters=MAX_ITERS)
    mx.eval(fp, fe, fs)
    elapsed = time.perf_counter() - start
    return elapsed, fe


# ---------- main ----------

def main():
    batch_sizes = [10, 50, 100, 200, 500, 1000]

    print()
    print("=" * 90)
    print("  Metal L-BFGS GPU  vs  RDKit ETKDGv3 8-CPU  (DG BFGS stage only)")
    print(f"  max_iters={MAX_ITERS}   RDKit workers={N_WORKERS}")
    print("=" * 90)
    print()
    print(f"{'N':>6}  {'L-BFGS GPU':>14}  {'dense BFGS':>14}  {'RDKit 8-CPU':>14}  {'L-BFGS speedup':>16}")
    print(f"{'':>6}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'vs RDKit 8-CPU':>16}")
    print("-" * 90)

    for n in batch_sizes:
        row = f"{n:>6}"

        # Metal L-BFGS
        try:
            t, _ = bench_metal_lbfgs(n)
            lbfgs_ms = t / n * 1000
            row += f"  {lbfgs_ms:>11.2f} ms"
        except Exception as ex:
            lbfgs_ms = None
            row += f"  {'ERR':>14}"
            print(f"  [L-BFGS err: {ex}]")

        # Metal dense BFGS
        try:
            t, _ = bench_metal_bfgs(n)
            bfgs_ms = t / n * 1000
            row += f"  {bfgs_ms:>11.2f} ms"
        except Exception as ex:
            bfgs_ms = None
            row += f"  {'ERR':>14}"

        # RDKit 8-CPU
        try:
            t, _ = bench_rdkit_parallel(n)
            rdkit_ms = t / n * 1000
            row += f"  {rdkit_ms:>11.2f} ms"
        except Exception as ex:
            rdkit_ms = None
            row += f"  {'ERR':>14}"

        # speedup
        if lbfgs_ms and rdkit_ms:
            if lbfgs_ms < rdkit_ms:
                row += f"  {rdkit_ms/lbfgs_ms:>12.1f}x faster"
            else:
                row += f"  {lbfgs_ms/rdkit_ms:>11.1f}x slower"
        else:
            row += f"  {'N/A':>16}"

        print(row)

    print()
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
