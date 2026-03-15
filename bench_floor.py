#!/usr/bin/env python3
"""Find the GPU floor: sweep batch sizes up to 100K."""

import multiprocessing as mp
import time
from functools import partial

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params, get_bounds_matrix
from mlxmolkit.metal_kernels.dg_lbfgs import metal_dg_lbfgs

SMILES = [
    "CCO", "CCC", "c1ccccc1", "CC(=O)O", "CC(C)O",
    "c1ccc(cc1)O", "CC(=O)NC", "C1CCCCC1", "CC(C)(C)O", "c1ccncc1",
    "c1ccc2ccccc2c1", "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(cc1)CC(=O)O", "OC(=O)c1ccccc1O",
]
MAX_ITERS = 200
N_WORKERS = 8


def _rdkit_worker(mol_block):
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return 0
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useRandomCoords = True
    params.maxIterations = MAX_ITERS
    params.numThreads = 0
    cid = AllChem.EmbedMolecule(mol, params)
    return 1 if cid >= 0 else 0


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


def main():
    # Warmup GPU
    system, pos = _make_batch(100)
    metal_dg_lbfgs(pos, system, max_iters=5)
    mx.eval(pos)

    # Warmup RDKit pool
    mols_w = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in SMILES[:N_WORKERS]]
    blocks_w = [Chem.MolToMolBlock(m) for m in mols_w]
    with mp.Pool(N_WORKERS) as pool:
        list(pool.map(_rdkit_worker, blocks_w))

    batch_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    print()
    print("=" * 100)
    print("  Finding the floor: Metal L-BFGS GPU vs RDKit 8-CPU")
    print(f"  max_iters={MAX_ITERS}   workers={N_WORKERS}")
    print("=" * 100)
    print()
    hdr = f"{'N':>8}  {'L-BFGS GPU':>14}  {'RDKit 8-CPU':>14}  {'Speedup':>10}  {'GPU wall':>10}  {'CPU wall':>10}"
    print(hdr)
    sub = f"{'':>8}  {'(ms/mol)':>14}  {'(ms/mol)':>14}  {'':>10}  {'(sec)':>10}  {'(sec)':>10}"
    print(sub)
    print("-" * 100)

    rdkit_ms_10k = None

    for n in batch_sizes:
        row = f"{n:>8}"

        # ---- Metal L-BFGS ----
        try:
            system, pos = _make_batch(n)
            start = time.perf_counter()
            fp, fe, fs = metal_dg_lbfgs(pos, system, max_iters=MAX_ITERS)
            mx.eval(fp, fe, fs)
            t_gpu = time.perf_counter() - start
            lbfgs_ms = t_gpu / n * 1000
            row += f"  {lbfgs_ms:>11.3f} ms"
        except Exception as ex:
            lbfgs_ms = None
            t_gpu = None
            row += f"  {'ERR':>14}"
            print(f"  [GPU err at N={n}: {ex}]")
            row += "  --  --  --  --"
            print(row)
            continue

        # ---- RDKit 8-CPU ----
        if n <= 10000:
            try:
                mols = [Chem.AddHs(Chem.MolFromSmiles(SMILES[i % len(SMILES)])) for i in range(n)]
                mol_blocks = [Chem.MolToMolBlock(m) for m in mols]
                start = time.perf_counter()
                with mp.Pool(N_WORKERS) as pool:
                    results = list(pool.map(_rdkit_worker, mol_blocks))
                t_cpu = time.perf_counter() - start
                rdkit_ms = t_cpu / n * 1000
                row += f"  {rdkit_ms:>11.3f} ms"
            except Exception:
                rdkit_ms = None
                t_cpu = None
                row += f"  {'ERR':>14}"
        else:
            # Extrapolate from N=10000
            rdkit_ms = rdkit_ms_10k
            t_cpu = rdkit_ms * n / 1000 if rdkit_ms else None
            if rdkit_ms:
                row += f"  {rdkit_ms:>10.3f}ms*"
            else:
                row += f"  {'N/A':>14}"

        if n == 10000 and rdkit_ms:
            rdkit_ms_10k = rdkit_ms

        # Speedup
        if lbfgs_ms and rdkit_ms:
            row += f"  {rdkit_ms / lbfgs_ms:>7.1f}x"
        else:
            row += f"  {'N/A':>10}"

        # Wall clock
        if t_gpu is not None:
            row += f"  {t_gpu:>7.2f} s"
        else:
            row += f"  {'--':>10}"
        if t_cpu is not None:
            row += f"  {t_cpu:>7.2f} s"
        else:
            row += f"  {'--':>10}"

        print(row)

    print()
    print("* = extrapolated from N=10000 measurement")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
