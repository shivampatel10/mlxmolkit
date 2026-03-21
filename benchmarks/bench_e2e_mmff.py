"""End-to-end benchmark: Conformer generation + MMFF optimization.

Compares:
  - RDKit CPU (8 cores, multiprocessing): EmbedMultipleConfs + MMFFOptimizeMoleculeConfs
  - mlxmolkit GPU: EmbedMolecules + metal_mmff_bfgs_tg

Dataset: All valid molecules from MMFF94_dative.sdf (~761 mols)
Conformers: 20 per molecule
"""

import multiprocessing as mp
import os
import time

import mlx.core as mx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

# Suppress noisy RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

SDF_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data", "MMFF94_dative.sdf")
CONFS_PER_MOL = 20
N_CPU = 8


def load_molecules():
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
    mols = []
    for mol in suppl:
        if mol is None:
            continue
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if props is None:
            continue
        mols.append(mol)
    return mols


# ---------------------------------------------------------------------------
# RDKit CPU (8-core multiprocessing)
# ---------------------------------------------------------------------------
def _rdkit_worker(mol_bytes):
    RDLogger.logger().setLevel(RDLogger.ERROR)
    mol = Chem.Mol(mol_bytes)
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    params.numThreads = 1
    AllChem.EmbedMultipleConfs(mol, numConfs=CONFS_PER_MOL, params=params)
    if mol.GetNumConformers() > 0:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=1)
    return mol.GetNumConformers(), mol.GetNumAtoms()


def bench_rdkit_cpu(mols):
    mol_bytes_list = [mol.ToBinary() for mol in mols]
    t0 = time.perf_counter()
    with mp.Pool(N_CPU) as pool:
        results = pool.map(_rdkit_worker, mol_bytes_list)
    elapsed = time.perf_counter() - t0
    total_confs = sum(r[0] for r in results)
    total_atoms = sum(r[0] * r[1] for r in results)
    return elapsed, total_confs, total_atoms


# ---------------------------------------------------------------------------
# mlxmolkit GPU
# ---------------------------------------------------------------------------
def bench_mlx_gpu(mols):
    from mlxmolkit.embed_molecules import EmbedMolecules
    from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs_tg
    from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
    from mlxmolkit.preprocessing.mmff_extract import extract_mmff_params

    mlx_mols = [Chem.Mol(m) for m in mols]

    t0 = time.perf_counter()

    # Phase 1: Conformer generation (GPU)
    embed_params = rdDistGeom.ETKDGv3()
    embed_params.useRandomCoords = True
    embed_params.randomSeed = 42
    EmbedMolecules(mlx_mols, embed_params, confsPerMolecule=CONFS_PER_MOL)
    t_embed = time.perf_counter() - t0

    # Phase 2: MMFF param extraction + batching
    t1 = time.perf_counter()
    all_params, all_positions, conf_counts = [], [], []
    for mol in mlx_mols:
        n_confs = mol.GetNumConformers()
        conf_counts.append(n_confs)
        if n_confs == 0:
            continue
        params = extract_mmff_params(mol)
        if params is None:
            continue
        n_atoms = mol.GetNumAtoms()
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
    t_extract = time.perf_counter() - t1

    # Phase 3: MMFF optimization (TG Metal kernel)
    t2 = time.perf_counter()
    if all_params:
        system = batch_mmff_params(all_params)
        pos = mx.array(np.concatenate(all_positions), dtype=mx.float32)
        final_pos, final_e, statuses = metal_mmff_bfgs_tg(pos, system, max_iters=200)
        mx.eval(final_pos, final_e, statuses)
    t_mmff = time.perf_counter() - t2

    total_elapsed = time.perf_counter() - t0
    total_confs = sum(conf_counts)
    total_atoms = sum(c * mol.GetNumAtoms() for c, mol in zip(conf_counts, mlx_mols) if c > 0)
    return total_elapsed, t_embed, t_extract, t_mmff, total_confs, total_atoms


def main():
    mols = load_molecules()
    n_mols = len(mols)

    print("=" * 78)
    print("End-to-End Benchmark: Conformer Generation + MMFF94 Optimization")
    print("=" * 78)
    print(f"  Dataset:         MMFF94_dative.sdf")
    print(f"  Molecules:       {n_mols}")
    print(f"  Confs/molecule:  {CONFS_PER_MOL}")
    print(f"  Expected confs:  {n_mols * CONFS_PER_MOL:,}")
    print(f"  RDKit workers:   {N_CPU}")
    print()

    # --- Warmup Metal with a tiny batch (no embed, just kernel compile) ---
    print("Warming up Metal kernels...", flush=True)
    from mlxmolkit.metal_kernels.mmff_bfgs import metal_mmff_bfgs_tg
    from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
    from mlxmolkit.preprocessing.mmff_extract import extract_mmff_params
    warmup_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(warmup_mol, randomSeed=42)
    wp = extract_mmff_params(warmup_mol)
    ws = batch_mmff_params([wp])
    wpos = mx.array(np.array([list(warmup_mol.GetConformer().GetAtomPosition(i))
                               for i in range(warmup_mol.GetNumAtoms())],
                              dtype=np.float32).flatten())
    for _ in range(3):
        r = metal_mmff_bfgs_tg(wpos, ws, max_iters=50)
        mx.eval(r)
    print("Warmup done.\n", flush=True)

    # --- mlxmolkit GPU ---
    print("Running mlxmolkit GPU...", flush=True)
    mlx_total, mlx_embed, mlx_extract, mlx_mmff, mlx_confs, mlx_atoms = bench_mlx_gpu(mols)
    print(f"  Done in {mlx_total:.1f}s\n", flush=True)

    # --- RDKit CPU ---
    print(f"Running RDKit CPU ({N_CPU} cores)...", flush=True)
    rdk_total, rdk_confs, rdk_atoms = bench_rdkit_cpu(mols)
    print(f"  Done in {rdk_total:.1f}s\n", flush=True)

    # --- Results ---
    print("=" * 78)
    print(f"{'':>30} {'RDKit CPU':>16} {'mlxmolkit GPU':>16}")
    print("-" * 78)
    print(f"{'Conformers generated':>30} {rdk_confs:>16,} {mlx_confs:>16,}")
    print(f"{'Total atoms optimized':>30} {rdk_atoms:>16,} {mlx_atoms:>16,}")
    print()
    print(f"{'Embed time (s)':>30} {'(included)':>16} {mlx_embed:>16.2f}")
    print(f"{'Param extraction (s)':>30} {'(included)':>16} {mlx_extract:>16.2f}")
    print(f"{'MMFF optimization (s)':>30} {'(included)':>16} {mlx_mmff:>16.2f}")
    print(f"{'Total wall time (s)':>30} {rdk_total:>16.2f} {mlx_total:>16.2f}")
    print()
    confs_sec_rdk = rdk_confs / rdk_total if rdk_total > 0 else 0
    confs_sec_mlx = mlx_confs / mlx_total if mlx_total > 0 else 0
    print(f"{'Throughput (confs/sec)':>30} {confs_sec_rdk:>16,.0f} {confs_sec_mlx:>16,.0f}")
    if mlx_total > 0 and rdk_total > 0:
        print(f"{'Speedup':>30} {'1.0x':>16} {rdk_total / mlx_total:>15.1f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
