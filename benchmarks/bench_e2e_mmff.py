"""End-to-end benchmark: Conformer generation + MMFF optimization.

Compares:
  - RDKit CPU (8 cores, multiprocessing): EmbedMultipleConfs + MMFFOptimizeMoleculeConfs
  - mlxmolkit GPU: EmbedMolecules + metal_mmff_bfgs_tg

Dataset: All valid molecules from MMFF94_dative.sdf (~761 mols)
Conformers: 20 per molecule
"""

import argparse
import dataclasses
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
def _stats_value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _find_embed_stats(embed_module, embed_fn):
    for owner in (embed_fn, embed_module):
        for name in ("get_last_embed_stats", "get_embed_stats"):
            candidate = getattr(owner, name, None)
            if callable(candidate):
                return candidate()
        for name in ("last_embed_stats", "LAST_EMBED_STATS", "embed_stats", "EMBED_STATS"):
            if hasattr(owner, name):
                return getattr(owner, name)
    return None


def _print_embed_stats(stats):
    print("Embed retry stats:", flush=True)
    if stats is None:
        print("  unavailable: public embed stats API was not found", flush=True)
        return

    if dataclasses.is_dataclass(stats):
        stats = dataclasses.asdict(stats)

    passes = _stats_value(stats, "passes")
    if not passes:
        print(f"  {stats}", flush=True)
        return

    total_attempts = sum(_stats_value(p, "attempts", 0) for p in passes)
    total_successes = sum(_stats_value(p, "successes", 0) for p in passes)
    total_written = sum(_stats_value(p, "conformers_written", 0) for p in passes)
    print(f"  passes:             {len(passes):,}", flush=True)
    print(f"  attempts:           {total_attempts:,}", flush=True)
    print(f"  successful attempts:{total_successes:>10,}", flush=True)
    print(f"  conformers written: {total_written:,}", flush=True)
    print("  per pass:", flush=True)
    print("    pass  attempts  mols  successes  written  context_s  pipeline_s  writeback_s", flush=True)
    for i, pass_stats in enumerate(passes, start=1):
        pass_index = _stats_value(pass_stats, "pass_index", i)
        print(
            f"    {pass_index:>4}  "
            f"{_stats_value(pass_stats, 'attempts', 0):>8,}  "
            f"{_stats_value(pass_stats, 'unique_mols', 0):>4,}  "
            f"{_stats_value(pass_stats, 'successes', 0):>9,}  "
            f"{_stats_value(pass_stats, 'conformers_written', 0):>7,}  "
            f"{_stats_value(pass_stats, 'context_seconds', 0.0):>9.2f}  "
            f"{_stats_value(pass_stats, 'pipeline_seconds', 0.0):>10.2f}  "
            f"{_stats_value(pass_stats, 'writeback_seconds', 0.0):>10.2f}",
            flush=True,
        )


def bench_mlx_gpu(mols, confs_per_mol, print_embed_stats=False):
    import mlxmolkit.embed_molecules as embed_module
    from mlxmolkit.mmff_optimize import MMFFOptimizeMoleculesConfs

    mlx_mols = [Chem.Mol(m) for m in mols]

    t0 = time.perf_counter()

    # Phase 1: Conformer generation (GPU, auto-chunked)
    embed_params = rdDistGeom.ETKDGv3()
    embed_params.useRandomCoords = True
    embed_params.randomSeed = 42
    embed_module.EmbedMolecules(mlx_mols, embed_params, confsPerMolecule=confs_per_mol)
    t_embed = time.perf_counter() - t0
    if print_embed_stats:
        _print_embed_stats(_find_embed_stats(embed_module, embed_module.EmbedMolecules))

    # Phase 2+3: MMFF optimization (GPU, auto-chunked)
    t1 = time.perf_counter()
    MMFFOptimizeMoleculesConfs(mlx_mols, maxIters=200)
    t_mmff = time.perf_counter() - t1

    total_elapsed = time.perf_counter() - t0
    total_confs = sum(mol.GetNumConformers() for mol in mlx_mols)
    total_atoms = sum(mol.GetNumConformers() * mol.GetNumAtoms() for mol in mlx_mols
                      if mol.GetNumConformers() > 0)
    return total_elapsed, t_embed, 0.0, t_mmff, total_confs, total_atoms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-mols",
        type=int,
        default=100,
        help="number of valid molecules to benchmark",
    )
    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="run only the mlxmolkit GPU path and skip RDKit CPU",
    )
    parser.add_argument(
        "--embed-stats",
        action="store_true",
        help="print embed retry stats when exposed by the public API",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mols = load_molecules()[: args.num_mols]
    n_mols = len(mols)

    print("=" * 78)
    print("End-to-End Benchmark: Conformer Generation + MMFF94 Optimization")
    print("=" * 78)
    print(f"  Dataset:         MMFF94_dative.sdf")
    print(f"  Molecules:       {n_mols}")
    print(f"  Confs/molecule:  {CONFS_PER_MOL}")
    print(f"  Expected confs:  {n_mols * CONFS_PER_MOL:,}")
    if args.mlx_only:
        print("  RDKit workers:   skipped (--mlx-only)")
    else:
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
    mlx_total, mlx_embed, mlx_extract, mlx_mmff, mlx_confs, mlx_atoms = bench_mlx_gpu(
        mols,
        confs_per_mol=CONFS_PER_MOL,
        print_embed_stats=args.embed_stats,
    )
    print(f"  Done in {mlx_total:.1f}s\n", flush=True)

    # --- RDKit CPU ---
    if not args.mlx_only:
        print(f"Running RDKit CPU ({N_CPU} cores)...", flush=True)
        rdk_total, rdk_confs, rdk_atoms = bench_rdkit_cpu(mols)
        print(f"  Done in {rdk_total:.1f}s\n", flush=True)

    # --- Results ---
    print("=" * 78)
    if args.mlx_only:
        print(f"{'':>30} {'mlxmolkit GPU':>16}")
        print("-" * 78)
        print(f"{'Conformers generated':>30} {mlx_confs:>16,}")
        print(f"{'Total atoms optimized':>30} {mlx_atoms:>16,}")
        print()
        print(f"{'Embed time (s)':>30} {mlx_embed:>16.2f}")
        print(f"{'Param extraction (s)':>30} {mlx_extract:>16.2f}")
        print(f"{'MMFF optimization (s)':>30} {mlx_mmff:>16.2f}")
        print(f"{'Total mlx time (s)':>30} {mlx_total:>16.2f}")
    else:
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
    confs_sec_mlx = mlx_confs / mlx_total if mlx_total > 0 else 0
    if args.mlx_only:
        print(f"{'Throughput (confs/sec)':>30} {confs_sec_mlx:>16,.0f}")
    else:
        confs_sec_rdk = rdk_confs / rdk_total if rdk_total > 0 else 0
        print(f"{'Throughput (confs/sec)':>30} {confs_sec_rdk:>16,.0f} {confs_sec_mlx:>16,.0f}")
        if mlx_total > 0 and rdk_total > 0:
            print(f"{'Speedup':>30} {'1.0x':>16} {rdk_total / mlx_total:>15.1f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
