"""Benchmark: mlxmolkit pipeline (stages 1-4) vs RDKit EmbedMultipleConfs.

Compares wall-clock time for conformer generation:
  - RDKit CPU: EmbedMultipleConfs (full ETKDG pipeline)
  - mlxmolkit MLX: stages 1-4 (coordgen + DG minimize + stereo checks + 4th dim minimize)

Note: mlxmolkit currently implements stages 1-4 only (no ETK torsion minimization,
no double bond checks, no retry loop). RDKit does the full ETKDG pipeline.
This benchmark measures the DG core, which is the computationally dominant part.
"""

import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.pipeline.context import create_pipeline_context
from mlxmolkit.pipeline.driver import run_dg_pipeline

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


def load_smiles_molecules():
    """Simple molecules for quick benchmarking."""
    smiles_list = [
        "CCO",           # ethanol
        "CCC",           # propane
        "CCCC",          # butane
        "c1ccccc1",      # benzene
        "CC(=O)O",       # acetic acid
        "CCN",           # ethylamine
        "CC(C)C",        # isobutane
        "C1CCCCC1",      # cyclohexane
        "CC(O)CC",       # 2-butanol
        "CCOC",          # diethyl ether
    ]
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            mols.append(mol)
    return mols


def benchmark_rdkit(mols, n_confs=1, n_repeats=3):
    """Benchmark RDKit EmbedMultipleConfs (full ETKDG)."""
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True

    # Warmup
    for mol in mols[:min(2, len(mols))]:
        m = Chem.RWMol(mol)
        rdDistGeom.EmbedMultipleConfs(m, n_confs, params)

    times = []
    for rep in range(n_repeats):
        t0 = time.perf_counter()
        n_success = 0
        for mol in mols:
            m = Chem.RWMol(mol)
            cids = rdDistGeom.EmbedMultipleConfs(m, n_confs, params)
            n_success += len(cids)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times), n_success


def benchmark_mlxmolkit(mols, n_repeats=3):
    """Benchmark mlxmolkit pipeline stages 1-4 (1 conformer per mol)."""
    # Warmup: create context once to JIT-compile MLX operations
    ctx = create_pipeline_context(mols[:min(2, len(mols))])
    run_dg_pipeline(ctx, seed=0)

    times = []
    for rep in range(n_repeats):
        t0 = time.perf_counter()
        ctx = create_pipeline_context(mols)
        t_preprocess = time.perf_counter() - t0

        t_pipeline_start = time.perf_counter()
        run_dg_pipeline(ctx, seed=rep)
        mx.eval(ctx.positions)
        t_pipeline = time.perf_counter() - t_pipeline_start

        t_total = time.perf_counter() - t0
        n_success = sum(ctx.active)
        times.append((t_total, t_preprocess, t_pipeline, n_success))

    # Pick the run with minimum total time
    best = min(times, key=lambda x: x[0])
    return best


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    print("=" * 72)
    print("  Pipeline Benchmark: mlxmolkit (MLX/Metal) vs RDKit (CPU)")
    print("=" * 72)
    print()
    print("  mlxmolkit: stages 1-4 (coordgen + DG minimize + stereo + 4th dim)")
    print("  RDKit:     full ETKDGv3 (all stages including ETK torsion)")
    print()

    # --- Simple molecules ---
    print("-" * 72)
    print("  Simple molecules (SMILES)")
    print("-" * 72)

    simple_mols = load_smiles_molecules()
    n_atoms_total = sum(m.GetNumAtoms() for m in simple_mols)
    print(f"  {len(simple_mols)} molecules, {n_atoms_total} total atoms")
    print()

    # RDKit: 1 conformer per molecule
    t_rdkit, n_rdkit = benchmark_rdkit(simple_mols, n_confs=1, n_repeats=5)
    print(f"  RDKit CPU (1 conf/mol):  {format_time(t_rdkit):>10s}  ({n_rdkit}/{len(simple_mols)} succeeded)")

    # mlxmolkit: 1 conformer per molecule
    t_mlx, t_pre, t_pipe, n_mlx = benchmark_mlxmolkit(simple_mols, n_repeats=5)
    print(f"  mlxmolkit MLX (1 conf):  {format_time(t_mlx):>10s}  ({n_mlx}/{len(simple_mols)} succeeded)")
    print(f"    preprocessing:         {format_time(t_pre):>10s}")
    print(f"    pipeline (GPU):        {format_time(t_pipe):>10s}")

    if t_mlx > 0:
        ratio = t_rdkit / t_mlx
        print(f"  Ratio (RDKit/mlxmolkit): {ratio:.2f}x {'(mlxmolkit faster)' if ratio > 1 else '(RDKit faster)'}")
    print()

    # --- SDF molecules (larger, more diverse) ---
    for n_mols in [5, 10, 20, 50]:
        print("-" * 72)
        sdf_mols = load_molecules(n_mols)
        actual_n = len(sdf_mols)
        if actual_n == 0:
            print(f"  No SDF molecules available, skipping")
            continue

        n_atoms = sum(m.GetNumAtoms() for m in sdf_mols)
        avg_atoms = n_atoms / actual_n
        print(f"  SDF: {actual_n} molecules, {n_atoms} total atoms (avg {avg_atoms:.0f}/mol)")
        print("-" * 72)

        # RDKit
        n_repeats = max(2, 10 // actual_n)
        t_rdkit, n_rdkit = benchmark_rdkit(sdf_mols, n_confs=1, n_repeats=n_repeats)
        print(f"  RDKit CPU (1 conf/mol):  {format_time(t_rdkit):>10s}  ({n_rdkit}/{actual_n} succeeded)")

        # mlxmolkit
        t_mlx, t_pre, t_pipe, n_mlx = benchmark_mlxmolkit(sdf_mols, n_repeats=n_repeats)
        print(f"  mlxmolkit MLX (1 conf):  {format_time(t_mlx):>10s}  ({n_mlx}/{actual_n} succeeded)")
        print(f"    preprocessing:         {format_time(t_pre):>10s}")
        print(f"    pipeline (GPU):        {format_time(t_pipe):>10s}")

        if t_mlx > 0:
            ratio = t_rdkit / t_mlx
            label = "(mlxmolkit faster)" if ratio > 1 else "(RDKit faster)"
            print(f"  Ratio (RDKit/mlxmolkit): {ratio:.2f}x {label}")

        # Per-molecule timing
        print(f"  Per molecule:")
        print(f"    RDKit:     {format_time(t_rdkit / actual_n)}/mol")
        print(f"    mlxmolkit: {format_time(t_mlx / actual_n)}/mol")
        print()

    # --- Multi-conformer comparison ---
    print("=" * 72)
    print("  Multi-conformer: 5 molecules x 5 conformers")
    print("=" * 72)
    mols_5 = load_molecules(5)
    if mols_5:
        n_atoms_5 = sum(m.GetNumAtoms() for m in mols_5)

        # RDKit: 5 conformers
        t_rdkit_5, n_rdkit_5 = benchmark_rdkit(mols_5, n_confs=5, n_repeats=5)
        print(f"  RDKit CPU (5 confs/mol): {format_time(t_rdkit_5):>10s}  ({n_rdkit_5}/25 conformers)")

        # mlxmolkit: 5 conformers = 5 pipeline runs (no retry loop yet)
        # Simulate by running 5x with different seeds
        t0 = time.perf_counter()
        total_success = 0
        for seed in range(5):
            ctx = create_pipeline_context(mols_5)
            run_dg_pipeline(ctx, seed=seed)
            mx.eval(ctx.positions)
            total_success += sum(ctx.active)
        t_mlx_5 = time.perf_counter() - t0
        print(f"  mlxmolkit MLX (5 runs):  {format_time(t_mlx_5):>10s}  ({total_success}/25 conformers)")

        if t_mlx_5 > 0:
            ratio = t_rdkit_5 / t_mlx_5
            label = "(mlxmolkit faster)" if ratio > 1 else "(RDKit faster)"
            print(f"  Ratio (RDKit/mlxmolkit): {ratio:.2f}x {label}")
    print()


if __name__ == "__main__":
    main()
