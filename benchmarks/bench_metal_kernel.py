"""Prototype: Metal kernel BFGS vs Python BFGS for DG minimization.

Validates whether a single Metal kernel dispatch running 600 iterations
can achieve competitive per-molecule latency vs RDKit (~7ms/mol).

Approach:
  - Metal kernel: one thread per molecule, full gradient descent loop
    with DG energy (distance violations + 4th dim penalty) inside a
    single mx.fast.metal_kernel call
  - Python/MLX: our current bfgs_minimize for comparison
  - RDKit: EmbedMultipleConfs for reference

The Metal kernel uses gradient descent (not full BFGS) to keep the
prototype simple. Full BFGS adds Hessian tracking but similar loop
structure.
"""

import time

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.forcefields.dist_geom import dg_energy_and_grad
from mlxmolkit.minimizer.bfgs import bfgs_minimize
from mlxmolkit.pipeline.context import create_pipeline_context
from mlxmolkit.pipeline.driver import run_dg_pipeline
from mlxmolkit.preprocessing.batching import batch_dg_params
from mlxmolkit.preprocessing.rdkit_extract import extract_dg_params

# ---------------------------------------------------------------------------
# Metal kernel: gradient descent on DG energy, 600 iters, one thread per mol
# ---------------------------------------------------------------------------

# We pack distance terms as [idx1, idx2, lb2, ub2, weight] (5 floats each)
# and pass int data as float (exact for small ints).
METAL_DG_MINIMIZE_SOURCE = """
    uint mol_idx = thread_position_in_grid.x;

    // Unpack per-molecule info: [atom_start, atom_end, term_start, term_end]
    int atom_start = (int)mol_info[mol_idx * 4 + 0];
    int atom_end   = (int)mol_info[mol_idx * 4 + 1];
    int term_start = (int)mol_info[mol_idx * 4 + 2];
    int term_end   = (int)mol_info[mol_idx * 4 + 3];

    int n_atoms = atom_end - atom_start;
    int n_terms = term_end - term_start;
    int n_vars = n_atoms * 4;  // dim=4

    // Copy positions to thread-local array (max 128 atoms * 4 = 512 floats)
    float pos[512];
    float grad[512];
    for (int i = 0; i < n_vars; i++) {
        pos[i] = positions[atom_start * 4 + i];
    }

    float energy = 0.0f;
    float fourth_dim_weight = 0.1f;
    int max_iters = 600;

    for (int iter = 0; iter < max_iters; iter++) {
        // Zero gradient and energy
        energy = 0.0f;
        for (int i = 0; i < n_vars; i++) grad[i] = 0.0f;

        // Distance violation energy + gradient
        for (int t = term_start; t < term_end; t++) {
            int i1 = ((int)terms[t * 5 + 0] - atom_start);  // local index
            int i2 = ((int)terms[t * 5 + 1] - atom_start);
            float lb2 = terms[t * 5 + 2];
            float ub2 = terms[t * 5 + 3];
            float w   = terms[t * 5 + 4];

            float diff[4];
            float d2 = 0.0f;
            for (int d = 0; d < 4; d++) {
                diff[d] = pos[i1 * 4 + d] - pos[i2 * 4 + d];
                d2 += diff[d] * diff[d];
            }

            float pf = 0.0f;

            // Upper bound violation
            if (d2 > ub2) {
                float val = d2 / ub2 - 1.0f;
                energy += w * val * val;
                pf += 4.0f * w * val / ub2;
            }

            // Lower bound violation
            if (d2 < lb2) {
                float l2d2 = d2 + lb2;
                float val = 2.0f * lb2 / l2d2 - 1.0f;
                energy += w * val * val;
                pf += w * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
            }

            if (pf != 0.0f) {
                for (int d = 0; d < 4; d++) {
                    grad[i1 * 4 + d] += pf * diff[d];
                    grad[i2 * 4 + d] -= pf * diff[d];
                }
            }
        }

        // Fourth dimension penalty: E = w * pos_w^2, dE/dw = 2*w*pos_w
        for (int i = 0; i < n_atoms; i++) {
            float w_coord = pos[i * 4 + 3];
            energy += fourth_dim_weight * w_coord * w_coord;
            grad[i * 4 + 3] += 2.0f * fourth_dim_weight * w_coord;
        }

        // Convergence check
        float max_grad = 0.0f;
        for (int i = 0; i < n_vars; i++) {
            float ag = metal::abs(grad[i]);
            if (ag > max_grad) max_grad = ag;
        }
        if (max_grad < 1e-4f) break;

        // --- Backtracking line search ---
        float step = 1.0f;
        float slope = 0.0f;
        for (int i = 0; i < n_vars; i++) {
            slope -= grad[i] * grad[i];  // dir = -grad, slope = dir.grad
        }

        for (int ls = 0; ls < 20; ls++) {
            float trial_energy = 0.0f;
            // Compute energy at trial point
            for (int t = term_start; t < term_end; t++) {
                int i1 = ((int)terms[t * 5 + 0] - atom_start);
                int i2 = ((int)terms[t * 5 + 1] - atom_start);
                float ub2 = terms[t * 5 + 3];
                float lb2 = terms[t * 5 + 2];
                float w   = terms[t * 5 + 4];

                float d2 = 0.0f;
                for (int d = 0; d < 4; d++) {
                    float p1 = pos[i1 * 4 + d] - step * grad[i1 * 4 + d];
                    float p2 = pos[i2 * 4 + d] - step * grad[i2 * 4 + d];
                    float df = p1 - p2;
                    d2 += df * df;
                }
                if (d2 > ub2) {
                    float val = d2 / ub2 - 1.0f;
                    trial_energy += w * val * val;
                }
                if (d2 < lb2) {
                    float l2d2 = d2 + lb2;
                    float val = 2.0f * lb2 / l2d2 - 1.0f;
                    trial_energy += w * val * val;
                }
            }
            for (int i = 0; i < n_atoms; i++) {
                float w_coord = pos[i * 4 + 3] - step * grad[i * 4 + 3];
                trial_energy += fourth_dim_weight * w_coord * w_coord;
            }

            // Armijo condition
            if (trial_energy <= energy + 1e-4f * step * slope) {
                break;
            }
            step *= 0.5f;
        }

        // Update positions
        for (int i = 0; i < n_vars; i++) {
            pos[i] -= step * grad[i];
        }
    }

    // Write output
    for (int i = 0; i < n_vars; i++) {
        out_positions[atom_start * 4 + i] = pos[i];
    }
    out_energies[mol_idx] = energy;
"""


def create_metal_kernel():
    """Create the Metal DG minimize kernel."""
    return mx.fast.metal_kernel(
        name="dg_minimize_gd",
        input_names=["positions", "terms", "mol_info"],
        output_names=["out_positions", "out_energies"],
        source=METAL_DG_MINIMIZE_SOURCE,
    )


def prepare_metal_inputs(mols):
    """Prepare packed arrays for the Metal kernel."""
    dim = 4
    dg_params_list = [extract_dg_params(mol, dim=dim) for mol in mols]
    system = batch_dg_params(dg_params_list, dim)

    atom_starts = np.array(system.atom_starts.tolist(), dtype=np.int32)
    n_atoms_total = system.n_atoms_total

    # Pack distance terms: [idx1, idx2, lb2, ub2, weight] per term
    idx1 = np.array(system.dist_idx1)
    idx2 = np.array(system.dist_idx2)
    lb2 = np.array(system.dist_lb2)
    ub2 = np.array(system.dist_ub2)
    weight = np.array(system.dist_weight)
    n_terms = len(idx1)

    terms = np.zeros((n_terms, 5), dtype=np.float32)
    terms[:, 0] = idx1.astype(np.float32)
    terms[:, 1] = idx2.astype(np.float32)
    terms[:, 2] = lb2
    terms[:, 3] = ub2
    terms[:, 4] = weight

    # Build term_starts from dist_term_starts
    term_starts = np.array(system.dist_term_starts.tolist(), dtype=np.int32)

    # Mol info: [atom_start, atom_end, term_start, term_end] per mol
    n_mols = len(mols)
    mol_info = np.zeros((n_mols, 4), dtype=np.float32)
    for i in range(n_mols):
        mol_info[i, 0] = atom_starts[i]
        mol_info[i, 1] = atom_starts[i + 1]
        mol_info[i, 2] = term_starts[i]
        mol_info[i, 3] = term_starts[i + 1]

    # Random initial positions
    np.random.seed(42)
    positions = (np.random.random(n_atoms_total * dim).astype(np.float32) - 0.5) * 10.0

    return {
        "positions": mx.array(positions),
        "terms": mx.array(terms.reshape(-1)),
        "mol_info": mx.array(mol_info.reshape(-1)),
        "n_atoms_total": n_atoms_total,
        "n_mols": n_mols,
        "system": system,
    }


def run_metal_kernel(kernel, data, n_mols):
    """Run the Metal kernel and return outputs."""
    outputs = kernel(
        inputs=[data["positions"], data["terms"], data["mol_info"]],
        template=[("T", mx.float32)],
        grid=(n_mols, 1, 1),
        threadgroup=(min(n_mols, 256), 1, 1),
        output_shapes=[(data["n_atoms_total"] * 4,), (n_mols,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs


def benchmark_metal(kernel, data, n_mols, n_repeats=20):
    """Benchmark Metal kernel with warmup."""
    # Warmup
    for _ in range(3):
        out = run_metal_kernel(kernel, data, n_mols)
        mx.eval(out[0], out[1])

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = run_metal_kernel(kernel, data, n_mols)
        mx.eval(out[0], out[1])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times), out


def benchmark_python_bfgs(data, n_repeats=3):
    """Benchmark our Python BFGS for comparison."""
    system = data["system"]

    def eg(pos):
        return dg_energy_and_grad(pos, system, 1.0, 0.1)

    atom_starts = [int(x) for x in np.array(system.atom_starts)]

    # Warmup
    pos, e, s = bfgs_minimize(
        eg, data["positions"], atom_starts, data["n_mols"], 4,
        max_iters=400, scale_grads=False,
    )
    mx.eval(pos, e)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        pos, e, s = bfgs_minimize(
            eg, data["positions"], atom_starts, data["n_mols"], 4,
            max_iters=400, scale_grads=False,
        )
        mx.eval(pos, e)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times), e


def benchmark_python_pipeline(mols, n_repeats=3):
    """Benchmark our full Python pipeline (stages 1-4)."""
    # Warmup
    ctx = create_pipeline_context(mols)
    run_dg_pipeline(ctx, seed=0)

    times = []
    for rep in range(n_repeats):
        t0 = time.perf_counter()
        ctx = create_pipeline_context(mols)
        run_dg_pipeline(ctx, seed=rep)
        mx.eval(ctx.positions)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


def benchmark_rdkit(mols, n_confs=1, n_repeats=5):
    """Benchmark RDKit EmbedMultipleConfs."""
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True

    # Warmup
    for mol in mols[:2]:
        m = Chem.RWMol(mol)
        rdDistGeom.EmbedMultipleConfs(m, n_confs, params)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for mol in mols:
            m = Chem.RWMol(mol)
            rdDistGeom.EmbedMultipleConfs(m, n_confs, params)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} us"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    print("=" * 72)
    print("  Metal Kernel Prototype: DG Minimize Performance Validation")
    print("=" * 72)
    print()
    print("  Metal kernel: gradient descent + line search, 600 iters, 1 thread/mol")
    print("  Python BFGS:  our current bfgs_minimize (400 iters, no grad scaling)")
    print("  Python pipeline: full stages 1-4 (coordgen + 2x minimize + stereo)")
    print("  RDKit: full ETKDGv3")
    print()

    # Create Metal kernel once
    kernel = create_metal_kernel()

    # --- Test with simple molecules ---
    smiles_list = ["CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O",
                   "CCN", "CC(C)C", "C1CCCCC1", "CC(O)CC", "CCOC"]
    simple_mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            simple_mols.append(Chem.AddHs(mol))

    for n_mols in [1, 10, 50, 100]:
        mols = (simple_mols * ((n_mols // len(simple_mols)) + 1))[:n_mols]
        n_atoms = sum(m.GetNumAtoms() for m in mols)
        data = prepare_metal_inputs(mols)

        print("-" * 72)
        print(f"  {n_mols} molecules, {n_atoms} total atoms "
              f"(avg {n_atoms/n_mols:.0f} atoms/mol)")
        print("-" * 72)

        # Metal kernel
        t_metal, (out_pos, out_energy) = benchmark_metal(
            kernel, data, n_mols, n_repeats=50
        )
        mx.eval(out_energy)
        metal_energies = np.array(out_energy)
        avg_e = np.mean(metal_energies)
        print(f"  Metal kernel (GD, 600it): {format_time(t_metal):>10s}  "
              f"(avg E={avg_e:.4f}, {format_time(t_metal/n_mols)}/mol)")

        # Python BFGS
        n_bfgs_repeats = 3 if n_mols <= 10 else 1
        t_bfgs, bfgs_energies = benchmark_python_bfgs(data, n_repeats=n_bfgs_repeats)
        mx.eval(bfgs_energies)
        avg_e_bfgs = np.mean(np.array(bfgs_energies))
        print(f"  Python BFGS (400 it):     {format_time(t_bfgs):>10s}  "
              f"(avg E={avg_e_bfgs:.6f}, {format_time(t_bfgs/n_mols)}/mol)")

        # Python pipeline (only for small counts)
        if n_mols <= 10:
            t_pipe = benchmark_python_pipeline(mols, n_repeats=2)
            print(f"  Python pipeline (stg1-4): {format_time(t_pipe):>10s}  "
                  f"({format_time(t_pipe/n_mols)}/mol)")

        # RDKit
        t_rdkit = benchmark_rdkit(mols, n_confs=1, n_repeats=5)
        print(f"  RDKit ETKDGv3 (CPU):      {format_time(t_rdkit):>10s}  "
              f"({format_time(t_rdkit/n_mols)}/mol)")

        # Ratios
        print()
        print(f"  Metal vs RDKit:  {t_rdkit/t_metal:.1f}x "
              f"{'(Metal faster)' if t_metal < t_rdkit else '(RDKit faster)'}")
        print(f"  Metal vs PyBFGS: {t_bfgs/t_metal:.0f}x speedup")
        print()

    # --- Scaling test ---
    print("=" * 72)
    print("  Scaling: Metal kernel with increasing molecule count")
    print("=" * 72)
    for n_mols in [1, 10, 50, 100, 200, 500]:
        mols = (simple_mols * ((n_mols // len(simple_mols)) + 1))[:n_mols]
        data = prepare_metal_inputs(mols)
        t, _ = benchmark_metal(kernel, data, n_mols, n_repeats=50)
        print(f"  {n_mols:>4d} mols: {format_time(t):>10s} total, "
              f"{format_time(t/n_mols):>10s}/mol")


if __name__ == "__main__":
    main()
