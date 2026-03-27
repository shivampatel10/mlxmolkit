"""Metal kernels for stereochemistry validation checks.

GPU-accelerated replacements for the CPU Python loops in
stage_stereochem_checks.py. One thread per term, writes 1.0 to
failed[mol_idx] on failure (idempotent — no atomics needed).

Uses mx.fast.metal_kernel() to launch MSL code.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from ..pipeline.context import BatchedTetrahedralData

# Load MSL source from external .metal file and split into sections
_KERNEL_DIR = Path(__file__).parent
_sections = (_KERNEL_DIR / "stereo_checks.metal").read_text().split(
    "// ---- STEREO_CHECKS_SPLIT ----\n"
)
_MSL_HEADER = _sections[0]
_MSL_TETRAHEDRAL = _sections[1]
_MSL_FIRST_CHIRAL = _sections[2]
_MSL_DOUBLE_BOND_GEOM = _sections[3]
_MSL_DOUBLE_BOND_STEREO = _sections[4]
_MSL_CHIRAL_DIST = _sections[5]
del _sections


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

MIN_TETRAHEDRAL_CHIRAL_VOL = 0.02


def metal_tetrahedral_check(
    pos: mx.array,
    tet_data: BatchedTetrahedralData,
    active: mx.array,
    n_mols: int,
    dim: int,
    tol: float = 0.3,
    do_volume_test: bool = True,
) -> mx.array:
    """GPU tetrahedral geometry check. Returns failed flags (float32, n_mols)."""
    n_terms = tet_data.idx0.shape[0]
    if n_terms == 0:
        return mx.zeros(n_mols, dtype=mx.float32)

    config = mx.array([
        float(n_terms), float(dim), tol,
        MIN_TETRAHEDRAL_CHIRAL_VOL,
        1.0 if do_volume_test else 0.0,
    ], dtype=mx.float32)

    in_fused = tet_data.in_fused_small_rings.astype(mx.float32)

    kernel = mx.fast.metal_kernel(
        name="stereo_tetrahedral_check",
        input_names=["pos", "idx0", "idx1", "idx2", "idx3", "idx4",
                     "in_fused", "mol_indices", "active", "config"],
        output_names=["failed"],
        header=_MSL_HEADER,
        source=_MSL_TETRAHEDRAL,
    )

    outputs = kernel(
        inputs=[pos, tet_data.idx0, tet_data.idx1, tet_data.idx2,
                tet_data.idx3, tet_data.idx4, in_fused,
                tet_data.mol_indices, active, config],
        output_shapes=[(n_mols,)],
        output_dtypes=[mx.float32],
        grid=(n_terms, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


def metal_first_chiral_check(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    vol_lower: mx.array,
    vol_upper: mx.array,
    mol_indices: mx.array,
    active: mx.array,
    n_mols: int,
    dim: int,
) -> mx.array:
    """GPU first chiral check. Returns failed flags (float32, n_mols)."""
    n_terms = idx1.shape[0]
    if n_terms == 0:
        return mx.zeros(n_mols, dtype=mx.float32)

    config = mx.array([float(n_terms), float(dim)], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="stereo_first_chiral_check",
        input_names=["pos", "idx1", "idx2", "idx3", "idx4",
                     "vol_lower", "vol_upper", "mol_indices", "active", "config"],
        output_names=["failed"],
        header=_MSL_HEADER,
        source=_MSL_FIRST_CHIRAL,
    )

    outputs = kernel(
        inputs=[pos, idx1, idx2, idx3, idx4,
                vol_lower, vol_upper, mol_indices, active, config],
        output_shapes=[(n_mols,)],
        output_dtypes=[mx.float32],
        grid=(n_terms, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


def metal_double_bond_geom_check(
    pos: mx.array,
    idx0: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    mol_indices: mx.array,
    active: mx.array,
    n_mols: int,
    dim: int,
    linear_tol: float = 1e-3,
) -> mx.array:
    """GPU double bond geometry check. Returns failed flags (float32, n_mols)."""
    n_terms = idx0.shape[0]
    if n_terms == 0:
        return mx.zeros(n_mols, dtype=mx.float32)

    config = mx.array([float(n_terms), float(dim), linear_tol], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="stereo_double_bond_geom",
        input_names=["pos", "idx0", "idx1", "idx2",
                     "mol_indices", "active", "config"],
        output_names=["failed"],
        header=_MSL_HEADER,
        source=_MSL_DOUBLE_BOND_GEOM,
    )

    outputs = kernel(
        inputs=[pos, idx0, idx1, idx2, mol_indices, active, config],
        output_shapes=[(n_mols,)],
        output_dtypes=[mx.float32],
        grid=(n_terms, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


def metal_double_bond_stereo_check(
    pos: mx.array,
    idx0: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    signs: mx.array,
    mol_indices: mx.array,
    active: mx.array,
    n_mols: int,
    dim: int,
) -> mx.array:
    """GPU double bond stereo (E/Z) check. Returns failed flags (float32, n_mols)."""
    n_terms = idx0.shape[0]
    if n_terms == 0:
        return mx.zeros(n_mols, dtype=mx.float32)

    config = mx.array([float(n_terms), float(dim)], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="stereo_double_bond_stereo",
        input_names=["pos", "idx0", "idx1", "idx2", "idx3",
                     "signs", "mol_indices", "active", "config"],
        output_names=["failed"],
        header=_MSL_HEADER,
        source=_MSL_DOUBLE_BOND_STEREO,
    )

    outputs = kernel(
        inputs=[pos, idx0, idx1, idx2, idx3,
                signs.astype(mx.float32), mol_indices, active, config],
        output_shapes=[(n_mols,)],
        output_dtypes=[mx.float32],
        grid=(n_terms, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return outputs[0]


def metal_chiral_dist_check(
    pos: mx.array,
    idx0: mx.array,
    idx1: mx.array,
    lower: mx.array,
    upper: mx.array,
    mol_indices: mx.array,
    active: mx.array,
    n_mols: int,
    dim: int,
) -> mx.array:
    """GPU chiral distance matrix check. Returns failed flags (float32, n_mols)."""
    n_terms = idx0.shape[0]
    if n_terms == 0:
        return mx.zeros(n_mols, dtype=mx.float32)

    config = mx.array([float(n_terms), float(dim)], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="stereo_chiral_dist",
        input_names=["pos", "idx0", "idx1", "lower", "upper",
                     "mol_indices", "active", "config"],
        output_names=["failed"],
        header=_MSL_HEADER,
        source=_MSL_CHIRAL_DIST,
    )

    outputs = kernel(
        inputs=[pos, idx0, idx1,
                lower.astype(mx.float32), upper.astype(mx.float32),
                mol_indices, active, config],
        output_shapes=[(n_mols,)],
        output_dtypes=[mx.float32],
        grid=(n_terms, 1, 1),
        threadgroup=(1, 1, 1),
        init_value=0.0,
    )
    return outputs[0]
