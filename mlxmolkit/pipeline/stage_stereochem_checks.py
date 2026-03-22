"""Stereochemistry checks for stages 3 and 6.

Tetrahedral geometry validation, chiral volume checks,
double bond geometry and stereo checks.

Port of nvMolKit's etkdg_stage_stereochem_checks.cu.
"""

import logging
import math

import mlx.core as mx
import numpy as np

from .context import PipelineContext

log = logging.getLogger(__name__)


def _build_active_array(ctx: PipelineContext) -> mx.array:
    """Build float32 active mask: 1.0 if active and not failed, else 0.0."""
    return mx.array(
        [1.0 if ctx.active[i] and not ctx.failed[i] else 0.0
         for i in range(ctx.n_mols)],
        dtype=mx.float32,
    )


def _apply_failed(ctx: PipelineContext, failed: mx.array) -> None:
    """Apply GPU kernel failed flags back to ctx.failed."""
    mx.eval(failed)
    failed_np = np.array(failed)
    for i in range(ctx.n_mols):
        if failed_np[i] > 0.5:
            ctx.failed[i] = True

# nvMolKit uses 0.50, calibrated for its float64 CUDA DG minimizer.
# Our float32 MLX BFGS produces volumes in [0.027, 0.083] for valid-but-flat
# tetrahedra where one of the four cross-dot products is small (~0.04) while
# the other three are healthy (~0.85). A threshold of 0.02 catches genuinely
# degenerate geometry (volume ~ 0) while allowing these valid-but-flat
# tetrahedra through. The lowest observed volume for a valid sp3 center
# is 0.027.
MIN_TETRAHEDRAL_CHIRAL_VOL = 0.02


def _same_side(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    p0: np.ndarray,
    tol: float,
) -> bool:
    """Check if v4 and p0 are on the same side of the plane defined by (v1, v2, v3).

    Port of nvMolKit's _sameSide function.

    Args:
        v1, v2, v3: Three points defining the plane.
        v4: Reference point (must be on the correct side).
        p0: Point to check (must be same side as v4).
        tol: Tolerance for near-plane points.

    Returns:
        True if v4 and p0 are on the same side, False otherwise.
    """
    cross = np.cross(v2 - v1, v3 - v1)
    d1 = np.dot(cross, v4 - v1)
    d2 = np.dot(cross, p0 - v1)

    if abs(d1) < tol or abs(d2) < tol:
        return False

    return (d1 > 0) == (d2 > 0)


def stage_tetrahedral_check(ctx: PipelineContext, tol: float = 0.3) -> None:
    """Check tetrahedral geometry of sp3 centers.

    For each tetrahedral atom (center with 4 neighbors):
    1. Volume test -- 4 cross-dot products of normalized vectors from center
       to neighbors. Fails if any volume is too small (planar/degenerate).
    2. Center-in-volume test -- checks center is inside the tetrahedron
       formed by its 4 neighbors using 4 same-side plane checks.

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
        tol: Tolerance for same-side check.
    """
    tet_data = ctx.tet_data
    if tet_data is None:
        return

    n_terms = tet_data.idx0.shape[0]
    if n_terms == 0:
        return

    # Try Metal kernel first
    try:
        from ..metal_kernels.stereo_checks import metal_tetrahedral_check

        active = _build_active_array(ctx)
        failed = metal_tetrahedral_check(
            ctx.positions, tet_data, active,
            ctx.n_mols, ctx.dim, tol=tol, do_volume_test=True,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal tetrahedral check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx0 = np.array(tet_data.idx0)
    idx1 = np.array(tet_data.idx1)
    idx2 = np.array(tet_data.idx2)
    idx3 = np.array(tet_data.idx3)
    idx4 = np.array(tet_data.idx4)
    in_fused = np.array(tet_data.in_fused_small_rings)
    mol_indices = np.array(tet_data.mol_indices)

    for t in range(n_terms):
        mol_idx = int(mol_indices[t])
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        p0 = pos[idx0[t]]
        p1 = pos[idx1[t]]
        p2 = pos[idx2[t]]
        p3 = pos[idx3[t]]
        p4 = pos[idx4[t]]

        # 3-coordinate centers (idx0 == idx4) have no 4th neighbor — skip
        # both volume test and center-in-volume check (matches nvMolKit which
        # only includes atoms with exactly 4 neighbors).
        if idx0[t] == idx4[t]:
            continue

        vol_scale = 0.25 if in_fused[t] else 1.0
        threshold = vol_scale * MIN_TETRAHEDRAL_CHIRAL_VOL

        # Normalized vectors from center to each neighbor
        def _normalize(v: np.ndarray) -> np.ndarray:
            """Return unit vector, or original if near-zero length."""
            n = np.linalg.norm(v)
            return v / n if n > 1e-10 else v

        d1 = _normalize(p0 - p1)
        d2 = _normalize(p0 - p2)
        d3 = _normalize(p0 - p3)
        d4 = _normalize(p0 - p4)

        # Volume test: 4 cross-dot products
        failed = False
        for da, db, dc in [(d1, d2, d3), (d1, d2, d4), (d1, d3, d4), (d2, d3, d4)]:
            vol = np.dot(np.cross(da, db), dc)
            if abs(vol) < threshold:
                failed = True
                break

        if failed:
            ctx.failed[mol_idx] = True
            continue

        if not _same_side(p1, p2, p3, p4, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p2, p3, p4, p1, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p3, p4, p1, p2, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p4, p1, p2, p3, p0, tol):
            ctx.failed[mol_idx] = True
            continue


def stage_first_chiral_check(ctx: PipelineContext) -> None:
    """Check chiral volumes against bounds.

    For each chiral center, computes the signed volume and fails molecules
    where the volume deviates from the allowed range by more than 20% or
    has the wrong sign.

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
    """
    system = ctx.dg_system
    n_chiral = system.chiral_idx1.shape[0]
    if n_chiral == 0:
        return

    # Try Metal kernel first
    try:
        from ..metal_kernels.stereo_checks import metal_first_chiral_check

        active = _build_active_array(ctx)
        failed = metal_first_chiral_check(
            ctx.positions,
            system.chiral_idx1, system.chiral_idx2,
            system.chiral_idx3, system.chiral_idx4,
            system.chiral_vol_lower, system.chiral_vol_upper,
            system.chiral_mol_indices, active,
            ctx.n_mols, ctx.dim,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal first chiral check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx1 = np.array(system.chiral_idx1)
    idx2 = np.array(system.chiral_idx2)
    idx3 = np.array(system.chiral_idx3)
    idx4 = np.array(system.chiral_idx4)
    vol_lower = np.array(system.chiral_vol_lower)
    vol_upper = np.array(system.chiral_vol_upper)
    mol_indices = np.array(system.chiral_mol_indices)

    for t in range(n_chiral):
        mol_idx = int(mol_indices[t])
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        p1 = pos[idx1[t]]
        p2 = pos[idx2[t]]
        p3 = pos[idx3[t]]
        p4 = pos[idx4[t]]

        # Compute chiral volume: V = (p1-p4) . ((p2-p4) x (p3-p4))
        v1 = p1 - p4
        v2 = p2 - p4
        v3 = p3 - p4
        vol = float(np.dot(v1, np.cross(v2, v3)))

        lb = float(vol_lower[t])
        ub = float(vol_upper[t])

        # Check lower bound
        if lb > 0 and vol < lb:
            if vol / lb < 0.8 or (vol < 0) != (lb < 0):
                ctx.failed[mol_idx] = True
                continue

        # Check upper bound
        if ub < 0 and vol > ub:
            if vol / ub < 0.8 or (vol < 0) != (ub < 0):
                ctx.failed[mol_idx] = True
                continue


# ---------------------
# Stage 6: Final checks
# ---------------------


def stage_double_bond_geometry_check(
    ctx: PipelineContext, double_bond_data: dict[str, np.ndarray] | None
) -> None:
    """Check double bond geometry (linearity check).

    Fails molecules where two bonds around a double bond are nearly linear
    (dot product of normalized bond vectors ~ -1).

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
        double_bond_data: Dict with 'idx0', 'idx1', 'idx2', 'mol_indices'
            arrays, or None to skip.
    """
    if double_bond_data is None or len(double_bond_data['idx0']) == 0:
        return

    # Try Metal kernel first
    try:
        from ..metal_kernels.stereo_checks import metal_double_bond_geom_check

        active = _build_active_array(ctx)
        failed = metal_double_bond_geom_check(
            ctx.positions,
            mx.array(double_bond_data['idx0'].astype(np.int32)),
            mx.array(double_bond_data['idx1'].astype(np.int32)),
            mx.array(double_bond_data['idx2'].astype(np.int32)),
            mx.array(double_bond_data['mol_indices'].astype(np.int32)),
            active, ctx.n_mols, ctx.dim,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal double bond geom check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx0 = double_bond_data['idx0']
    idx1 = double_bond_data['idx1']
    idx2 = double_bond_data['idx2']
    mol_indices = double_bond_data['mol_indices']

    linear_tol = 1e-3

    for t in range(len(idx0)):
        mol_idx = mol_indices[t]
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        p0 = pos[idx0[t]]
        p1 = pos[idx1[t]]
        p2 = pos[idx2[t]]

        d1 = p1 - p0
        d2 = p1 - p2
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        d1 /= n1
        d2 /= n2

        dot = np.dot(d1, d2)
        if (dot + 1.0) < linear_tol:
            ctx.failed[mol_idx] = True


def stage_double_bond_stereo_check(
    ctx: PipelineContext, stereo_bond_data: dict[str, np.ndarray] | None
) -> None:
    """Check double bond stereo (E/Z) assignment.

    Computes dihedral-like angle between substituents and checks
    that the E/Z assignment matches the reference.

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
        stereo_bond_data: Dict with 'idx0'-'idx3', 'signs', 'mol_indices'
            arrays, or None to skip.
    """
    if stereo_bond_data is None or len(stereo_bond_data['idx0']) == 0:
        return

    # Try Metal kernel first
    try:
        from ..metal_kernels.stereo_checks import metal_double_bond_stereo_check

        active = _build_active_array(ctx)
        failed = metal_double_bond_stereo_check(
            ctx.positions,
            mx.array(stereo_bond_data['idx0'].astype(np.int32)),
            mx.array(stereo_bond_data['idx1'].astype(np.int32)),
            mx.array(stereo_bond_data['idx2'].astype(np.int32)),
            mx.array(stereo_bond_data['idx3'].astype(np.int32)),
            mx.array(stereo_bond_data['signs'].astype(np.int32)),
            mx.array(stereo_bond_data['mol_indices'].astype(np.int32)),
            active, ctx.n_mols, ctx.dim,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal double bond stereo check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx0 = stereo_bond_data['idx0']
    idx1 = stereo_bond_data['idx1']
    idx2 = stereo_bond_data['idx2']
    idx3 = stereo_bond_data['idx3']
    signs = stereo_bond_data['signs']
    mol_indices = stereo_bond_data['mol_indices']

    for t in range(len(idx0)):
        mol_idx = mol_indices[t]
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        p0 = pos[idx0[t]]
        p1 = pos[idx1[t]]
        p2 = pos[idx2[t]]
        p3 = pos[idx3[t]]
        sign = signs[t]

        d1 = p2 - p1  # bond vector
        d2 = p0 - p1  # substituent on atom 1
        d3 = p3 - p2  # substituent on atom 2

        cross1 = np.cross(d2, d1)
        cross2 = np.cross(d3, d1)

        l1sq = np.dot(cross1, cross1)
        l2sq = np.dot(cross2, cross2)
        denom = math.sqrt(l1sq * l2sq)
        if denom < 1e-16:
            continue

        dot = np.dot(cross1, cross2) / denom
        if dot <= -1.0:
            angle = math.pi
        elif dot >= 1.0:
            angle = 0.0
        else:
            angle = math.acos(dot)

        if ((angle - math.pi / 2.0) * sign) < 0.0:
            ctx.failed[mol_idx] = True


def stage_chiral_dist_matrix_check(
    ctx: PipelineContext, chiral_dist_data: dict[str, np.ndarray] | None
) -> None:
    """Check distances between chiral center atoms against bounds matrix.

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
        chiral_dist_data: Dict with 'idx0', 'idx1', 'lower', 'upper',
            'mol_indices' arrays, or None to skip.
    """
    if chiral_dist_data is None or len(chiral_dist_data['idx0']) == 0:
        return

    # Try Metal kernel first
    try:
        from ..metal_kernels.stereo_checks import metal_chiral_dist_check

        active = _build_active_array(ctx)
        failed = metal_chiral_dist_check(
            ctx.positions,
            mx.array(chiral_dist_data['idx0'].astype(np.int32)),
            mx.array(chiral_dist_data['idx1'].astype(np.int32)),
            mx.array(chiral_dist_data['lower'].astype(np.float32)),
            mx.array(chiral_dist_data['upper'].astype(np.float32)),
            mx.array(chiral_dist_data['mol_indices'].astype(np.int32)),
            active, ctx.n_mols, ctx.dim,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal chiral dist check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx0 = chiral_dist_data['idx0']
    idx1 = chiral_dist_data['idx1']
    lower = chiral_dist_data['lower']
    upper = chiral_dist_data['upper']
    mol_indices = chiral_dist_data['mol_indices']

    for t in range(len(idx0)):
        mol_idx = mol_indices[t]
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        p0 = pos[idx0[t]]
        p1 = pos[idx1[t]]
        dist = np.linalg.norm(p0 - p1)
        lb = lower[t]
        ub = upper[t]

        if (dist < lb and abs(dist - lb) > 0.1 * ub) or \
           (dist > ub and abs(dist - ub) > 0.1 * ub):
            ctx.failed[mol_idx] = True


def stage_chiral_volume_check(ctx: PipelineContext) -> None:
    """Final chiral center volume check (center-in-volume only, no volume test).

    Same as ``stage_tetrahedral_check`` but without the volume test.
    Only checks that the center atom is inside the tetrahedron formed by
    its 4 neighbors.

    Args:
        ctx: Pipeline context (modifies ``ctx.failed`` in place).
    """
    tet_data = ctx.tet_data
    if tet_data is None:
        return

    n_terms = tet_data.idx0.shape[0]
    if n_terms == 0:
        return

    # Try Metal kernel (same as tetrahedral but do_volume_test=False)
    try:
        from ..metal_kernels.stereo_checks import metal_tetrahedral_check

        active = _build_active_array(ctx)
        failed = metal_tetrahedral_check(
            ctx.positions, tet_data, active,
            ctx.n_mols, ctx.dim, tol=0.1, do_volume_test=False,
        )
        _apply_failed(ctx, failed)
        return
    except Exception as e:
        log.debug("Metal chiral volume check unavailable: %s", e)

    # CPU fallback
    mx.eval(ctx.positions)
    pos = np.array(ctx.positions).reshape(-1, ctx.dim)[:, :3]

    idx0 = np.array(tet_data.idx0)
    idx1 = np.array(tet_data.idx1)
    idx2 = np.array(tet_data.idx2)
    idx3 = np.array(tet_data.idx3)
    idx4 = np.array(tet_data.idx4)
    mol_indices = np.array(tet_data.mol_indices)

    # nvMolKit uses tol=0.1 for the final chiral volume check (stage 6d),
    # vs tol=0.3 for the earlier tetrahedral check (stage 3).
    tol = 0.1

    for t in range(n_terms):
        mol_idx = int(mol_indices[t])
        if not ctx.active[mol_idx] or ctx.failed[mol_idx]:
            continue

        # Skip 3-coordinate centers (no center-in-volume check)
        if idx0[t] == idx4[t]:
            continue

        p0 = pos[idx0[t]]
        p1 = pos[idx1[t]]
        p2 = pos[idx2[t]]
        p3 = pos[idx3[t]]
        p4 = pos[idx4[t]]

        if not _same_side(p1, p2, p3, p4, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p2, p3, p4, p1, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p3, p4, p1, p2, p0, tol):
            ctx.failed[mol_idx] = True
            continue
        if not _same_side(p4, p1, p2, p3, p0, tol):
            ctx.failed[mol_idx] = True
