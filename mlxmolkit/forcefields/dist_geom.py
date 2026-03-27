"""Distance geometry energy and gradient functions for 4D ETKDG.

All computations use float32 MLX arrays. Manual gradients are used
(not mx.grad) for numerical fidelity and scatter-add compatibility.

Port of nvMolKit's dist_geom_kernels_device.cuh.
"""

import mlx.core as mx

from ..preprocessing.batching import BatchedDGSystem


# ---------------------
# Distance Violation
# ---------------------


def dist_violation_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    lb2: mx.array,
    ub2: mx.array,
    weight: mx.array,
    dim: int,
) -> mx.array:
    """Compute distance violation energy for each term.

    E = weight * val^2  where:
      val = d^2/ub^2 - 1    if d^2 > ub^2
      val = 2*lb^2/(lb^2+d^2) - 1  if d^2 < lb^2
      val = 0                otherwise

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1, idx2: Atom index arrays, shape (n_terms,), int32.
        lb2, ub2: Squared lower/upper bounds, shape (n_terms,), float32.
        weight: Per-term weights, shape (n_terms,), float32.
        dim: Coordinate dimension (3 or 4).

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    # Gather positions: (n_terms, dim)
    pos_r = pos.reshape(-1, dim)
    p1 = pos_r[idx1]
    p2 = pos_r[idx2]

    diff = p1 - p2
    d2 = mx.sum(diff * diff, axis=1)

    # Upper bound violation
    val_ub = (d2 / ub2) - 1.0
    e_ub = mx.where(d2 > ub2, weight * val_ub * val_ub, 0.0)

    # Lower bound violation
    val_lb = (2.0 * lb2) / (lb2 + d2) - 1.0
    e_lb = mx.where(d2 < lb2, weight * val_lb * val_lb, 0.0)

    return e_ub + e_lb


def dist_violation_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    lb2: mx.array,
    ub2: mx.array,
    weight: mx.array,
    dim: int,
) -> mx.array:
    """Compute distance violation gradient, accumulated into flat gradient array.

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1, idx2: Atom index arrays, shape (n_terms,), int32.
        lb2, ub2: Squared lower/upper bounds, shape (n_terms,), float32.
        weight: Per-term weights, shape (n_terms,), float32.
        dim: Coordinate dimension (3 or 4).

    Returns:
        Gradient array, same shape as pos.
    """
    pos_r = pos.reshape(-1, dim)
    p1 = pos_r[idx1]
    p2 = pos_r[idx2]

    diff = p1 - p2  # (n_terms, dim)
    d2 = mx.sum(diff * diff, axis=1)  # (n_terms,)

    # Upper bound prefactor: 4 * (d^2/ub^2 - 1) / ub^2
    pf_ub = mx.where(d2 > ub2, 4.0 * ((d2 / ub2) - 1.0) / ub2, 0.0)

    # Lower bound prefactor: 8 * lb^2 * (1 - 2*lb^2/(lb^2+d^2)) / (lb^2+d^2)^2
    l2d2 = d2 + lb2
    pf_lb = mx.where(
        d2 < lb2,
        8.0 * lb2 * (1.0 - 2.0 * lb2 / l2d2) / (l2d2 * l2d2),
        0.0,
    )

    prefactor = weight * (pf_ub + pf_lb)  # (n_terms,)
    g = prefactor[:, None] * diff  # (n_terms, dim)

    # Scatter-add to gradient
    grad = mx.zeros_like(pos_r)
    for d_idx in range(dim):
        grad_col = grad[:, d_idx]
        grad_col = grad_col.at[idx1].add(g[:, d_idx])
        grad_col = grad_col.at[idx2].add(-g[:, d_idx])
        grad = grad.at[:, d_idx].add(grad_col - grad[:, d_idx])

    return grad.reshape(-1)


def dist_violation_grad_v2(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    lb2: mx.array,
    ub2: mx.array,
    weight: mx.array,
    dim: int,
) -> mx.array:
    """Compute distance violation gradient using flat index scatter-add.

    Uses flat indexing for scatter-add, matching the nvMolKit pattern.

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1, idx2: Atom index arrays, shape (n_terms,), int32.
        lb2, ub2: Squared lower/upper bounds, shape (n_terms,), float32.
        weight: Per-term weights, shape (n_terms,), float32.
        dim: Coordinate dimension (3 or 4).

    Returns:
        Gradient array, same shape as pos.
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]
    p1 = pos_r[idx1]
    p2 = pos_r[idx2]

    diff = p1 - p2
    d2 = mx.sum(diff * diff, axis=1)

    pf_ub = mx.where(d2 > ub2, 4.0 * ((d2 / ub2) - 1.0) / ub2, 0.0)
    l2d2 = d2 + lb2
    pf_lb = mx.where(
        d2 < lb2,
        8.0 * lb2 * (1.0 - 2.0 * lb2 / l2d2) / (l2d2 * l2d2),
        0.0,
    )

    prefactor = weight * (pf_ub + pf_lb)
    g = prefactor[:, None] * diff  # (n_terms, dim)

    # Scatter-add using flat indices
    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    for d_idx in range(dim):
        flat_idx1 = idx1 * dim + d_idx
        flat_idx2 = idx2 * dim + d_idx
        grad = grad.at[flat_idx1].add(g[:, d_idx])
        grad = grad.at[flat_idx2].add(-g[:, d_idx])

    return grad


# ---------------------
# Chiral Violation
# ---------------------


def _calc_chiral_volume(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute signed chiral volume and intermediate vectors.

    V = (p1-p4) . ((p2-p4) x (p3-p4))

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1, idx2, idx3, idx4: Atom indices, shape (n_terms,), int32.
        dim: Coordinate dimension (3 or 4).

    Returns:
        (volume, v1, v2, v3) where v_i are difference vectors (n_terms, 3).
    """
    pos_r = pos.reshape(-1, dim)
    # Only use xyz components for volume computation
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]
    p4 = pos_r[idx4][:, :3]

    v1 = p1 - p4  # (n_terms, 3)
    v2 = p2 - p4
    v3 = p3 - p4

    # Cross product v2 x v3
    cross_x = v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]
    cross_y = v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2]
    cross_z = v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]

    # Dot product v1 . (v2 x v3)
    vol = v1[:, 0] * cross_x + v1[:, 1] * cross_y + v1[:, 2] * cross_z

    return vol, v1, v2, v3


def chiral_violation_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    vol_lower: mx.array,
    vol_upper: mx.array,
    weight: float,
    dim: int,
) -> mx.array:
    """Compute chiral violation energy for each term.

    E = weight * (vol - bound)^2  if vol outside [vol_lower, vol_upper]
    E = 0                         otherwise

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1-idx4: Atom indices, shape (n_terms,), int32.
        vol_lower, vol_upper: Volume bounds, shape (n_terms,), float32.
        weight: Scalar chiral weight.
        dim: Coordinate dimension (3 or 4).

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    vol, _, _, _ = _calc_chiral_volume(pos, idx1, idx2, idx3, idx4, dim)

    e_lower = mx.where(vol < vol_lower, weight * (vol - vol_lower) ** 2, 0.0)
    e_upper = mx.where(vol > vol_upper, weight * (vol - vol_upper) ** 2, 0.0)

    return e_lower + e_upper


def chiral_violation_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    vol_lower: mx.array,
    vol_upper: mx.array,
    weight: float,
    dim: int,
) -> mx.array:
    """Compute chiral violation gradient, accumulated into flat gradient array.

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx1, idx2, idx3, idx4: Atom indices, shape (n_terms,), int32.
        vol_lower, vol_upper: Volume bounds, shape (n_terms,), float32.
        weight: Scalar chiral weight.
        dim: Coordinate dimension (3 or 4).

    Returns:
        Gradient array, same shape as pos.
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]

    vol, v1, v2, v3 = _calc_chiral_volume(pos, idx1, idx2, idx3, idx4, dim)

    # Compute prefactor: dE/dvol = 2 * weight * (vol - bound)
    pf_lower = mx.where(vol < vol_lower, 2.0 * weight * (vol - vol_lower), 0.0)
    pf_upper = mx.where(vol > vol_upper, 2.0 * weight * (vol - vol_upper), 0.0)
    prefactor = pf_lower + pf_upper  # (n_terms,)

    # Gradient for idx1: prefactor * (v2 x v3)
    g1_x = prefactor * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1])
    g1_y = prefactor * (v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2])
    g1_z = prefactor * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])

    # Gradient for idx2: prefactor * (v3 x v1)
    g2_x = prefactor * (v3[:, 1] * v1[:, 2] - v3[:, 2] * v1[:, 1])
    g2_y = prefactor * (v3[:, 2] * v1[:, 0] - v3[:, 0] * v1[:, 2])
    g2_z = prefactor * (v3[:, 0] * v1[:, 1] - v3[:, 1] * v1[:, 0])

    # Gradient for idx3: prefactor * (v1 x v2) -- NOTE: nvMolKit uses specific ordering
    # In CUDA: grad[idx3] += pf * (v2z*v1y - v2y*v1z, v2x*v1z - v2z*v1x, v2y*v1x - v2x*v1y)
    # This is -(v1 x v2) = v2 x v1
    g3_x = prefactor * (v2[:, 2] * v1[:, 1] - v2[:, 1] * v1[:, 2])
    g3_y = prefactor * (v2[:, 0] * v1[:, 2] - v2[:, 2] * v1[:, 0])
    g3_z = prefactor * (v2[:, 1] * v1[:, 0] - v2[:, 0] * v1[:, 1])

    # Gradient for idx4: -(g1 + g2 + g3)
    # In CUDA this is computed from the original coordinates:
    # gx4 = pf * (z1*(y2-y3) + z2*(y3-y1) + z3*(y1-y2))
    # gy4 = pf * (x1*(z2-z3) + x2*(z3-z1) + x3*(z1-z2))
    # gz4 = pf * (y1*(x2-x3) + y2*(x3-x1) + y3*(x1-x2))
    # This equals -(g1 + g2 + g3) by the chain rule.
    g4_x = -(g1_x + g2_x + g3_x)
    g4_y = -(g1_y + g2_y + g3_y)
    g4_z = -(g1_z + g2_z + g3_z)

    # Scatter-add to gradient
    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)

    # idx1
    grad = grad.at[idx1 * dim + 0].add(g1_x)
    grad = grad.at[idx1 * dim + 1].add(g1_y)
    grad = grad.at[idx1 * dim + 2].add(g1_z)

    # idx2
    grad = grad.at[idx2 * dim + 0].add(g2_x)
    grad = grad.at[idx2 * dim + 1].add(g2_y)
    grad = grad.at[idx2 * dim + 2].add(g2_z)

    # idx3
    grad = grad.at[idx3 * dim + 0].add(g3_x)
    grad = grad.at[idx3 * dim + 1].add(g3_y)
    grad = grad.at[idx3 * dim + 2].add(g3_z)

    # idx4
    grad = grad.at[idx4 * dim + 0].add(g4_x)
    grad = grad.at[idx4 * dim + 1].add(g4_y)
    grad = grad.at[idx4 * dim + 2].add(g4_z)

    return grad


# ---------------------
# Fourth Dimension
# ---------------------


def fourth_dim_energy(
    pos: mx.array,
    idx: mx.array,
    weight: float,
    dim: int,
) -> mx.array:
    """Compute fourth dimension penalty energy for each term.

    E = weight * w^2  where w is the 4th coordinate.
    Only active when dim == 4.

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx: Atom indices, shape (n_terms,), int32.
        weight: Fourth dimension weight scalar.
        dim: Coordinate dimension (must be 4).

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    if dim != 4:
        return mx.zeros(idx.shape[0], dtype=pos.dtype)

    w = pos.reshape(-1, dim)[idx, 3]  # 4th coordinate
    return weight * w * w


def fourth_dim_grad(
    pos: mx.array,
    idx: mx.array,
    weight: float,
    dim: int,
) -> mx.array:
    """Compute fourth dimension penalty gradient.

    dE/dw = 2 * weight * w for E = weight * w^2.

    Note: nvMolKit's CUDA code uses weight * w (without the factor of 2),
    but the mathematically correct derivative produces better results for
    our float32 BFGS optimizer — the consistent gradient gives the Hessian
    approximation better curvature information for the 4th dimension.

    Args:
        pos: Flat positions array, shape (n_atoms * dim,), float32.
        idx: Atom indices, shape (n_terms,), int32.
        weight: Fourth dimension weight scalar.
        dim: Coordinate dimension (must be 4).

    Returns:
        Gradient array, same shape as pos.
    """
    if dim != 4:
        return mx.zeros_like(pos)

    n_atoms = pos.shape[0] // dim
    w = pos.reshape(-1, dim)[idx, 3]

    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    grad = grad.at[idx * dim + 3].add(2.0 * weight * w)

    return grad


# ---------------------
# Combined DG Energy/Gradient
# ---------------------


def dg_energy(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
) -> mx.array:
    """Compute total DG energy per molecule.

    Sums distance violation, chiral violation, and fourth dimension penalties.

    Args:
        pos: Flat positions, shape (n_atoms_total * dim,), float32.
        system: Batched DG system with all terms.
        chiral_weight: Weight for chiral violation terms.
        fourth_dim_weight: Weight for fourth dimension terms.

    Returns:
        Per-molecule energies, shape (n_mols,), float32.
    """
    dim = system.dim
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)

    # Distance violation
    if system.dist_idx1.size > 0:
        e_dist = dist_violation_energy(
            pos,
            system.dist_idx1,
            system.dist_idx2,
            system.dist_lb2,
            system.dist_ub2,
            system.dist_weight,
            dim,
        )
        energies = energies.at[system.dist_mol_indices].add(e_dist)

    # Chiral violation
    if system.chiral_idx1.size > 0:
        e_chiral = chiral_violation_energy(
            pos,
            system.chiral_idx1,
            system.chiral_idx2,
            system.chiral_idx3,
            system.chiral_idx4,
            system.chiral_vol_lower,
            system.chiral_vol_upper,
            chiral_weight,
            dim,
        )
        energies = energies.at[system.chiral_mol_indices].add(e_chiral)

    # Fourth dimension
    if system.fourth_idx.size > 0:
        e_fourth = fourth_dim_energy(pos, system.fourth_idx, fourth_dim_weight, dim)
        energies = energies.at[system.fourth_mol_indices].add(e_fourth)

    return energies


def dg_energy_and_grad(
    pos: mx.array,
    system: BatchedDGSystem,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
) -> tuple[mx.array, mx.array]:
    """Compute total DG energy and gradient.

    Args:
        pos: Flat positions, shape (n_atoms_total * dim,), float32.
        system: Batched DG system with all terms.
        chiral_weight: Weight for chiral violation terms.
        fourth_dim_weight: Weight for fourth dimension terms.

    Returns:
        (energies, grad) where energies is (n_mols,) and grad is same shape as pos.
    """
    dim = system.dim
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)
    grad = mx.zeros_like(pos)

    # Distance violation
    if system.dist_idx1.size > 0:
        e_dist = dist_violation_energy(
            pos,
            system.dist_idx1,
            system.dist_idx2,
            system.dist_lb2,
            system.dist_ub2,
            system.dist_weight,
            dim,
        )
        energies = energies.at[system.dist_mol_indices].add(e_dist)

        g_dist = dist_violation_grad_v2(
            pos,
            system.dist_idx1,
            system.dist_idx2,
            system.dist_lb2,
            system.dist_ub2,
            system.dist_weight,
            dim,
        )
        grad = grad + g_dist

    # Chiral violation
    if system.chiral_idx1.size > 0:
        e_chiral = chiral_violation_energy(
            pos,
            system.chiral_idx1,
            system.chiral_idx2,
            system.chiral_idx3,
            system.chiral_idx4,
            system.chiral_vol_lower,
            system.chiral_vol_upper,
            chiral_weight,
            dim,
        )
        energies = energies.at[system.chiral_mol_indices].add(e_chiral)

        g_chiral = chiral_violation_grad(
            pos,
            system.chiral_idx1,
            system.chiral_idx2,
            system.chiral_idx3,
            system.chiral_idx4,
            system.chiral_vol_lower,
            system.chiral_vol_upper,
            chiral_weight,
            dim,
        )
        grad = grad + g_chiral

    # Fourth dimension
    if system.fourth_idx.size > 0:
        e_fourth = fourth_dim_energy(
            pos, system.fourth_idx, fourth_dim_weight, dim
        )
        energies = energies.at[system.fourth_mol_indices].add(e_fourth)

        g_fourth = fourth_dim_grad(
            pos, system.fourth_idx, fourth_dim_weight, dim
        )
        grad = grad + g_fourth

    return energies, grad
