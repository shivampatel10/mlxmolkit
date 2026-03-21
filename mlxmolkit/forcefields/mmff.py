"""MMFF94 energy and gradient functions.

All computations use float32 MLX arrays. Manual gradients are used
(not mx.grad) for numerical fidelity and scatter-add compatibility.
Positions are flat arrays of shape (n_atoms_total * 3,).

Port of nvMolKit's mmff_kernels_device.cuh.
"""

from __future__ import annotations

import math

import mlx.core as mx

from ..preprocessing.mmff_batching import BatchedMMFFSystem

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# =====================
# Bond Stretch
# =====================


def _bond_stretch_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    kb: mx.array,
    r0: mx.array,
) -> mx.array:
    """Compute MMFF bond stretch energy per term.

    E = (143.9325/2) * kb * ΔR² * (1 + cs*ΔR + cs2*ΔR²)
    cs = -2.0, cs2 = (7/12)*cs² = 7/3
    """
    pos_r = pos.reshape(-1, 3)
    diff = pos_r[idx1] - pos_r[idx2]
    d = mx.sqrt(mx.maximum(mx.sum(diff * diff, axis=1), 1e-16))
    delta_r = d - r0
    delta_r2 = delta_r * delta_r
    cs = -2.0
    cs2 = 7.0 / 12.0 * cs * cs  # 7/3
    return (143.9325 / 2.0) * kb * delta_r2 * (1.0 + cs * delta_r + cs2 * delta_r2)


def _bond_stretch_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    kb: mx.array,
    r0: mx.array,
) -> mx.array:
    """Compute MMFF bond stretch gradient.

    dE/dr = 143.9325 * kb * ΔR * (1 - 3*ΔR + (14/3)*ΔR²)
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    diff = pos_r[idx1] - pos_r[idx2]
    d2 = mx.sum(diff * diff, axis=1)
    d = mx.sqrt(mx.maximum(d2, 1e-16))
    inv_d = mx.where(d > 1e-8, 1.0 / d, 0.0)
    delta_r = d - r0

    cs = -2.0
    cs_15 = cs * 1.5  # -3.0
    last = 2.0 * 7.0 / 12.0 * cs * cs  # 14/3

    de_dr = 143.9325 * kb * delta_r * (1.0 + cs_15 * delta_r + last * delta_r * delta_r)

    # grad_i = dE/dr * (p_i - p_j) / d
    g = (de_dr * inv_d)[:, None] * diff  # (n_terms, 3)

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(-g[:, d_idx])
    return grad


# =====================
# Angle Bend
# =====================


def _angle_bend_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    ka: mx.array,
    theta0: mx.array,
    is_linear: mx.array,
) -> mx.array:
    """Compute MMFF angle bend energy per term.

    Non-linear: E = 0.5 * 143.9325 * (π/180)² * ka * Δθ² * (1 + cb*Δθ)
    Linear:     E = 143.9325 * ka * (1 + cosθ)
    cb = -0.4 * (π/180)
    """
    pos_r = pos.reshape(-1, 3)
    r1 = pos_r[idx1] - pos_r[idx2]
    r2 = pos_r[idx3] - pos_r[idx2]
    d1 = mx.sqrt(mx.maximum(mx.sum(r1 * r1, axis=1), 1e-16))
    d2 = mx.sqrt(mx.maximum(mx.sum(r2 * r2, axis=1), 1e-16))
    cos_theta = mx.clip(mx.sum(r1 * r2, axis=1) / (d1 * d2), -1.0, 1.0)
    theta_deg = RAD_TO_DEG * mx.arccos(cos_theta)
    delta_theta = theta_deg - theta0

    cb = -0.4 * DEG_TO_RAD
    prefactor = 0.5 * 143.9325 * DEG_TO_RAD * DEG_TO_RAD
    e_nonlin = prefactor * ka * delta_theta * delta_theta * (1.0 + cb * delta_theta)
    e_linear = 143.9325 * ka * (1.0 + cos_theta)

    is_lin = is_linear.astype(mx.float32)
    return mx.where(is_lin > 0.5, e_linear, e_nonlin)


def _angle_bend_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    ka: mx.array,
    theta0: mx.array,
    is_linear: mx.array,
) -> mx.array:
    """Compute MMFF angle bend gradient.

    Port of nvMolKit angleBendGrad.
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    r1 = pos_r[idx1] - pos_r[idx2]
    r2 = pos_r[idx3] - pos_r[idx2]
    d1_sq = mx.sum(r1 * r1, axis=1)
    d2_sq = mx.sum(r2 * r2, axis=1)
    inv_d1 = mx.rsqrt(mx.maximum(d1_sq, 1e-16))
    inv_d2 = mx.rsqrt(mx.maximum(d2_sq, 1e-16))
    cos_theta = mx.clip(mx.sum(r1 * r2, axis=1) * inv_d1 * inv_d2, -1.0, 1.0)
    sin_theta_sq = 1.0 - cos_theta * cos_theta
    # Skip degenerate cases
    degenerate = (sin_theta_sq < 1e-16) | (d1_sq < 1e-16) | (d2_sq < 1e-16)
    inv_neg_sin_theta = mx.where(
        degenerate, 0.0, -mx.rsqrt(mx.maximum(sin_theta_sq, 1e-16))
    )

    theta_deg = RAD_TO_DEG * mx.arccos(cos_theta)
    delta_theta = theta_deg - theta0

    c1 = 143.9325 * DEG_TO_RAD
    cb_factor = -0.4 * DEG_TO_RAD * 1.5  # -0.006981317 * 1.5

    is_lin = is_linear.astype(mx.float32)
    sin_theta = mx.sqrt(mx.maximum(sin_theta_sq, 0.0))
    de_nonlin = c1 * ka * delta_theta * (1.0 + cb_factor * delta_theta)
    de_linear = -143.9325 * ka * sin_theta
    de_d_delta_theta = mx.where(is_lin > 0.5, de_linear, de_nonlin)

    constant_factor = de_d_delta_theta * inv_neg_sin_theta

    # Unit vectors
    r1_hat = r1 * inv_d1[:, None]
    r2_hat = r2 * inv_d2[:, None]

    # dcosθ/dp1 = (1/|r1|) * (r2_hat - cosθ * r1_hat)
    inter1 = inv_d1[:, None] * (r2_hat - cos_theta[:, None] * r1_hat)
    # dcosθ/dp3 = (1/|r2|) * (r1_hat - cosθ * r2_hat)
    inter3 = inv_d2[:, None] * (r1_hat - cos_theta[:, None] * r2_hat)

    g1 = constant_factor[:, None] * inter1
    g3 = constant_factor[:, None] * inter3
    g2 = -(g1 + g3)

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g1[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(g2[:, d_idx])
        grad = grad.at[idx3 * 3 + d_idx].add(g3[:, d_idx])
    return grad


# =====================
# Stretch-Bend
# =====================


def _stretch_bend_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    r0_ij: mx.array,
    r0_kj: mx.array,
    theta0: mx.array,
    kba_ij: mx.array,
    kba_kj: mx.array,
) -> mx.array:
    """Compute MMFF stretch-bend energy per term.

    E = 2.51210 * Δθ * (ΔR_ij * kba_ij + ΔR_kj * kba_kj)
    """
    pos_r = pos.reshape(-1, 3)
    r1 = pos_r[idx1] - pos_r[idx2]
    r2 = pos_r[idx3] - pos_r[idx2]
    d1 = mx.sqrt(mx.maximum(mx.sum(r1 * r1, axis=1), 1e-16))
    d2 = mx.sqrt(mx.maximum(mx.sum(r2 * r2, axis=1), 1e-16))
    cos_theta = mx.clip(mx.sum(r1 * r2, axis=1) / (d1 * d2), -1.0, 1.0)
    theta_deg = RAD_TO_DEG * mx.arccos(cos_theta)
    delta_theta = theta_deg - theta0
    delta_r_ij = d1 - r0_ij
    delta_r_kj = d2 - r0_kj
    return 2.51210 * delta_theta * (delta_r_ij * kba_ij + delta_r_kj * kba_kj)


def _stretch_bend_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    r0_ij: mx.array,
    r0_kj: mx.array,
    theta0: mx.array,
    kba_ij: mx.array,
    kba_kj: mx.array,
) -> mx.array:
    """Compute MMFF stretch-bend gradient.

    Port of nvMolKit bendStretchGrad.
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    r1 = pos_r[idx1] - pos_r[idx2]
    r2 = pos_r[idx3] - pos_r[idx2]
    d1_sq = mx.sum(r1 * r1, axis=1)
    d2_sq = mx.sum(r2 * r2, axis=1)
    d1 = mx.sqrt(mx.maximum(d1_sq, 1e-16))
    d2 = mx.sqrt(mx.maximum(d2_sq, 1e-16))
    inv_d1 = 1.0 / mx.maximum(d1, 1e-8)
    inv_d2 = 1.0 / mx.maximum(d2, 1e-8)
    cos_theta = mx.clip(mx.sum(r1 * r2, axis=1) * inv_d1 * inv_d2, -1.0, 1.0)
    inv_sin_theta = mx.minimum(
        mx.rsqrt(mx.maximum(1.0 - cos_theta * cos_theta, 1e-16)), 1e8
    )

    theta_deg = RAD_TO_DEG * mx.arccos(cos_theta)
    delta_theta = theta_deg - theta0
    delta_r_ij = d1 - r0_ij
    delta_r_kj = d2 - r0_kj

    prefactor = 143.9325 * DEG_TO_RAD
    bond_factor = RAD_TO_DEG
    bond_energy_term = bond_factor * (kba_ij * delta_r_ij + kba_kj * delta_r_kj)
    bond_e_inv_sin = bond_energy_term * inv_sin_theta

    r1_hat = r1 * inv_d1[:, None]
    r2_hat = r2 * inv_d2[:, None]

    inter1 = inv_d1[:, None] * (r2_hat - cos_theta[:, None] * r1_hat)
    inter3 = inv_d2[:, None] * (r1_hat - cos_theta[:, None] * r2_hat)

    g1 = prefactor * (
        delta_theta[:, None] * r1_hat * kba_ij[:, None]
        - inter1 * bond_e_inv_sin[:, None]
    )
    g3 = prefactor * (
        delta_theta[:, None] * r2_hat * kba_kj[:, None]
        - inter3 * bond_e_inv_sin[:, None]
    )
    g2 = prefactor * (
        -delta_theta[:, None]
        * (r1_hat * kba_ij[:, None] + r2_hat * kba_kj[:, None])
        + (inter1 + inter3) * bond_e_inv_sin[:, None]
    )

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g1[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(g2[:, d_idx])
        grad = grad.at[idx3 * 3 + d_idx].add(g3[:, d_idx])
    return grad


# =====================
# Out-of-Plane Bend
# =====================


def _oop_bend_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    koop: mx.array,
) -> mx.array:
    """Compute MMFF out-of-plane bend energy per term.

    E = 0.5 * 143.9325 * (π/180)² * koop * χ²
    χ = out-of-plane angle in degrees
    idx2 is the central atom.
    """
    pos_r = pos.reshape(-1, 3)
    rJI = pos_r[idx1] - pos_r[idx2]
    rJK = pos_r[idx3] - pos_r[idx2]
    rJL = pos_r[idx4] - pos_r[idx2]

    l2JI = mx.sum(rJI * rJI, axis=1)
    l2JK = mx.sum(rJK * rJK, axis=1)
    l2JL = mx.sum(rJL * rJL, axis=1)

    inv_dJI = mx.rsqrt(mx.maximum(l2JI, 1e-16))
    inv_dJK = mx.rsqrt(mx.maximum(l2JK, 1e-16))
    inv_dJL = mx.rsqrt(mx.maximum(l2JL, 1e-16))

    # Normalize
    nJI = rJI * inv_dJI[:, None]
    nJK = rJK * inv_dJK[:, None]
    nJL = rJL * inv_dJL[:, None]

    # Normal to plane: (-rJI) x rJK
    normal = mx.stack(
        [
            (-nJI[:, 1]) * nJK[:, 2] - (-nJI[:, 2]) * nJK[:, 1],
            (-nJI[:, 2]) * nJK[:, 0] - (-nJI[:, 0]) * nJK[:, 2],
            (-nJI[:, 0]) * nJK[:, 1] - (-nJI[:, 1]) * nJK[:, 0],
        ],
        axis=1,
    )
    norm_len_sq = mx.sum(normal * normal, axis=1)
    inv_norm_len = mx.rsqrt(mx.maximum(norm_len_sq, 1e-16))
    normal = normal * inv_norm_len[:, None]

    sin_chi = mx.clip(mx.sum(nJL * normal, axis=1), -1.0, 1.0)
    chi_deg = RAD_TO_DEG * mx.arcsin(sin_chi)

    return 0.5 * 143.9325 * DEG_TO_RAD * DEG_TO_RAD * koop * chi_deg * chi_deg


def _oop_bend_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    koop: mx.array,
) -> mx.array:
    """Compute MMFF out-of-plane bend gradient.

    Port of nvMolKit oopGrad.
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]

    rJI = pos_r[idx1] - pos_r[idx2]
    rJK = pos_r[idx3] - pos_r[idx2]
    rJL = pos_r[idx4] - pos_r[idx2]

    inv_dJI = mx.rsqrt(mx.maximum(mx.sum(rJI * rJI, axis=1), 1e-16))
    inv_dJK = mx.rsqrt(mx.maximum(mx.sum(rJK * rJK, axis=1), 1e-16))
    inv_dJL = mx.rsqrt(mx.maximum(mx.sum(rJL * rJL, axis=1), 1e-16))

    dJI = rJI * inv_dJI[:, None]
    dJK = rJK * inv_dJK[:, None]
    dJL = rJL * inv_dJL[:, None]

    # Normal: (-dJI) x dJK
    normal = mx.stack(
        [
            (-dJI[:, 1]) * dJK[:, 2] - (-dJI[:, 2]) * dJK[:, 1],
            (-dJI[:, 2]) * dJK[:, 0] - (-dJI[:, 0]) * dJK[:, 2],
            (-dJI[:, 0]) * dJK[:, 1] - (-dJI[:, 1]) * dJK[:, 0],
        ],
        axis=1,
    )
    inv_norm = mx.rsqrt(mx.maximum(mx.sum(normal * normal, axis=1), 1e-16))
    normal = normal * inv_norm[:, None]

    sin_chi = mx.clip(mx.sum(dJL * normal, axis=1), -1.0, 1.0)
    cos_chi_sq = 1.0 - sin_chi * sin_chi
    inv_cos_chi = mx.where(cos_chi_sq > 0, mx.rsqrt(mx.maximum(cos_chi_sq, 1e-16)), 1e8)
    chi_deg = RAD_TO_DEG * mx.arcsin(sin_chi)

    cos_theta = mx.clip(mx.sum(dJI * dJK, axis=1), -1.0, 1.0)
    inv_sin_theta = mx.rsqrt(mx.maximum(1.0 - cos_theta * cos_theta, 1e-8))

    dE_dChi = 143.9325 * DEG_TO_RAD * koop * chi_deg

    term1 = inv_cos_chi * inv_sin_theta
    term2 = sin_chi * inv_cos_chi * (inv_sin_theta * inv_sin_theta)

    # t1 = dJL x dJK
    t1 = mx.stack(
        [
            dJL[:, 1] * dJK[:, 2] - dJL[:, 2] * dJK[:, 1],
            dJL[:, 2] * dJK[:, 0] - dJL[:, 0] * dJK[:, 2],
            dJL[:, 0] * dJK[:, 1] - dJL[:, 1] * dJK[:, 0],
        ],
        axis=1,
    )
    # t2 = dJI x dJL
    t2 = mx.stack(
        [
            dJI[:, 1] * dJL[:, 2] - dJI[:, 2] * dJL[:, 1],
            dJI[:, 2] * dJL[:, 0] - dJI[:, 0] * dJL[:, 2],
            dJI[:, 0] * dJL[:, 1] - dJI[:, 1] * dJL[:, 0],
        ],
        axis=1,
    )
    # t3 = dJK x dJI
    t3 = mx.stack(
        [
            dJK[:, 1] * dJI[:, 2] - dJK[:, 2] * dJI[:, 1],
            dJK[:, 2] * dJI[:, 0] - dJK[:, 0] * dJI[:, 2],
            dJK[:, 0] * dJI[:, 1] - dJK[:, 1] * dJI[:, 0],
        ],
        axis=1,
    )

    tg1 = (t1 * term1[:, None] - (dJI - dJK * cos_theta[:, None]) * term2[:, None]) * inv_dJI[:, None]
    tg3 = (t2 * term1[:, None] - (dJK - dJI * cos_theta[:, None]) * term2[:, None]) * inv_dJK[:, None]
    tg4 = (t3 * term1[:, None] - dJL * (sin_chi * inv_cos_chi)[:, None]) * inv_dJL[:, None]

    g1 = dE_dChi[:, None] * tg1
    g3 = dE_dChi[:, None] * tg3
    g4 = dE_dChi[:, None] * tg4
    g2 = -(g1 + g3 + g4)

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g1[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(g2[:, d_idx])
        grad = grad.at[idx3 * 3 + d_idx].add(g3[:, d_idx])
        grad = grad.at[idx4 * 3 + d_idx].add(g4[:, d_idx])
    return grad


# =====================
# Torsion
# =====================


def _torsion_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    V1: mx.array,
    V2: mx.array,
    V3: mx.array,
) -> mx.array:
    """Compute MMFF torsion energy per term.

    E = 0.5 * (V1*(1+cosφ) + V2*(1-cos2φ) + V3*(1+cos3φ))
    """
    pos_r = pos.reshape(-1, 3)
    p1, p2, p3, p4 = pos_r[idx1], pos_r[idx2], pos_r[idx3], pos_r[idx4]

    r1 = p1 - p2
    r2 = p3 - p2
    r3 = -r2
    r4 = p4 - p3

    # cross products
    t1 = mx.stack(
        [
            r1[:, 1] * r2[:, 2] - r1[:, 2] * r2[:, 1],
            r1[:, 2] * r2[:, 0] - r1[:, 0] * r2[:, 2],
            r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0],
        ],
        axis=1,
    )
    t2 = mx.stack(
        [
            r3[:, 1] * r4[:, 2] - r3[:, 2] * r4[:, 1],
            r3[:, 2] * r4[:, 0] - r3[:, 0] * r4[:, 2],
            r3[:, 0] * r4[:, 1] - r3[:, 1] * r4[:, 0],
        ],
        axis=1,
    )

    t1_sq = mx.sum(t1 * t1, axis=1)
    t2_sq = mx.sum(t2 * t2, axis=1)
    combined = t1_sq * t2_sq
    inv_len = mx.rsqrt(mx.maximum(combined, 1e-16))
    cos_phi = mx.clip(mx.sum(t1 * t2, axis=1) * inv_len, -1.0, 1.0)
    cos_phi = mx.where(combined < 1e-16, 1.0, cos_phi)

    cos2_phi = 2.0 * cos_phi * cos_phi - 1.0
    cos3_phi = cos_phi * (4.0 * cos_phi * cos_phi - 3.0)

    return 0.5 * (V1 * (1.0 + cos_phi) + V2 * (1.0 - cos2_phi) + V3 * (1.0 + cos3_phi))


def _torsion_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    idx3: mx.array,
    idx4: mx.array,
    V1: mx.array,
    V2: mx.array,
    V3: mx.array,
) -> mx.array:
    """Compute MMFF torsion gradient.

    Port of nvMolKit torsionGrad.
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    p1, p2, p3, p4 = pos_r[idx1], pos_r[idx2], pos_r[idx3], pos_r[idx4]

    dx1 = p1 - p2  # r1
    dx2 = p3 - p2  # r2
    dx4 = p4 - p3  # r4

    # cross1 = r1 x r2
    cross1 = mx.stack(
        [
            dx1[:, 1] * dx2[:, 2] - dx1[:, 2] * dx2[:, 1],
            dx1[:, 2] * dx2[:, 0] - dx1[:, 0] * dx2[:, 2],
            dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0],
        ],
        axis=1,
    )
    inv_norm1 = mx.minimum(mx.rsqrt(mx.maximum(mx.sum(cross1 * cross1, axis=1), 1e-30)), 1e5)
    cross1 = cross1 * inv_norm1[:, None]

    # cross2 = (-r2) x r4
    cross2 = mx.stack(
        [
            (-dx2[:, 1]) * dx4[:, 2] - (-dx2[:, 2]) * dx4[:, 1],
            (-dx2[:, 2]) * dx4[:, 0] - (-dx2[:, 0]) * dx4[:, 2],
            (-dx2[:, 0]) * dx4[:, 1] - (-dx2[:, 1]) * dx4[:, 0],
        ],
        axis=1,
    )
    inv_norm2 = mx.minimum(mx.rsqrt(mx.maximum(mx.sum(cross2 * cross2, axis=1), 1e-30)), 1e5)
    cross2 = cross2 * inv_norm2[:, None]

    cos_phi = mx.clip(mx.sum(cross1 * cross2, axis=1), -1.0, 1.0)
    sin_phi_sq = 1.0 - cos_phi * cos_phi

    # sinTerm = dE/d(cosPhi) when sinPhi != 0
    sin2_phi = 2.0 * cos_phi
    sin3_phi = 3.0 - 4.0 * sin_phi_sq
    sin_term = mx.where(
        sin_phi_sq > 0.0,
        0.5 * (V1 - 2.0 * V2 * sin2_phi + 3.0 * V3 * sin3_phi),
        0.0,
    )

    # dCos/dT0 = invNorm1 * (cross2 - cosPhi * cross1)
    dCos_dT0 = inv_norm1[:, None] * (cross2 - cos_phi[:, None] * cross1)
    # dCos/dT1 = invNorm2 * (cross1 - cosPhi * cross2)
    dCos_dT1 = inv_norm2[:, None] * (cross1 - cos_phi[:, None] * cross2)

    # Atom 1: sinTerm * (dCos_dT0 x r2) — note: this is dCos_dT0 cross r2 with swapped order
    g1 = sin_term[:, None] * mx.stack(
        [
            dCos_dT0[:, 2] * dx2[:, 1] - dCos_dT0[:, 1] * dx2[:, 2],
            dCos_dT0[:, 0] * dx2[:, 2] - dCos_dT0[:, 2] * dx2[:, 0],
            dCos_dT0[:, 1] * dx2[:, 0] - dCos_dT0[:, 0] * dx2[:, 1],
        ],
        axis=1,
    )

    # Atom 4: sinTerm * (dCos_dT1 x (-r2)) with specific ordering
    g4 = sin_term[:, None] * mx.stack(
        [
            dCos_dT1[:, 1] * (-dx2[:, 2]) - dCos_dT1[:, 2] * (-dx2[:, 1]),
            dCos_dT1[:, 2] * (-dx2[:, 0]) - dCos_dT1[:, 0] * (-dx2[:, 2]),
            dCos_dT1[:, 0] * (-dx2[:, 1]) - dCos_dT1[:, 1] * (-dx2[:, 0]),
        ],
        axis=1,
    )

    # Atom 2: complex, involves both cross product derivatives
    g2 = sin_term[:, None] * mx.stack(
        [
            dCos_dT0[:, 1] * (dx2[:, 2] - dx1[:, 2])
            + dCos_dT0[:, 2] * (dx1[:, 1] - dx2[:, 1])
            + dCos_dT1[:, 1] * (-dx4[:, 2])
            + dCos_dT1[:, 2] * dx4[:, 1],
            dCos_dT0[:, 0] * (dx1[:, 2] - dx2[:, 2])
            + dCos_dT0[:, 2] * (dx2[:, 0] - dx1[:, 0])
            + dCos_dT1[:, 0] * dx4[:, 2]
            + dCos_dT1[:, 2] * (-dx4[:, 0]),
            dCos_dT0[:, 0] * (dx2[:, 1] - dx1[:, 1])
            + dCos_dT0[:, 1] * (dx1[:, 0] - dx2[:, 0])
            + dCos_dT1[:, 0] * (-dx4[:, 1])
            + dCos_dT1[:, 1] * dx4[:, 0],
        ],
        axis=1,
    )

    # Atom 3
    g3 = sin_term[:, None] * mx.stack(
        [
            dCos_dT0[:, 1] * dx1[:, 2]
            + dCos_dT0[:, 2] * (-dx1[:, 1])
            + dCos_dT1[:, 1] * (dx4[:, 2] + dx2[:, 2])
            + dCos_dT1[:, 2] * (-dx4[:, 1] - dx2[:, 1]),
            dCos_dT0[:, 0] * (-dx1[:, 2])
            + dCos_dT0[:, 2] * dx1[:, 0]
            + dCos_dT1[:, 0] * (-dx4[:, 2] - dx2[:, 2])
            + dCos_dT1[:, 2] * (dx4[:, 0] + dx2[:, 0]),
            dCos_dT0[:, 0] * dx1[:, 1]
            + dCos_dT0[:, 1] * (-dx1[:, 0])
            + dCos_dT1[:, 0] * (dx4[:, 1] + dx2[:, 1])
            + dCos_dT1[:, 1] * (-dx4[:, 0] - dx2[:, 0]),
        ],
        axis=1,
    )

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g1[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(g2[:, d_idx])
        grad = grad.at[idx3 * 3 + d_idx].add(g3[:, d_idx])
        grad = grad.at[idx4 * 3 + d_idx].add(g4[:, d_idx])
    return grad


# =====================
# Van der Waals (Buffered 14-7)
# =====================


def _vdw_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    R_star: mx.array,
    epsilon: mx.array,
) -> mx.array:
    """Compute MMFF Van der Waals energy (Buffered 14-7) per term.

    E = ε * (1.07*R*/(d + 0.07*R*))^7 * (1.12*R*^7/(d^7 + 0.12*R*^7) - 2)
    """
    pos_r = pos.reshape(-1, 3)
    diff = pos_r[idx1] - pos_r[idx2]
    d = mx.sqrt(mx.maximum(mx.sum(diff * diff, axis=1), 1e-16))

    rho = d / mx.maximum(R_star, 1e-8)
    rho7 = rho * rho * rho * rho * rho * rho * rho

    t = 1.07 / (rho + 0.07)
    t7 = t * t * t * t * t * t * t

    return epsilon * t7 * (1.12 / (rho7 + 0.12) - 2.0)


def _vdw_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    R_star: mx.array,
    epsilon: mx.array,
) -> mx.array:
    """Compute MMFF Van der Waals gradient.

    Port of nvMolKit vDWGrad.
    """
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    diff = pos_r[idx1] - pos_r[idx2]
    d2 = mx.sum(diff * diff, axis=1)
    d = mx.sqrt(mx.maximum(d2, 1e-16))
    inv_d = mx.where(d > 1e-8, 1.0 / d, 0.0)

    inv_R_star = 1.0 / mx.maximum(R_star, 1e-8)
    q = d * inv_R_star
    q2 = q * q
    q6 = q2 * q2 * q2
    q7 = q6 * q
    q7p = q7 + 0.12  # q^7 + vdw2m1
    inv_q7p = 1.0 / mx.maximum(q7p, 1e-30)

    t = 1.07 / (q + 0.07)
    t2 = t * t
    t7 = t2 * t2 * t2 * t

    dE_dr = epsilon * inv_R_star * t7 * (
        -7.84 * q6 * inv_q7p * inv_q7p
        + (-7.84 * inv_q7p + 14.0) / (q + 0.07)
    )

    g = (dE_dr * inv_d)[:, None] * diff

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(-g[:, d_idx])
    return grad


# =====================
# Electrostatic
# =====================


def _ele_energy(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    charge_term: mx.array,
    diel_model: mx.array,
    is_1_4: mx.array,
) -> mx.array:
    """Compute MMFF electrostatic energy per term.

    E = 332.0716 * charge_term / (d + 0.05)^n * scale
    n = diel_model (1 or 2), scale = 0.75 for 1,4 pairs
    """
    pos_r = pos.reshape(-1, 3)
    diff = pos_r[idx1] - pos_r[idx2]
    d = mx.sqrt(mx.maximum(mx.sum(diff * diff, axis=1), 1e-16))

    d_plus_delta = d + 0.05
    # n=1: 1/(d+δ), n=2: 1/(d+δ)²
    is_model_2 = (diel_model.astype(mx.float32) > 1.5)
    denom = mx.where(is_model_2, d_plus_delta * d_plus_delta, d_plus_delta)
    scale = mx.where(is_1_4.astype(mx.float32) > 0.5, 0.75, 1.0)

    return 332.0716 * charge_term * scale / denom


def _ele_grad(
    pos: mx.array,
    idx1: mx.array,
    idx2: mx.array,
    charge_term: mx.array,
    diel_model: mx.array,
    is_1_4: mx.array,
) -> mx.array:
    """Compute MMFF electrostatic gradient."""
    pos_r = pos.reshape(-1, 3)
    n_atoms = pos_r.shape[0]
    diff = pos_r[idx1] - pos_r[idx2]
    d2 = mx.sum(diff * diff, axis=1)
    d = mx.sqrt(mx.maximum(d2, 1e-16))
    inv_d = mx.where(d > 1e-8, 1.0 / d, 0.0)

    d_plus_delta = d + 0.05
    n = diel_model.astype(mx.float32)
    scale = mx.where(is_1_4.astype(mx.float32) > 0.5, 0.75, 1.0)

    # dE/dd = -332.0716 * n * charge_term * scale / (d + δ)^(n+1)
    is_model_2 = n > 1.5
    denom_n_plus_1 = mx.where(
        is_model_2,
        d_plus_delta * d_plus_delta * d_plus_delta,
        d_plus_delta * d_plus_delta,
    )
    dE_dd = -332.0716 * n * charge_term * scale / mx.maximum(denom_n_plus_1, 1e-30)

    g = (dE_dd * inv_d)[:, None] * diff

    grad = mx.zeros((n_atoms * 3,), dtype=pos.dtype)
    for d_idx in range(3):
        grad = grad.at[idx1 * 3 + d_idx].add(g[:, d_idx])
        grad = grad.at[idx2 * 3 + d_idx].add(-g[:, d_idx])
    return grad


# =====================
# Combined Energy/Gradient
# =====================


def mmff_energy_and_grad(
    pos: mx.array,
    system: BatchedMMFFSystem,
) -> tuple[mx.array, mx.array]:
    """Compute total MMFF94 energy and gradient.

    Args:
        pos: Flat positions, shape (n_atoms_total * 3,), float32.
        system: Batched MMFF system with all terms.

    Returns:
        (per_mol_energies, flat_gradient) where energies is (n_mols,)
        and gradient is same shape as pos.
    """
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)
    grad = mx.zeros_like(pos)

    # Bond stretch
    if system.bond_idx1.size > 0:
        e = _bond_stretch_energy(
            pos, system.bond_idx1, system.bond_idx2,
            system.bond_kb, system.bond_r0,
        )
        energies = energies.at[system.bond_mol_indices].add(e)
        grad = grad + _bond_stretch_grad(
            pos, system.bond_idx1, system.bond_idx2,
            system.bond_kb, system.bond_r0,
        )

    # Angle bend
    if system.angle_idx1.size > 0:
        e = _angle_bend_energy(
            pos, system.angle_idx1, system.angle_idx2, system.angle_idx3,
            system.angle_ka, system.angle_theta0, system.angle_is_linear,
        )
        energies = energies.at[system.angle_mol_indices].add(e)
        grad = grad + _angle_bend_grad(
            pos, system.angle_idx1, system.angle_idx2, system.angle_idx3,
            system.angle_ka, system.angle_theta0, system.angle_is_linear,
        )

    # Stretch-bend
    if system.sb_idx1.size > 0:
        e = _stretch_bend_energy(
            pos, system.sb_idx1, system.sb_idx2, system.sb_idx3,
            system.sb_r0_ij, system.sb_r0_kj, system.sb_theta0,
            system.sb_kba_ij, system.sb_kba_kj,
        )
        energies = energies.at[system.sb_mol_indices].add(e)
        grad = grad + _stretch_bend_grad(
            pos, system.sb_idx1, system.sb_idx2, system.sb_idx3,
            system.sb_r0_ij, system.sb_r0_kj, system.sb_theta0,
            system.sb_kba_ij, system.sb_kba_kj,
        )

    # Out-of-plane bend
    if system.oop_idx1.size > 0:
        e = _oop_bend_energy(
            pos, system.oop_idx1, system.oop_idx2,
            system.oop_idx3, system.oop_idx4, system.oop_koop,
        )
        energies = energies.at[system.oop_mol_indices].add(e)
        grad = grad + _oop_bend_grad(
            pos, system.oop_idx1, system.oop_idx2,
            system.oop_idx3, system.oop_idx4, system.oop_koop,
        )

    # Torsion
    if system.tor_idx1.size > 0:
        e = _torsion_energy(
            pos, system.tor_idx1, system.tor_idx2,
            system.tor_idx3, system.tor_idx4,
            system.tor_V1, system.tor_V2, system.tor_V3,
        )
        energies = energies.at[system.tor_mol_indices].add(e)
        grad = grad + _torsion_grad(
            pos, system.tor_idx1, system.tor_idx2,
            system.tor_idx3, system.tor_idx4,
            system.tor_V1, system.tor_V2, system.tor_V3,
        )

    # Van der Waals
    if system.vdw_idx1.size > 0:
        e = _vdw_energy(
            pos, system.vdw_idx1, system.vdw_idx2,
            system.vdw_R_star, system.vdw_epsilon,
        )
        energies = energies.at[system.vdw_mol_indices].add(e)
        grad = grad + _vdw_grad(
            pos, system.vdw_idx1, system.vdw_idx2,
            system.vdw_R_star, system.vdw_epsilon,
        )

    # Electrostatic
    if system.ele_idx1.size > 0:
        e = _ele_energy(
            pos, system.ele_idx1, system.ele_idx2,
            system.ele_charge_term, system.ele_diel_model, system.ele_is_1_4,
        )
        energies = energies.at[system.ele_mol_indices].add(e)
        grad = grad + _ele_grad(
            pos, system.ele_idx1, system.ele_idx2,
            system.ele_charge_term, system.ele_diel_model, system.ele_is_1_4,
        )

    return energies, grad
