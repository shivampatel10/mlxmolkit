"""3D ETK energy and gradient functions for ETKDG.

All computations use float32 MLX arrays. Manual gradients are used
(not mx.grad) for numerical fidelity and scatter-add compatibility.

Port of nvMolKit's dist_geom_kernels_device.cuh (lines 237-830).

Energy terms:
  - Experimental torsion (6-term Fourier cosine series)
  - Inversion/improper (planar enforcement)
  - Distance constraint (flat-bottom penalty)
  - Angle constraint (triple bond linearity)
"""

import math

import mlx.core as mx


# ---------------------
# Torsion Angle Energy
# ---------------------

def _calc_torsion_cos_phi(pos, idx1, idx2, idx3, idx4):
    """Compute cos(phi) for torsion angle defined by 4 atoms.

    Uses only xyz (first 3) coordinates regardless of pos dimension.
    Torsion angle is the dihedral angle between planes (1,2,3) and (2,3,4).

    Returns:
        cos_phi: (n_terms,) float32
    """
    # Work in 3D only — pos may be 4D but torsion uses xyz
    dim = pos.shape[1] if pos.ndim == 2 else 3
    pos_r = pos.reshape(-1, dim)

    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]
    p4 = pos_r[idx4][:, :3]

    r1 = p1 - p2  # (N, 3)
    r2 = p3 - p2
    r3 = -r2      # p2 - p3
    r4 = p4 - p3

    # t1 = r1 x r2
    t1 = mx.stack([
        r1[:, 1] * r2[:, 2] - r1[:, 2] * r2[:, 1],
        r1[:, 2] * r2[:, 0] - r1[:, 0] * r2[:, 2],
        r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0],
    ], axis=1)

    # t2 = r3 x r4
    t2 = mx.stack([
        r3[:, 1] * r4[:, 2] - r3[:, 2] * r4[:, 1],
        r3[:, 2] * r4[:, 0] - r3[:, 0] * r4[:, 2],
        r3[:, 0] * r4[:, 1] - r3[:, 1] * r4[:, 0],
    ], axis=1)

    t1_len_sq = mx.sum(t1 * t1, axis=1)
    t2_len_sq = mx.sum(t2 * t2, axis=1)
    combined = t1_len_sq * t2_len_sq

    inv_len = mx.rsqrt(mx.maximum(combined, 1e-16))
    dot = mx.sum(t1 * t2, axis=1)
    cos_phi = mx.clip(dot * inv_len, -1.0, 1.0)
    # Zero out degenerate cases
    cos_phi = mx.where(combined < 1e-16, 0.0, cos_phi)

    return cos_phi


def _calc_torsion_energy_m6(fc, signs, cos_phi):
    """Compute 6-term Fourier torsion energy.

    E = sum_k fc[k] * (1 + signs[k] * cos(k*phi))

    Uses Chebyshev recurrence for cos(n*phi) from cos(phi).

    Args:
        fc: (n_terms, 6) force constants
        signs: (n_terms, 6) sign factors
        cos_phi: (n_terms,) cosine of torsion angle

    Returns:
        (n_terms,) energy values
    """
    c = cos_phi
    c2 = c * c
    c3 = c * c2
    c4 = c * c3
    c5 = c * c4
    c6 = c * c5

    cos1 = c
    cos2 = 2.0 * c2 - 1.0
    cos3 = 4.0 * c3 - 3.0 * c
    cos4 = 8.0 * c4 - 8.0 * c2 + 1.0
    cos5 = 16.0 * c5 - 20.0 * c3 + 5.0 * c
    cos6 = 32.0 * c6 - 48.0 * c4 + 18.0 * c2 - 1.0

    e = (fc[:, 0] * (1.0 + signs[:, 0] * cos1) +
         fc[:, 1] * (1.0 + signs[:, 1] * cos2) +
         fc[:, 2] * (1.0 + signs[:, 2] * cos3) +
         fc[:, 3] * (1.0 + signs[:, 3] * cos4) +
         fc[:, 4] * (1.0 + signs[:, 4] * cos5) +
         fc[:, 5] * (1.0 + signs[:, 5] * cos6))

    return e


def torsion_angle_energy(pos, idx1, idx2, idx3, idx4, fc, signs, dim):
    """Compute experimental torsion angle energy for each term.

    Args:
        pos: Flat positions, shape (n_atoms * dim,), float32.
        idx1-idx4: Atom indices, shape (n_terms,), int32.
        fc: Force constants, shape (n_terms, 6), float32.
        signs: Sign factors, shape (n_terms, 6), int32.
        dim: Coordinate dimension.

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    pos_r = pos.reshape(-1, dim)
    cos_phi = _calc_torsion_cos_phi(pos_r, idx1, idx2, idx3, idx4)
    signs_f = signs.astype(mx.float32)
    return _calc_torsion_energy_m6(fc, signs_f, cos_phi)


def torsion_angle_grad(pos, idx1, idx2, idx3, idx4, fc, signs, dim):
    """Compute experimental torsion angle gradient.

    Port of nvMolKit's torsionAngleGrad (dist_geom_kernels_device.cuh:451-575).
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]

    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]
    p4 = pos_r[idx4][:, :3]

    r1 = p1 - p2
    r2 = p3 - p2
    r3 = -r2
    r4 = p4 - p3

    # t0 = r1 x r2
    t0 = mx.stack([
        r1[:, 1] * r2[:, 2] - r1[:, 2] * r2[:, 1],
        r1[:, 2] * r2[:, 0] - r1[:, 0] * r2[:, 2],
        r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0],
    ], axis=1)

    # t1 = r3 x r4
    t1 = mx.stack([
        r3[:, 1] * r4[:, 2] - r3[:, 2] * r4[:, 1],
        r3[:, 2] * r4[:, 0] - r3[:, 0] * r4[:, 2],
        r3[:, 0] * r4[:, 1] - r3[:, 1] * r4[:, 0],
    ], axis=1)

    d02 = mx.sum(t0 * t0, axis=1)
    d12 = mx.sum(t1 * t1, axis=1)

    degenerate = (d02 < 1e-16) | (d12 < 1e-16)

    inv_d0 = mx.rsqrt(mx.maximum(d02, 1e-16))
    inv_d1 = mx.rsqrt(mx.maximum(d12, 1e-16))

    # Normalize
    t0n = t0 * inv_d0[:, None]
    t1n = t1 * inv_d1[:, None]

    cos_phi = mx.clip(mx.sum(t0n * t1n, axis=1), -1.0, 1.0)

    sin_phi_sq = 1.0 - cos_phi * cos_phi
    sin_phi = mx.sqrt(mx.maximum(sin_phi_sq, 0.0))

    # dE/dPhi from the M6 formula
    signs_f = signs.astype(mx.float32)
    c = cos_phi
    c2 = c * c
    c3 = c * c2
    c4 = c * c3

    dE_dPhi = (
        -signs_f[:, 0] * fc[:, 0] * sin_phi
        - 2.0 * signs_f[:, 1] * fc[:, 1] * (2.0 * c * sin_phi)
        - 3.0 * signs_f[:, 2] * fc[:, 2] * (4.0 * c2 * sin_phi - sin_phi)
        - 4.0 * signs_f[:, 3] * fc[:, 3] * (8.0 * c3 * sin_phi - 4.0 * c * sin_phi)
        - 5.0 * signs_f[:, 4] * fc[:, 4] * (16.0 * c4 * sin_phi - 12.0 * c2 * sin_phi + sin_phi)
        - 6.0 * signs_f[:, 5] * fc[:, 5] * (32.0 * c4 * c * sin_phi - 32.0 * c3 * sin_phi + 6.0 * sin_phi)
    )

    # sinTerm = -dE_dPhi / sin_phi  (or / cos_phi if sin_phi ~ 0)
    sin_term = mx.where(
        mx.abs(sin_phi) > 1e-8,
        -dE_dPhi / mx.maximum(mx.abs(sin_phi), 1e-16) * mx.sign(sin_phi + 1e-30),
        -dE_dPhi / mx.maximum(mx.abs(cos_phi), 1e-16) * mx.sign(cos_phi + 1e-30),
    )
    sin_term = mx.where(degenerate, 0.0, sin_term)

    # dCos/dT0 = inv_d0 * (t1n - cos_phi * t0n)
    dCos_dT0 = inv_d0[:, None] * (t1n - cos_phi[:, None] * t0n)
    # dCos/dT1 = inv_d1 * (t0n - cos_phi * t1n)
    dCos_dT1 = inv_d1[:, None] * (t0n - cos_phi[:, None] * t1n)

    # Atom 1: sin_term * (dCos_dT0 x r2)
    g1 = sin_term[:, None] * mx.stack([
        dCos_dT0[:, 2] * r2[:, 1] - dCos_dT0[:, 1] * r2[:, 2],
        dCos_dT0[:, 0] * r2[:, 2] - dCos_dT0[:, 2] * r2[:, 0],
        dCos_dT0[:, 1] * r2[:, 0] - dCos_dT0[:, 0] * r2[:, 1],
    ], axis=1)

    # Atom 4: sin_term * (dCos_dT1 x r3)
    g4 = sin_term[:, None] * mx.stack([
        dCos_dT1[:, 1] * r3[:, 2] - dCos_dT1[:, 2] * r3[:, 1],
        dCos_dT1[:, 2] * r3[:, 0] - dCos_dT1[:, 0] * r3[:, 2],
        dCos_dT1[:, 0] * r3[:, 1] - dCos_dT1[:, 1] * r3[:, 0],
    ], axis=1)

    # Atom 2: complex, involves both cross product derivatives
    g2 = sin_term[:, None] * mx.stack([
        dCos_dT0[:, 1] * (r2[:, 2] - r1[:, 2]) + dCos_dT0[:, 2] * (r1[:, 1] - r2[:, 1])
        + dCos_dT1[:, 1] * (-r4[:, 2]) + dCos_dT1[:, 2] * r4[:, 1],
        dCos_dT0[:, 0] * (r1[:, 2] - r2[:, 2]) + dCos_dT0[:, 2] * (r2[:, 0] - r1[:, 0])
        + dCos_dT1[:, 0] * r4[:, 2] + dCos_dT1[:, 2] * (-r4[:, 0]),
        dCos_dT0[:, 0] * (r2[:, 1] - r1[:, 1]) + dCos_dT0[:, 1] * (r1[:, 0] - r2[:, 0])
        + dCos_dT1[:, 0] * (-r4[:, 1]) + dCos_dT1[:, 1] * r4[:, 0],
    ], axis=1)

    # Atom 3: from the derivative chain
    g3 = sin_term[:, None] * mx.stack([
        dCos_dT0[:, 1] * r1[:, 2] + dCos_dT0[:, 2] * (-r1[:, 1])
        + dCos_dT1[:, 1] * (r4[:, 2] - r3[:, 2]) + dCos_dT1[:, 2] * (r3[:, 1] - r4[:, 1]),
        dCos_dT0[:, 0] * (-r1[:, 2]) + dCos_dT0[:, 2] * r1[:, 0]
        + dCos_dT1[:, 0] * (r3[:, 2] - r4[:, 2]) + dCos_dT1[:, 2] * (r4[:, 0] - r3[:, 0]),
        dCos_dT0[:, 0] * r1[:, 1] + dCos_dT0[:, 1] * (-r1[:, 0])
        + dCos_dT1[:, 0] * (r4[:, 1] - r3[:, 1]) + dCos_dT1[:, 1] * (r3[:, 0] - r4[:, 0]),
    ], axis=1)

    # Scatter-add to gradient (3D components only)
    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    for d in range(3):
        grad = grad.at[idx1 * dim + d].add(g1[:, d])
        grad = grad.at[idx2 * dim + d].add(g2[:, d])
        grad = grad.at[idx3 * dim + d].add(g3[:, d])
        grad = grad.at[idx4 * dim + d].add(g4[:, d])

    return grad


# ---------------------
# Inversion Energy
# ---------------------

def _calc_inversion_cos_y(pos_r, idx1, idx2, idx3, idx4):
    """Compute cosY for inversion (improper torsion).

    idx2 is the central atom. cosY measures out-of-plane angle.

    Returns:
        cos_y: (n_terms,) float32
    """
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]
    p4 = pos_r[idx4][:, :3]

    rJI = p1 - p2
    rJK = p3 - p2
    rJL = p4 - p2

    l2JI = mx.sum(rJI * rJI, axis=1)
    l2JK = mx.sum(rJK * rJK, axis=1)
    l2JL = mx.sum(rJL * rJL, axis=1)

    zero_tol = 1e-16
    degenerate = (l2JI < zero_tol) | (l2JK < zero_tol) | (l2JL < zero_tol)

    # n = rJI x rJK (not negated — matches nvMolKit which uses -rJI x rJK but then
    # the cosY sign is consistent with the energy formula)
    # Actually looking at nvMolKit: crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, ...)
    # which is (-rJI) x rJK = -(rJI x rJK) = rJK x rJI
    n = mx.stack([
        (-rJI[:, 1]) * rJK[:, 2] - (-rJI[:, 2]) * rJK[:, 1],
        (-rJI[:, 2]) * rJK[:, 0] - (-rJI[:, 0]) * rJK[:, 2],
        (-rJI[:, 0]) * rJK[:, 1] - (-rJI[:, 1]) * rJK[:, 0],
    ], axis=1)

    norm_factor = mx.rsqrt(mx.maximum(l2JI * l2JK, zero_tol))
    n = n * norm_factor[:, None]

    l2n = mx.sum(n * n, axis=1)
    degenerate = degenerate | (l2n < zero_tol)

    dot_n_rJL = mx.sum(n * rJL, axis=1)
    cos_y = dot_n_rJL * mx.rsqrt(mx.maximum(l2JL, zero_tol)) * mx.rsqrt(mx.maximum(l2n, zero_tol))
    cos_y = mx.clip(cos_y, -1.0, 1.0)
    cos_y = mx.where(degenerate, 0.0, cos_y)

    return cos_y


def inversion_energy(pos, idx1, idx2, idx3, idx4, C0, C1, C2, force_constant, dim):
    """Compute inversion (improper torsion) energy.

    E = forceConstant * (C0 + C1 * sinY + C2 * cos2W)
    where sinY = sqrt(1 - cosY^2), cos2W = 2*sinY^2 - 1

    Args:
        pos: Flat positions, shape (n_atoms * dim,), float32.
        idx1-idx4: Atom indices, shape (n_terms,), int32. idx2 is central.
        C0, C1, C2: Inversion coefficients, shape (n_terms,), float32.
        force_constant: Force constants, shape (n_terms,), float32.
        dim: Coordinate dimension.

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    pos_r = pos.reshape(-1, dim)
    cos_y = _calc_inversion_cos_y(pos_r, idx1, idx2, idx3, idx4)

    sin_y_sq = 1.0 - cos_y * cos_y
    sin_y = mx.where(sin_y_sq > 0.0, mx.sqrt(sin_y_sq), 0.0)
    cos_2w = 2.0 * sin_y * sin_y - 1.0

    return force_constant * (C0 + C1 * sin_y + C2 * cos_2w)


def inversion_grad(pos, idx1, idx2, idx3, idx4, C0, C1, C2, force_constant, dim):
    """Compute inversion (improper torsion) gradient.

    Port of nvMolKit's inversionGrad (dist_geom_kernels_device.cuh:577-704).
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]

    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]
    p4 = pos_r[idx4][:, :3]

    rJI = p1 - p2
    rJK = p3 - p2
    rJL = p4 - p2

    dJIsq = mx.sum(rJI * rJI, axis=1)
    dJKsq = mx.sum(rJK * rJK, axis=1)
    dJLsq = mx.sum(rJL * rJL, axis=1)

    zero_tol = 1e-16
    degenerate = (dJIsq < zero_tol) | (dJKsq < zero_tol) | (dJLsq < zero_tol)

    invdJI = mx.rsqrt(mx.maximum(dJIsq, zero_tol))
    invdJK = mx.rsqrt(mx.maximum(dJKsq, zero_tol))
    invdJL = mx.rsqrt(mx.maximum(dJLsq, zero_tol))

    # Normalize
    rJIn = rJI * invdJI[:, None]
    rJKn = rJK * invdJK[:, None]
    rJLn = rJL * invdJL[:, None]

    # n = (-rJI) x rJK
    n = mx.stack([
        (-rJIn[:, 1]) * rJKn[:, 2] - (-rJIn[:, 2]) * rJKn[:, 1],
        (-rJIn[:, 2]) * rJKn[:, 0] - (-rJIn[:, 0]) * rJKn[:, 2],
        (-rJIn[:, 0]) * rJKn[:, 1] - (-rJIn[:, 1]) * rJKn[:, 0],
    ], axis=1)

    inv_n_len = mx.rsqrt(mx.maximum(mx.sum(n * n, axis=1), zero_tol))
    nn = n * inv_n_len[:, None]

    cos_y = mx.clip(mx.sum(nn * rJLn, axis=1), -1.0, 1.0)

    sin_y_sq = 1.0 - cos_y * cos_y
    sin_y = mx.maximum(mx.sqrt(mx.maximum(sin_y_sq, 0.0)), 1e-8)

    cos_theta = mx.clip(mx.sum(rJIn * rJKn, axis=1), -1.0, 1.0)
    sin_theta_sq = 1.0 - cos_theta * cos_theta
    sin_theta = mx.maximum(mx.sqrt(mx.maximum(sin_theta_sq, 0.0)), 1e-8)

    # dE_dW = -fc * (C1*cosY - 4*C2*cosY*sinY)
    dE_dW = -force_constant * (C1 * cos_y - 4.0 * C2 * cos_y * sin_y)
    dE_dW = mx.where(degenerate, 0.0, dE_dW)

    # Cross products for gradient terms
    # t1 = rJL x rJK
    t1 = mx.stack([
        rJLn[:, 1] * rJKn[:, 2] - rJLn[:, 2] * rJKn[:, 1],
        rJLn[:, 2] * rJKn[:, 0] - rJLn[:, 0] * rJKn[:, 2],
        rJLn[:, 0] * rJKn[:, 1] - rJLn[:, 1] * rJKn[:, 0],
    ], axis=1)

    # t2 = rJI x rJL
    t2 = mx.stack([
        rJIn[:, 1] * rJLn[:, 2] - rJIn[:, 2] * rJLn[:, 1],
        rJIn[:, 2] * rJLn[:, 0] - rJIn[:, 0] * rJLn[:, 2],
        rJIn[:, 0] * rJLn[:, 1] - rJIn[:, 1] * rJLn[:, 0],
    ], axis=1)

    # t3 = rJK x rJI
    t3 = mx.stack([
        rJKn[:, 1] * rJIn[:, 2] - rJKn[:, 2] * rJIn[:, 1],
        rJKn[:, 2] * rJIn[:, 0] - rJKn[:, 0] * rJIn[:, 2],
        rJKn[:, 0] * rJIn[:, 1] - rJKn[:, 1] * rJIn[:, 0],
    ], axis=1)

    inverseTerm1 = 1.0 / (sin_y * sin_theta)
    term2 = cos_y / (sin_y * sin_theta_sq)
    cos_y_over_sin_y = cos_y / sin_y

    # Atom 1 gradient
    tg1 = (t1 * inverseTerm1[:, None] - (rJIn - rJKn * cos_theta[:, None]) * term2[:, None]) * invdJI[:, None]

    # Atom 3 gradient
    tg3 = (t2 * inverseTerm1[:, None] - (rJKn - rJIn * cos_theta[:, None]) * term2[:, None]) * invdJK[:, None]

    # Atom 4 gradient
    tg4 = (t3 * inverseTerm1[:, None] - rJLn * cos_y_over_sin_y[:, None]) * invdJL[:, None]

    g1 = dE_dW[:, None] * tg1
    g3 = dE_dW[:, None] * tg3
    g4 = dE_dW[:, None] * tg4
    g2 = -(g1 + g3 + g4)

    # Scatter-add to gradient
    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    for d in range(3):
        grad = grad.at[idx1 * dim + d].add(g1[:, d])
        grad = grad.at[idx2 * dim + d].add(g2[:, d])
        grad = grad.at[idx3 * dim + d].add(g3[:, d])
        grad = grad.at[idx4 * dim + d].add(g4[:, d])

    return grad


# ---------------------
# Distance Constraint
# ---------------------

def distance_constraint_energy(pos, idx1, idx2, min_len, max_len, force_constant, dim):
    """Compute flat-bottom distance constraint energy.

    E = 0.5 * fc * (d - bound)^2  if d < minLen or d > maxLen
    E = 0                          otherwise

    Args:
        pos: Flat positions, shape (n_atoms * dim,), float32.
        idx1, idx2: Atom indices, shape (n_terms,), int32.
        min_len, max_len: Distance bounds, shape (n_terms,), float32.
        force_constant: Shape (n_terms,), float32.
        dim: Coordinate dimension.

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    pos_r = pos.reshape(-1, dim)
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    diff = p1 - p2
    d2 = mx.sum(diff * diff, axis=1)
    d = mx.sqrt(mx.maximum(d2, 1e-16))

    min2 = min_len * min_len
    max2 = max_len * max_len

    diff_lo = min_len - d
    diff_hi = d - max_len

    e_lo = mx.where(d2 < min2, 0.5 * force_constant * diff_lo * diff_lo, 0.0)
    e_hi = mx.where(d2 > max2, 0.5 * force_constant * diff_hi * diff_hi, 0.0)

    return e_lo + e_hi


def distance_constraint_grad(pos, idx1, idx2, min_len, max_len, force_constant, dim):
    """Compute distance constraint gradient.

    Port of nvMolKit's distanceConstraintGrad.
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    diff = p1 - p2
    d2 = mx.sum(diff * diff, axis=1)
    d = mx.sqrt(mx.maximum(d2, 1e-16))

    min2 = min_len * min_len
    max2 = max_len * max_len

    # preFactor = fc * (d - bound) / d
    pf_lo = mx.where(d2 < min2, force_constant * (d - min_len) / mx.maximum(d, 1e-8), 0.0)
    pf_hi = mx.where(d2 > max2, force_constant * (d - max_len) / mx.maximum(d, 1e-8), 0.0)
    pf = pf_lo + pf_hi

    # grad contribution for each xyz component
    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    for d_idx in range(3):
        g = pf * diff[:, d_idx]
        grad = grad.at[idx1 * dim + d_idx].add(g)
        grad = grad.at[idx2 * dim + d_idx].add(-g)

    return grad


# ---------------------
# Angle Constraint
# ---------------------

RAD2DEG = 180.0 / math.pi


def angle_constraint_energy(pos, idx1, idx2, idx3, min_angle, max_angle,
                            force_constant, dim):
    """Compute angle constraint energy for triple bonds.

    E = fc * angleTerm^2  where angleTerm = angle - bound if outside [min, max]

    Angles in degrees.

    Args:
        pos: Flat positions, shape (n_atoms * dim,), float32.
        idx1, idx2 (central), idx3: Atom indices, shape (n_terms,), int32.
        min_angle, max_angle: Angle bounds in degrees, shape (n_terms,), float32.
        force_constant: Shape (n_terms,) or scalar, float32.
        dim: Coordinate dimension.

    Returns:
        Per-term energies, shape (n_terms,), float32.
    """
    pos_r = pos.reshape(-1, dim)
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]

    r1 = p1 - p2
    r2 = p3 - p2

    d1_sq = mx.sum(r1 * r1, axis=1)
    d2_sq = mx.sum(r2 * r2, axis=1)
    dist_term = d1_sq * d2_sq

    dot = mx.sum(r1 * r2, axis=1)
    cos_theta = mx.clip(dot * mx.rsqrt(mx.maximum(dist_term, 1e-16)), -1.0, 1.0)
    angle_deg = RAD2DEG * mx.arccos(cos_theta)

    # Flat-bottom: compute deviation from bounds
    angle_term = mx.where(angle_deg < min_angle, angle_deg - min_angle,
                          mx.where(angle_deg > max_angle, angle_deg - max_angle, 0.0))

    degenerate = dist_term < 1e-16
    angle_term = mx.where(degenerate, 0.0, angle_term)

    return force_constant * angle_term * angle_term


def angle_constraint_grad(pos, idx1, idx2, idx3, min_angle, max_angle,
                          force_constant, dim):
    """Compute angle constraint gradient.

    Port of nvMolKit's angleConstraintGrad.
    """
    pos_r = pos.reshape(-1, dim)
    n_atoms = pos_r.shape[0]
    p1 = pos_r[idx1][:, :3]
    p2 = pos_r[idx2][:, :3]
    p3 = pos_r[idx3][:, :3]

    r1 = p1 - p2
    r2 = p3 - p2

    r1_len_sq = mx.maximum(mx.sum(r1 * r1, axis=1), 1e-5)
    r2_len_sq = mx.maximum(mx.sum(r2 * r2, axis=1), 1e-5)
    denom = mx.rsqrt(r1_len_sq * r2_len_sq)

    dot = mx.sum(r1 * r2, axis=1)
    cos_theta = mx.clip(dot * denom, -1.0, 1.0)
    angle_deg = RAD2DEG * mx.arccos(cos_theta)

    angle_term = mx.where(angle_deg < min_angle, angle_deg - min_angle,
                          mx.where(angle_deg > max_angle, angle_deg - max_angle, 0.0))

    dE_dTheta = 2.0 * RAD2DEG * force_constant * angle_term

    # rp = r2 x r1
    rp = mx.stack([
        r2[:, 1] * r1[:, 2] - r2[:, 2] * r1[:, 1],
        r2[:, 2] * r1[:, 0] - r2[:, 0] * r1[:, 2],
        r2[:, 0] * r1[:, 1] - r2[:, 1] * r1[:, 0],
    ], axis=1)

    rp_len_sq = mx.maximum(mx.sum(rp * rp, axis=1), 1e-10)
    rp_len_inv = mx.rsqrt(rp_len_sq)
    prefactor = dE_dTheta * rp_len_inv

    t1 = -prefactor / r1_len_sq
    t2 = prefactor / r2_len_sq

    # dedp1 = t1 * (r1 x rp)
    dedp1 = mx.stack([
        r1[:, 1] * rp[:, 2] - r1[:, 2] * rp[:, 1],
        r1[:, 2] * rp[:, 0] - r1[:, 0] * rp[:, 2],
        r1[:, 0] * rp[:, 1] - r1[:, 1] * rp[:, 0],
    ], axis=1) * t1[:, None]

    # dedp3 = t2 * (r2 x rp)
    dedp3 = mx.stack([
        r2[:, 1] * rp[:, 2] - r2[:, 2] * rp[:, 1],
        r2[:, 2] * rp[:, 0] - r2[:, 0] * rp[:, 2],
        r2[:, 0] * rp[:, 1] - r2[:, 1] * rp[:, 0],
    ], axis=1) * t2[:, None]

    dedp2 = -(dedp1 + dedp3)

    grad = mx.zeros((n_atoms * dim,), dtype=pos.dtype)
    for d in range(3):
        grad = grad.at[idx1 * dim + d].add(dedp1[:, d])
        grad = grad.at[idx2 * dim + d].add(dedp2[:, d])
        grad = grad.at[idx3 * dim + d].add(dedp3[:, d])

    return grad


# ---------------------
# Combined ETK Energy/Gradient
# ---------------------

def etk_energy(pos, system, use_basic_knowledge=True):
    """Compute total ETK energy per molecule.

    Args:
        pos: Flat positions, shape (n_atoms_total * dim,), float32.
        system: BatchedETKSystem with all terms.
        use_basic_knowledge: Whether to include improper torsion terms.

    Returns:
        Per-molecule energies, shape (n_mols,), float32.
    """
    dim = system.dim
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)

    # Experimental torsion
    if system.torsion_idx1.size > 0:
        e_torsion = torsion_angle_energy(
            pos, system.torsion_idx1, system.torsion_idx2,
            system.torsion_idx3, system.torsion_idx4,
            system.torsion_fc, system.torsion_signs, dim)
        energies = energies.at[system.torsion_mol_indices].add(e_torsion)

    # Improper torsion
    if use_basic_knowledge and system.improper_idx1.size > 0:
        e_improper = inversion_energy(
            pos, system.improper_idx1, system.improper_idx2,
            system.improper_idx3, system.improper_idx4,
            system.improper_C0, system.improper_C1, system.improper_C2,
            system.improper_fc, dim)
        energies = energies.at[system.improper_mol_indices].add(e_improper)

    # 1-2 distance constraints
    if system.dist12_idx1.size > 0:
        e_d12 = distance_constraint_energy(
            pos, system.dist12_idx1, system.dist12_idx2,
            system.dist12_min, system.dist12_max,
            system.dist12_fc, dim)
        energies = energies.at[system.dist12_mol_indices].add(e_d12)

    # 1-3 distance constraints
    if system.dist13_idx1.size > 0:
        e_d13 = distance_constraint_energy(
            pos, system.dist13_idx1, system.dist13_idx2,
            system.dist13_min, system.dist13_max,
            system.dist13_fc, dim)
        energies = energies.at[system.dist13_mol_indices].add(e_d13)

    # Angle constraints (triple bonds)
    if system.angle13_idx1.size > 0:
        e_ang = angle_constraint_energy(
            pos, system.angle13_idx1, system.angle13_idx2,
            system.angle13_idx3, system.angle13_min_angle,
            system.angle13_max_angle, system.angle13_fc, dim)
        energies = energies.at[system.angle13_mol_indices].add(e_ang)

    # Long-range distance constraints
    if system.long_range_idx1.size > 0:
        e_lr = distance_constraint_energy(
            pos, system.long_range_idx1, system.long_range_idx2,
            system.long_range_min, system.long_range_max,
            system.long_range_fc, dim)
        energies = energies.at[system.long_range_mol_indices].add(e_lr)

    return energies


def etk_energy_and_grad(pos, system, use_basic_knowledge=True):
    """Compute total ETK energy and gradient.

    Args:
        pos: Flat positions, shape (n_atoms_total * dim,), float32.
        system: BatchedETKSystem with all terms.
        use_basic_knowledge: Whether to include improper torsion terms.

    Returns:
        (energies, grad) where energies is (n_mols,) and grad is same shape as pos.
    """
    dim = system.dim
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)
    grad = mx.zeros_like(pos)

    # Experimental torsion
    if system.torsion_idx1.size > 0:
        e_torsion = torsion_angle_energy(
            pos, system.torsion_idx1, system.torsion_idx2,
            system.torsion_idx3, system.torsion_idx4,
            system.torsion_fc, system.torsion_signs, dim)
        energies = energies.at[system.torsion_mol_indices].add(e_torsion)
        g_torsion = torsion_angle_grad(
            pos, system.torsion_idx1, system.torsion_idx2,
            system.torsion_idx3, system.torsion_idx4,
            system.torsion_fc, system.torsion_signs, dim)
        grad = grad + g_torsion

    # Improper torsion
    if use_basic_knowledge and system.improper_idx1.size > 0:
        e_improper = inversion_energy(
            pos, system.improper_idx1, system.improper_idx2,
            system.improper_idx3, system.improper_idx4,
            system.improper_C0, system.improper_C1, system.improper_C2,
            system.improper_fc, dim)
        energies = energies.at[system.improper_mol_indices].add(e_improper)
        g_improper = inversion_grad(
            pos, system.improper_idx1, system.improper_idx2,
            system.improper_idx3, system.improper_idx4,
            system.improper_C0, system.improper_C1, system.improper_C2,
            system.improper_fc, dim)
        grad = grad + g_improper

    # 1-2 distance constraints
    if system.dist12_idx1.size > 0:
        e_d12 = distance_constraint_energy(
            pos, system.dist12_idx1, system.dist12_idx2,
            system.dist12_min, system.dist12_max,
            system.dist12_fc, dim)
        energies = energies.at[system.dist12_mol_indices].add(e_d12)
        g_d12 = distance_constraint_grad(
            pos, system.dist12_idx1, system.dist12_idx2,
            system.dist12_min, system.dist12_max,
            system.dist12_fc, dim)
        grad = grad + g_d12

    # 1-3 distance constraints
    if system.dist13_idx1.size > 0:
        e_d13 = distance_constraint_energy(
            pos, system.dist13_idx1, system.dist13_idx2,
            system.dist13_min, system.dist13_max,
            system.dist13_fc, dim)
        energies = energies.at[system.dist13_mol_indices].add(e_d13)
        g_d13 = distance_constraint_grad(
            pos, system.dist13_idx1, system.dist13_idx2,
            system.dist13_min, system.dist13_max,
            system.dist13_fc, dim)
        grad = grad + g_d13

    # Angle constraints
    if system.angle13_idx1.size > 0:
        e_ang = angle_constraint_energy(
            pos, system.angle13_idx1, system.angle13_idx2,
            system.angle13_idx3, system.angle13_min_angle,
            system.angle13_max_angle, system.angle13_fc, dim)
        energies = energies.at[system.angle13_mol_indices].add(e_ang)
        g_ang = angle_constraint_grad(
            pos, system.angle13_idx1, system.angle13_idx2,
            system.angle13_idx3, system.angle13_min_angle,
            system.angle13_max_angle, system.angle13_fc, dim)
        grad = grad + g_ang

    # Long-range distance constraints
    if system.long_range_idx1.size > 0:
        e_lr = distance_constraint_energy(
            pos, system.long_range_idx1, system.long_range_idx2,
            system.long_range_min, system.long_range_max,
            system.long_range_fc, dim)
        energies = energies.at[system.long_range_mol_indices].add(e_lr)
        g_lr = distance_constraint_grad(
            pos, system.long_range_idx1, system.long_range_idx2,
            system.long_range_min, system.long_range_max,
            system.long_range_fc, dim)
        grad = grad + g_lr

    return energies, grad


def compute_planar_energy(pos, system, dim):
    """Compute total improper torsion energy per molecule (for planar check).

    Returns:
        Per-molecule energies, shape (n_mols,), float32.
    """
    n_mols = system.n_mols
    energies = mx.zeros(n_mols, dtype=pos.dtype)

    if system.improper_idx1.size > 0:
        e_improper = inversion_energy(
            pos, system.improper_idx1, system.improper_idx2,
            system.improper_idx3, system.improper_idx4,
            system.improper_C0, system.improper_C1, system.improper_C2,
            system.improper_fc, dim)
        energies = energies.at[system.improper_mol_indices].add(e_improper)

    return energies
