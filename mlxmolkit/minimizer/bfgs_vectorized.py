"""Vectorized batched BFGS minimizer — no per-molecule Python loops.

Replaces the per-molecule loops in bfgs.py with padded batch tensor
operations. All molecules are processed simultaneously via (n_mols, max_dim)
padded arrays. The only .item() call per iteration is one
mx.any(active).item() to check early exit.

Key data structures:
    flat_to_padded_idx: (total_pos_size,) int32 — maps flat[i] -> padded row
    dim_mask: (n_mols, max_dim) bool — True for real dimensions
    H_batch: (n_mols, max_dim, max_dim) float32 — padded inverse Hessians
"""

from typing import Callable

import mlx.core as mx
import numpy as np

from .bfgs import (
    DEFAULT_GRAD_TOL,
    EPS,
    FUNCTOL,
    MAX_LINESEARCH_ITERS,
    MAX_STEP_FACTOR,
    MOVETOL,
    TOLX,
)


def _build_padding_maps(
    atom_starts_list: list[int],
    n_mols: int,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, int, list[int]]:
    """Build flat-to-padded index mapping and dimension mask.

    Args:
        atom_starts_list: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.

    Returns:
        Tuple of ``(flat_to_padded_idx, padded_to_flat_idx, dim_mask,
        max_dim, mol_dims)``.
    """
    mol_dims = [(atom_starts_list[i + 1] - atom_starts_list[i]) * dim
                for i in range(n_mols)]
    max_dim = max(mol_dims) if mol_dims else 0

    total_pos_size = atom_starts_list[-1] * dim

    # Build flat -> padded index mapping
    flat_to_padded = np.empty(total_pos_size, dtype=np.int32)
    padded_to_flat = np.full(n_mols * max_dim, -1, dtype=np.int32)
    dim_mask_np = np.zeros((n_mols, max_dim), dtype=bool)

    for mol_idx in range(n_mols):
        start = atom_starts_list[mol_idx] * dim
        d = mol_dims[mol_idx]
        for j in range(d):
            padded_idx = mol_idx * max_dim + j
            flat_to_padded[start + j] = padded_idx
            padded_to_flat[padded_idx] = start + j
            dim_mask_np[mol_idx, j] = True

    flat_to_padded_idx = mx.array(flat_to_padded)
    padded_to_flat_idx = mx.array(padded_to_flat)
    dim_mask = mx.array(dim_mask_np)

    return flat_to_padded_idx, padded_to_flat_idx, dim_mask, max_dim, mol_dims


def _flat_to_padded(
    flat_arr: mx.array,
    flat_to_padded_idx: mx.array,
    n_mols: int,
    max_dim: int,
) -> mx.array:
    """Convert flat array to padded ``(n_mols, max_dim)`` via scatter.

    Args:
        flat_arr: Flat input array.
        flat_to_padded_idx: Mapping from flat indices to padded indices.
        n_mols: Number of molecules.
        max_dim: Maximum coordinate dimension across molecules.

    Returns:
        Padded array of shape ``(n_mols, max_dim)``.
    """
    padded = mx.zeros(n_mols * max_dim, dtype=flat_arr.dtype)
    padded = padded.at[flat_to_padded_idx].add(flat_arr)
    return padded.reshape(n_mols, max_dim)


def _padded_to_flat(
    padded: mx.array,
    padded_to_flat_idx: mx.array,
    total_pos_size: int,
) -> mx.array:
    """Convert padded ``(n_mols, max_dim)`` to flat array via gather.

    Args:
        padded: Padded array of shape ``(n_mols, max_dim)``.
        padded_to_flat_idx: Mapping from padded indices to flat indices.
            Uses -1 for padding positions.
        total_pos_size: Length of the output flat array.

    Returns:
        Flat array of shape ``(total_pos_size,)``.
    """
    flat_padded = padded.reshape(-1)
    # Clamp -1 indices to 0, then zero out via mask
    safe_idx = mx.maximum(padded_to_flat_idx, 0)
    valid = padded_to_flat_idx >= 0
    values = flat_padded[mx.arange(flat_padded.shape[0])] * valid
    result = mx.zeros(total_pos_size, dtype=padded.dtype)
    result = result.at[safe_idx].add(values)
    return result


def _compute_max_step_vec(
    pos_padded: mx.array,
    dim_mask: mx.array,
    n_mols: int,
    max_dim: int,
    dim: int,
) -> mx.array:
    """Compute max step size per molecule in vectorized form.

    Args:
        pos_padded: Padded positions, shape ``(n_mols, max_dim)``.
        dim_mask: Boolean mask for valid dimensions, shape ``(n_mols, max_dim)``.
        n_mols: Number of molecules.
        max_dim: Maximum coordinate dimension across molecules.
        dim: Coordinate dimension.

    Returns:
        Per-molecule max step sizes, shape ``(n_mols,)``.
    """
    # sum of squares per molecule
    sum_sq = mx.sum(pos_padded * pos_padded * dim_mask, axis=1)  # (n_mols,)
    # n_terms per molecule = number of True entries in dim_mask
    n_terms = mx.sum(dim_mask.astype(mx.float32), axis=1)  # (n_mols,)
    return MAX_STEP_FACTOR * mx.maximum(mx.sqrt(sum_sq), n_terms)


def bfgs_minimize_vectorized(
    energy_and_grad_fn: Callable[[mx.array], tuple[mx.array, mx.array]],
    pos: mx.array,
    atom_starts_list: list[int] | mx.array,
    n_mols: int,
    dim: int,
    max_iters: int = 400,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Vectorized batched BFGS minimizer with no per-molecule Python loops.

    Args:
        energy_and_grad_fn: Callable taking flat positions and returning
            ``(energies, grad)`` with shapes ``(n_mols,)`` and ``(flat,)``.
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        atom_starts_list: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension (3 or 4).
        max_iters: Maximum BFGS iterations.
        grad_tol: Gradient convergence tolerance. Defaults to ``1e-3``.

    Returns:
        Tuple of ``(final_pos, final_energies, statuses)`` where statuses is
        ``(n_mols,)`` int with 0=converged, 1=not converged.
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    if isinstance(atom_starts_list, mx.array):
        atom_starts_list = atom_starts_list.tolist()
    else:
        atom_starts_list = list(atom_starts_list)

    total_pos_size = atom_starts_list[-1] * dim

    # Build padding maps
    (flat_to_padded_idx, padded_to_flat_idx, dim_mask,
     max_dim, mol_dims) = _build_padding_maps(atom_starts_list, n_mols, dim)

    if max_dim == 0:
        return pos, mx.zeros(n_mols, dtype=mx.float32), mx.zeros(n_mols, dtype=mx.int32)

    # Active mask (True = still optimizing)
    active = mx.ones(n_mols, dtype=mx.bool_)

    # Initialize inverse Hessians to identity (padded)
    H_batch = mx.broadcast_to(
        mx.eye(max_dim, dtype=mx.float32)[None, :, :],
        (n_mols, max_dim, max_dim),
    ) * 1  # force copy

    # Compute initial energy and gradient
    energies, grad_flat = energy_and_grad_fn(pos)
    mx.eval(energies, grad_flat)

    # Convert to padded
    grad_padded = _flat_to_padded(grad_flat, flat_to_padded_idx, n_mols, max_dim)
    pos_padded = _flat_to_padded(pos, flat_to_padded_idx, n_mols, max_dim)

    # Initial direction = -grad
    dir_padded = -grad_padded * dim_mask

    # Compute max step per molecule
    max_steps = _compute_max_step_vec(pos_padded, dim_mask, n_mols, max_dim, dim)
    mx.eval(H_batch, grad_padded, pos_padded, dir_padded, max_steps)

    # Grad scale = 1 (no scaling in vectorized version, matching pipeline usage)
    grad_scale = mx.ones(n_mols, dtype=mx.float32)

    # Main BFGS loop
    for iteration in range(max_iters):
        # Check if all converged — single .item() call per iteration
        if not mx.any(active).item():
            break

        # === LINE SEARCH (vectorized) ===
        (pos_padded, pos, energies, xi_padded, active) = _line_search_vec(
            energy_and_grad_fn, pos_padded, pos, grad_padded, dir_padded,
            energies, flat_to_padded_idx, padded_to_flat_idx, dim_mask,
            n_mols, max_dim, dim, total_pos_size, max_steps, active,
        )

        # === TOLX CHECK (vectorized) ===
        abs_xi = mx.abs(xi_padded)
        abs_pos = mx.maximum(mx.abs(pos_padded), 1.0)
        test_tolx = mx.max(abs_xi / abs_pos * dim_mask, axis=1)  # (n_mols,)
        converged_tolx = test_tolx < TOLX
        active = mx.where(converged_tolx & active, False, active)

        # === NEW GRADIENT ===
        old_grad_padded = grad_padded
        energies, grad_flat = energy_and_grad_fn(pos)
        mx.eval(energies, grad_flat)
        grad_padded = _flat_to_padded(grad_flat, flat_to_padded_idx, n_mols, max_dim)

        # === GRADIENT CONVERGENCE CHECK ===
        abs_grad = mx.abs(grad_padded) * mx.maximum(mx.abs(pos_padded), 1.0)
        test_grad = mx.max(abs_grad * dim_mask, axis=1)  # (n_mols,)
        denom = mx.maximum(energies * grad_scale, 1.0)
        test_grad = test_grad / denom
        converged_grad = test_grad < grad_tol
        active = mx.where(converged_grad & active, False, active)

        # === BFGS HESSIAN UPDATE (vectorized) ===
        d_grad_padded = grad_padded - old_grad_padded

        H_batch, dir_padded = _bfgs_update_vec(
            xi_padded, d_grad_padded, grad_padded, H_batch,
            dim_mask, n_mols, max_dim, active,
        )
        mx.eval(pos_padded, pos, energies, grad_padded, dir_padded, H_batch, active)

    # Convert active -> statuses (0=converged, 1=not)
    statuses = active.astype(mx.int32)

    return pos, energies, statuses


def _line_search_vec(
    energy_and_grad_fn: Callable[[mx.array], tuple[mx.array, mx.array]],
    pos_padded: mx.array,
    pos_flat: mx.array,
    grad_padded: mx.array,
    dir_padded: mx.array,
    energies: mx.array,
    flat_to_padded_idx: mx.array,
    padded_to_flat_idx: mx.array,
    dim_mask: mx.array,
    n_mols: int,
    max_dim: int,
    dim: int,
    total_pos_size: int,
    max_steps: mx.array,
    active: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Vectorized backtracking line search for all molecules simultaneously.

    Args:
        energy_and_grad_fn: Callable taking flat positions and returning
            ``(energies, grad)``.
        pos_padded: Padded positions, shape ``(n_mols, max_dim)``.
        pos_flat: Flat positions, shape ``(total_pos_size,)``.
        grad_padded: Padded gradient, shape ``(n_mols, max_dim)``.
        dir_padded: Padded search direction, shape ``(n_mols, max_dim)``.
        energies: Current per-molecule energies, shape ``(n_mols,)``.
        flat_to_padded_idx: Flat-to-padded index mapping.
        padded_to_flat_idx: Padded-to-flat index mapping.
        dim_mask: Boolean mask for valid dimensions, shape ``(n_mols, max_dim)``.
        n_mols: Number of molecules.
        max_dim: Maximum coordinate dimension across molecules.
        dim: Coordinate dimension.
        total_pos_size: Total flat position array length.
        max_steps: Per-molecule max step sizes, shape ``(n_mols,)``.
        active: Boolean mask of active molecules, shape ``(n_mols,)``.

    Returns:
        Tuple of ``(pos_padded, pos_flat, energies, xi_padded, active)``.
    """
    old_pos_padded = pos_padded
    old_pos_flat = pos_flat
    old_energies = energies

    # Scale direction if norm > max_step (vectorized)
    dir_sq = mx.sum(dir_padded * dir_padded * dim_mask, axis=1)  # (n_mols,)
    dir_norm = mx.sqrt(mx.maximum(dir_sq, 1e-30))
    scale_needed = dir_norm > max_steps
    scale_factor = mx.where(scale_needed, max_steps / dir_norm, 1.0)
    dir_scaled = dir_padded * scale_factor[:, None]

    # Compute slope = dir . grad (vectorized)
    slopes = mx.sum(dir_scaled * grad_padded * dim_mask, axis=1)  # (n_mols,)

    # Compute lambda_min (vectorized)
    abs_dir = mx.abs(dir_scaled)
    abs_pos = mx.maximum(mx.abs(pos_padded), 1.0)
    # Per-molecule max of abs_dir/abs_pos, masking padding
    test_vals = (abs_dir / abs_pos) * dim_mask
    test_max = mx.max(test_vals, axis=1)  # (n_mols,)
    lambda_mins = MOVETOL / mx.maximum(test_max, 1e-30)

    # Initialize line search state
    lambdas = mx.ones(n_mols, dtype=mx.float32)
    prev_lambdas = mx.ones(n_mols, dtype=mx.float32)
    prev_energies = old_energies
    best_pos_padded = old_pos_padded
    best_pos_flat = old_pos_flat

    # ls_active: True = still searching
    ls_active = active

    for ls_iter in range(MAX_LINESEARCH_ITERS):
        if not mx.any(ls_active).item():
            break

        # Check lambda_min
        too_small = lambdas < lambda_mins
        ls_active = mx.where(too_small & ls_active, False, ls_active)

        if not mx.any(ls_active).item():
            break

        # Trial positions: old_pos + lambda * dir_scaled
        trial_padded = old_pos_padded + lambdas[:, None] * dir_scaled
        trial_padded = mx.where(dim_mask, trial_padded, 0.0)

        # Convert to flat for energy evaluation
        trial_flat = _padded_to_flat(trial_padded, padded_to_flat_idx, total_pos_size)
        mx.eval(trial_flat)

        # Compute trial energy
        trial_energies, _ = energy_and_grad_fn(trial_flat)
        mx.eval(trial_energies)

        # Armijo condition: new_e - old_e <= FUNCTOL * lam * slope
        sufficient_decrease = (trial_energies - old_energies) <= FUNCTOL * lambdas * slopes
        accepted = sufficient_decrease & ls_active

        # Update best positions for accepted molecules
        best_pos_padded = mx.where(
            (accepted[:, None] & dim_mask), trial_padded, best_pos_padded
        )
        best_pos_flat = _padded_to_flat(best_pos_padded, padded_to_flat_idx, total_pos_size)

        ls_active = mx.where(accepted, False, ls_active)

        if not mx.any(ls_active).item():
            break

        # Backtrack: compute new lambda for rejected molecules
        is_first = ls_iter == 0

        if is_first:
            # Quadratic interpolation: -slope / (2 * (new_e - old_e - slope))
            quad_denom = 2.0 * (trial_energies - old_energies - slopes)
            tmp_lam_quad = mx.where(
                mx.abs(quad_denom) > 1e-30,
                -slopes / quad_denom,
                0.5 * lambdas,
            )
            new_lambdas = tmp_lam_quad
        else:
            # Cubic interpolation
            lam = lambdas
            lam2 = prev_lambdas
            rhs1 = trial_energies - old_energies - lam * slopes
            rhs2 = prev_energies - old_energies - lam2 * slopes

            lam_sq = lam * lam
            lam2_sq = lam2 * lam2
            denom = lam - lam2
            safe_denom = mx.where(mx.abs(denom) < 1e-30, 1.0, denom)

            a = (rhs1 / lam_sq - rhs2 / lam2_sq) / safe_denom
            b = (-lam2 * rhs1 / lam_sq + lam * rhs2 / lam2_sq) / safe_denom

            # Different cases for cubic
            disc = b * b - 3.0 * a * slopes
            disc_safe = mx.maximum(disc, 0.0)

            # Case 1: |a| < 1e-30 -> -slope / (2*b)
            case_a_zero = mx.where(
                mx.abs(b) > 1e-30,
                -slopes / (2.0 * b),
                0.5 * lam,
            )
            # Case 2: disc < 0 -> 0.5 * lam
            # Case 3: b <= 0 -> (-b + sqrt(disc)) / (3*a)
            safe_a = mx.where(mx.abs(a) < 1e-30, 1.0, a)
            case_b_neg = (-b + mx.sqrt(disc_safe)) / (3.0 * safe_a)
            # Case 4: b > 0 -> -slope / (b + sqrt(disc))
            case_b_pos = -slopes / mx.maximum(b + mx.sqrt(disc_safe), 1e-30)

            cubic_lam = mx.where(disc < 0.0, 0.5 * lam,
                                 mx.where(b <= 0.0, case_b_neg, case_b_pos))
            new_lambdas = mx.where(mx.abs(a) < 1e-30, case_a_zero, cubic_lam)

            # Handle degenerate denom
            new_lambdas = mx.where(mx.abs(denom) < 1e-30, 0.5 * lam, new_lambdas)

        # Clamp: min(tmp, 0.5*lam), max(result, 0.1*lam)
        new_lambdas = mx.minimum(new_lambdas, 0.5 * lambdas)
        new_lambdas = mx.maximum(new_lambdas, 0.1 * lambdas)

        # Only update for still-searching molecules
        prev_lambdas = mx.where(ls_active, lambdas, prev_lambdas)
        prev_energies = mx.where(ls_active, trial_energies, prev_energies)
        lambdas = mx.where(ls_active, new_lambdas, lambdas)

        mx.eval(lambdas, prev_lambdas, prev_energies, ls_active, best_pos_padded, best_pos_flat)

    # xi = best_pos - old_pos (step taken)
    xi_padded = (best_pos_padded - old_pos_padded) * dim_mask

    # Compute final energies at best positions
    final_energies, _ = energy_and_grad_fn(best_pos_flat)
    mx.eval(final_energies, best_pos_padded, best_pos_flat, xi_padded)

    return best_pos_padded, best_pos_flat, final_energies, xi_padded, active


def _bfgs_update_vec(
    xi_padded: mx.array,
    d_grad_padded: mx.array,
    grad_padded: mx.array,
    H_batch: mx.array,
    dim_mask: mx.array,
    n_mols: int,
    max_dim: int,
    active: mx.array,
) -> tuple[mx.array, mx.array]:
    """Vectorized BFGS rank-2 inverse Hessian update.

    Args:
        xi_padded: Step taken, shape ``(n_mols, max_dim)``.
        d_grad_padded: Gradient difference, shape ``(n_mols, max_dim)``.
        grad_padded: Current gradient, shape ``(n_mols, max_dim)``.
        H_batch: Batched inverse Hessians, shape ``(n_mols, max_dim, max_dim)``.
        dim_mask: Boolean mask for valid dimensions, shape ``(n_mols, max_dim)``.
        n_mols: Number of molecules.
        max_dim: Maximum coordinate dimension across molecules.
        active: Boolean mask of active molecules, shape ``(n_mols,)``.

    Returns:
        Tuple of ``(H_batch, dir_padded)`` with updated inverse Hessians and
        new search direction.
    """
    # hess_dg = H @ dGrad: (n_mols, max_dim)
    hess_dg = mx.squeeze(H_batch @ d_grad_padded[:, :, None], axis=-1)
    hess_dg = hess_dg * dim_mask

    # Dot products (masked for padding)
    fac = mx.sum(d_grad_padded * xi_padded * dim_mask, axis=1)      # (n_mols,)
    fae = mx.sum(d_grad_padded * hess_dg * dim_mask, axis=1)        # (n_mols,)
    sum_dg = mx.sum(d_grad_padded * d_grad_padded * dim_mask, axis=1)
    sum_xi = mx.sum(xi_padded * xi_padded * dim_mask, axis=1)

    # Guard condition: fac^2 > EPS * sum_dg * sum_xi AND fac > 0
    guard = (fac * fac > EPS * sum_dg * sum_xi) & (fac > 0)
    do_update = guard & active

    # Compute update terms
    fac_inv = mx.where(do_update, 1.0 / mx.maximum(mx.abs(fac), 1e-30) * mx.sign(fac + 1e-30), 0.0)
    fae_inv = mx.where(do_update, 1.0 / mx.maximum(mx.abs(fae), 1e-30) * mx.sign(fae + 1e-30), 0.0)

    # aux = fac_inv * xi - fae_inv * hess_dg
    aux = fac_inv[:, None] * xi_padded - fae_inv[:, None] * hess_dg  # (n_mols, max_dim)

    # Rank-2 update: H += fac_inv * xi@xi.T - fae_inv * hess_dg@hess_dg.T + fae * aux@aux.T
    # Use batched outer products
    xi_outer = xi_padded[:, :, None] * xi_padded[:, None, :]          # (n_mols, max_dim, max_dim)
    hd_outer = hess_dg[:, :, None] * hess_dg[:, None, :]
    aux_outer = aux[:, :, None] * aux[:, None, :]

    H_update = (
        fac_inv[:, None, None] * xi_outer
        - fae_inv[:, None, None] * hd_outer
        + fae[:, None, None] * aux_outer
    )

    # Only apply update where do_update is True
    H_batch = mx.where(do_update[:, None, None], H_batch + H_update, H_batch)

    # New direction = -H @ grad
    dir_padded = -mx.squeeze(H_batch @ grad_padded[:, :, None], axis=-1)
    dir_padded = dir_padded * dim_mask

    # For inactive molecules, zero out direction
    dir_padded = mx.where(active[:, None], dir_padded, 0.0)

    return H_batch, dir_padded
