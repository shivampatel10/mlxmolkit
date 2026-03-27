"""Batched BFGS minimizer for molecular force fields on MLX.

Port of nvMolKit's bfgs_minimize.cu + bfgs_hessian.cu.
All computation uses float32. Per-molecule inverse Hessian stored as
dense matrices with variable sizes.

Algorithm:
1. Initialize H^-1 = I, compute energy + grad, scale grad
2. direction = -grad
3. Main loop:
   a. Line search (backtracking with quadratic/cubic interpolation)
   b. Update positions, check TOLX convergence
   c. Compute new gradient, check gradient convergence
   d. BFGS rank-2 inverse Hessian update
   e. New direction = -H^-1 @ grad
"""

from typing import Callable

import mlx.core as mx

# ---------------------
# Constants (float32-adjusted, while preserving RDKit scaling/convergence form)
# ---------------------
TOLX = 1.2e-6
FUNCTOL = 1e-4
MOVETOL = 1e-6
EPS = 3e-7
MAX_STEP_FACTOR = 100.0
MAX_LINESEARCH_ITERS = 1000
GRAD_SCALE_INIT = 0.1
GRAD_CAP = 10.0
DEFAULT_GRAD_TOL = 1e-3


def _scale_grad(
    grad: mx.array,
    atom_starts: list[int],
    n_mols: int,
    dim: int,
    grad_scale: mx.array,
    pre_loop: bool,
) -> tuple[mx.array, mx.array]:
    """Scale gradient per-molecule: multiply by grad_scale, halve while max > 10.

    Args:
        grad: Flat gradient array, shape ``(n_atoms_total * dim,)``.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.
        grad_scale: Per-molecule scale factors, shape ``(n_mols,)``.
        pre_loop: If True, initialize scale to 0.1 first.

    Returns:
        Tuple of ``(scaled_grad, grad_scale)`` with updated arrays.
    """
    grad_r = grad.reshape(-1, dim)
    new_grad = grad_r * 1  # MLX copy

    for mol_idx in range(n_mols):
        start = atom_starts[mol_idx]
        end = atom_starts[mol_idx + 1]
        mol_grad = new_grad[start:end]  # (n_atoms_mol, dim)

        if pre_loop:
            scale = mx.array(GRAD_SCALE_INIT, dtype=mx.float32)
        else:
            scale = grad_scale[mol_idx]

        mol_grad = mol_grad * scale
        max_g = mx.max(mx.abs(mol_grad))

        # Halve scale while max gradient > GRAD_CAP
        while max_g.item() > GRAD_CAP:
            scale = scale * 0.5
            mol_grad = grad_r[start:end] * scale
            max_g = mx.max(mx.abs(mol_grad))

        new_grad = new_grad.at[start:end].add(mol_grad - new_grad[start:end])
        grad_scale = grad_scale.at[mol_idx].add(scale - grad_scale[mol_idx])

    return new_grad.reshape(-1), grad_scale


def _compute_max_step(
    pos: mx.array,
    atom_starts: list[int],
    n_mols: int,
    dim: int,
) -> mx.array:
    """Compute max step size per molecule: ``100 * max(||pos||, n_terms)``.

    Args:
        pos: Flat positions array, shape ``(n_atoms_total * dim,)``.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.

    Returns:
        Per-molecule max step sizes, shape ``(n_mols,)``.
    """
    max_steps = mx.zeros(n_mols, dtype=mx.float32)
    pos_r = pos.reshape(-1, dim)

    for mol_idx in range(n_mols):
        start = atom_starts[mol_idx]
        end = atom_starts[mol_idx + 1]
        n_terms = (end - start) * dim
        sum_sq = mx.sum(pos_r[start:end] ** 2)
        max_step = MAX_STEP_FACTOR * mx.maximum(mx.sqrt(sum_sq), mx.array(float(n_terms)))
        max_steps = max_steps.at[mol_idx].add(max_step)

    return max_steps


def bfgs_minimize(
    energy_and_grad_fn: Callable[[mx.array], tuple[mx.array, mx.array]],
    pos: mx.array,
    atom_starts_list: list[int] | mx.array,
    n_mols: int,
    dim: int,
    max_iters: int = 400,
    grad_tol: float | None = None,
    scale_grads: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched BFGS minimizer.

    Args:
        energy_and_grad_fn: Callable taking flat positions and returning
            ``(energies, grad)`` with shapes ``(n_mols,)`` and ``(flat,)``.
        pos: Initial flat positions, shape ``(n_atoms_total * dim,)``, float32.
        atom_starts_list: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension (3 or 4).
        max_iters: Maximum BFGS iterations.
        grad_tol: Gradient convergence tolerance. Defaults to ``1e-3``.
        scale_grads: Whether to apply gradient scaling (RDKit compat).

    Returns:
        Tuple of ``(final_pos, final_energies, statuses)`` where statuses is
        ``(n_mols,)`` int with 0=converged, 1=not converged.
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    # Ensure atom_starts is a plain list for Python indexing
    if isinstance(atom_starts_list, mx.array):
        atom_starts = atom_starts_list.tolist()
    else:
        atom_starts = list(atom_starts_list)

    # Per-molecule sizes
    mol_sizes = [atom_starts[i + 1] - atom_starts[i] for i in range(n_mols)]
    mol_dims = [s * dim for s in mol_sizes]  # n_terms per mol

    # Status: 0=converged, 1=active
    statuses = mx.ones(n_mols, dtype=mx.int32)

    # Gradient scale factors
    grad_scale = mx.ones(n_mols, dtype=mx.float32)

    # --- Initialize ---
    # Compute initial energy and gradient
    energies, grad = energy_and_grad_fn(pos)
    mx.eval(energies, grad)

    # Scale gradient
    if scale_grads:
        grad, grad_scale = _scale_grad(grad, atom_starts, n_mols, dim, grad_scale, pre_loop=True)
        mx.eval(grad, grad_scale)

    # Initial direction = -grad
    direction = -grad

    # Initialize per-molecule inverse Hessians to identity
    inv_hessians = []
    for i in range(n_mols):
        d = mol_dims[i]
        inv_hessians.append(mx.eye(d, dtype=mx.float32))

    # Compute max step per molecule
    max_steps = _compute_max_step(pos, atom_starts, n_mols, dim)
    mx.eval(max_steps)

    # --- Main BFGS loop ---
    for iteration in range(max_iters):
        # Check if all converged
        n_active = mx.sum(statuses).item()
        if n_active == 0:
            break

        # === LINE SEARCH ===
        pos, energies, direction, statuses = _line_search(
            energy_and_grad_fn, pos, grad, direction, energies,
            atom_starts, n_mols, dim, mol_dims, max_steps, statuses,
        )
        mx.eval(pos, energies)

        # === SET DIRECTION (xi = new_pos - old_pos done inside line search) ===
        # direction now holds xi (step taken) from _line_search

        # Check TOLX convergence
        statuses = _check_tolx(pos, direction, atom_starts, n_mols, dim, statuses)

        # Save old gradient
        old_grad = grad

        # === COMPUTE NEW GRADIENT ===
        energies, grad = energy_and_grad_fn(pos)
        mx.eval(energies, grad)

        if scale_grads:
            grad, grad_scale = _scale_grad(
                grad, atom_starts, n_mols, dim, grad_scale, pre_loop=False
            )
            mx.eval(grad, grad_scale)

        # === UPDATE dGrad and check gradient convergence ===
        d_grad = grad - old_grad
        statuses = _check_grad_convergence(
            grad, pos, energies, grad_scale, atom_starts, n_mols, dim,
            grad_tol, statuses,
        )

        # === BFGS HESSIAN UPDATE + NEW DIRECTION ===
        inv_hessians, direction = _bfgs_hessian_update(
            direction, d_grad, grad, inv_hessians,
            atom_starts, n_mols, dim, mol_dims, statuses,
        )
        mx.eval(direction)

    return pos, energies, statuses


def _line_search(
    energy_and_grad_fn: Callable[[mx.array], tuple[mx.array, mx.array]],
    pos: mx.array,
    grad: mx.array,
    direction: mx.array,
    energies: mx.array,
    atom_starts: list[int],
    n_mols: int,
    dim: int,
    mol_dims: list[int],
    max_steps: mx.array,
    statuses: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Backtracking line search with quadratic/cubic interpolation.

    Args:
        energy_and_grad_fn: Callable taking flat positions and returning
            ``(energies, grad)``.
        pos: Flat positions, shape ``(n_atoms_total * dim,)``.
        grad: Flat gradient, shape ``(n_atoms_total * dim,)``.
        direction: Search direction, flat array.
        energies: Current per-molecule energies, shape ``(n_mols,)``.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.
        mol_dims: Number of coordinate terms per molecule.
        max_steps: Per-molecule max step sizes, shape ``(n_mols,)``.
        statuses: Per-molecule convergence statuses, shape ``(n_mols,)``.

    Returns:
        Tuple of ``(new_pos, new_energies, xi, statuses)`` where
        ``xi = new_pos - old_pos``.
    """
    old_pos = pos
    old_energies = energies

    # Per-molecule: scale direction if ||dir|| > maxStep, compute slope
    lambdas = mx.ones(n_mols, dtype=mx.float32)
    lambda_mins = mx.zeros(n_mols, dtype=mx.float32)
    slopes = mx.zeros(n_mols, dtype=mx.float32)
    ls_status = mx.full((n_mols,), -2, dtype=mx.int32)  # -2 = in progress

    dir_scaled = direction * 1  # MLX copy

    for mol_idx in range(n_mols):
        if statuses[mol_idx].item() == 0:
            ls_status = ls_status.at[mol_idx].add(2)  # set to 0
            continue

        start = atom_starts[mol_idx] * dim
        end = atom_starts[mol_idx + 1] * dim
        mol_dir = dir_scaled[start:end]
        mol_grad = grad[start:end]
        mol_pos = pos[start:end]

        dir_norm = mx.sqrt(mx.sum(mol_dir * mol_dir))
        max_step = max_steps[mol_idx]

        # Scale direction if too large
        if dir_norm.item() > max_step.item():
            scale = max_step / dir_norm
            mol_dir = mol_dir * scale
            dir_scaled = dir_scaled.at[start:end].add(mol_dir - dir_scaled[start:end])

        # Compute slope = dir . grad (must be negative)
        slope = mx.sum(mol_dir * mol_grad)
        slopes = slopes.at[mol_idx].add(slope)

        # Compute lambda_min
        abs_dir = mx.abs(mol_dir)
        abs_pos = mx.maximum(mx.abs(mol_pos), 1.0)
        test = mx.max(abs_dir / abs_pos)
        lam_min = MOVETOL / mx.maximum(test, mx.array(1e-30, dtype=mx.float32))
        lambda_mins = lambda_mins.at[mol_idx].add(lam_min)

    mx.eval(dir_scaled, slopes, lambda_mins)

    # Line search iterations
    prev_lambdas = mx.ones(n_mols, dtype=mx.float32)
    prev_energies = old_energies * 1  # MLX copy
    best_pos = old_pos * 1  # MLX copy

    for ls_iter in range(MAX_LINESEARCH_ITERS):
        # Check if all done
        n_searching = mx.sum(ls_status == -2).item()
        if n_searching == 0:
            break

        # Compute trial positions: trial = old_pos + lambda * dir
        trial_pos = old_pos * 1  # MLX copy
        for mol_idx in range(n_mols):
            if ls_status[mol_idx].item() != -2:
                continue

            lam = lambdas[mol_idx]

            # Check lambda_min
            if lam.item() < lambda_mins[mol_idx].item():
                ls_status = ls_status.at[mol_idx].add(3)  # -2 + 3 = 1 (converged by step)
                continue

            start = atom_starts[mol_idx] * dim
            end = atom_starts[mol_idx + 1] * dim
            mol_dir = dir_scaled[start:end]
            trial_mol = old_pos[start:end] + lam * mol_dir
            trial_pos = trial_pos.at[start:end].add(trial_mol - trial_pos[start:end])

        mx.eval(trial_pos)

        # Compute energy at trial positions
        trial_energies, _ = energy_and_grad_fn(trial_pos)
        mx.eval(trial_energies)

        # Post-energy: check sufficient decrease or backtrack
        new_lambdas = lambdas * 1  # MLX copy
        for mol_idx in range(n_mols):
            if ls_status[mol_idx].item() != -2:
                continue

            lam = lambdas[mol_idx].item()
            slope = slopes[mol_idx].item()
            old_e = old_energies[mol_idx].item()
            new_e = trial_energies[mol_idx].item()

            # Sufficient decrease (Armijo condition)
            if new_e - old_e <= FUNCTOL * lam * slope:
                ls_status = ls_status.at[mol_idx].add(2)  # -2 + 2 = 0 (success)
                best_pos = _copy_mol_pos(best_pos, trial_pos, atom_starts, mol_idx, dim)
                continue

            # Backtrack
            if ls_iter == 0:
                # Quadratic interpolation
                tmp_lam = -slope / (2.0 * (new_e - old_e - slope))
            else:
                # Cubic interpolation
                lam2 = prev_lambdas[mol_idx].item()
                val2 = prev_energies[mol_idx].item()

                rhs1 = new_e - old_e - lam * slope
                rhs2 = val2 - old_e - lam2 * slope

                lam_sq = lam * lam
                lam2_sq = lam2 * lam2
                denom = lam - lam2
                if abs(denom) < 1e-30:
                    tmp_lam = 0.5 * lam
                else:
                    a = (rhs1 / lam_sq - rhs2 / lam2_sq) / denom
                    b = (-lam2 * rhs1 / lam_sq + lam * rhs2 / lam2_sq) / denom

                    if abs(a) < 1e-30:
                        tmp_lam = -slope / (2.0 * b) if abs(b) > 1e-30 else 0.5 * lam
                    else:
                        disc = b * b - 3.0 * a * slope
                        if disc < 0.0:
                            tmp_lam = 0.5 * lam
                        elif b <= 0.0:
                            tmp_lam = (-b + disc ** 0.5) / (3.0 * a)
                        else:
                            tmp_lam = -slope / (b + disc ** 0.5)

            # Clamp: min(tmp, 0.5*lam), max(result, 0.1*lam)
            tmp_lam = min(tmp_lam, 0.5 * lam)
            tmp_lam = max(tmp_lam, 0.1 * lam)

            prev_lambdas = prev_lambdas.at[mol_idx].add(
                mx.array(lam, dtype=mx.float32) - prev_lambdas[mol_idx]
            )
            prev_energies = prev_energies.at[mol_idx].add(
                trial_energies[mol_idx] - prev_energies[mol_idx]
            )
            new_lambdas = new_lambdas.at[mol_idx].add(
                mx.array(tmp_lam, dtype=mx.float32) - new_lambdas[mol_idx]
            )

        lambdas = new_lambdas
        mx.eval(lambdas, prev_lambdas, prev_energies, ls_status, best_pos)

    # Post-loop: for molecules still searching (-2), accept best known
    for mol_idx in range(n_mols):
        status = ls_status[mol_idx].item()
        if status == -2:
            # Exhausted — revert to old position
            best_pos = _copy_mol_pos(best_pos, old_pos, atom_starts, mol_idx, dim)
        elif status == 1:
            # Lambda too small — revert to old position
            best_pos = _copy_mol_pos(best_pos, old_pos, atom_starts, mol_idx, dim)

    # xi = new_pos - old_pos (step taken)
    xi = best_pos - old_pos

    # Compute energies at best positions
    final_energies, _ = energy_and_grad_fn(best_pos)
    mx.eval(final_energies, best_pos, xi)

    return best_pos, final_energies, xi, statuses


def _copy_mol_pos(
    dest: mx.array,
    src: mx.array,
    atom_starts: list[int],
    mol_idx: int,
    dim: int,
) -> mx.array:
    """Copy one molecule's positions from src to dest.

    Args:
        dest: Destination flat positions array.
        src: Source flat positions array.
        atom_starts: CSR atom boundaries.
        mol_idx: Index of the molecule to copy.
        dim: Coordinate dimension.

    Returns:
        Updated destination array with the molecule's positions replaced.
    """
    start = atom_starts[mol_idx] * dim
    end = atom_starts[mol_idx + 1] * dim
    dest = dest.at[start:end].add(src[start:end] - dest[start:end])
    return dest


def _check_tolx(
    pos: mx.array,
    xi: mx.array,
    atom_starts: list[int],
    n_mols: int,
    dim: int,
    statuses: mx.array,
) -> mx.array:
    """Check position change convergence against TOLX threshold.

    Args:
        pos: Current flat positions, shape ``(n_atoms_total * dim,)``.
        xi: Step taken (new_pos - old_pos), flat array.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.
        statuses: Per-molecule convergence statuses, shape ``(n_mols,)``.

    Returns:
        Updated statuses array with newly converged molecules set to 0.
    """
    for mol_idx in range(n_mols):
        if statuses[mol_idx].item() == 0:
            continue

        start = atom_starts[mol_idx] * dim
        end = atom_starts[mol_idx + 1] * dim
        mol_xi = xi[start:end]
        mol_pos = pos[start:end]

        test = mx.max(mx.abs(mol_xi) / mx.maximum(mx.abs(mol_pos), 1.0))
        if test.item() < TOLX:
            statuses = statuses.at[mol_idx].add(-1)  # 1 -> 0

    return statuses


def _check_grad_convergence(
    grad: mx.array,
    pos: mx.array,
    energies: mx.array,
    grad_scale: mx.array,
    atom_starts: list[int],
    n_mols: int,
    dim: int,
    grad_tol: float,
    statuses: mx.array,
) -> mx.array:
    """Check gradient convergence against tolerance.

    Args:
        grad: Current flat gradient, shape ``(n_atoms_total * dim,)``.
        pos: Current flat positions, shape ``(n_atoms_total * dim,)``.
        energies: Per-molecule energies, shape ``(n_mols,)``.
        grad_scale: Per-molecule gradient scale factors, shape ``(n_mols,)``.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.
        grad_tol: Gradient convergence tolerance.
        statuses: Per-molecule convergence statuses, shape ``(n_mols,)``.

    Returns:
        Updated statuses array with newly converged molecules set to 0.
    """
    for mol_idx in range(n_mols):
        if statuses[mol_idx].item() == 0:
            continue

        start = atom_starts[mol_idx] * dim
        end = atom_starts[mol_idx + 1] * dim
        mol_grad = grad[start:end]
        mol_pos = pos[start:end]

        # test = max_i(|grad_i| * max(|pos_i|, 1.0))
        test = mx.max(mx.abs(mol_grad) * mx.maximum(mx.abs(mol_pos), 1.0))
        e = energies[mol_idx]
        gs = grad_scale[mol_idx]
        denom = mx.maximum(e * gs, mx.array(1.0, dtype=mx.float32))
        test = test / denom

        if test.item() < grad_tol:
            statuses = statuses.at[mol_idx].add(-1)  # 1 -> 0

    return statuses


def _bfgs_hessian_update(
    xi: mx.array,
    d_grad: mx.array,
    grad: mx.array,
    inv_hessians: list[mx.array],
    atom_starts: list[int],
    n_mols: int,
    dim: int,
    mol_dims: list[int],
    statuses: mx.array,
) -> tuple[list[mx.array], mx.array]:
    """BFGS rank-2 inverse Hessian update and compute new search direction.

    Args:
        xi: Step taken (new_pos - old_pos), flat array.
        d_grad: Gradient difference (grad_new - grad_old), flat array.
        grad: Current gradient, flat array.
        inv_hessians: Per-molecule inverse Hessian matrices, each ``(d, d)``.
        atom_starts: CSR atom boundaries of length ``n_mols + 1``.
        n_mols: Number of molecules.
        dim: Coordinate dimension.
        mol_dims: Number of coordinate terms per molecule.
        statuses: Per-molecule convergence statuses, shape ``(n_mols,)``.

    Returns:
        Tuple of ``(inv_hessians, direction)`` with updated inverse Hessians
        and new search direction as a flat array.
    """
    direction = mx.zeros_like(grad)

    for mol_idx in range(n_mols):
        if statuses[mol_idx].item() == 0:
            continue

        start = atom_starts[mol_idx] * dim
        end = atom_starts[mol_idx + 1] * dim
        d = mol_dims[mol_idx]

        mol_xi = xi[start:end]       # (d,)
        mol_dg = d_grad[start:end]   # (d,)
        mol_grad = grad[start:end]   # (d,)
        H = inv_hessians[mol_idx]    # (d, d)

        # Step 1: hessDGrad = H @ dGrad
        hess_dg = H @ mol_dg  # (d,)

        # Step 2: dot products
        fac = mx.sum(mol_dg * mol_xi)      # dGrad . xi
        fae = mx.sum(mol_dg * hess_dg)     # dGrad . hessDGrad
        sum_dg = mx.sum(mol_dg * mol_dg)   # ||dGrad||^2
        sum_xi = mx.sum(mol_xi * mol_xi)   # ||xi||^2

        # Step 3: guard condition
        fac_val = fac.item()
        guard = fac_val * fac_val > EPS * sum_dg.item() * sum_xi.item()

        if guard and fac_val > 0:
            # Step 4: compute auxiliary vector
            fac_inv = 1.0 / fac
            fae_inv = 1.0 / fae
            aux = fac_inv * mol_xi - fae_inv * hess_dg  # (d,)

            # Step 5: rank-2 update
            # H += fac_inv * xi @ xi.T - fae_inv * hessDGrad @ hessDGrad.T + fae * aux @ aux.T
            H = (
                H
                + fac_inv * mx.outer(mol_xi, mol_xi)
                - fae_inv * mx.outer(hess_dg, hess_dg)
                + fae * mx.outer(aux, aux)
            )
            inv_hessians[mol_idx] = H

        # Step 6: new direction = -H @ grad
        mol_dir = -(H @ mol_grad)
        direction = direction.at[start:end].add(mol_dir)

    return inv_hessians, direction
