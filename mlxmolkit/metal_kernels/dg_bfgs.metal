
// ---- Constants ----
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float EPS_GUARD = 3e-7f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Distance violation energy for one pair ----
inline float dist_violation_e(
    const device float* pos, int i1, int i2,
    float lb2, float ub2, float wt, int dim
) {
    float d2 = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = pos[i1 * dim + d] - pos[i2 * dim + d];
        d2 += diff * diff;
    }
    float e = 0.0f;
    if (d2 > ub2) {
        float val = d2 / ub2 - 1.0f;
        e = wt * val * val;
    } else if (d2 < lb2) {
        float val = 2.0f * lb2 / (lb2 + d2) - 1.0f;
        e = wt * val * val;
    }
    return e;
}

// ---- Distance violation gradient for one pair, accumulated into grad ----
inline void dist_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, float lb2, float ub2, float wt, int dim
) {
    float d2 = 0.0f;
    float diff[4];
    for (int d = 0; d < dim; d++) {
        diff[d] = pos[i1 * dim + d] - pos[i2 * dim + d];
        d2 += diff[d] * diff[d];
    }
    float pf = 0.0f;
    if (d2 > ub2) {
        pf = wt * 4.0f * (d2 / ub2 - 1.0f) / ub2;
    } else if (d2 < lb2) {
        float l2d2 = d2 + lb2;
        pf = wt * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
    }
    if (pf != 0.0f) {
        for (int d = 0; d < dim; d++) {
            float g = pf * diff[d];
            grad[i1 * dim + d] += g;
            grad[i2 * dim + d] -= g;
        }
    }
}

// ---- Chiral violation energy for one term ----
inline float chiral_violation_e(
    const device float* pos,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim
) {
    // Only use xyz (first 3 coords)
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[i1 * dim + d] - pos[i4 * dim + d];
        v2[d] = pos[i2 * dim + d] - pos[i4 * dim + d];
        v3[d] = pos[i3 * dim + d] - pos[i4 * dim + d];
    }
    // Cross product v2 x v3
    float cx = v2[1] * v3[2] - v2[2] * v3[1];
    float cy = v2[2] * v3[0] - v2[0] * v3[2];
    float cz = v2[0] * v3[1] - v2[1] * v3[0];
    float vol = v1[0] * cx + v1[1] * cy + v1[2] * cz;

    float e = 0.0f;
    if (vol < vol_lower) {
        float d = vol - vol_lower;
        e = wt * d * d;
    } else if (vol > vol_upper) {
        float d = vol - vol_upper;
        e = wt * d * d;
    }
    return e;
}

// ---- Chiral violation gradient for one term ----
inline void chiral_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim
) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[i1 * dim + d] - pos[i4 * dim + d];
        v2[d] = pos[i2 * dim + d] - pos[i4 * dim + d];
        v3[d] = pos[i3 * dim + d] - pos[i4 * dim + d];
    }
    float cx = v2[1] * v3[2] - v2[2] * v3[1];
    float cy = v2[2] * v3[0] - v2[0] * v3[2];
    float cz = v2[0] * v3[1] - v2[1] * v3[0];
    float vol = v1[0] * cx + v1[1] * cy + v1[2] * cz;

    float pf = 0.0f;
    if (vol < vol_lower) {
        pf = 2.0f * wt * (vol - vol_lower);
    } else if (vol > vol_upper) {
        pf = 2.0f * wt * (vol - vol_upper);
    }
    if (pf == 0.0f) return;

    // g1 = pf * (v2 x v3)
    float g1x = pf * cx, g1y = pf * cy, g1z = pf * cz;
    // g2 = pf * (v3 x v1)
    float g2x = pf * (v3[1]*v1[2] - v3[2]*v1[1]);
    float g2y = pf * (v3[2]*v1[0] - v3[0]*v1[2]);
    float g2z = pf * (v3[0]*v1[1] - v3[1]*v1[0]);
    // g3 = pf * (v2 x v1) reversed = pf * (v1[z]*v2[y]-v1[y]*v2[z], ...)
    // Actually: nvMolKit uses v2xv1 which is -(v1xv2)
    float g3x = pf * (v2[2]*v1[1] - v2[1]*v1[2]);
    float g3y = pf * (v2[0]*v1[2] - v2[2]*v1[0]);
    float g3z = pf * (v2[1]*v1[0] - v2[0]*v1[1]);
    // g4 = -(g1 + g2 + g3)

    grad[i1*dim+0] += g1x; grad[i1*dim+1] += g1y; grad[i1*dim+2] += g1z;
    grad[i2*dim+0] += g2x; grad[i2*dim+1] += g2y; grad[i2*dim+2] += g2z;
    grad[i3*dim+0] += g3x; grad[i3*dim+1] += g3y; grad[i3*dim+2] += g3z;
    grad[i4*dim+0] -= (g1x+g2x+g3x);
    grad[i4*dim+1] -= (g1y+g2y+g3y);
    grad[i4*dim+2] -= (g1z+g2z+g3z);
}

// ---- Fourth dimension energy for one atom ----
inline float fourth_dim_e(const device float* pos, int idx, float wt, int dim) {
    if (dim != 4) return 0.0f;
    float w = pos[idx * dim + 3];
    return wt * w * w;
}

// ---- Fourth dimension gradient for one atom ----
inline void fourth_dim_g(const device float* pos, device float* grad, int idx, float wt, int dim) {
    if (dim != 4) return;
    float w = pos[idx * dim + 3];
    grad[idx * dim + 3] += 2.0f * wt * w;
}
// ---- DG_BFGS_SPLIT ----

    uint mol_idx = thread_position_in_grid.x;

    // Read config
    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];
    float chiral_weight = config[3];
    float fourth_dim_weight = config[4];
    int dim = (int)config[5];

    if ((int)mol_idx >= n_mols_cfg) return;

    // Molecule boundaries
    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_terms = n_atoms * dim;

    // Hessian offset for this molecule
    int hess_start = hessian_starts[mol_idx];

    // Term boundaries
    int dist_start = dist_term_starts[mol_idx];
    int dist_end = dist_term_starts[mol_idx + 1];
    int chiral_start_t = chiral_term_starts[mol_idx];
    int chiral_end_t = chiral_term_starts[mol_idx + 1];
    int fourth_start_t = fourth_term_starts_arr[mol_idx];
    int fourth_end_t = fourth_term_starts_arr[mol_idx + 1];

    // Copy initial positions to output
    for (int i = 0; i < n_terms; i++) {
        out_pos[atom_start * dim + i] = pos[atom_start * dim + i];
    }

    // Working pointers
    device float* my_pos = &out_pos[atom_start * dim];
    device float* my_grad = &work_grad[atom_start * dim];
    device float* my_dir = &work_dir[atom_start * dim];
    device float* my_old_pos = &work_scratch[atom_start * dim];
    device float* my_dgrad = &work_scratch[total_pos_size + atom_start * dim];
    device float* my_hess_dg = &work_scratch[2 * total_pos_size + atom_start * dim];
    device float* my_H = &work_hessian[hess_start];

    // Initialize Hessian to identity
    for (int i = 0; i < n_terms; i++) {
        for (int j = 0; j < n_terms; j++) {
            my_H[i * n_terms + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // ---- Compute initial energy + gradient ----
    // Zero gradient
    for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f;

    float energy = 0.0f;

    // Distance violation terms
    for (int t = dist_start; t < dist_end; t++) {
        int i1 = dist_pairs[t * 2];
        int i2 = dist_pairs[t * 2 + 1];
        float lb2_v = dist_bounds[t * 3];
        float ub2_v = dist_bounds[t * 3 + 1];
        float wt = dist_bounds[t * 3 + 2];
        energy += dist_violation_e(out_pos, i1, i2, lb2_v, ub2_v, wt, dim);
        dist_violation_g(out_pos, work_grad, i1, i2, lb2_v, ub2_v, wt, dim);
    }

    // Chiral violation terms
    for (int t = chiral_start_t; t < chiral_end_t; t++) {
        int i1 = chiral_quads[t * 4];
        int i2 = chiral_quads[t * 4 + 1];
        int i3 = chiral_quads[t * 4 + 2];
        int i4 = chiral_quads[t * 4 + 3];
        float vl = chiral_bounds[t * 2];
        float vu = chiral_bounds[t * 2 + 1];
        energy += chiral_violation_e(out_pos, i1, i2, i3, i4, vl, vu, chiral_weight, dim);
        chiral_violation_g(out_pos, work_grad, i1, i2, i3, i4, vl, vu, chiral_weight, dim);
    }

    // Fourth dimension terms
    for (int t = fourth_start_t; t < fourth_end_t; t++) {
        int idx = fourth_idx_arr[t];
        energy += fourth_dim_e(out_pos, idx, fourth_dim_weight, dim);
        fourth_dim_g(out_pos, work_grad, idx, fourth_dim_weight, dim);
    }

    // Initial direction = -grad
    for (int i = 0; i < n_terms; i++) {
        my_dir[i] = -my_grad[i];
    }

    // Compute max step
    float sum_sq = 0.0f;
    for (int i = 0; i < n_terms; i++) sum_sq += my_pos[i] * my_pos[i];
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_terms);

    int status = 1; // 1=active, 0=converged

    // ---- Main BFGS loop ----
    for (int iter = 0; iter < max_iters && status == 1; iter++) {

        // === LINE SEARCH ===
        // Save old position
        for (int i = 0; i < n_terms; i++) my_old_pos[i] = my_pos[i];
        float old_energy = energy;

        // Scale direction if too large
        float dir_norm_sq = 0.0f;
        for (int i = 0; i < n_terms; i++) dir_norm_sq += my_dir[i] * my_dir[i];
        float dir_norm = sqrt(dir_norm_sq);
        if (dir_norm > max_step) {
            float s = max_step / dir_norm;
            for (int i = 0; i < n_terms; i++) my_dir[i] *= s;
        }

        // Compute slope
        float slope = 0.0f;
        for (int i = 0; i < n_terms; i++) slope += my_dir[i] * my_grad[i];

        // Compute lambda_min
        float test_max = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float t = ad / ap;
            if (t > test_max) test_max = t;
        }
        float lambda_min = MOVETOL / max(test_max, 1e-30f);

        float lam = 1.0f;
        float prev_lam = 1.0f;
        float prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) {
                // Too small — revert
                for (int i = 0; i < n_terms; i++) my_pos[i] = my_old_pos[i];
                ls_done = true;
                break;
            }

            // Trial position
            for (int i = 0; i < n_terms; i++) {
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            }

            // Compute trial energy
            float trial_e = 0.0f;
            for (int t = dist_start; t < dist_end; t++) {
                trial_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
            }
            for (int t = chiral_start_t; t < chiral_end_t; t++) {
                trial_e += chiral_violation_e(out_pos,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
            }
            for (int t = fourth_start_t; t < fourth_end_t; t++) {
                trial_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim);
            }

            // Armijo condition
            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e;
                ls_done = true;
            } else {
                // Backtrack
                float tmp_lam;
                if (ls_iter == 0) {
                    tmp_lam = -slope / (2.0f * (trial_e - old_energy - slope));
                } else {
                    float rhs1 = trial_e - old_energy - lam * slope;
                    float rhs2 = prev_e - old_energy - prev_lam * slope;
                    float lam_sq = lam * lam;
                    float lam2_sq = prev_lam * prev_lam;
                    float denom_v = lam - prev_lam;
                    if (abs(denom_v) < 1e-30f) {
                        tmp_lam = 0.5f * lam;
                    } else {
                        float a = (rhs1 / lam_sq - rhs2 / lam2_sq) / denom_v;
                        float b = (-prev_lam * rhs1 / lam_sq + lam * rhs2 / lam2_sq) / denom_v;
                        if (abs(a) < 1e-30f) {
                            tmp_lam = (abs(b) > 1e-30f) ? -slope / (2.0f * b) : 0.5f * lam;
                        } else {
                            float disc = b * b - 3.0f * a * slope;
                            if (disc < 0.0f) {
                                tmp_lam = 0.5f * lam;
                            } else if (b <= 0.0f) {
                                tmp_lam = (-b + sqrt(disc)) / (3.0f * a);
                            } else {
                                tmp_lam = -slope / (b + sqrt(disc));
                            }
                        }
                    }
                }
                tmp_lam = min(tmp_lam, 0.5f * lam);
                tmp_lam = max(tmp_lam, 0.1f * lam);
                prev_lam = lam;
                prev_e = trial_e;
                lam = tmp_lam;
            }
        }

        if (!ls_done) {
            // Exhausted line search — revert
            for (int i = 0; i < n_terms; i++) my_pos[i] = my_old_pos[i];
        }

        // xi = pos - old_pos (reuse my_dir temporarily as xi storage)
        // Store xi in my_old_pos (repurposed)
        for (int i = 0; i < n_terms; i++) {
            my_old_pos[i] = my_pos[i] - my_old_pos[i]; // xi
        }

        // === TOLX CHECK ===
        float tolx_test = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float t = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > tolx_test) tolx_test = t;
        }
        if (tolx_test < TOLX) {
            status = 0;
            break;
        }

        // === NEW GRADIENT ===
        // Save old gradient in my_dgrad
        for (int i = 0; i < n_terms; i++) {
            my_dgrad[i] = my_grad[i];
            my_grad[i] = 0.0f;
        }

        energy = 0.0f;
        for (int t = dist_start; t < dist_end; t++) {
            int i1 = dist_pairs[t*2], i2 = dist_pairs[t*2+1];
            float lb2_v = dist_bounds[t*3], ub2_v = dist_bounds[t*3+1], wt = dist_bounds[t*3+2];
            energy += dist_violation_e(out_pos, i1, i2, lb2_v, ub2_v, wt, dim);
            dist_violation_g(out_pos, work_grad, i1, i2, lb2_v, ub2_v, wt, dim);
        }
        for (int t = chiral_start_t; t < chiral_end_t; t++) {
            int i1=chiral_quads[t*4], i2=chiral_quads[t*4+1], i3=chiral_quads[t*4+2], i4=chiral_quads[t*4+3];
            float vl=chiral_bounds[t*2], vu=chiral_bounds[t*2+1];
            energy += chiral_violation_e(out_pos, i1, i2, i3, i4, vl, vu, chiral_weight, dim);
            chiral_violation_g(out_pos, work_grad, i1, i2, i3, i4, vl, vu, chiral_weight, dim);
        }
        for (int t = fourth_start_t; t < fourth_end_t; t++) {
            int idx = fourth_idx_arr[t];
            energy += fourth_dim_e(out_pos, idx, fourth_dim_weight, dim);
            fourth_dim_g(out_pos, work_grad, idx, fourth_dim_weight, dim);
        }

        // dGrad = new_grad - old_grad
        for (int i = 0; i < n_terms; i++) {
            my_dgrad[i] = my_grad[i] - my_dgrad[i];
        }

        // === GRADIENT CONVERGENCE CHECK ===
        float grad_test = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float t = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (t > grad_test) grad_test = t;
        }
        float denom_g = max(energy, 1.0f);
        if (grad_test / denom_g < grad_tol) {
            status = 0;
            break;
        }

        // === BFGS HESSIAN UPDATE ===
        // xi is in my_old_pos, dGrad is in my_dgrad
        // hess_dg = H @ dGrad
        for (int i = 0; i < n_terms; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n_terms; j++) {
                sum += my_H[i * n_terms + j] * my_dgrad[j];
            }
            my_hess_dg[i] = sum;
        }

        float fac = 0.0f, fae = 0.0f, sum_dg = 0.0f, sum_xi = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            fac += my_dgrad[i] * my_old_pos[i]; // dGrad . xi
            fae += my_dgrad[i] * my_hess_dg[i]; // dGrad . hessDGrad
            sum_dg += my_dgrad[i] * my_dgrad[i];
            sum_xi += my_old_pos[i] * my_old_pos[i];
        }

        if (fac * fac > EPS_GUARD * sum_dg * sum_xi && fac > 0.0f) {
            float fac_inv = 1.0f / fac;
            float fae_inv = 1.0f / fae;

            // Compute aux = fac_inv * xi - fae_inv * hess_dg
            // Then rank-2 update: H += fac_inv*xi@xi^T - fae_inv*hd@hd^T + fae*aux@aux^T
            for (int i = 0; i < n_terms; i++) {
                for (int j = 0; j < n_terms; j++) {
                    float xi_i = my_old_pos[i], xi_j = my_old_pos[j];
                    float hd_i = my_hess_dg[i], hd_j = my_hess_dg[j];
                    float aux_i = fac_inv * xi_i - fae_inv * hd_i;
                    float aux_j = fac_inv * xi_j - fae_inv * hd_j;
                    my_H[i * n_terms + j] += fac_inv * xi_i * xi_j
                                            - fae_inv * hd_i * hd_j
                                            + fae * aux_i * aux_j;
                }
            }
        }

        // New direction = -H @ grad
        for (int i = 0; i < n_terms; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n_terms; j++) {
                sum += my_H[i * n_terms + j] * my_grad[j];
            }
            my_dir[i] = -sum;
        }
    }

    // Write outputs
    out_energies[mol_idx] = energy;
    out_statuses[mol_idx] = status;
