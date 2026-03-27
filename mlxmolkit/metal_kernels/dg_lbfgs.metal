
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
// Called by thread 0 only — no atomics needed
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

    float g1x = pf * cx, g1y = pf * cy, g1z = pf * cz;
    float g2x = pf * (v3[1]*v1[2] - v3[2]*v1[1]);
    float g2y = pf * (v3[2]*v1[0] - v3[0]*v1[2]);
    float g2z = pf * (v3[0]*v1[1] - v3[1]*v1[0]);
    float g3x = pf * (v2[2]*v1[1] - v2[1]*v1[2]);
    float g3y = pf * (v2[0]*v1[2] - v2[2]*v1[0]);
    float g3z = pf * (v2[1]*v1[0] - v2[0]*v1[1]);

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

// ---- Threadgroup reduction: sum shared[0..n-1] -> shared[0] ----
inline float tg_reduce_sum(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[0];
}

// ---- Parallel dot product: dot(a, b) across threads ----
inline float parallel_dot(const device float* a, const device float* b,
    int n, uint tid, uint tpm, threadgroup float* s) {
    float sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) {
        sum += a[i] * b[i];
    }
    s[tid] = sum;
    return tg_reduce_sum(s, tid, tpm);
}

// ---- Parallel saxpy: a[i] += alpha * b[i] ----
inline void parallel_saxpy(device float* a, float alpha, const device float* b,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) {
        a[i] += alpha * b[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- Parallel scale: a[i] *= alpha ----
inline void parallel_scale(device float* a, float alpha,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) {
        a[i] *= alpha;
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- Parallel copy: dst[i] = src[i] ----
inline void parallel_copy(device float* dst, const device float* src,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) {
        dst[i] = src[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- Parallel set: a[i] = val ----
inline void parallel_set(device float* a, float val,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) {
        a[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- Parallel negate copy: dst[i] = -src[i] ----
inline void parallel_neg_copy(device float* dst, const device float* src,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) {
        dst[i] = -src[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
}
// ---- DG_LBFGS_SPLIT ----

    uint tid = thread_position_in_threadgroup.x;  // 0..TPM-1
    uint mol_idx = threadgroup_position_in_grid.x; // which molecule
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;

    // Shared memory for reductions
    threadgroup float shared[TPM];

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
    int n_vars = n_atoms * dim;

    // L-BFGS history offset for this molecule
    int lbfgs_start = lbfgs_history_starts[mol_idx];

    // Term boundaries
    int dist_start = dist_term_starts[mol_idx];
    int dist_end = dist_term_starts[mol_idx + 1];
    int chiral_start_t = chiral_term_starts[mol_idx];
    int chiral_end_t = chiral_term_starts[mol_idx + 1];
    int fourth_start_t = fourth_term_starts_arr[mol_idx];
    int fourth_end_t = fourth_term_starts_arr[mol_idx + 1];

    // Copy initial positions to output (parallel)
    parallel_copy(&out_pos[atom_start * dim], &pos[atom_start * dim], n_vars, tid, tpm);

    // Working pointers
    device float* my_pos = &out_pos[atom_start * dim];
    device float* my_grad = &work_grad[atom_start * dim];
    device float* my_dir = &work_dir[atom_start * dim];
    device float* my_old_pos = &work_scratch[atom_start * dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_start * dim];
    device float* my_q = &work_scratch[2 * total_pos_size + atom_start * dim];

    // L-BFGS history: s_k and y_k vectors, rho values
    // Layout: work_lbfgs[lbfgs_start + k * n_vars] for s_k (k=0..m-1)
    //         work_lbfgs[lbfgs_start + m * n_vars + k * n_vars] for y_k
    device float* my_S = &work_lbfgs[lbfgs_start];                      // m * n_vars
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m * n_vars];   // m * n_vars
    device float* my_rho = &work_rho[mol_idx * lbfgs_m];                // m

    // ---- Compute initial energy (parallel) + gradient (thread 0) ----
    parallel_set(my_grad, 0.0f, n_vars, tid, tpm);

    // Energy: all threads sum a stripe, then reduce
    float local_energy = 0.0f;
    for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm) {
        local_energy += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
            dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
    }
    for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm) {
        local_energy += chiral_violation_e(out_pos,
            chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
            chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
    }
    for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm) {
        local_energy += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim);
    }
    shared[tid] = local_energy;
    float energy = tg_reduce_sum(shared, tid, tpm);

    // Gradient: thread 0 only (no atomics needed)
    if (tid == 0) {
        for (int t = dist_start; t < dist_end; t++)
            dist_violation_g(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1],
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
        for (int t = chiral_start_t; t < chiral_end_t; t++)
            chiral_violation_g(out_pos, work_grad,
                chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
        for (int t = fourth_start_t; t < fourth_end_t; t++)
            fourth_dim_g(out_pos, work_grad, fourth_idx_arr[t], fourth_dim_weight, dim);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Initial direction = -grad
    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);

    // Compute max step
    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_vars; i += (int)tpm) {
        local_sum_sq += my_pos[i] * my_pos[i];
    }
    shared[tid] = local_sum_sq;
    float sum_sq = tg_reduce_sum(shared, tid, tpm);
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_vars);

    int status = 1;
    int hist_count = 0;  // Number of stored L-BFGS pairs
    int hist_idx = 0;    // Circular buffer index

    // ---- Main L-BFGS loop ----
    for (int iter = 0; iter < max_iters && status == 1; iter++) {

        // === LINE SEARCH ===
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        // Scale direction if too large (parallel norm)
        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            local_dir_sq += my_dir[i] * my_dir[i];
        }
        shared[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(shared, tid, tpm));
        if (dir_norm > max_step) {
            parallel_scale(my_dir, max_step / dir_norm, n_vars, tid, tpm);
        }

        // Compute slope = dir . grad (parallel)
        float slope = parallel_dot(my_dir, my_grad, n_vars, tid, tpm, shared);

        // Compute lambda_min (parallel max)
        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float t = ad / ap;
            if (t > local_test_max) local_test_max = t;
        }
        shared[tid] = local_test_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Parallel max reduction
        for (uint stride = tpm / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float lambda_min = MOVETOL / max(shared[0], 1e-30f);

        float lam = 1.0f;
        float prev_lam = 1.0f;
        float prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) {
                parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);
                ls_done = true;
                break;
            }

            // Trial position (parallel)
            for (int i = (int)tid; i < n_vars; i += (int)tpm) {
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Compute trial energy (parallel, energy-only — biggest threadgroup win)
            float local_trial_e = 0.0f;
            for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm) {
                local_trial_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
            }
            for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm) {
                local_trial_e += chiral_violation_e(out_pos,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
            }
            for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm) {
                local_trial_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim);
            }
            shared[tid] = local_trial_e;
            float trial_e = tg_reduce_sum(shared, tid, tpm);

            // Armijo condition
            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e;
                ls_done = true;
            } else {
                // Backtrack (only thread 0 computes, all threads read result)
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
            parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);
        }

        // s_k = pos - old_pos (store in old_pos temporarily, then copy to history)
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            my_old_pos[i] = my_pos[i] - my_old_pos[i];  // s_k
        }
        threadgroup_barrier(mem_flags::mem_device);

        // === TOLX CHECK (parallel max) ===
        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > local_tolx) local_tolx = t;
        }
        shared[tid] = local_tolx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpm / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shared[0] < TOLX) {
            status = 0;
            break;
        }

        // === SAVE OLD GRADIENT, COMPUTE NEW ENERGY (parallel) + GRADIENT (thread 0) ===
        parallel_copy(my_old_grad, my_grad, n_vars, tid, tpm);
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm);

        // Energy: parallel across all threads
        float local_new_e = 0.0f;
        for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
            local_new_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
        for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
            local_new_e += chiral_violation_e(out_pos,
                chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
        for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
            local_new_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim);
        shared[tid] = local_new_e;
        energy = tg_reduce_sum(shared, tid, tpm);

        // Gradient: thread 0 only
        if (tid == 0) {
            for (int t = dist_start; t < dist_end; t++)
                dist_violation_g(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim);
            for (int t = chiral_start_t; t < chiral_end_t; t++)
                chiral_violation_g(out_pos, work_grad,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim);
            for (int t = fourth_start_t; t < fourth_end_t; t++)
                fourth_dim_g(out_pos, work_grad, fourth_idx_arr[t], fourth_dim_weight, dim);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // === GRADIENT CONVERGENCE CHECK (parallel) ===
        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (t > local_grad_test) local_grad_test = t;
        }
        shared[tid] = local_grad_test;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpm / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shared[0] / max(energy, 1.0f) < grad_tol) {
            status = 0;
            break;
        }

        // === L-BFGS UPDATE ===
        // y_k = grad_new - grad_old (compute into my_q temporarily)
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            my_q[i] = my_grad[i] - my_old_grad[i];  // y_k
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Curvature check: dot(y_k, s_k) > 0
        float ys_dot = parallel_dot(my_q, my_old_pos, n_vars, tid, tpm, shared);

        if (ys_dot > 1e-10f) {
            // Store s_k and y_k in circular buffer
            int slot = hist_idx % lbfgs_m;
            device float* s_slot = &my_S[slot * n_vars];
            device float* y_slot = &my_Y[slot * n_vars];
            parallel_copy(s_slot, my_old_pos, n_vars, tid, tpm);  // s_k
            parallel_copy(y_slot, my_q, n_vars, tid, tpm);        // y_k
            if (tid == 0) {
                my_rho[slot] = 1.0f / ys_dot;
            }
            threadgroup_barrier(mem_flags::mem_device);
            hist_idx++;
            if (hist_count < lbfgs_m) hist_count++;
        }

        // === L-BFGS TWO-LOOP RECURSION (parallel dot/saxpy) ===
        // q = grad
        parallel_copy(my_q, my_grad, n_vars, tid, tpm);

        // Temporary storage for alpha values (in threadgroup memory is limited,
        // use work_alpha array in device memory)
        device float* my_alpha = &work_alpha[mol_idx * lbfgs_m];

        // First loop: i = newest to oldest
        for (int j = hist_count - 1; j >= 0; j--) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            device float* s_j = &my_S[slot * n_vars];
            device float* y_j = &my_Y[slot * n_vars];
            float rho_j = my_rho[slot];

            float alpha_j = rho_j * parallel_dot(s_j, my_q, n_vars, tid, tpm, shared);
            if (tid == 0) my_alpha[j] = alpha_j;
            threadgroup_barrier(mem_flags::mem_device);

            // q -= alpha_j * y_j
            parallel_saxpy(my_q, -alpha_j, y_j, n_vars, tid, tpm);
        }

        // Scale q by gamma = dot(s_{k-1}, y_{k-1}) / dot(y_{k-1}, y_{k-1})
        if (hist_count > 0) {
            int newest = (hist_idx - 1) % lbfgs_m;
            if (newest < 0) newest += lbfgs_m;
            device float* s_new = &my_S[newest * n_vars];
            device float* y_new = &my_Y[newest * n_vars];
            float sy = parallel_dot(s_new, y_new, n_vars, tid, tpm, shared);
            float yy = parallel_dot(y_new, y_new, n_vars, tid, tpm, shared);
            float gamma = sy / max(yy, 1e-30f);
            parallel_scale(my_q, gamma, n_vars, tid, tpm);
        }

        // Second loop: i = oldest to newest
        for (int j = 0; j < hist_count; j++) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            device float* s_j = &my_S[slot * n_vars];
            device float* y_j = &my_Y[slot * n_vars];
            float rho_j = my_rho[slot];

            float beta_j = rho_j * parallel_dot(y_j, my_q, n_vars, tid, tpm, shared);
            float alpha_j = my_alpha[j];

            // q += (alpha_j - beta_j) * s_j
            parallel_saxpy(my_q, alpha_j - beta_j, s_j, n_vars, tid, tpm);
        }

        // direction = -q
        parallel_neg_copy(my_dir, my_q, n_vars, tid, tpm);
    }

    // Write outputs (only thread 0)
    if (tid == 0) {
        out_energies[mol_idx] = energy;
        out_statuses[mol_idx] = status;
    }
