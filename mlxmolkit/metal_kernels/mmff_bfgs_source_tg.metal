
    uint mol_idx = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint tg_size = threads_per_threadgroup.x;

    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];

    if ((int)mol_idx >= n_mols_cfg) return;

    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_terms = n_atoms * 3;
    int hess_start = hessian_starts[mol_idx];

    int ts_stride = n_mols_cfg + 1;
    int b_s = all_term_starts[0*ts_stride+mol_idx], b_e = all_term_starts[0*ts_stride+mol_idx+1];
    int a_s = all_term_starts[1*ts_stride+mol_idx], a_e = all_term_starts[1*ts_stride+mol_idx+1];
    int sb_s = all_term_starts[2*ts_stride+mol_idx], sb_e = all_term_starts[2*ts_stride+mol_idx+1];
    int o_s = all_term_starts[3*ts_stride+mol_idx], o_e = all_term_starts[3*ts_stride+mol_idx+1];
    int t_s = all_term_starts[4*ts_stride+mol_idx], t_e = all_term_starts[4*ts_stride+mol_idx+1];
    int v_s = all_term_starts[5*ts_stride+mol_idx], v_e = all_term_starts[5*ts_stride+mol_idx+1];
    int e_s = all_term_starts[6*ts_stride+mol_idx], e_e = all_term_starts[6*ts_stride+mol_idx+1];

    // Copy initial positions (parallel)
    for (int i = (int)tid; i < n_terms; i += (int)tg_size)
        out_pos[atom_start * 3 + i] = pos[atom_start * 3 + i];
    threadgroup_barrier(mem_flags::mem_device);

    device float* my_pos = &out_pos[atom_start * 3];
    device float* my_grad = &work_grad[atom_start * 3];
    device float* my_dir = &work_dir[atom_start * 3];
    device float* my_old_pos = &work_scratch[atom_start * 3];
    device float* my_dgrad = &work_scratch[total_pos_size + atom_start * 3];
    device float* my_hess_dg = &work_scratch[2 * total_pos_size + atom_start * 3];
    device float* my_H = &work_hessian[hess_start];

    // Only tg_reduce (work array) and tg_status_shared remain as threadgroup-shared
    threadgroup float tg_reduce[TG_SIZE_VAL];
    threadgroup float tg_grad_scale_shared;
    threadgroup int tg_status_shared;

    // Initialize Hessian to identity (ALL threads)
    int hess_total = n_terms * n_terms;
    for (int k = (int)tid; k < hess_total; k += (int)tg_size) {
        int i = k / n_terms, j = k % n_terms;
        my_H[k] = (i == j) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Serial energy+gradient macro — thread 0 only (gradient funcs use non-atomic +=)
    #define SEQ_COMPUTE_EG(OUT_E) \
        if (tid == 0) { \
            for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f; \
            OUT_E = 0.0f; \
            for (int t = b_s; t < b_e; t++) { \
                OUT_E += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
                bond_stretch_g(out_pos, work_grad, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
            } \
            for (int t = a_s; t < a_e; t++) { \
                bool lin = angle_params[t*3+2] > 0.5f; \
                OUT_E += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
                angle_bend_g(out_pos, work_grad, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
            } \
            for (int t = sb_s; t < sb_e; t++) { \
                OUT_E += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
                stretch_bend_g(out_pos, work_grad, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
            } \
            for (int t = o_s; t < o_e; t++) { \
                OUT_E += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
                oop_bend_g(out_pos, work_grad, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
            } \
            for (int t = t_s; t < t_e; t++) { \
                OUT_E += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
                torsion_g(out_pos, work_grad, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
            } \
            for (int t = v_s; t < v_e; t++) { \
                OUT_E += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
                vdw_g(out_pos, work_grad, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
            } \
            for (int t = e_s; t < e_e; t++) { \
                OUT_E += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
                ele_g(out_pos, work_grad, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_device);

    // Parallel energy-only macro — all threads sum a stripe, then reduce.
    // Replaces serial SEQ_COMPUTE_E. Energy functions are read-only on positions.
    #define PAR_COMPUTE_E(OUT_E) \
        { \
            float _local_e = 0.0f; \
            for (int t = b_s + (int)tid; t < b_e; t += (int)tg_size) \
                _local_e += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
            for (int t = a_s + (int)tid; t < a_e; t += (int)tg_size) { \
                bool lin = angle_params[t*3+2] > 0.5f; \
                _local_e += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \
            } \
            for (int t = sb_s + (int)tid; t < sb_e; t += (int)tg_size) \
                _local_e += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
            for (int t = o_s + (int)tid; t < o_e; t += (int)tg_size) \
                _local_e += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
            for (int t = t_s + (int)tid; t < t_e; t += (int)tg_size) \
                _local_e += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
            for (int t = v_s + (int)tid; t < v_e; t += (int)tg_size) \
                _local_e += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
            for (int t = e_s + (int)tid; t < e_e; t += (int)tg_size) \
                _local_e += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
            tg_reduce[tid] = _local_e; \
            OUT_E = tg_reduce_sum(tg_reduce, tid, tg_size); \
        }

    // ---- Initial energy + gradient (thread 0 computes, broadcast energy) ----
    float energy = 0.0f;
    SEQ_COMPUTE_EG(energy);
    if (tid == 0) {
        float grad_scale = 1.0f;
        scale_grad_serial(my_grad, n_terms, grad_scale, true);
        tg_grad_scale_shared = grad_scale;
        tg_reduce[0] = energy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    energy = tg_reduce[0];

    // Initial direction = -grad (ALL threads)
    parallel_neg_copy(my_dir, my_grad, n_terms, tid, tg_size);

    // Compute max step (parallel reduction)
    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_terms; i += (int)tg_size) local_sum_sq += my_pos[i] * my_pos[i];
    tg_reduce[tid] = local_sum_sq;
    float sum_sq = tg_reduce_sum(tg_reduce, tid, tg_size);
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_terms);

    if (tid == 0) tg_status_shared = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Main BFGS loop ----
    for (int iter = 0; iter < max_iters && tg_status_shared == 1; iter++) {

        // === LINE SEARCH ===
        parallel_copy(my_old_pos, my_pos, n_terms, tid, tg_size);
        float old_energy = energy;

        // Scale direction if too large (parallel norm)
        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) local_dir_sq += my_dir[i] * my_dir[i];
        tg_reduce[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(tg_reduce, tid, tg_size));
        if (dir_norm > max_step) {
            float sc = max_step / dir_norm;
            for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_dir[i] *= sc;
            threadgroup_barrier(mem_flags::mem_device);
        }

        // Slope = dir . grad (parallel dot — all threads get same result)
        float slope = parallel_dot(my_dir, my_grad, n_terms, tid, tg_size, tg_reduce);
        if (slope >= 0.0f) {
            if (tid == 0) tg_status_shared = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }

        // Lambda min (parallel max reduction)
        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float tv = ad / ap;
            if (tv > local_test_max) local_test_max = tv;
        }
        tg_reduce[tid] = local_test_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) tg_reduce[tid] = max(tg_reduce[tid], tg_reduce[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float lambda_min = MOVETOL / max(tg_reduce[0], 1e-30f);

        // Line search — ALL local variables (no threadgroup-shared state)
        // All threads get the same trial_e from PAR_COMPUTE_E reduction,
        // so they all independently compute the same lambda update.
        float lam = 1.0f;
        float prev_lam = 1.0f;
        float prev_e = old_energy;
        bool ls_done = false;

        for (int ls = 0; ls < MAX_LS_ITERS && !ls_done; ls++) {
            if (lam < lambda_min) {
                parallel_copy(my_pos, my_old_pos, n_terms, tid, tg_size);
                ls_done = true;
                break;
            }

            // Trial position (parallel)
            for (int i = (int)tid; i < n_terms; i += (int)tg_size)
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            // Trial energy (parallel — all threads get same value via reduction)
            float trial_e;
            PAR_COMPUTE_E(trial_e);

            // Armijo condition — all threads evaluate identically
            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e;
                ls_done = true;
            } else {
                // Cubic interpolation backtrack — pure arithmetic on identical scalars
                float tmp_lam;
                if (ls == 0) {
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
                            if (disc < 0.0f) tmp_lam = 0.5f * lam;
                            else if (b <= 0.0f) tmp_lam = (-b + sqrt(disc)) / (3.0f * a);
                            else tmp_lam = -slope / (b + sqrt(disc));
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
            parallel_copy(my_pos, my_old_pos, n_terms, tid, tg_size);
        }

        // xi = pos - old_pos (ALL threads)
        for (int i = (int)tid; i < n_terms; i += (int)tg_size)
            my_old_pos[i] = my_pos[i] - my_old_pos[i];
        threadgroup_barrier(mem_flags::mem_device);

        // TOLX check (parallel max)
        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float tv = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (tv > local_tolx) local_tolx = tv;
        }
        tg_reduce[tid] = local_tolx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) tg_reduce[tid] = max(tg_reduce[tid], tg_reduce[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tg_reduce[0] < TOLX) {
            if (tid == 0) tg_status_shared = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }

        // Save old grad, compute new energy+gradient (thread 0), broadcast energy
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_dgrad[i] = my_grad[i];
        threadgroup_barrier(mem_flags::mem_device);
        float new_e = 0.0f;
        SEQ_COMPUTE_EG(new_e);
        if (tid == 0) {
            float grad_scale = tg_grad_scale_shared;
            scale_grad_serial(my_grad, n_terms, grad_scale, false);
            tg_grad_scale_shared = grad_scale;
            tg_reduce[0] = new_e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        energy = tg_reduce[0];
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) my_dgrad[i] = my_grad[i] - my_dgrad[i];
        threadgroup_barrier(mem_flags::mem_device);

        // Grad convergence check (parallel max)
        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float tv = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (tv > local_grad_test) local_grad_test = tv;
        }
        tg_reduce[tid] = local_grad_test;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) tg_reduce[tid] = max(tg_reduce[tid], tg_reduce[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tg_reduce[0] / max(energy * tg_grad_scale_shared, 1.0f) < grad_tol) {
            if (tid == 0) tg_status_shared = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }

        // BFGS: hessDGrad = H @ dGrad (ALL threads, parallel rows)
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float s = 0;
            for (int j = 0; j < n_terms; j++) s += my_H[i * n_terms + j] * my_dgrad[j];
            my_hess_dg[i] = s;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // 4 dot products (parallel)
        threadgroup float tg_fac[TG_SIZE_VAL], tg_fae[TG_SIZE_VAL], tg_sdg[TG_SIZE_VAL], tg_sxi[TG_SIZE_VAL];
        tg_fac[tid] = 0; tg_fae[tid] = 0; tg_sdg[tid] = 0; tg_sxi[tid] = 0;
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            tg_fac[tid] += my_dgrad[i] * my_old_pos[i];
            tg_fae[tid] += my_dgrad[i] * my_hess_dg[i];
            tg_sdg[tid] += my_dgrad[i] * my_dgrad[i];
            tg_sxi[tid] += my_old_pos[i] * my_old_pos[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                tg_fac[tid] += tg_fac[tid + s];
                tg_fae[tid] += tg_fae[tid + s];
                tg_sdg[tid] += tg_sdg[tid + s];
                tg_sxi[tid] += tg_sxi[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float fac = tg_fac[0], fae = tg_fae[0], sum_dg = tg_sdg[0], sum_xi = tg_sxi[0];

        if (fac * fac > EPS_GUARD * sum_dg * sum_xi && fac > 0) {
            float fi = 1.0f / fac, fei = 1.0f / fae;
            // Rank-2 Hessian update (ALL threads, parallel matrix elements)
            for (int k = (int)tid; k < hess_total; k += (int)tg_size) {
                int i = k / n_terms, j = k % n_terms;
                float xi_i = my_old_pos[i], xi_j = my_old_pos[j];
                float hd_i = my_hess_dg[i], hd_j = my_hess_dg[j];
                float ai = fi * xi_i - fei * hd_i, aj = fi * xi_j - fei * hd_j;
                my_H[k] += fi * xi_i * xi_j - fei * hd_i * hd_j + fae * ai * aj;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }

        // Direction = -H @ grad (ALL threads, parallel rows)
        for (int i = (int)tid; i < n_terms; i += (int)tg_size) {
            float s = 0;
            for (int j = 0; j < n_terms; j++) s += my_H[i * n_terms + j] * my_grad[j];
            my_dir[i] = -s;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    if (tid == 0) {
        out_energies[mol_idx] = energy;
        out_statuses[mol_idx] = tg_status_shared;
    }
