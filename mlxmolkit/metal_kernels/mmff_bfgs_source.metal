
    uint mol_idx = thread_position_in_grid.x;

    // Read config
    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];

    if ((int)mol_idx >= n_mols_cfg) return;

    // Molecule boundaries
    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_terms = n_atoms * 3;

    // Hessian offset
    int hess_start = hessian_starts[mol_idx];

    // Term boundaries for each of the 7 term types (packed in all_term_starts)
    // Layout: 7 consecutive (n_mols+1) arrays
    int ts_stride = n_mols_cfg + 1;
    int b_s = all_term_starts[0*ts_stride+mol_idx], b_e = all_term_starts[0*ts_stride+mol_idx+1];
    int a_s = all_term_starts[1*ts_stride+mol_idx], a_e = all_term_starts[1*ts_stride+mol_idx+1];
    int sb_s = all_term_starts[2*ts_stride+mol_idx], sb_e = all_term_starts[2*ts_stride+mol_idx+1];
    int o_s = all_term_starts[3*ts_stride+mol_idx], o_e = all_term_starts[3*ts_stride+mol_idx+1];
    int t_s = all_term_starts[4*ts_stride+mol_idx], t_e = all_term_starts[4*ts_stride+mol_idx+1];
    int v_s = all_term_starts[5*ts_stride+mol_idx], v_e = all_term_starts[5*ts_stride+mol_idx+1];
    int e_s = all_term_starts[6*ts_stride+mol_idx], e_e = all_term_starts[6*ts_stride+mol_idx+1];

    // Copy initial positions to output
    for (int i = 0; i < n_terms; i++) {
        out_pos[atom_start * 3 + i] = pos[atom_start * 3 + i];
    }

    // Working pointers
    device float* my_pos = &out_pos[atom_start * 3];
    device float* my_grad = &work_grad[atom_start * 3];
    device float* my_dir = &work_dir[atom_start * 3];
    device float* my_old_pos = &work_scratch[atom_start * 3];
    device float* my_dgrad = &work_scratch[total_pos_size + atom_start * 3];
    device float* my_hess_dg = &work_scratch[2 * total_pos_size + atom_start * 3];
    device float* my_H = &work_hessian[hess_start];

    // Initialize Hessian to identity
    for (int i = 0; i < n_terms; i++)
        for (int j = 0; j < n_terms; j++)
            my_H[i * n_terms + j] = (i == j) ? 1.0f : 0.0f;

    // ---- Macro to compute energy + gradient for all 7 terms ----
    // (used at initial compute and after each line search step)
    #define COMPUTE_ENERGY_GRAD(OUT_E) \
        for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f; \
        OUT_E = 0.0f; \
        for (int t = b_s; t < b_e; t++) { \
            int a1=bond_pairs[t*2], a2=bond_pairs[t*2+1]; \
            OUT_E += bond_stretch_e(out_pos, a1, a2, bond_params[t*2], bond_params[t*2+1]); \
            bond_stretch_g(out_pos, work_grad, a1, a2, bond_params[t*2], bond_params[t*2+1]); \
        } \
        for (int t = a_s; t < a_e; t++) { \
            int a1=angle_trips[t*3], a2=angle_trips[t*3+1], a3=angle_trips[t*3+2]; \
            bool lin = angle_params[t*3+2] > 0.5f; \
            OUT_E += angle_bend_e(out_pos, a1, a2, a3, angle_params[t*3], angle_params[t*3+1], lin); \
            angle_bend_g(out_pos, work_grad, a1, a2, a3, angle_params[t*3], angle_params[t*3+1], lin); \
        } \
        for (int t = sb_s; t < sb_e; t++) { \
            int a1=sb_trips[t*3], a2=sb_trips[t*3+1], a3=sb_trips[t*3+2]; \
            OUT_E += stretch_bend_e(out_pos, a1, a2, a3, sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
            stretch_bend_g(out_pos, work_grad, a1, a2, a3, sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
        } \
        for (int t = o_s; t < o_e; t++) { \
            int a1=oop_quads[t*4], a2=oop_quads[t*4+1], a3=oop_quads[t*4+2], a4=oop_quads[t*4+3]; \
            OUT_E += oop_bend_e(out_pos, a1, a2, a3, a4, oop_params[t]); \
            oop_bend_g(out_pos, work_grad, a1, a2, a3, a4, oop_params[t]); \
        } \
        for (int t = t_s; t < t_e; t++) { \
            int a1=tor_quads[t*4], a2=tor_quads[t*4+1], a3=tor_quads[t*4+2], a4=tor_quads[t*4+3]; \
            OUT_E += torsion_e(out_pos, a1, a2, a3, a4, tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
            torsion_g(out_pos, work_grad, a1, a2, a3, a4, tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
        } \
        for (int t = v_s; t < v_e; t++) { \
            int a1=vdw_pairs[t*2], a2=vdw_pairs[t*2+1]; \
            OUT_E += vdw_e(out_pos, a1, a2, vdw_params[t*2], vdw_params[t*2+1]); \
            vdw_g(out_pos, work_grad, a1, a2, vdw_params[t*2], vdw_params[t*2+1]); \
        } \
        for (int t = e_s; t < e_e; t++) { \
            int a1=ele_pairs[t*2], a2=ele_pairs[t*2+1]; \
            OUT_E += ele_e(out_pos, a1, a2, ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
            ele_g(out_pos, work_grad, a1, a2, ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \
        }

    // Macro for energy-only (line search trial)
    #define COMPUTE_ENERGY_ONLY(OUT_E) \
        OUT_E = 0.0f; \
        for (int t = b_s; t < b_e; t++) \
            OUT_E += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \
        for (int t = a_s; t < a_e; t++) \
            OUT_E += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], angle_params[t*3+2]>0.5f); \
        for (int t = sb_s; t < sb_e; t++) \
            OUT_E += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \
        for (int t = o_s; t < o_e; t++) \
            OUT_E += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \
        for (int t = t_s; t < t_e; t++) \
            OUT_E += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \
        for (int t = v_s; t < v_e; t++) \
            OUT_E += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \
        for (int t = e_s; t < e_e; t++) \
            OUT_E += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f);

    // ---- Compute initial energy + gradient ----
    float energy;
    float grad_scale = 1.0f;
    COMPUTE_ENERGY_GRAD(energy);
    scale_grad_serial(my_grad, n_terms, grad_scale, true);

    // Initial direction = -grad
    for (int i = 0; i < n_terms; i++) my_dir[i] = -my_grad[i];

    // Compute max step
    float sum_sq = 0.0f;
    for (int i = 0; i < n_terms; i++) sum_sq += my_pos[i] * my_pos[i];
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_terms);

    int status = 1;

    // ---- Main BFGS loop ----
    for (int iter = 0; iter < max_iters && status == 1; iter++) {

        // === LINE SEARCH ===
        for (int i = 0; i < n_terms; i++) my_old_pos[i] = my_pos[i];
        float old_energy = energy;

        float dir_norm_sq = 0.0f;
        for (int i = 0; i < n_terms; i++) dir_norm_sq += my_dir[i] * my_dir[i];
        float dir_norm = sqrt(dir_norm_sq);
        if (dir_norm > max_step) {
            float s = max_step / dir_norm;
            for (int i = 0; i < n_terms; i++) my_dir[i] *= s;
        }

        float slope = 0.0f;
        for (int i = 0; i < n_terms; i++) slope += my_dir[i] * my_grad[i];
        if (slope >= 0.0f) { status = 0; break; }

        float test_max = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float tv = ad / ap;
            if (tv > test_max) test_max = tv;
        }
        float lambda_min = MOVETOL / max(test_max, 1e-30f);

        float lam = 1.0f, prev_lam = 1.0f, prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) {
                for (int i = 0; i < n_terms; i++) my_pos[i] = my_old_pos[i];
                ls_done = true;
                break;
            }
            for (int i = 0; i < n_terms; i++) my_pos[i] = my_old_pos[i] + lam * my_dir[i];

            float trial_e;
            COMPUTE_ENERGY_ONLY(trial_e);

            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e;
                ls_done = true;
            } else {
                float tmp_lam;
                if (ls_iter == 0) {
                    tmp_lam = -slope / (2.0f * (trial_e - old_energy - slope));
                } else {
                    float rhs1 = trial_e - old_energy - lam * slope;
                    float rhs2 = prev_e - old_energy - prev_lam * slope;
                    float lam_sq = lam*lam, lam2_sq = prev_lam*prev_lam;
                    float denom_v = lam - prev_lam;
                    if (abs(denom_v) < 1e-30f) { tmp_lam = 0.5f*lam; }
                    else {
                        float a = (rhs1/lam_sq - rhs2/lam2_sq) / denom_v;
                        float b = (-prev_lam*rhs1/lam_sq + lam*rhs2/lam2_sq) / denom_v;
                        if (abs(a) < 1e-30f) tmp_lam = (abs(b)>1e-30f) ? -slope/(2.0f*b) : 0.5f*lam;
                        else {
                            float disc = b*b - 3.0f*a*slope;
                            if (disc < 0.0f) tmp_lam = 0.5f*lam;
                            else if (b <= 0.0f) tmp_lam = (-b+sqrt(disc))/(3.0f*a);
                            else tmp_lam = -slope/(b+sqrt(disc));
                        }
                    }
                }
                tmp_lam = min(tmp_lam, 0.5f*lam);
                tmp_lam = max(tmp_lam, 0.1f*lam);
                prev_lam = lam; prev_e = trial_e; lam = tmp_lam;
            }
        }
        if (!ls_done) {
            for (int i = 0; i < n_terms; i++) my_pos[i] = my_old_pos[i];
        }

        // xi = pos - old_pos
        for (int i = 0; i < n_terms; i++) my_old_pos[i] = my_pos[i] - my_old_pos[i];

        // TOLX check
        float tolx_test = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float tv = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (tv > tolx_test) tolx_test = tv;
        }
        if (tolx_test < TOLX) { status = 0; break; }

        // New gradient
        for (int i = 0; i < n_terms; i++) { my_dgrad[i] = my_grad[i]; }
        COMPUTE_ENERGY_GRAD(energy);
        scale_grad_serial(my_grad, n_terms, grad_scale, false);
        for (int i = 0; i < n_terms; i++) my_dgrad[i] = my_grad[i] - my_dgrad[i];

        // Gradient convergence check
        float grad_test = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            float tv = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (tv > grad_test) grad_test = tv;
        }
        if (grad_test / max(energy * grad_scale, 1.0f) < grad_tol) { status = 0; break; }

        // BFGS Hessian update
        for (int i = 0; i < n_terms; i++) {
            float s = 0.0f;
            for (int j = 0; j < n_terms; j++) s += my_H[i*n_terms+j] * my_dgrad[j];
            my_hess_dg[i] = s;
        }
        float fac=0.0f, fae=0.0f, sum_dg=0.0f, sum_xi=0.0f;
        for (int i = 0; i < n_terms; i++) {
            fac += my_dgrad[i]*my_old_pos[i];
            fae += my_dgrad[i]*my_hess_dg[i];
            sum_dg += my_dgrad[i]*my_dgrad[i];
            sum_xi += my_old_pos[i]*my_old_pos[i];
        }
        if (fac*fac > EPS_GUARD*sum_dg*sum_xi && fac > 0.0f) {
            float fi = 1.0f/fac, fei = 1.0f/fae;
            for (int i = 0; i < n_terms; i++) {
                for (int j = 0; j < n_terms; j++) {
                    float xi_i=my_old_pos[i], xi_j=my_old_pos[j];
                    float hd_i=my_hess_dg[i], hd_j=my_hess_dg[j];
                    float ai = fi*xi_i - fei*hd_i;
                    float aj = fi*xi_j - fei*hd_j;
                    my_H[i*n_terms+j] += fi*xi_i*xi_j - fei*hd_i*hd_j + fae*ai*aj;
                }
            }
        }

        // New direction = -H @ grad
        for (int i = 0; i < n_terms; i++) {
            float s = 0.0f;
            for (int j = 0; j < n_terms; j++) s += my_H[i*n_terms+j] * my_grad[j];
            my_dir[i] = -s;
        }
    }

    out_energies[mol_idx] = energy;
    out_statuses[mol_idx] = status;
