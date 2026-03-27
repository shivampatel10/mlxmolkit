
// ---- same_side: check if v4 and p0 are on the same side of plane(v1,v2,v3) ----
inline bool same_side(float3 v1, float3 v2, float3 v3, float3 v4, float3 p0, float tol) {
    float3 edge1 = v2 - v1;
    float3 edge2 = v3 - v1;
    float3 cr = float3(edge1.y*edge2.z - edge1.z*edge2.y,
                        edge1.z*edge2.x - edge1.x*edge2.z,
                        edge1.x*edge2.y - edge1.y*edge2.x);
    float3 dv4 = v4 - v1;
    float3 dp0 = p0 - v1;
    float d1 = cr.x*dv4.x + cr.y*dv4.y + cr.z*dv4.z;
    float d2 = cr.x*dp0.x + cr.y*dp0.y + cr.z*dp0.z;
    if (abs(d1) < tol || abs(d2) < tol) return false;
    return (d1 > 0.0f) == (d2 > 0.0f);
}

inline float3 read_pos3(const device float* pos, int idx, int dim) {
    return float3(pos[idx*dim], pos[idx*dim+1], pos[idx*dim+2]);
}

inline float3 normalize_safe(float3 v) {
    float n = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return (n > 1e-10f) ? v / n : v;
}

inline float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 cross3(float3 a, float3 b) {
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
// ---- STEREO_CHECKS_SPLIT ----

    uint t = thread_position_in_grid.x;
    int n_terms = (int)config[0];
    int dim = (int)config[1];
    float tol = config[2];
    float min_vol = config[3];
    bool do_volume_test = config[4] > 0.5f;

    if ((int)t >= n_terms) return;

    int mol_idx = mol_indices[t];
    if (active[mol_idx] < 0.5f) return;
    if (failed[mol_idx] > 0.5f) return;

    int i0 = idx0[t], i1 = idx1[t], i2 = idx2[t], i3 = idx3[t], i4 = idx4[t];

    // Skip 3-coordinate centers (idx0 == idx4)
    if (i0 == i4) return;

    float3 p0 = read_pos3(pos, i0, dim);
    float3 p1 = read_pos3(pos, i1, dim);
    float3 p2 = read_pos3(pos, i2, dim);
    float3 p3 = read_pos3(pos, i3, dim);
    float3 p4 = read_pos3(pos, i4, dim);

    if (do_volume_test) {
        float vol_scale = (in_fused[t] > 0.5f) ? 0.25f : 1.0f;
        float threshold = vol_scale * min_vol;

        float3 d1 = normalize_safe(p0 - p1);
        float3 d2 = normalize_safe(p0 - p2);
        float3 d3 = normalize_safe(p0 - p3);
        float3 d4 = normalize_safe(p0 - p4);

        // 4 cross-dot volume tests
        if (abs(dot3(cross3(d1, d2), d3)) < threshold) { failed[mol_idx] = 1.0f; return; }
        if (abs(dot3(cross3(d1, d2), d4)) < threshold) { failed[mol_idx] = 1.0f; return; }
        if (abs(dot3(cross3(d1, d3), d4)) < threshold) { failed[mol_idx] = 1.0f; return; }
        if (abs(dot3(cross3(d2, d3), d4)) < threshold) { failed[mol_idx] = 1.0f; return; }
    }

    // Center-in-volume: 4 same_side checks
    if (!same_side(p1, p2, p3, p4, p0, tol)) { failed[mol_idx] = 1.0f; return; }
    if (!same_side(p2, p3, p4, p1, p0, tol)) { failed[mol_idx] = 1.0f; return; }
    if (!same_side(p3, p4, p1, p2, p0, tol)) { failed[mol_idx] = 1.0f; return; }
    if (!same_side(p4, p1, p2, p3, p0, tol)) { failed[mol_idx] = 1.0f; return; }
// ---- STEREO_CHECKS_SPLIT ----

    uint t = thread_position_in_grid.x;
    int n_terms = (int)config[0];
    int dim = (int)config[1];

    if ((int)t >= n_terms) return;

    int mol_idx = mol_indices[t];
    if (active[mol_idx] < 0.5f) return;
    if (failed[mol_idx] > 0.5f) return;

    float3 p1 = read_pos3(pos, idx1[t], dim);
    float3 p2 = read_pos3(pos, idx2[t], dim);
    float3 p3 = read_pos3(pos, idx3[t], dim);
    float3 p4 = read_pos3(pos, idx4[t], dim);

    float3 v1 = p1 - p4;
    float3 v2 = p2 - p4;
    float3 v3 = p3 - p4;
    float vol = dot3(v1, cross3(v2, v3));

    float lb = vol_lower[t];
    float ub = vol_upper[t];

    // Check lower bound
    if (lb > 0.0f && vol < lb) {
        bool wrong_sign = (vol < 0.0f) != (lb < 0.0f);
        if (vol / lb < 0.8f || wrong_sign) {
            failed[mol_idx] = 1.0f;
            return;
        }
    }

    // Check upper bound
    if (ub < 0.0f && vol > ub) {
        bool wrong_sign = (vol < 0.0f) != (ub < 0.0f);
        if (vol / ub < 0.8f || wrong_sign) {
            failed[mol_idx] = 1.0f;
            return;
        }
    }
// ---- STEREO_CHECKS_SPLIT ----

    uint t = thread_position_in_grid.x;
    int n_terms = (int)config[0];
    int dim = (int)config[1];
    float linear_tol = config[2];

    if ((int)t >= n_terms) return;

    int mol_idx = mol_indices[t];
    if (active[mol_idx] < 0.5f) return;
    if (failed[mol_idx] > 0.5f) return;

    float3 p0 = read_pos3(pos, idx0[t], dim);
    float3 p1 = read_pos3(pos, idx1[t], dim);
    float3 p2 = read_pos3(pos, idx2[t], dim);

    float3 d1 = p1 - p0;
    float3 d2 = p1 - p2;
    float n1 = sqrt(dot3(d1, d1));
    float n2 = sqrt(dot3(d2, d2));
    if (n1 < 1e-10f || n2 < 1e-10f) return;
    d1 = d1 / n1;
    d2 = d2 / n2;

    float dot_val = dot3(d1, d2);
    if ((dot_val + 1.0f) < linear_tol) {
        failed[mol_idx] = 1.0f;
    }
// ---- STEREO_CHECKS_SPLIT ----

    uint t = thread_position_in_grid.x;
    int n_terms = (int)config[0];
    int dim = (int)config[1];

    if ((int)t >= n_terms) return;

    int mol_idx = mol_indices[t];
    if (active[mol_idx] < 0.5f) return;
    if (failed[mol_idx] > 0.5f) return;

    float3 p0 = read_pos3(pos, idx0[t], dim);
    float3 p1 = read_pos3(pos, idx1[t], dim);
    float3 p2 = read_pos3(pos, idx2[t], dim);
    float3 p3 = read_pos3(pos, idx3[t], dim);
    float sign_val = signs[t];

    float3 d1 = p2 - p1;  // bond vector
    float3 d2 = p0 - p1;  // substituent on atom 1
    float3 d3 = p3 - p2;  // substituent on atom 2

    float3 c1 = cross3(d2, d1);
    float3 c2 = cross3(d3, d1);

    float l1sq = dot3(c1, c1);
    float l2sq = dot3(c2, c2);
    float denom = sqrt(l1sq * l2sq);
    if (denom < 1e-16f) return;

    float dot_val = clamp(dot3(c1, c2) / denom, -1.0f, 1.0f);
    float angle = acos(dot_val);

    if ((angle - 1.5707963267948966f) * sign_val < 0.0f) {
        failed[mol_idx] = 1.0f;
    }
// ---- STEREO_CHECKS_SPLIT ----

    uint t = thread_position_in_grid.x;
    int n_terms = (int)config[0];
    int dim = (int)config[1];

    if ((int)t >= n_terms) return;

    int mol_idx = mol_indices[t];
    if (active[mol_idx] < 0.5f) return;
    if (failed[mol_idx] > 0.5f) return;

    float3 p0 = read_pos3(pos, idx0[t], dim);
    float3 p1 = read_pos3(pos, idx1[t], dim);
    float3 diff = p0 - p1;
    float dist = sqrt(dot3(diff, diff));

    float lb = lower[t];
    float ub = upper[t];

    if (dist < lb && abs(dist - lb) > 0.1f * ub) {
        failed[mol_idx] = 1.0f;
        return;
    }
    if (dist > ub && abs(dist - ub) > 0.1f * ub) {
        failed[mol_idx] = 1.0f;
    }
