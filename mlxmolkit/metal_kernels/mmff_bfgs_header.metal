
// ---- Constants ----
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float EPS_GUARD = 3e-7f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;
constant float GRAD_SCALE_INIT = 0.1f;
constant float GRAD_CAP = 10.0f;
constant float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;
constant float RAD_TO_DEG = 180.0f / 3.14159265358979323846f;

// ---- Bond stretch energy for one pair ----
inline float bond_stretch_e(
    const device float* pos, int i1, int i2, float kb, float r0
) {
    float dx = pos[i1*3+0] - pos[i2*3+0];
    float dy = pos[i1*3+1] - pos[i2*3+1];
    float dz = pos[i1*3+2] - pos[i2*3+2];
    float d = sqrt(dx*dx + dy*dy + dz*dz);
    float dr = d - r0;
    float dr2 = dr * dr;
    // cs = -2, cs2 = 7/3
    return (143.9325f / 2.0f) * kb * dr2 * (1.0f - 2.0f * dr + (7.0f/3.0f) * dr2);
}

// ---- Bond stretch gradient for one pair ----
inline void bond_stretch_g(
    const device float* pos, device float* grad,
    int i1, int i2, float kb, float r0
) {
    float dx = pos[i1*3+0] - pos[i2*3+0];
    float dy = pos[i1*3+1] - pos[i2*3+1];
    float dz = pos[i1*3+2] - pos[i2*3+2];
    float d2 = dx*dx + dy*dy + dz*dz;
    float d = sqrt(max(d2, 1e-16f));
    float inv_d = (d > 1e-8f) ? 1.0f / d : 0.0f;
    float dr = d - r0;
    // dE/dr = 143.9325 * kb * dr * (1 - 3*dr + 14/3*dr^2)
    float de_dr = 143.9325f * kb * dr * (1.0f - 3.0f * dr + (14.0f/3.0f) * dr * dr);
    float f = de_dr * inv_d;
    float gx = f * dx, gy = f * dy, gz = f * dz;
    grad[i1*3+0] += gx; grad[i1*3+1] += gy; grad[i1*3+2] += gz;
    grad[i2*3+0] -= gx; grad[i2*3+1] -= gy; grad[i2*3+2] -= gz;
}

// ---- Angle internals struct (no references in MSL) ----
struct AngleData {
    float cosT;
    float theta_deg;
    float inv_d1;
    float inv_d2;
    float r1[3];
    float r2[3];
};

inline AngleData angle_internals(
    const device float* pos, int i1, int i2, int i3
) {
    AngleData ad;
    for (int d = 0; d < 3; d++) {
        ad.r1[d] = pos[i1*3+d] - pos[i2*3+d];
        ad.r2[d] = pos[i3*3+d] - pos[i2*3+d];
    }
    float d1sq = ad.r1[0]*ad.r1[0] + ad.r1[1]*ad.r1[1] + ad.r1[2]*ad.r1[2];
    float d2sq = ad.r2[0]*ad.r2[0] + ad.r2[1]*ad.r2[1] + ad.r2[2]*ad.r2[2];
    ad.inv_d1 = rsqrt(max(d1sq, 1e-16f));
    ad.inv_d2 = rsqrt(max(d2sq, 1e-16f));
    float dot = ad.r1[0]*ad.r2[0] + ad.r1[1]*ad.r2[1] + ad.r1[2]*ad.r2[2];
    ad.cosT = clamp(dot * ad.inv_d1 * ad.inv_d2, -1.0f, 1.0f);
    ad.theta_deg = RAD_TO_DEG * acos(ad.cosT);
    return ad;
}

// ---- Angle bend energy for one term ----
inline float angle_bend_e(
    const device float* pos, int i1, int i2, int i3,
    float ka, float theta0, bool is_linear
) {
    AngleData ad = angle_internals(pos, i1, i2, i3);
    if (is_linear) {
        return 143.9325f * ka * (1.0f + ad.cosT);
    }
    float dtheta = ad.theta_deg - theta0;
    float cb = -0.4f * DEG_TO_RAD;
    return 0.5f * 143.9325f * DEG_TO_RAD * DEG_TO_RAD * ka * dtheta * dtheta * (1.0f + cb * dtheta);
}

// ---- Angle bend gradient for one term ----
inline void angle_bend_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3,
    float ka, float theta0, bool is_linear
) {
    AngleData ad = angle_internals(pos, i1, i2, i3);
    float sinTsq = 1.0f - ad.cosT * ad.cosT;
    if (sinTsq < 1e-16f) return;
    float inv_neg_sinT = -rsqrt(max(sinTsq, 1e-16f));
    float dtheta = ad.theta_deg - theta0;

    float de;
    if (is_linear) {
        float sinT = sqrt(max(sinTsq, 0.0f));
        de = -143.9325f * ka * sinT;
    } else {
        float c1 = 143.9325f * DEG_TO_RAD;
        float cbf = -0.4f * DEG_TO_RAD * 1.5f;
        de = c1 * ka * dtheta * (1.0f + cbf * dtheta);
    }
    float cf = de * inv_neg_sinT;

    float r1h[3], r2h[3];
    for (int d = 0; d < 3; d++) { r1h[d] = ad.r1[d] * ad.inv_d1; r2h[d] = ad.r2[d] * ad.inv_d2; }

    for (int d = 0; d < 3; d++) {
        float inter1 = ad.inv_d1 * (r2h[d] - ad.cosT * r1h[d]);
        float inter3 = ad.inv_d2 * (r1h[d] - ad.cosT * r2h[d]);
        float g1 = cf * inter1;
        float g3 = cf * inter3;
        grad[i1*3+d] += g1;
        grad[i3*3+d] += g3;
        grad[i2*3+d] -= (g1 + g3);
    }
}

// ---- Stretch-bend energy for one term ----
inline float stretch_bend_e(
    const device float* pos, int i1, int i2, int i3,
    float r0_ij, float r0_kj, float theta0, float kba_ij, float kba_kj
) {
    AngleData ad = angle_internals(pos, i1, i2, i3);
    float d1 = 1.0f / max(ad.inv_d1, 1e-8f);
    float d2 = 1.0f / max(ad.inv_d2, 1e-8f);
    float dtheta = ad.theta_deg - theta0;
    return 2.51210f * dtheta * ((d1-r0_ij)*kba_ij + (d2-r0_kj)*kba_kj);
}

// ---- Stretch-bend gradient for one term ----
inline void stretch_bend_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3,
    float r0_ij, float r0_kj, float theta0, float kba_ij, float kba_kj
) {
    AngleData ad = angle_internals(pos, i1, i2, i3);
    float sinTsq = 1.0f - ad.cosT * ad.cosT;
    if (sinTsq < 1e-16f) return;
    float d1 = 1.0f / max(ad.inv_d1, 1e-8f);
    float d2 = 1.0f / max(ad.inv_d2, 1e-8f);
    float invSinT = min(rsqrt(max(sinTsq, 1e-16f)), 1e8f);
    float dtheta = ad.theta_deg - theta0;
    float dr_ij = d1 - r0_ij, dr_kj = d2 - r0_kj;

    float pf = 143.9325f * DEG_TO_RAD;
    float beis = RAD_TO_DEG * (kba_ij*dr_ij + kba_kj*dr_kj) * invSinT;

    float r1h[3], r2h[3];
    for (int d = 0; d < 3; d++) { r1h[d] = ad.r1[d]*ad.inv_d1; r2h[d] = ad.r2[d]*ad.inv_d2; }

    for (int d = 0; d < 3; d++) {
        float inter1 = ad.inv_d1 * (r2h[d] - ad.cosT*r1h[d]);
        float inter3 = ad.inv_d2 * (r1h[d] - ad.cosT*r2h[d]);
        grad[i1*3+d] += pf * (dtheta*r1h[d]*kba_ij - inter1*beis);
        grad[i3*3+d] += pf * (dtheta*r2h[d]*kba_kj - inter3*beis);
        grad[i2*3+d] += pf * (-dtheta*(r1h[d]*kba_ij+r2h[d]*kba_kj) + (inter1+inter3)*beis);
    }
}

// ---- OOP bend energy for one term ----
inline float oop_bend_e(
    const device float* pos, int i1, int i2, int i3, int i4, float koop
) {
    float rJI[3], rJK[3], rJL[3];
    for (int d = 0; d < 3; d++) {
        rJI[d] = pos[i1*3+d] - pos[i2*3+d];
        rJK[d] = pos[i3*3+d] - pos[i2*3+d];
        rJL[d] = pos[i4*3+d] - pos[i2*3+d];
    }
    float idJI = rsqrt(max(rJI[0]*rJI[0]+rJI[1]*rJI[1]+rJI[2]*rJI[2], 1e-16f));
    float idJK = rsqrt(max(rJK[0]*rJK[0]+rJK[1]*rJK[1]+rJK[2]*rJK[2], 1e-16f));
    float idJL = rsqrt(max(rJL[0]*rJL[0]+rJL[1]*rJL[1]+rJL[2]*rJL[2], 1e-16f));
    float nJI[3], nJK[3], nJL[3];
    for (int d = 0; d < 3; d++) { nJI[d]=rJI[d]*idJI; nJK[d]=rJK[d]*idJK; nJL[d]=rJL[d]*idJL; }
    // normal = (-nJI) x nJK
    float nx = (-nJI[1])*nJK[2] - (-nJI[2])*nJK[1];
    float ny = (-nJI[2])*nJK[0] - (-nJI[0])*nJK[2];
    float nz = (-nJI[0])*nJK[1] - (-nJI[1])*nJK[0];
    float inv_nl = rsqrt(max(nx*nx+ny*ny+nz*nz, 1e-16f));
    nx *= inv_nl; ny *= inv_nl; nz *= inv_nl;
    float sinChi = clamp(nJL[0]*nx + nJL[1]*ny + nJL[2]*nz, -1.0f, 1.0f);
    float chi_deg = RAD_TO_DEG * asin(sinChi);
    return 0.5f * 143.9325f * DEG_TO_RAD * DEG_TO_RAD * koop * chi_deg * chi_deg;
}

// ---- OOP bend gradient for one term ----
inline void oop_bend_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4, float koop
) {
    float rJI[3], rJK[3], rJL[3];
    for (int d = 0; d < 3; d++) {
        rJI[d] = pos[i1*3+d] - pos[i2*3+d];
        rJK[d] = pos[i3*3+d] - pos[i2*3+d];
        rJL[d] = pos[i4*3+d] - pos[i2*3+d];
    }
    float idJI = rsqrt(max(rJI[0]*rJI[0]+rJI[1]*rJI[1]+rJI[2]*rJI[2], 1e-16f));
    float idJK = rsqrt(max(rJK[0]*rJK[0]+rJK[1]*rJK[1]+rJK[2]*rJK[2], 1e-16f));
    float idJL = rsqrt(max(rJL[0]*rJL[0]+rJL[1]*rJL[1]+rJL[2]*rJL[2], 1e-16f));
    float dJI[3], dJK[3], dJL[3];
    for (int d = 0; d < 3; d++) { dJI[d]=rJI[d]*idJI; dJK[d]=rJK[d]*idJK; dJL[d]=rJL[d]*idJL; }

    float nx = (-dJI[1])*dJK[2] - (-dJI[2])*dJK[1];
    float ny = (-dJI[2])*dJK[0] - (-dJI[0])*dJK[2];
    float nz = (-dJI[0])*dJK[1] - (-dJI[1])*dJK[0];
    float inv_nl = rsqrt(max(nx*nx+ny*ny+nz*nz, 1e-16f));
    nx *= inv_nl; ny *= inv_nl; nz *= inv_nl;

    float sinChi = clamp(dJL[0]*nx + dJL[1]*ny + dJL[2]*nz, -1.0f, 1.0f);
    float cosChiSq = 1.0f - sinChi*sinChi;
    float invCosChi = cosChiSq > 0 ? rsqrt(max(cosChiSq, 1e-16f)) : 1e8f;
    float chi_deg = RAD_TO_DEG * asin(sinChi);

    float cosTheta = clamp(dJI[0]*dJK[0]+dJI[1]*dJK[1]+dJI[2]*dJK[2], -1.0f, 1.0f);
    float invSinTheta = rsqrt(max(1.0f - cosTheta*cosTheta, 1e-8f));

    float dE_dChi = 143.9325f * DEG_TO_RAD * koop * chi_deg;

    float term1 = invCosChi * invSinTheta;
    float term2 = sinChi * invCosChi * invSinTheta * invSinTheta;

    // t1 = dJL x dJK, t2 = dJI x dJL, t3 = dJK x dJI
    float t1[3], t2[3], t3[3];
    t1[0]=dJL[1]*dJK[2]-dJL[2]*dJK[1]; t1[1]=dJL[2]*dJK[0]-dJL[0]*dJK[2]; t1[2]=dJL[0]*dJK[1]-dJL[1]*dJK[0];
    t2[0]=dJI[1]*dJL[2]-dJI[2]*dJL[1]; t2[1]=dJI[2]*dJL[0]-dJI[0]*dJL[2]; t2[2]=dJI[0]*dJL[1]-dJI[1]*dJL[0];
    t3[0]=dJK[1]*dJI[2]-dJK[2]*dJI[1]; t3[1]=dJK[2]*dJI[0]-dJK[0]*dJI[2]; t3[2]=dJK[0]*dJI[1]-dJK[1]*dJI[0];

    for (int d = 0; d < 3; d++) {
        float tg1 = (t1[d]*term1 - (dJI[d]-dJK[d]*cosTheta)*term2) * idJI;
        float tg3 = (t2[d]*term1 - (dJK[d]-dJI[d]*cosTheta)*term2) * idJK;
        float tg4 = (t3[d]*term1 - dJL[d]*sinChi*invCosChi) * idJL;
        float g1 = dE_dChi * tg1;
        float g3 = dE_dChi * tg3;
        float g4 = dE_dChi * tg4;
        grad[i1*3+d] += g1;
        grad[i3*3+d] += g3;
        grad[i4*3+d] += g4;
        grad[i2*3+d] -= (g1 + g3 + g4);
    }
}

// ---- Torsion energy for one term ----
inline float torsion_e(
    const device float* pos, int i1, int i2, int i3, int i4,
    float V1, float V2, float V3
) {
    float dx1[3], dx2[3], dx4[3];
    for (int d = 0; d < 3; d++) {
        dx1[d] = pos[i1*3+d] - pos[i2*3+d];
        dx2[d] = pos[i3*3+d] - pos[i2*3+d];
        dx4[d] = pos[i4*3+d] - pos[i3*3+d];
    }
    float c1[3], c2[3];
    c1[0]=dx1[1]*dx2[2]-dx1[2]*dx2[1]; c1[1]=dx1[2]*dx2[0]-dx1[0]*dx2[2]; c1[2]=dx1[0]*dx2[1]-dx1[1]*dx2[0];
    c2[0]=(-dx2[1])*dx4[2]-(-dx2[2])*dx4[1]; c2[1]=(-dx2[2])*dx4[0]-(-dx2[0])*dx4[2]; c2[2]=(-dx2[0])*dx4[1]-(-dx2[1])*dx4[0];
    float n1sq = c1[0]*c1[0]+c1[1]*c1[1]+c1[2]*c1[2];
    float n2sq = c2[0]*c2[0]+c2[1]*c2[1]+c2[2]*c2[2];
    float comb = n1sq * n2sq;
    if (comb < 1e-16f) return 0.0f;
    float inv_len = rsqrt(comb);
    float cosPhi = clamp((c1[0]*c2[0]+c1[1]*c2[1]+c1[2]*c2[2]) * inv_len, -1.0f, 1.0f);
    float cos2 = 2.0f*cosPhi*cosPhi - 1.0f;
    float cos3 = cosPhi*(4.0f*cosPhi*cosPhi - 3.0f);
    return 0.5f * (V1*(1.0f+cosPhi) + V2*(1.0f-cos2) + V3*(1.0f+cos3));
}

// ---- Torsion gradient for one term ----
inline void torsion_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float V1, float V2, float V3
) {
    float dx1[3], dx2[3], dx4[3];
    for (int d = 0; d < 3; d++) {
        dx1[d] = pos[i1*3+d] - pos[i2*3+d];
        dx2[d] = pos[i3*3+d] - pos[i2*3+d];
        dx4[d] = pos[i4*3+d] - pos[i3*3+d];
    }
    float c1[3], c2[3];
    c1[0]=dx1[1]*dx2[2]-dx1[2]*dx2[1]; c1[1]=dx1[2]*dx2[0]-dx1[0]*dx2[2]; c1[2]=dx1[0]*dx2[1]-dx1[1]*dx2[0];
    c2[0]=(-dx2[1])*dx4[2]-(-dx2[2])*dx4[1]; c2[1]=(-dx2[2])*dx4[0]-(-dx2[0])*dx4[2]; c2[2]=(-dx2[0])*dx4[1]-(-dx2[1])*dx4[0];
    float n1sq = c1[0]*c1[0]+c1[1]*c1[1]+c1[2]*c1[2];
    float n2sq = c2[0]*c2[0]+c2[1]*c2[1]+c2[2]*c2[2];
    if (n1sq < 1e-30f || n2sq < 1e-30f) return;
    float in1 = min(rsqrt(n1sq), 1e5f);
    float in2 = min(rsqrt(n2sq), 1e5f);
    for (int d = 0; d < 3; d++) { c1[d]*=in1; c2[d]*=in2; }
    float cosPhi = clamp(c1[0]*c2[0]+c1[1]*c2[1]+c1[2]*c2[2], -1.0f, 1.0f);
    float sinPhiSq = 1.0f - cosPhi*cosPhi;
    float sinTerm = 0.0f;
    if (sinPhiSq > 0.0f) {
        float sin2 = 2.0f * cosPhi;
        float sin3 = 3.0f - 4.0f * sinPhiSq;
        sinTerm = 0.5f * (V1 - 2.0f*V2*sin2 + 3.0f*V3*sin3);
    }
    float dT0[3], dT1[3];
    for (int d = 0; d < 3; d++) {
        dT0[d] = in1 * (c2[d] - cosPhi*c1[d]);
        dT1[d] = in2 * (c1[d] - cosPhi*c2[d]);
    }
    // Atom 1
    grad[i1*3+0] += sinTerm*(dT0[2]*dx2[1]-dT0[1]*dx2[2]);
    grad[i1*3+1] += sinTerm*(dT0[0]*dx2[2]-dT0[2]*dx2[0]);
    grad[i1*3+2] += sinTerm*(dT0[1]*dx2[0]-dT0[0]*dx2[1]);
    // Atom 4
    grad[i4*3+0] += sinTerm*(dT1[1]*(-dx2[2])-dT1[2]*(-dx2[1]));
    grad[i4*3+1] += sinTerm*(dT1[2]*(-dx2[0])-dT1[0]*(-dx2[2]));
    grad[i4*3+2] += sinTerm*(dT1[0]*(-dx2[1])-dT1[1]*(-dx2[0]));
    // Atom 2
    grad[i2*3+0] += sinTerm*(dT0[1]*(dx2[2]-dx1[2])+dT0[2]*(dx1[1]-dx2[1])+dT1[1]*(-dx4[2])+dT1[2]*dx4[1]);
    grad[i2*3+1] += sinTerm*(dT0[0]*(dx1[2]-dx2[2])+dT0[2]*(dx2[0]-dx1[0])+dT1[0]*dx4[2]+dT1[2]*(-dx4[0]));
    grad[i2*3+2] += sinTerm*(dT0[0]*(dx2[1]-dx1[1])+dT0[1]*(dx1[0]-dx2[0])+dT1[0]*(-dx4[1])+dT1[1]*dx4[0]);
    // Atom 3
    grad[i3*3+0] += sinTerm*(dT0[1]*dx1[2]+dT0[2]*(-dx1[1])+dT1[1]*(dx4[2]+dx2[2])+dT1[2]*(-dx4[1]-dx2[1]));
    grad[i3*3+1] += sinTerm*(dT0[0]*(-dx1[2])+dT0[2]*dx1[0]+dT1[0]*(-dx4[2]-dx2[2])+dT1[2]*(dx4[0]+dx2[0]));
    grad[i3*3+2] += sinTerm*(dT0[0]*dx1[1]+dT0[1]*(-dx1[0])+dT1[0]*(dx4[1]+dx2[1])+dT1[1]*(-dx4[0]-dx2[0]));
}

// ---- Van der Waals energy (Buffered 14-7) for one pair ----
inline float vdw_e(
    const device float* pos, int i1, int i2, float R_star, float epsilon
) {
    float dx = pos[i1*3+0]-pos[i2*3+0];
    float dy = pos[i1*3+1]-pos[i2*3+1];
    float dz = pos[i1*3+2]-pos[i2*3+2];
    float d = sqrt(max(dx*dx+dy*dy+dz*dz, 1e-16f));
    float rho = d / max(R_star, 1e-8f);
    float rho7 = rho*rho*rho*rho*rho*rho*rho;
    float t = 1.07f / (rho + 0.07f);
    float t7 = t*t*t*t*t*t*t;
    return epsilon * t7 * (1.12f / (rho7 + 0.12f) - 2.0f);
}

// ---- Van der Waals gradient for one pair ----
inline void vdw_g(
    const device float* pos, device float* grad,
    int i1, int i2, float R_star, float epsilon
) {
    float dx = pos[i1*3+0]-pos[i2*3+0];
    float dy = pos[i1*3+1]-pos[i2*3+1];
    float dz = pos[i1*3+2]-pos[i2*3+2];
    float d2 = dx*dx+dy*dy+dz*dz;
    float d = sqrt(max(d2, 1e-16f));
    float inv_d = (d > 1e-8f) ? 1.0f / d : 0.0f;
    float inv_R = 1.0f / max(R_star, 1e-8f);
    float q = d * inv_R;
    float q2=q*q, q6=q2*q2*q2, q7=q6*q;
    float q7p = q7 + 0.12f;
    float inv_q7p = 1.0f / max(q7p, 1e-30f);
    float t = 1.07f / (q + 0.07f);
    float t2=t*t, t7=t2*t2*t2*t;
    float dE_dr = epsilon * inv_R * t7 * (-7.84f*q6*inv_q7p*inv_q7p + (-7.84f*inv_q7p+14.0f)/(q+0.07f));
    float f = dE_dr * inv_d;
    grad[i1*3+0]+=f*dx; grad[i1*3+1]+=f*dy; grad[i1*3+2]+=f*dz;
    grad[i2*3+0]-=f*dx; grad[i2*3+1]-=f*dy; grad[i2*3+2]-=f*dz;
}

// ---- Electrostatic energy for one pair ----
inline float ele_e(
    const device float* pos, int i1, int i2,
    float charge_term, int diel_model, bool is_1_4
) {
    float dx = pos[i1*3+0]-pos[i2*3+0];
    float dy = pos[i1*3+1]-pos[i2*3+1];
    float dz = pos[i1*3+2]-pos[i2*3+2];
    float d = sqrt(max(dx*dx+dy*dy+dz*dz, 1e-16f));
    float dpd = d + 0.05f;
    float denom = (diel_model == 2) ? dpd*dpd : dpd;
    float scale = is_1_4 ? 0.75f : 1.0f;
    return 332.0716f * charge_term * scale / denom;
}

// ---- Electrostatic gradient for one pair ----
inline void ele_g(
    const device float* pos, device float* grad,
    int i1, int i2,
    float charge_term, int diel_model, bool is_1_4
) {
    float dx = pos[i1*3+0]-pos[i2*3+0];
    float dy = pos[i1*3+1]-pos[i2*3+1];
    float dz = pos[i1*3+2]-pos[i2*3+2];
    float d2 = dx*dx+dy*dy+dz*dz;
    float d = sqrt(max(d2, 1e-16f));
    float inv_d = (d > 1e-8f) ? 1.0f / d : 0.0f;
    float dpd = d + 0.05f;
    float n = (float)diel_model;
    float scale = is_1_4 ? 0.75f : 1.0f;
    float denom_np1 = (diel_model == 2) ? dpd*dpd*dpd : dpd*dpd;
    float dE_dd = -332.0716f * n * charge_term * scale / max(denom_np1, 1e-30f);
    float f = dE_dd * inv_d;
    grad[i1*3+0]+=f*dx; grad[i1*3+1]+=f*dy; grad[i1*3+2]+=f*dz;
    grad[i2*3+0]-=f*dx; grad[i2*3+1]-=f*dy; grad[i2*3+2]-=f*dz;
}

// ---- Threadgroup parallel helpers (used by TG kernel) ----

inline float tg_reduce_sum(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[0];
}

inline void parallel_copy(device float* dst, const device float* src,
                           int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = src[i];
    threadgroup_barrier(mem_flags::mem_device);
}

inline void parallel_neg_copy(device float* dst, const device float* src,
                               int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = -src[i];
    threadgroup_barrier(mem_flags::mem_device);
}

inline float parallel_dot(const device float* a, const device float* b,
                           int n, uint tid, uint tpm, threadgroup float* shared) {
    float local_sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) local_sum += a[i] * b[i];
    shared[tid] = local_sum;
    return tg_reduce_sum(shared, tid, tpm);
}

inline void scale_grad_serial(device float* grad, int n_terms, thread float& grad_scale, bool pre_loop) {
    if (pre_loop) {
        grad_scale = GRAD_SCALE_INIT;
    }
    float scale = grad_scale;
    float max_grad = 0.0f;
    for (int i = 0; i < n_terms; i++) {
        grad[i] *= scale;
        max_grad = max(max_grad, abs(grad[i]));
    }
    while (max_grad > GRAD_CAP) {
        scale *= 0.5f;
        max_grad = 0.0f;
        for (int i = 0; i < n_terms; i++) {
            grad[i] *= 0.5f;
            max_grad = max(max_grad, abs(grad[i]));
        }
    }
    grad_scale = scale;
}
