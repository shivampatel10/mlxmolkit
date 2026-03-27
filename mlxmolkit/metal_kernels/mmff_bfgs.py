"""Metal kernel for MMFF94 BFGS minimization — entire optimization on-device.

One thread per molecule runs the complete BFGS loop (up to max_iters)
with all 7 MMFF energy+gradient terms computed inline. Zero Python round-trips.

Uses mx.fast.metal_kernel() to launch MSL code.
"""

import mlx.core as mx
import numpy as np

from ..minimizer.bfgs import DEFAULT_GRAD_TOL
from ..preprocessing.mmff_batching import BatchedMMFFSystem

# Maximum atoms per molecule for Metal kernel (Hessian fits in device memory)
# 64 atoms * 3 dim = 192 terms -> 192*192*4 = ~144KB per Hessian
MAX_ATOMS_METAL = 64

# MSL helper functions for all 7 MMFF energy+gradient terms
_MSL_HEADER = """
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
"""

# MSL kernel body — one thread per molecule
_MSL_SOURCE = """
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
    #define COMPUTE_ENERGY_GRAD(OUT_E) \\
        for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f; \\
        OUT_E = 0.0f; \\
        for (int t = b_s; t < b_e; t++) { \\
            int a1=bond_pairs[t*2], a2=bond_pairs[t*2+1]; \\
            OUT_E += bond_stretch_e(out_pos, a1, a2, bond_params[t*2], bond_params[t*2+1]); \\
            bond_stretch_g(out_pos, work_grad, a1, a2, bond_params[t*2], bond_params[t*2+1]); \\
        } \\
        for (int t = a_s; t < a_e; t++) { \\
            int a1=angle_trips[t*3], a2=angle_trips[t*3+1], a3=angle_trips[t*3+2]; \\
            bool lin = angle_params[t*3+2] > 0.5f; \\
            OUT_E += angle_bend_e(out_pos, a1, a2, a3, angle_params[t*3], angle_params[t*3+1], lin); \\
            angle_bend_g(out_pos, work_grad, a1, a2, a3, angle_params[t*3], angle_params[t*3+1], lin); \\
        } \\
        for (int t = sb_s; t < sb_e; t++) { \\
            int a1=sb_trips[t*3], a2=sb_trips[t*3+1], a3=sb_trips[t*3+2]; \\
            OUT_E += stretch_bend_e(out_pos, a1, a2, a3, sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
            stretch_bend_g(out_pos, work_grad, a1, a2, a3, sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
        } \\
        for (int t = o_s; t < o_e; t++) { \\
            int a1=oop_quads[t*4], a2=oop_quads[t*4+1], a3=oop_quads[t*4+2], a4=oop_quads[t*4+3]; \\
            OUT_E += oop_bend_e(out_pos, a1, a2, a3, a4, oop_params[t]); \\
            oop_bend_g(out_pos, work_grad, a1, a2, a3, a4, oop_params[t]); \\
        } \\
        for (int t = t_s; t < t_e; t++) { \\
            int a1=tor_quads[t*4], a2=tor_quads[t*4+1], a3=tor_quads[t*4+2], a4=tor_quads[t*4+3]; \\
            OUT_E += torsion_e(out_pos, a1, a2, a3, a4, tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
            torsion_g(out_pos, work_grad, a1, a2, a3, a4, tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
        } \\
        for (int t = v_s; t < v_e; t++) { \\
            int a1=vdw_pairs[t*2], a2=vdw_pairs[t*2+1]; \\
            OUT_E += vdw_e(out_pos, a1, a2, vdw_params[t*2], vdw_params[t*2+1]); \\
            vdw_g(out_pos, work_grad, a1, a2, vdw_params[t*2], vdw_params[t*2+1]); \\
        } \\
        for (int t = e_s; t < e_e; t++) { \\
            int a1=ele_pairs[t*2], a2=ele_pairs[t*2+1]; \\
            OUT_E += ele_e(out_pos, a1, a2, ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \\
            ele_g(out_pos, work_grad, a1, a2, ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \\
        }

    // Macro for energy-only (line search trial)
    #define COMPUTE_ENERGY_ONLY(OUT_E) \\
        OUT_E = 0.0f; \\
        for (int t = b_s; t < b_e; t++) \\
            OUT_E += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \\
        for (int t = a_s; t < a_e; t++) \\
            OUT_E += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], angle_params[t*3+2]>0.5f); \\
        for (int t = sb_s; t < sb_e; t++) \\
            OUT_E += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
        for (int t = o_s; t < o_e; t++) \\
            OUT_E += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \\
        for (int t = t_s; t < t_e; t++) \\
            OUT_E += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
        for (int t = v_s; t < v_e; t++) \\
            OUT_E += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \\
        for (int t = e_s; t < e_e; t++) \\
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
"""


def _pack_kernel_inputs(
    system: BatchedMMFFSystem,
    max_iters: int,
    grad_tol: float,
) -> dict[str, mx.array | int]:
    """Pack BatchedMMFFSystem fields into flat arrays for the Metal kernel."""
    n_mols = system.n_mols
    atom_starts_np = np.array(system.atom_starts, copy=False).astype(np.int32)

    # Compute hessian_starts from atom counts
    n_atoms_per_mol = np.diff(atom_starts_np)
    n_terms_per_mol = n_atoms_per_mol * 3
    hessian_sizes = n_terms_per_mol * n_terms_per_mol
    hessian_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    np.cumsum(hessian_sizes, out=hessian_starts_np[1:])

    total_pos_size = int(atom_starts_np[-1]) * 3
    total_hessian_size = int(hessian_starts_np[-1])

    def _to_np_i32(arr):
        return np.array(arr, copy=False).astype(np.int32) if arr.size > 0 else np.zeros(0, dtype=np.int32)

    def _to_np_f32(arr):
        return np.array(arr, copy=False).astype(np.float32) if arr.size > 0 else np.zeros(0, dtype=np.float32)

    def build_term_starts(mol_indices_arr):
        starts = np.zeros(n_mols + 1, dtype=np.int32)
        if mol_indices_arr.size > 0:
            mi = _to_np_i32(mol_indices_arr)
            counts = np.bincount(mi, minlength=n_mols)
            np.cumsum(counts, out=starts[1:])
        return starts

    bond_ts = build_term_starts(system.bond_mol_indices)
    angle_ts = build_term_starts(system.angle_mol_indices)
    sb_ts = build_term_starts(system.sb_mol_indices)
    oop_ts = build_term_starts(system.oop_mol_indices)
    tor_ts = build_term_starts(system.tor_mol_indices)
    vdw_ts = build_term_starts(system.vdw_mol_indices)
    ele_ts = build_term_starts(system.ele_mol_indices)

    # --- Pack term data using fast numpy conversion ---
    def _pack_pairs(idx1, idx2):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2)], axis=1).flatten()
        return np.zeros(2, dtype=np.int32)

    def _pack_trips(idx1, idx2, idx3):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2), _to_np_i32(idx3)], axis=1).flatten()
        return np.zeros(3, dtype=np.int32)

    def _pack_quads(idx1, idx2, idx3, idx4):
        if idx1.size > 0:
            return np.stack([_to_np_i32(idx1), _to_np_i32(idx2), _to_np_i32(idx3), _to_np_i32(idx4)], axis=1).flatten()
        return np.zeros(4, dtype=np.int32)

    def _pack_params(*arrs, fallback_stride=1):
        if arrs[0].size > 0:
            return np.stack([_to_np_f32(a) for a in arrs], axis=1).flatten()
        return np.zeros(fallback_stride, dtype=np.float32)

    bond_pairs_np = _pack_pairs(system.bond_idx1, system.bond_idx2)
    bond_params_np = _pack_params(system.bond_kb, system.bond_r0, fallback_stride=2)

    angle_trips_np = _pack_trips(system.angle_idx1, system.angle_idx2, system.angle_idx3)
    angle_params_np = _pack_params(system.angle_ka, system.angle_theta0, system.angle_is_linear, fallback_stride=3)

    sb_trips_np = _pack_trips(system.sb_idx1, system.sb_idx2, system.sb_idx3)
    sb_params_np = _pack_params(system.sb_r0_ij, system.sb_r0_kj, system.sb_theta0, system.sb_kba_ij, system.sb_kba_kj, fallback_stride=5)

    oop_quads_np = _pack_quads(system.oop_idx1, system.oop_idx2, system.oop_idx3, system.oop_idx4)
    oop_params_np = _to_np_f32(system.oop_koop) if system.oop_koop.size > 0 else np.zeros(1, dtype=np.float32)

    tor_quads_np = _pack_quads(system.tor_idx1, system.tor_idx2, system.tor_idx3, system.tor_idx4)
    tor_params_np = _pack_params(system.tor_V1, system.tor_V2, system.tor_V3, fallback_stride=3)

    vdw_pairs_np = _pack_pairs(system.vdw_idx1, system.vdw_idx2)
    vdw_params_np = _pack_params(system.vdw_R_star, system.vdw_epsilon, fallback_stride=2)

    ele_pairs_np = _pack_pairs(system.ele_idx1, system.ele_idx2)
    ele_params_np = _pack_params(system.ele_charge_term, system.ele_diel_model, system.ele_is_1_4, fallback_stride=3)

    # Config array
    config_np = np.array([n_mols, max_iters, grad_tol], dtype=np.float32)

    # Combine all 7 term_starts into a single flat array
    all_term_starts_np = np.concatenate([
        bond_ts, angle_ts, sb_ts, oop_ts, tor_ts, vdw_ts, ele_ts
    ]).astype(np.int32)

    return {
        'atom_starts': mx.array(atom_starts_np),
        'hessian_starts': mx.array(hessian_starts_np),
        'config': mx.array(config_np),
        'all_term_starts': mx.array(all_term_starts_np),
        # Term data
        'bond_pairs': mx.array(bond_pairs_np),
        'bond_params': mx.array(bond_params_np),
        'angle_trips': mx.array(angle_trips_np),
        'angle_params': mx.array(angle_params_np),
        'sb_trips': mx.array(sb_trips_np),
        'sb_params': mx.array(sb_params_np),
        'oop_quads': mx.array(oop_quads_np),
        'oop_params': mx.array(oop_params_np),
        'tor_quads': mx.array(tor_quads_np),
        'tor_params': mx.array(tor_params_np),
        'vdw_pairs': mx.array(vdw_pairs_np),
        'vdw_params': mx.array(vdw_params_np),
        'ele_pairs': mx.array(ele_pairs_np),
        'ele_params': mx.array(ele_params_np),
        'total_pos_size': total_pos_size,
        'total_hessian_size': total_hessian_size,
    }


def metal_mmff_bfgs(
    pos: mx.array,
    system: BatchedMMFFSystem,
    max_iters: int = 200,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run MMFF BFGS minimization entirely on-device via a Metal kernel.

    Args:
        pos: Initial flat positions, shape (n_atoms_total * 3,), float32.
        system: Batched MMFF system with all energy terms pre-packed.
        max_iters: Maximum number of BFGS iterations per molecule.
        grad_tol: Gradient convergence tolerance.

    Returns:
        Tuple of (final_pos, final_energies, statuses).
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    inputs = _pack_kernel_inputs(system, max_iters, grad_tol)
    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    kernel = mx.fast.metal_kernel(
        name="mmff_bfgs",
        input_names=[
            "pos", "atom_starts", "hessian_starts", "config",
            "all_term_starts",
            "bond_pairs", "bond_params",
            "angle_trips", "angle_params",
            "sb_trips", "sb_params",
            "oop_quads", "oop_params",
            "tor_quads", "tor_params",
            "vdw_pairs", "vdw_params",
            "ele_pairs", "ele_params",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=_MSL_HEADER,
        source=_MSL_SOURCE,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'], inputs['hessian_starts'], inputs['config'],
            inputs['all_term_starts'],
            inputs['bond_pairs'], inputs['bond_params'],
            inputs['angle_trips'], inputs['angle_params'],
            inputs['sb_trips'], inputs['sb_params'],
            inputs['oop_quads'], inputs['oop_params'],
            inputs['tor_quads'], inputs['tor_params'],
            inputs['vdw_pairs'], inputs['vdw_params'],
            inputs['ele_pairs'], inputs['ele_params'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_hessian_size, 1),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols, 1, 1),
        threadgroup=(1, 1, 1),
        template=[("total_pos_size", total_pos_size)],
    )

    return outputs[0], outputs[1], outputs[2]


# Threadgroup size for the parallel kernel (Hessian/direction parallelism)
TG_SIZE = 32

# Threadgroup-parallel MSL kernel — TG_SIZE threads per molecule.
# Energy+gradient: thread 0 only (gradient functions use non-atomic +=).
# Hessian update + direction: all threads (embarrassingly parallel O(n²) ops).
_MSL_SOURCE_TG = """
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
    #define SEQ_COMPUTE_EG(OUT_E) \\
        if (tid == 0) { \\
            for (int i = 0; i < n_terms; i++) my_grad[i] = 0.0f; \\
            OUT_E = 0.0f; \\
            for (int t = b_s; t < b_e; t++) { \\
                OUT_E += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \\
                bond_stretch_g(out_pos, work_grad, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \\
            } \\
            for (int t = a_s; t < a_e; t++) { \\
                bool lin = angle_params[t*3+2] > 0.5f; \\
                OUT_E += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \\
                angle_bend_g(out_pos, work_grad, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \\
            } \\
            for (int t = sb_s; t < sb_e; t++) { \\
                OUT_E += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
                stretch_bend_g(out_pos, work_grad, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
            } \\
            for (int t = o_s; t < o_e; t++) { \\
                OUT_E += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \\
                oop_bend_g(out_pos, work_grad, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \\
            } \\
            for (int t = t_s; t < t_e; t++) { \\
                OUT_E += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
                torsion_g(out_pos, work_grad, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
            } \\
            for (int t = v_s; t < v_e; t++) { \\
                OUT_E += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \\
                vdw_g(out_pos, work_grad, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \\
            } \\
            for (int t = e_s; t < e_e; t++) { \\
                OUT_E += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \\
                ele_g(out_pos, work_grad, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \\
            } \\
        } \\
        threadgroup_barrier(mem_flags::mem_device);

    // Parallel energy-only macro — all threads sum a stripe, then reduce.
    // Replaces serial SEQ_COMPUTE_E. Energy functions are read-only on positions.
    #define PAR_COMPUTE_E(OUT_E) \\
        { \\
            float _local_e = 0.0f; \\
            for (int t = b_s + (int)tid; t < b_e; t += (int)tg_size) \\
                _local_e += bond_stretch_e(out_pos, bond_pairs[t*2], bond_pairs[t*2+1], bond_params[t*2], bond_params[t*2+1]); \\
            for (int t = a_s + (int)tid; t < a_e; t += (int)tg_size) { \\
                bool lin = angle_params[t*3+2] > 0.5f; \\
                _local_e += angle_bend_e(out_pos, angle_trips[t*3], angle_trips[t*3+1], angle_trips[t*3+2], angle_params[t*3], angle_params[t*3+1], lin); \\
            } \\
            for (int t = sb_s + (int)tid; t < sb_e; t += (int)tg_size) \\
                _local_e += stretch_bend_e(out_pos, sb_trips[t*3], sb_trips[t*3+1], sb_trips[t*3+2], sb_params[t*5], sb_params[t*5+1], sb_params[t*5+2], sb_params[t*5+3], sb_params[t*5+4]); \\
            for (int t = o_s + (int)tid; t < o_e; t += (int)tg_size) \\
                _local_e += oop_bend_e(out_pos, oop_quads[t*4], oop_quads[t*4+1], oop_quads[t*4+2], oop_quads[t*4+3], oop_params[t]); \\
            for (int t = t_s + (int)tid; t < t_e; t += (int)tg_size) \\
                _local_e += torsion_e(out_pos, tor_quads[t*4], tor_quads[t*4+1], tor_quads[t*4+2], tor_quads[t*4+3], tor_params[t*3], tor_params[t*3+1], tor_params[t*3+2]); \\
            for (int t = v_s + (int)tid; t < v_e; t += (int)tg_size) \\
                _local_e += vdw_e(out_pos, vdw_pairs[t*2], vdw_pairs[t*2+1], vdw_params[t*2], vdw_params[t*2+1]); \\
            for (int t = e_s + (int)tid; t < e_e; t += (int)tg_size) \\
                _local_e += ele_e(out_pos, ele_pairs[t*2], ele_pairs[t*2+1], ele_params[t*3], (int)ele_params[t*3+1], ele_params[t*3+2]>0.5f); \\
            tg_reduce[tid] = _local_e; \\
            OUT_E = tg_reduce_sum(tg_reduce, tid, tg_size); \\
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
"""


def metal_mmff_bfgs_tg(
    pos: mx.array,
    system: BatchedMMFFSystem,
    max_iters: int = 200,
    grad_tol: float | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run MMFF BFGS with threadgroup parallelism (TG_SIZE threads per molecule)."""
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    inputs = _pack_kernel_inputs(system, max_iters, grad_tol)
    total_pos_size = inputs['total_pos_size']
    total_hessian_size = inputs['total_hessian_size']

    tg_header = _MSL_HEADER + f"\nconstant int TG_SIZE_VAL = {TG_SIZE};\n"

    kernel = mx.fast.metal_kernel(
        name="mmff_bfgs_tg",
        input_names=[
            "pos", "atom_starts", "hessian_starts", "config",
            "all_term_starts",
            "bond_pairs", "bond_params",
            "angle_trips", "angle_params",
            "sb_trips", "sb_params",
            "oop_quads", "oop_params",
            "tor_quads", "tor_params",
            "vdw_pairs", "vdw_params",
            "ele_pairs", "ele_params",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch", "work_hessian",
        ],
        header=tg_header,
        source=_MSL_SOURCE_TG,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'], inputs['hessian_starts'], inputs['config'],
            inputs['all_term_starts'],
            inputs['bond_pairs'], inputs['bond_params'],
            inputs['angle_trips'], inputs['angle_params'],
            inputs['sb_trips'], inputs['sb_params'],
            inputs['oop_quads'], inputs['oop_params'],
            inputs['tor_quads'], inputs['tor_params'],
            inputs['vdw_pairs'], inputs['vdw_params'],
            inputs['ele_pairs'], inputs['ele_params'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_hessian_size, 1),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols * TG_SIZE, 1, 1),
        threadgroup=(TG_SIZE, 1, 1),
        template=[("total_pos_size", total_pos_size)],
    )

    return outputs[0], outputs[1], outputs[2]
