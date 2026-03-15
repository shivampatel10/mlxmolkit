"""Metal kernel for ETK L-BFGS minimization with threadgroup parallelism.

Same threadgroup-parallel L-BFGS architecture as dg_lbfgs.py, but with ETK
energy terms (torsion, improper, dist12/13, angle, long-range).

The BFGS loop body is identical — only energy/gradient evaluation differs.
"""

import mlx.core as mx
import numpy as np

from ..minimizer.bfgs import (
    DEFAULT_GRAD_TOL,
    EPS,
    FUNCTOL,
    MAX_STEP_FACTOR,
    MOVETOL,
    TOLX,
)
from ..preprocessing.etk_batching import BatchedETKSystem

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8
MAX_ATOMS_METAL = 64

_MSL_HEADER = """
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float EPS_GUARD = 3e-7f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Distance constraint energy (flat-bottom) ----
inline float dist_constraint_e(
    const device float* pos, int i1, int i2,
    float min_len, float max_len, float fc, int dim
) {
    float d2 = 0.0f;
    for (int d = 0; d < 3; d++) {
        float diff = pos[i1*dim+d] - pos[i2*dim+d];
        d2 += diff*diff;
    }
    float dist = sqrt(max(d2, 1e-16f));
    float min2 = min_len*min_len;
    float max2 = max_len*max_len;
    if (d2 < min2) {
        float dv = min_len - dist;
        return 0.5f * fc * dv * dv;
    } else if (d2 > max2) {
        float dv = dist - max_len;
        return 0.5f * fc * dv * dv;
    }
    return 0.0f;
}

// ---- Distance constraint gradient ----
inline void dist_constraint_g(
    const device float* pos, device float* grad,
    int i1, int i2, float min_len, float max_len, float fc, int dim
) {
    float diff[3];
    float d2 = 0.0f;
    for (int d = 0; d < 3; d++) {
        diff[d] = pos[i1*dim+d] - pos[i2*dim+d];
        d2 += diff[d]*diff[d];
    }
    float dist = sqrt(max(d2, 1e-16f));
    float min2 = min_len*min_len, max2 = max_len*max_len;
    float pf = 0.0f;
    if (d2 < min2) pf = fc * (dist - min_len) / max(dist, 1e-8f);
    else if (d2 > max2) pf = fc * (dist - max_len) / max(dist, 1e-8f);
    if (pf != 0.0f) {
        for (int d = 0; d < 3; d++) {
            float g = pf * diff[d];
            grad[i1*dim+d] += g;
            grad[i2*dim+d] -= g;
        }
    }
}

// ---- Torsion angle cos(phi) ----
inline float calc_cos_phi(const device float* pos, int i1, int i2, int i3, int i4, int dim) {
    float r1[3], r2[3], r3[3], r4[3];
    for (int d = 0; d < 3; d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
        r3[d] = -(pos[i3*dim+d] - pos[i2*dim+d]);
        r4[d] = pos[i4*dim+d] - pos[i3*dim+d];
    }
    float t1x = r1[1]*r2[2] - r1[2]*r2[1];
    float t1y = r1[2]*r2[0] - r1[0]*r2[2];
    float t1z = r1[0]*r2[1] - r1[1]*r2[0];
    float t2x = r3[1]*r4[2] - r3[2]*r4[1];
    float t2y = r3[2]*r4[0] - r3[0]*r4[2];
    float t2z = r3[0]*r4[1] - r3[1]*r4[0];

    float t1sq = t1x*t1x + t1y*t1y + t1z*t1z;
    float t2sq = t2x*t2x + t2y*t2y + t2z*t2z;
    float comb = t1sq * t2sq;
    if (comb < 1e-16f) return 0.0f;
    float dot = t1x*t2x + t1y*t2y + t1z*t2z;
    float cp = dot * rsqrt(comb);
    return clamp(cp, -1.0f, 1.0f);
}

// ---- 6-term Fourier torsion energy ----
inline float torsion_e(float cos_phi,
    float fc0, float fc1, float fc2, float fc3, float fc4, float fc5,
    float s0, float s1, float s2, float s3, float s4, float s5
) {
    float c = cos_phi;
    float c2 = c*c, c3=c*c2, c4=c*c3, c5_v=c*c4, c6=c*c5_v;
    float cos1=c, cos2=2.0f*c2-1.0f, cos3=4.0f*c3-3.0f*c;
    float cos4=8.0f*c4-8.0f*c2+1.0f, cos5=16.0f*c5_v-20.0f*c3+5.0f*c;
    float cos6=32.0f*c6-48.0f*c4+18.0f*c2-1.0f;
    return fc0*(1.0f+s0*cos1) + fc1*(1.0f+s1*cos2) + fc2*(1.0f+s2*cos3)
         + fc3*(1.0f+s3*cos4) + fc4*(1.0f+s4*cos5) + fc5*(1.0f+s5*cos6);
}

// ---- Torsion gradient (full 4-atom) ----
inline void torsion_g(const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float fc0, float fc1, float fc2, float fc3, float fc4, float fc5,
    float s0, float s1, float s2, float s3, float s4, float s5,
    int dim
) {
    float r1[3], r2[3], r3[3], r4[3];
    for (int d=0;d<3;d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
        r3[d] = -r2[d];
        r4[d] = pos[i4*dim+d] - pos[i3*dim+d];
    }
    float t0x=r1[1]*r2[2]-r1[2]*r2[1], t0y=r1[2]*r2[0]-r1[0]*r2[2], t0z=r1[0]*r2[1]-r1[1]*r2[0];
    float t1x=r3[1]*r4[2]-r3[2]*r4[1], t1y=r3[2]*r4[0]-r3[0]*r4[2], t1z=r3[0]*r4[1]-r3[1]*r4[0];

    float d02=t0x*t0x+t0y*t0y+t0z*t0z;
    float d12=t1x*t1x+t1y*t1y+t1z*t1z;
    if (d02 < 1e-16f || d12 < 1e-16f) return;

    float inv_d0 = rsqrt(max(d02,1e-16f));
    float inv_d1 = rsqrt(max(d12,1e-16f));
    float t0nx=t0x*inv_d0, t0ny=t0y*inv_d0, t0nz=t0z*inv_d0;
    float t1nx=t1x*inv_d1, t1ny=t1y*inv_d1, t1nz=t1z*inv_d1;

    float cos_phi = clamp(t0nx*t1nx+t0ny*t1ny+t0nz*t1nz, -1.0f, 1.0f);
    float sin_phi_sq = 1.0f - cos_phi*cos_phi;
    float sin_phi = sqrt(max(sin_phi_sq, 0.0f));

    float c=cos_phi, c2=c*c, c3=c*c2, c4=c*c3;
    float dE_dPhi =
        -s0*fc0*sin_phi
        - 2.0f*s1*fc1*(2.0f*c*sin_phi)
        - 3.0f*s2*fc2*(4.0f*c2*sin_phi - sin_phi)
        - 4.0f*s3*fc3*(8.0f*c3*sin_phi - 4.0f*c*sin_phi)
        - 5.0f*s4*fc4*(16.0f*c4*sin_phi - 12.0f*c2*sin_phi + sin_phi)
        - 6.0f*s5*fc5*(32.0f*c4*c*sin_phi - 32.0f*c3*sin_phi + 6.0f*sin_phi);

    float sin_term;
    if (abs(sin_phi) > 1e-8f) {
        sin_term = -dE_dPhi / sin_phi;
    } else {
        float abs_cp = max(abs(cos_phi), 1e-16f);
        sin_term = -dE_dPhi / abs_cp * sign(cos_phi + 1e-30f);
    }

    float dc_t0x = inv_d0*(t1nx - cos_phi*t0nx);
    float dc_t0y = inv_d0*(t1ny - cos_phi*t0ny);
    float dc_t0z = inv_d0*(t1nz - cos_phi*t0nz);
    float dc_t1x = inv_d1*(t0nx - cos_phi*t1nx);
    float dc_t1y = inv_d1*(t0ny - cos_phi*t1ny);
    float dc_t1z = inv_d1*(t0nz - cos_phi*t1nz);

    float g1x = sin_term * (dc_t0z*r2[1] - dc_t0y*r2[2]);
    float g1y = sin_term * (dc_t0x*r2[2] - dc_t0z*r2[0]);
    float g1z = sin_term * (dc_t0y*r2[0] - dc_t0x*r2[1]);

    float g4x = sin_term * (dc_t1y*r3[2] - dc_t1z*r3[1]);
    float g4y = sin_term * (dc_t1z*r3[0] - dc_t1x*r3[2]);
    float g4z = sin_term * (dc_t1x*r3[1] - dc_t1y*r3[0]);

    float g2x = sin_term * (
        dc_t0y*(r2[2]-r1[2]) + dc_t0z*(r1[1]-r2[1])
        + dc_t1y*(-r4[2]) + dc_t1z*r4[1]);
    float g2y = sin_term * (
        dc_t0x*(r1[2]-r2[2]) + dc_t0z*(r2[0]-r1[0])
        + dc_t1x*r4[2] + dc_t1z*(-r4[0]));
    float g2z = sin_term * (
        dc_t0x*(r2[1]-r1[1]) + dc_t0y*(r1[0]-r2[0])
        + dc_t1x*(-r4[1]) + dc_t1y*r4[0]);

    float g3x = sin_term * (
        dc_t0y*r1[2] + dc_t0z*(-r1[1])
        + dc_t1y*(r4[2]-r3[2]) + dc_t1z*(r3[1]-r4[1]));
    float g3y = sin_term * (
        dc_t0x*(-r1[2]) + dc_t0z*r1[0]
        + dc_t1x*(r3[2]-r4[2]) + dc_t1z*(r4[0]-r3[0]));
    float g3z = sin_term * (
        dc_t0x*r1[1] + dc_t0y*(-r1[0])
        + dc_t1x*(r4[1]-r3[1]) + dc_t1y*(r3[0]-r4[0]));

    grad[i1*dim+0] += g1x; grad[i1*dim+1] += g1y; grad[i1*dim+2] += g1z;
    grad[i2*dim+0] += g2x; grad[i2*dim+1] += g2y; grad[i2*dim+2] += g2z;
    grad[i3*dim+0] += g3x; grad[i3*dim+1] += g3y; grad[i3*dim+2] += g3z;
    grad[i4*dim+0] += g4x; grad[i4*dim+1] += g4y; grad[i4*dim+2] += g4z;
}

// ---- Inversion (improper) energy ----
inline float inversion_e(const device float* pos,
    int i1, int i2, int i3, int i4,
    float C0, float C1, float C2, float fc, int dim
) {
    float rJI[3], rJK[3], rJL[3];
    for (int d=0;d<3;d++) {
        rJI[d] = pos[i1*dim+d] - pos[i2*dim+d];
        rJK[d] = pos[i3*dim+d] - pos[i2*dim+d];
        rJL[d] = pos[i4*dim+d] - pos[i2*dim+d];
    }
    float l2JI = rJI[0]*rJI[0]+rJI[1]*rJI[1]+rJI[2]*rJI[2];
    float l2JK = rJK[0]*rJK[0]+rJK[1]*rJK[1]+rJK[2]*rJK[2];
    float l2JL = rJL[0]*rJL[0]+rJL[1]*rJL[1]+rJL[2]*rJL[2];
    if (l2JI < 1e-16f || l2JK < 1e-16f || l2JL < 1e-16f) return 0.0f;

    float nx = (-rJI[1])*rJK[2] - (-rJI[2])*rJK[1];
    float ny = (-rJI[2])*rJK[0] - (-rJI[0])*rJK[2];
    float nz = (-rJI[0])*rJK[1] - (-rJI[1])*rJK[0];
    float nf = rsqrt(max(l2JI*l2JK, 1e-16f));
    nx *= nf; ny *= nf; nz *= nf;
    float l2n = nx*nx+ny*ny+nz*nz;
    if (l2n < 1e-16f) return 0.0f;

    float dot = nx*rJL[0]+ny*rJL[1]+nz*rJL[2];
    float cos_y = dot * rsqrt(max(l2JL, 1e-16f)) * rsqrt(max(l2n, 1e-16f));
    cos_y = clamp(cos_y, -1.0f, 1.0f);

    float sin_y_sq = 1.0f - cos_y*cos_y;
    float sin_y = (sin_y_sq > 0.0f) ? sqrt(sin_y_sq) : 0.0f;
    float cos_2w = 2.0f*sin_y*sin_y - 1.0f;

    return fc * (C0 + C1*sin_y + C2*cos_2w);
}

// ---- Inversion gradient ----
inline void inversion_g(const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float C0, float C1, float C2, float fc, int dim
) {
    float rJI[3], rJK[3], rJL[3];
    for (int d=0;d<3;d++) {
        rJI[d] = pos[i1*dim+d] - pos[i2*dim+d];
        rJK[d] = pos[i3*dim+d] - pos[i2*dim+d];
        rJL[d] = pos[i4*dim+d] - pos[i2*dim+d];
    }
    float dJIsq = rJI[0]*rJI[0]+rJI[1]*rJI[1]+rJI[2]*rJI[2];
    float dJKsq = rJK[0]*rJK[0]+rJK[1]*rJK[1]+rJK[2]*rJK[2];
    float dJLsq = rJL[0]*rJL[0]+rJL[1]*rJL[1]+rJL[2]*rJL[2];
    if (dJIsq < 1e-16f || dJKsq < 1e-16f || dJLsq < 1e-16f) return;

    float invdJI = rsqrt(max(dJIsq, 1e-16f));
    float invdJK = rsqrt(max(dJKsq, 1e-16f));
    float invdJL = rsqrt(max(dJLsq, 1e-16f));

    float rJIn[3], rJKn[3], rJLn[3];
    for (int d=0;d<3;d++) {
        rJIn[d] = rJI[d]*invdJI;
        rJKn[d] = rJK[d]*invdJK;
        rJLn[d] = rJL[d]*invdJL;
    }

    float nx = (-rJIn[1])*rJKn[2] - (-rJIn[2])*rJKn[1];
    float ny = (-rJIn[2])*rJKn[0] - (-rJIn[0])*rJKn[2];
    float nz = (-rJIn[0])*rJKn[1] - (-rJIn[1])*rJKn[0];
    float inv_n_len = rsqrt(max(nx*nx+ny*ny+nz*nz, 1e-16f));
    float nnx=nx*inv_n_len, nny=ny*inv_n_len, nnz=nz*inv_n_len;

    float cos_y = clamp(nnx*rJLn[0]+nny*rJLn[1]+nnz*rJLn[2], -1.0f, 1.0f);
    float sin_y = max(sqrt(max(1.0f-cos_y*cos_y, 0.0f)), 1e-8f);

    float cos_theta = clamp(rJIn[0]*rJKn[0]+rJIn[1]*rJKn[1]+rJIn[2]*rJKn[2], -1.0f, 1.0f);
    float sin_theta = max(sqrt(max(1.0f-cos_theta*cos_theta, 0.0f)), 1e-8f);

    float dE_dW = -fc * (C1*cos_y - 4.0f*C2*cos_y*sin_y);

    float inverseTerm1 = 1.0f / (sin_y * sin_theta);
    float term2 = cos_y / (sin_y * sin_theta * sin_theta);
    float cos_y_over_sin_y = cos_y / sin_y;

    float t1x=rJLn[1]*rJKn[2]-rJLn[2]*rJKn[1];
    float t1y=rJLn[2]*rJKn[0]-rJLn[0]*rJKn[2];
    float t1z=rJLn[0]*rJKn[1]-rJLn[1]*rJKn[0];
    float t2x=rJIn[1]*rJLn[2]-rJIn[2]*rJLn[1];
    float t2y=rJIn[2]*rJLn[0]-rJIn[0]*rJLn[2];
    float t2z=rJIn[0]*rJLn[1]-rJIn[1]*rJLn[0];
    float t3x=rJKn[1]*rJIn[2]-rJKn[2]*rJIn[1];
    float t3y=rJKn[2]*rJIn[0]-rJKn[0]*rJIn[2];
    float t3z=rJKn[0]*rJIn[1]-rJKn[1]*rJIn[0];

    float tg1[3], tg3[3], tg4[3];
    tg1[0] = (t1x*inverseTerm1 - (rJIn[0]-rJKn[0]*cos_theta)*term2)*invdJI;
    tg1[1] = (t1y*inverseTerm1 - (rJIn[1]-rJKn[1]*cos_theta)*term2)*invdJI;
    tg1[2] = (t1z*inverseTerm1 - (rJIn[2]-rJKn[2]*cos_theta)*term2)*invdJI;
    tg3[0] = (t2x*inverseTerm1 - (rJKn[0]-rJIn[0]*cos_theta)*term2)*invdJK;
    tg3[1] = (t2y*inverseTerm1 - (rJKn[1]-rJIn[1]*cos_theta)*term2)*invdJK;
    tg3[2] = (t2z*inverseTerm1 - (rJKn[2]-rJIn[2]*cos_theta)*term2)*invdJK;
    tg4[0] = (t3x*inverseTerm1 - rJLn[0]*cos_y_over_sin_y)*invdJL;
    tg4[1] = (t3y*inverseTerm1 - rJLn[1]*cos_y_over_sin_y)*invdJL;
    tg4[2] = (t3z*inverseTerm1 - rJLn[2]*cos_y_over_sin_y)*invdJL;

    for (int d=0;d<3;d++) {
        float g1 = dE_dW * tg1[d];
        float g3 = dE_dW * tg3[d];
        float g4 = dE_dW * tg4[d];
        float g2 = -(g1+g3+g4);
        grad[i1*dim+d] += g1;
        grad[i2*dim+d] += g2;
        grad[i3*dim+d] += g3;
        grad[i4*dim+d] += g4;
    }
}

// ---- Angle constraint energy ----
inline float angle_e(const device float* pos,
    int i1, int i2, int i3, float min_ang, float max_ang, float fc, int dim
) {
    float r1[3], r2[3];
    for (int d=0;d<3;d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
    }
    float d1sq=r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2];
    float d2sq=r2[0]*r2[0]+r2[1]*r2[1]+r2[2]*r2[2];
    float dt = d1sq*d2sq;
    if (dt < 1e-16f) return 0.0f;
    float dot = r1[0]*r2[0]+r1[1]*r2[1]+r1[2]*r2[2];
    float cos_t = clamp(dot*rsqrt(dt), -1.0f, 1.0f);
    float angle_deg = 57.29577951f * acos(cos_t);
    float at = 0.0f;
    if (angle_deg < min_ang) at = angle_deg - min_ang;
    else if (angle_deg > max_ang) at = angle_deg - max_ang;
    return fc * at * at;
}

// ---- Angle constraint gradient ----
inline void angle_g(const device float* pos, device float* grad,
    int i1, int i2, int i3, float min_ang, float max_ang, float fc, int dim
) {
    float r1[3], r2[3];
    for (int d=0;d<3;d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
    }
    float r1sq=max(r1[0]*r1[0]+r1[1]*r1[1]+r1[2]*r1[2], 1e-5f);
    float r2sq=max(r2[0]*r2[0]+r2[1]*r2[1]+r2[2]*r2[2], 1e-5f);
    float denom=rsqrt(r1sq*r2sq);
    float dot=r1[0]*r2[0]+r1[1]*r2[1]+r1[2]*r2[2];
    float cos_t=clamp(dot*denom, -1.0f, 1.0f);
    float angle_deg = 57.29577951f * acos(cos_t);
    float at = 0.0f;
    if (angle_deg < min_ang) at = angle_deg - min_ang;
    else if (angle_deg > max_ang) at = angle_deg - max_ang;
    if (at == 0.0f) return;

    float dE_dTheta = 2.0f * 57.29577951f * fc * at;

    float rpx=r2[1]*r1[2]-r2[2]*r1[1], rpy=r2[2]*r1[0]-r2[0]*r1[2], rpz=r2[0]*r1[1]-r2[1]*r1[0];
    float rpsq = max(rpx*rpx+rpy*rpy+rpz*rpz, 1e-10f);
    float rpinv = rsqrt(rpsq);
    float pf = dE_dTheta * rpinv;
    float t1 = -pf / r1sq;
    float t2 = pf / r2sq;

    float dp1x = t1*(r1[1]*rpz-r1[2]*rpy);
    float dp1y = t1*(r1[2]*rpx-r1[0]*rpz);
    float dp1z = t1*(r1[0]*rpy-r1[1]*rpx);
    float dp3x = t2*(r2[1]*rpz-r2[2]*rpy);
    float dp3y = t2*(r2[2]*rpx-r2[0]*rpz);
    float dp3z = t2*(r2[0]*rpy-r2[1]*rpx);

    grad[i1*dim+0]+=dp1x; grad[i1*dim+1]+=dp1y; grad[i1*dim+2]+=dp1z;
    grad[i3*dim+0]+=dp3x; grad[i3*dim+1]+=dp3y; grad[i3*dim+2]+=dp3z;
    grad[i2*dim+0]+=-(dp1x+dp3x); grad[i2*dim+1]+=-(dp1y+dp3y); grad[i2*dim+2]+=-(dp1z+dp3z);
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

inline float parallel_dot(const device float* a, const device float* b,
    int n, uint tid, uint tpm, threadgroup float* s) {
    float sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) sum += a[i] * b[i];
    s[tid] = sum;
    return tg_reduce_sum(s, tid, tpm);
}

inline void parallel_saxpy(device float* a, float alpha, const device float* b,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] += alpha * b[i];
    threadgroup_barrier(mem_flags::mem_device);
}

inline void parallel_scale(device float* a, float alpha, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] *= alpha;
    threadgroup_barrier(mem_flags::mem_device);
}

inline void parallel_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = src[i];
    threadgroup_barrier(mem_flags::mem_device);
}

inline void parallel_set(device float* a, float val, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] = val;
    threadgroup_barrier(mem_flags::mem_device);
}

inline void parallel_neg_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = -src[i];
    threadgroup_barrier(mem_flags::mem_device);
}
"""

_MSL_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint mol_idx = threadgroup_position_in_grid.x;
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;

    threadgroup float shared[TPM];

    int n_mols_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol_v = config[2];
    int dim = (int)config[3];
    int use_bk = (int)config[4];

    if ((int)mol_idx >= n_mols_cfg) return;

    int atom_start = atom_starts[mol_idx];
    int atom_end = atom_starts[mol_idx + 1];
    int n_atoms = atom_end - atom_start;
    int n_vars = n_atoms * dim;
    int lbfgs_start = lbfgs_history_starts[mol_idx];

    int tor_s = torsion_starts[mol_idx], tor_e = torsion_starts[mol_idx+1];
    int imp_s = improper_starts[mol_idx], imp_e = improper_starts[mol_idx+1];
    int d12_s = dist12_starts[mol_idx], d12_e = dist12_starts[mol_idx+1];
    int d13_s = dist13_starts[mol_idx], d13_e = dist13_starts[mol_idx+1];
    int ang_s = angle_starts[mol_idx], ang_e = angle_starts[mol_idx+1];
    int lr_s = lr_starts[mol_idx], lr_e = lr_starts[mol_idx+1];

    parallel_copy(&out_pos[atom_start * dim], &pos[atom_start * dim], n_vars, tid, tpm);

    device float* my_pos = &out_pos[atom_start * dim];
    device float* my_grad = &work_grad[atom_start * dim];
    device float* my_dir = &work_dir[atom_start * dim];
    device float* my_old_pos = &work_scratch[atom_start * dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_start * dim];
    device float* my_q = &work_scratch[2 * total_pos_size + atom_start * dim];

    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m * n_vars];
    device float* my_rho = &work_rho[mol_idx * lbfgs_m];

    // ---- Initial energy (parallel across all threads) ----
    parallel_set(my_grad, 0.0f, n_vars, tid, tpm);
    float local_energy = 0.0f;

    for (int t = tor_s + (int)tid; t < tor_e; t += (int)tpm) {
        float cp = calc_cos_phi(out_pos, torsion_quads[t*4], torsion_quads[t*4+1],
            torsion_quads[t*4+2], torsion_quads[t*4+3], dim);
        local_energy += torsion_e(cp,
            torsion_fc[t*6], torsion_fc[t*6+1], torsion_fc[t*6+2],
            torsion_fc[t*6+3], torsion_fc[t*6+4], torsion_fc[t*6+5],
            torsion_signs_arr[t*6], torsion_signs_arr[t*6+1], torsion_signs_arr[t*6+2],
            torsion_signs_arr[t*6+3], torsion_signs_arr[t*6+4], torsion_signs_arr[t*6+5]);
    }
    if (use_bk) {
        for (int t = imp_s + (int)tid; t < imp_e; t += (int)tpm) {
            local_energy += inversion_e(out_pos, improper_quads[t*4], improper_quads[t*4+1],
                improper_quads[t*4+2], improper_quads[t*4+3],
                improper_coeffs[t*4], improper_coeffs[t*4+1], improper_coeffs[t*4+2], improper_coeffs[t*4+3], dim);
        }
    }
    for (int t = d12_s + (int)tid; t < d12_e; t += (int)tpm)
        local_energy += dist_constraint_e(out_pos, dist12_pairs[t*2], dist12_pairs[t*2+1],
            dist12_bounds[t*3], dist12_bounds[t*3+1], dist12_bounds[t*3+2], dim);
    for (int t = d13_s + (int)tid; t < d13_e; t += (int)tpm)
        local_energy += dist_constraint_e(out_pos, dist13_pairs[t*2], dist13_pairs[t*2+1],
            dist13_bounds[t*3], dist13_bounds[t*3+1], dist13_bounds[t*3+2], dim);
    for (int t = ang_s + (int)tid; t < ang_e; t += (int)tpm)
        local_energy += angle_e(out_pos, angle_triples[t*3], angle_triples[t*3+1], angle_triples[t*3+2],
            angle_bounds[t*3], angle_bounds[t*3+1], angle_bounds[t*3+2], dim);
    for (int t = lr_s + (int)tid; t < lr_e; t += (int)tpm)
        local_energy += dist_constraint_e(out_pos, lr_pairs[t*2], lr_pairs[t*2+1],
            lr_bounds[t*3], lr_bounds[t*3+1], lr_bounds[t*3+2], dim);

    shared[tid] = local_energy;
    float energy = tg_reduce_sum(shared, tid, tpm);

    // ---- Initial gradient (thread 0 only, serial) ----
    if (tid == 0) {
        for (int t = tor_s; t < tor_e; t++) {
            int i1=torsion_quads[t*4], i2=torsion_quads[t*4+1], i3=torsion_quads[t*4+2], i4=torsion_quads[t*4+3];
            torsion_g(out_pos, work_grad, i1, i2, i3, i4,
                torsion_fc[t*6], torsion_fc[t*6+1], torsion_fc[t*6+2],
                torsion_fc[t*6+3], torsion_fc[t*6+4], torsion_fc[t*6+5],
                torsion_signs_arr[t*6], torsion_signs_arr[t*6+1], torsion_signs_arr[t*6+2],
                torsion_signs_arr[t*6+3], torsion_signs_arr[t*6+4], torsion_signs_arr[t*6+5], dim);
        }
        if (use_bk) {
            for (int t = imp_s; t < imp_e; t++) {
                int i1=improper_quads[t*4], i2=improper_quads[t*4+1], i3=improper_quads[t*4+2], i4=improper_quads[t*4+3];
                inversion_g(out_pos, work_grad, i1,i2,i3,i4,
                    improper_coeffs[t*4], improper_coeffs[t*4+1], improper_coeffs[t*4+2], improper_coeffs[t*4+3], dim);
            }
        }
        for (int t = d12_s; t < d12_e; t++)
            dist_constraint_g(out_pos, work_grad, dist12_pairs[t*2], dist12_pairs[t*2+1],
                dist12_bounds[t*3], dist12_bounds[t*3+1], dist12_bounds[t*3+2], dim);
        for (int t = d13_s; t < d13_e; t++)
            dist_constraint_g(out_pos, work_grad, dist13_pairs[t*2], dist13_pairs[t*2+1],
                dist13_bounds[t*3], dist13_bounds[t*3+1], dist13_bounds[t*3+2], dim);
        for (int t = ang_s; t < ang_e; t++)
            angle_g(out_pos, work_grad, angle_triples[t*3], angle_triples[t*3+1], angle_triples[t*3+2],
                angle_bounds[t*3], angle_bounds[t*3+1], angle_bounds[t*3+2], dim);
        for (int t = lr_s; t < lr_e; t++)
            dist_constraint_g(out_pos, work_grad, lr_pairs[t*2], lr_pairs[t*2+1],
                lr_bounds[t*3], lr_bounds[t*3+1], lr_bounds[t*3+2], dim);
    }
    threadgroup_barrier(mem_flags::mem_device);

    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);

    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_vars; i += (int)tpm) local_sum_sq += my_pos[i]*my_pos[i];
    shared[tid] = local_sum_sq;
    float max_step = MAX_STEP_FACTOR * max(sqrt(tg_reduce_sum(shared, tid, tpm)), (float)n_vars);

    int status = 1;
    int hist_count = 0;
    int hist_idx = 0;

    // ---- Main L-BFGS loop ----
    for (int iter = 0; iter < max_iters && status == 1; iter++) {
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        // Scale direction
        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) local_dir_sq += my_dir[i]*my_dir[i];
        shared[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(shared, tid, tpm));
        if (dir_norm > max_step) parallel_scale(my_dir, max_step / dir_norm, n_vars, tid, tpm);

        float slope = parallel_dot(my_dir, my_grad, n_vars, tid, tpm, shared);

        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_dir[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > local_test_max) local_test_max = t;
        }
        shared[tid] = local_test_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpm/2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid+stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float lambda_min = MOVETOL / max(shared[0], 1e-30f);

        float lam = 1.0f, prev_lam = 1.0f, prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) {
                parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);
                ls_done = true; break;
            }
            for (int i = (int)tid; i < n_vars; i += (int)tpm)
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            // Parallel trial energy
            float local_trial_e = 0.0f;
            for (int t = tor_s+(int)tid; t < tor_e; t += (int)tpm) {
                float cp = calc_cos_phi(out_pos, torsion_quads[t*4], torsion_quads[t*4+1],
                    torsion_quads[t*4+2], torsion_quads[t*4+3], dim);
                local_trial_e += torsion_e(cp, torsion_fc[t*6], torsion_fc[t*6+1], torsion_fc[t*6+2],
                    torsion_fc[t*6+3], torsion_fc[t*6+4], torsion_fc[t*6+5],
                    torsion_signs_arr[t*6], torsion_signs_arr[t*6+1], torsion_signs_arr[t*6+2],
                    torsion_signs_arr[t*6+3], torsion_signs_arr[t*6+4], torsion_signs_arr[t*6+5]);
            }
            if (use_bk) {
                for (int t = imp_s+(int)tid; t < imp_e; t += (int)tpm)
                    local_trial_e += inversion_e(out_pos, improper_quads[t*4], improper_quads[t*4+1],
                        improper_quads[t*4+2], improper_quads[t*4+3],
                        improper_coeffs[t*4], improper_coeffs[t*4+1], improper_coeffs[t*4+2], improper_coeffs[t*4+3], dim);
            }
            for (int t = d12_s+(int)tid; t < d12_e; t += (int)tpm)
                local_trial_e += dist_constraint_e(out_pos, dist12_pairs[t*2], dist12_pairs[t*2+1],
                    dist12_bounds[t*3], dist12_bounds[t*3+1], dist12_bounds[t*3+2], dim);
            for (int t = d13_s+(int)tid; t < d13_e; t += (int)tpm)
                local_trial_e += dist_constraint_e(out_pos, dist13_pairs[t*2], dist13_pairs[t*2+1],
                    dist13_bounds[t*3], dist13_bounds[t*3+1], dist13_bounds[t*3+2], dim);
            for (int t = ang_s+(int)tid; t < ang_e; t += (int)tpm)
                local_trial_e += angle_e(out_pos, angle_triples[t*3], angle_triples[t*3+1], angle_triples[t*3+2],
                    angle_bounds[t*3], angle_bounds[t*3+1], angle_bounds[t*3+2], dim);
            for (int t = lr_s+(int)tid; t < lr_e; t += (int)tpm)
                local_trial_e += dist_constraint_e(out_pos, lr_pairs[t*2], lr_pairs[t*2+1],
                    lr_bounds[t*3], lr_bounds[t*3+1], lr_bounds[t*3+2], dim);

            shared[tid] = local_trial_e;
            float trial_e = tg_reduce_sum(shared, tid, tpm);

            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e; ls_done = true;
            } else {
                float tmp_lam;
                if (ls_iter == 0) {
                    tmp_lam = -slope / (2.0f * (trial_e - old_energy - slope));
                } else {
                    float rhs1 = trial_e - old_energy - lam*slope;
                    float rhs2 = prev_e - old_energy - prev_lam*slope;
                    float lam_sq=lam*lam, lam2_sq=prev_lam*prev_lam;
                    float dv = lam - prev_lam;
                    if (abs(dv) < 1e-30f) tmp_lam = 0.5f*lam;
                    else {
                        float a = (rhs1/lam_sq - rhs2/lam2_sq)/dv;
                        float b = (-prev_lam*rhs1/lam_sq + lam*rhs2/lam2_sq)/dv;
                        if (abs(a) < 1e-30f) tmp_lam = (abs(b)>1e-30f) ? -slope/(2.0f*b) : 0.5f*lam;
                        else {
                            float disc = b*b - 3.0f*a*slope;
                            if (disc<0.0f) tmp_lam = 0.5f*lam;
                            else if (b<=0.0f) tmp_lam = (-b+sqrt(disc))/(3.0f*a);
                            else tmp_lam = -slope/(b+sqrt(disc));
                        }
                    }
                }
                tmp_lam = min(tmp_lam, 0.5f*lam);
                tmp_lam = max(tmp_lam, 0.1f*lam);
                prev_lam = lam; prev_e = trial_e; lam = tmp_lam;
            }
        }
        if (!ls_done) parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);

        // s_k = pos - old_pos
        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_old_pos[i] = my_pos[i] - my_old_pos[i];
        threadgroup_barrier(mem_flags::mem_device);

        // TOLX check
        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > local_tolx) local_tolx = t;
        }
        shared[tid] = local_tolx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpm/2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid+stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shared[0] < TOLX) { status = 0; break; }

        // Save old grad, compute new energy (parallel) + gradient (thread 0)
        parallel_copy(my_old_grad, my_grad, n_vars, tid, tpm);
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm);

        float local_new_e = 0.0f;
        for (int t = tor_s+(int)tid; t < tor_e; t += (int)tpm) {
            float cp = calc_cos_phi(out_pos, torsion_quads[t*4], torsion_quads[t*4+1],
                torsion_quads[t*4+2], torsion_quads[t*4+3], dim);
            local_new_e += torsion_e(cp, torsion_fc[t*6], torsion_fc[t*6+1], torsion_fc[t*6+2],
                torsion_fc[t*6+3], torsion_fc[t*6+4], torsion_fc[t*6+5],
                torsion_signs_arr[t*6], torsion_signs_arr[t*6+1], torsion_signs_arr[t*6+2],
                torsion_signs_arr[t*6+3], torsion_signs_arr[t*6+4], torsion_signs_arr[t*6+5]);
        }
        if (use_bk) {
            for (int t = imp_s+(int)tid; t < imp_e; t += (int)tpm)
                local_new_e += inversion_e(out_pos, improper_quads[t*4], improper_quads[t*4+1],
                    improper_quads[t*4+2], improper_quads[t*4+3],
                    improper_coeffs[t*4], improper_coeffs[t*4+1], improper_coeffs[t*4+2], improper_coeffs[t*4+3], dim);
        }
        for (int t = d12_s+(int)tid; t < d12_e; t += (int)tpm)
            local_new_e += dist_constraint_e(out_pos, dist12_pairs[t*2], dist12_pairs[t*2+1],
                dist12_bounds[t*3], dist12_bounds[t*3+1], dist12_bounds[t*3+2], dim);
        for (int t = d13_s+(int)tid; t < d13_e; t += (int)tpm)
            local_new_e += dist_constraint_e(out_pos, dist13_pairs[t*2], dist13_pairs[t*2+1],
                dist13_bounds[t*3], dist13_bounds[t*3+1], dist13_bounds[t*3+2], dim);
        for (int t = ang_s+(int)tid; t < ang_e; t += (int)tpm)
            local_new_e += angle_e(out_pos, angle_triples[t*3], angle_triples[t*3+1], angle_triples[t*3+2],
                angle_bounds[t*3], angle_bounds[t*3+1], angle_bounds[t*3+2], dim);
        for (int t = lr_s+(int)tid; t < lr_e; t += (int)tpm)
            local_new_e += dist_constraint_e(out_pos, lr_pairs[t*2], lr_pairs[t*2+1],
                lr_bounds[t*3], lr_bounds[t*3+1], lr_bounds[t*3+2], dim);

        shared[tid] = local_new_e;
        energy = tg_reduce_sum(shared, tid, tpm);

        // Gradient recomputation (thread 0 only, serial)
        if (tid == 0) {
            for (int t = tor_s; t < tor_e; t++) {
                int i1=torsion_quads[t*4], i2=torsion_quads[t*4+1], i3=torsion_quads[t*4+2], i4=torsion_quads[t*4+3];
                torsion_g(out_pos, work_grad, i1,i2,i3,i4,
                    torsion_fc[t*6], torsion_fc[t*6+1], torsion_fc[t*6+2],
                    torsion_fc[t*6+3], torsion_fc[t*6+4], torsion_fc[t*6+5],
                    torsion_signs_arr[t*6], torsion_signs_arr[t*6+1], torsion_signs_arr[t*6+2],
                    torsion_signs_arr[t*6+3], torsion_signs_arr[t*6+4], torsion_signs_arr[t*6+5], dim);
            }
            if (use_bk) {
                for (int t = imp_s; t < imp_e; t++) {
                    int i1=improper_quads[t*4], i2=improper_quads[t*4+1], i3=improper_quads[t*4+2], i4=improper_quads[t*4+3];
                    inversion_g(out_pos, work_grad, i1,i2,i3,i4,
                        improper_coeffs[t*4], improper_coeffs[t*4+1], improper_coeffs[t*4+2], improper_coeffs[t*4+3], dim);
                }
            }
            for (int t = d12_s; t < d12_e; t++)
                dist_constraint_g(out_pos, work_grad, dist12_pairs[t*2], dist12_pairs[t*2+1],
                    dist12_bounds[t*3], dist12_bounds[t*3+1], dist12_bounds[t*3+2], dim);
            for (int t = d13_s; t < d13_e; t++)
                dist_constraint_g(out_pos, work_grad, dist13_pairs[t*2], dist13_pairs[t*2+1],
                    dist13_bounds[t*3], dist13_bounds[t*3+1], dist13_bounds[t*3+2], dim);
            for (int t = ang_s; t < ang_e; t++)
                angle_g(out_pos, work_grad, angle_triples[t*3], angle_triples[t*3+1], angle_triples[t*3+2],
                    angle_bounds[t*3], angle_bounds[t*3+1], angle_bounds[t*3+2], dim);
            for (int t = lr_s; t < lr_e; t++)
                dist_constraint_g(out_pos, work_grad, lr_pairs[t*2], lr_pairs[t*2+1],
                    lr_bounds[t*3], lr_bounds[t*3+1], lr_bounds[t*3+2], dim);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Gradient convergence check
        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (t > local_grad_test) local_grad_test = t;
        }
        shared[tid] = local_grad_test;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tpm/2; stride > 0; stride >>= 1) {
            if (tid < stride) shared[tid] = max(shared[tid], shared[tid+stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shared[0] / max(energy, 1.0f) < grad_tol_v) { status = 0; break; }

        // L-BFGS update: y_k = grad_new - grad_old
        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_q[i] = my_grad[i] - my_old_grad[i];
        threadgroup_barrier(mem_flags::mem_device);

        float ys_dot = parallel_dot(my_q, my_old_pos, n_vars, tid, tpm, shared);

        if (ys_dot > 1e-10f) {
            int slot = hist_idx % lbfgs_m;
            parallel_copy(&my_S[slot * n_vars], my_old_pos, n_vars, tid, tpm);
            parallel_copy(&my_Y[slot * n_vars], my_q, n_vars, tid, tpm);
            if (tid == 0) my_rho[slot] = 1.0f / ys_dot;
            threadgroup_barrier(mem_flags::mem_device);
            hist_idx++;
            if (hist_count < lbfgs_m) hist_count++;
        }

        // L-BFGS two-loop recursion
        parallel_copy(my_q, my_grad, n_vars, tid, tpm);
        device float* my_alpha = &work_alpha[mol_idx * lbfgs_m];

        for (int j = hist_count - 1; j >= 0; j--) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float alpha_j = my_rho[slot] * parallel_dot(&my_S[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            if (tid == 0) my_alpha[j] = alpha_j;
            threadgroup_barrier(mem_flags::mem_device);
            parallel_saxpy(my_q, -alpha_j, &my_Y[slot*n_vars], n_vars, tid, tpm);
        }

        if (hist_count > 0) {
            int newest = (hist_idx - 1) % lbfgs_m;
            if (newest < 0) newest += lbfgs_m;
            float sy = parallel_dot(&my_S[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            float yy = parallel_dot(&my_Y[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            parallel_scale(my_q, sy / max(yy, 1e-30f), n_vars, tid, tpm);
        }

        for (int j = 0; j < hist_count; j++) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float beta_j = my_rho[slot] * parallel_dot(&my_Y[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            parallel_saxpy(my_q, my_alpha[j] - beta_j, &my_S[slot*n_vars], n_vars, tid, tpm);
        }

        parallel_neg_copy(my_dir, my_q, n_vars, tid, tpm);
    }

    if (tid == 0) {
        out_energies[mol_idx] = energy;
        out_statuses[mol_idx] = status;
    }
"""


def _pack_etk_inputs(system, use_basic_knowledge, max_iters, grad_tol, lbfgs_m):
    """Pack BatchedETKSystem into kernel input arrays."""
    dim = system.dim
    n_mols = system.n_mols
    atom_starts = system.atom_starts

    # L-BFGS history starts
    lbfgs_history_starts_np = np.zeros(n_mols + 1, dtype=np.int32)
    for i in range(n_mols):
        n_atoms = atom_starts[i + 1] - atom_starts[i]
        n_vars = n_atoms * dim
        lbfgs_history_starts_np[i + 1] = (
            lbfgs_history_starts_np[i] + 2 * lbfgs_m * n_vars
        )

    total_pos_size = atom_starts[-1] * dim
    total_lbfgs_size = int(lbfgs_history_starts_np[-1])

    config_np = np.array([n_mols, max_iters, grad_tol, dim,
                          1 if use_basic_knowledge else 0], dtype=np.float32)

    def _pack_quads(idx1, idx2, idx3, idx4):
        if idx1.size > 0:
            return np.stack([np.array(idx1.tolist()), np.array(idx2.tolist()),
                           np.array(idx3.tolist()), np.array(idx4.tolist())], axis=1).flatten().astype(np.int32)
        return np.zeros(4, dtype=np.int32)

    def _pack_pairs(idx1, idx2):
        if idx1.size > 0:
            return np.stack([np.array(idx1.tolist()), np.array(idx2.tolist())], axis=1).flatten().astype(np.int32)
        return np.zeros(2, dtype=np.int32)

    torsion_quads_np = _pack_quads(system.torsion_idx1, system.torsion_idx2,
                                    system.torsion_idx3, system.torsion_idx4)
    n_tor = system.torsion_idx1.size
    if n_tor > 0:
        torsion_fc_np = np.array(system.torsion_fc.tolist(), dtype=np.float32).flatten()
        torsion_signs_np = np.array(system.torsion_signs.tolist(), dtype=np.float32).flatten()
    else:
        torsion_fc_np = np.zeros(6, dtype=np.float32)
        torsion_signs_np = np.zeros(6, dtype=np.float32)

    improper_quads_np = _pack_quads(system.improper_idx1, system.improper_idx2,
                                     system.improper_idx3, system.improper_idx4)
    n_imp = system.improper_idx1.size
    if n_imp > 0:
        improper_coeffs_np = np.stack([
            np.array(system.improper_C0.tolist()), np.array(system.improper_C1.tolist()),
            np.array(system.improper_C2.tolist()), np.array(system.improper_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        improper_coeffs_np = np.zeros(4, dtype=np.float32)

    dist12_pairs_np = _pack_pairs(system.dist12_idx1, system.dist12_idx2)
    n_d12 = system.dist12_idx1.size
    if n_d12 > 0:
        dist12_bounds_np = np.stack([
            np.array(system.dist12_min.tolist()), np.array(system.dist12_max.tolist()),
            np.array(system.dist12_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        dist12_bounds_np = np.zeros(3, dtype=np.float32)

    dist13_pairs_np = _pack_pairs(system.dist13_idx1, system.dist13_idx2)
    n_d13 = system.dist13_idx1.size
    if n_d13 > 0:
        dist13_bounds_np = np.stack([
            np.array(system.dist13_min.tolist()), np.array(system.dist13_max.tolist()),
            np.array(system.dist13_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        dist13_bounds_np = np.zeros(3, dtype=np.float32)

    n_ang = system.angle13_idx1.size
    if n_ang > 0:
        angle_triples_np = np.stack([
            np.array(system.angle13_idx1.tolist()), np.array(system.angle13_idx2.tolist()),
            np.array(system.angle13_idx3.tolist()),
        ], axis=1).flatten().astype(np.int32)
        angle_bounds_np = np.stack([
            np.array(system.angle13_min_angle.tolist()), np.array(system.angle13_max_angle.tolist()),
            np.array(system.angle13_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        angle_triples_np = np.zeros(3, dtype=np.int32)
        angle_bounds_np = np.zeros(3, dtype=np.float32)

    lr_pairs_np = _pack_pairs(system.long_range_idx1, system.long_range_idx2)
    n_lr = system.long_range_idx1.size
    if n_lr > 0:
        lr_bounds_np = np.stack([
            np.array(system.long_range_min.tolist()), np.array(system.long_range_max.tolist()),
            np.array(system.long_range_fc.tolist()),
        ], axis=1).flatten().astype(np.float32)
    else:
        lr_bounds_np = np.zeros(3, dtype=np.float32)

    def _build_term_starts(mol_indices, n_terms_total):
        starts = np.zeros(n_mols + 1, dtype=np.int32)
        if n_terms_total > 0:
            mi = np.array(mol_indices.tolist(), dtype=np.int32)
            for m in mi:
                starts[m + 1] += 1
            np.cumsum(starts, out=starts)
        return starts

    torsion_starts_np = _build_term_starts(system.torsion_mol_indices, n_tor)
    improper_starts_np = _build_term_starts(system.improper_mol_indices, n_imp)
    dist12_starts_np = _build_term_starts(system.dist12_mol_indices, n_d12)
    dist13_starts_np = _build_term_starts(system.dist13_mol_indices, n_d13)
    angle_starts_np = _build_term_starts(system.angle13_mol_indices, n_ang)
    lr_starts_np = _build_term_starts(system.long_range_mol_indices, n_lr)

    return {
        'atom_starts': mx.array(np.array(atom_starts, dtype=np.int32)),
        'lbfgs_history_starts': mx.array(lbfgs_history_starts_np),
        'config': mx.array(config_np),
        'torsion_quads': mx.array(torsion_quads_np),
        'torsion_fc': mx.array(torsion_fc_np),
        'torsion_signs_arr': mx.array(torsion_signs_np),
        'torsion_starts': mx.array(torsion_starts_np),
        'improper_quads': mx.array(improper_quads_np),
        'improper_coeffs': mx.array(improper_coeffs_np),
        'improper_starts': mx.array(improper_starts_np),
        'dist12_pairs': mx.array(dist12_pairs_np),
        'dist12_bounds': mx.array(dist12_bounds_np),
        'dist12_starts': mx.array(dist12_starts_np),
        'dist13_pairs': mx.array(dist13_pairs_np),
        'dist13_bounds': mx.array(dist13_bounds_np),
        'dist13_starts': mx.array(dist13_starts_np),
        'angle_triples': mx.array(angle_triples_np),
        'angle_bounds': mx.array(angle_bounds_np),
        'angle_starts': mx.array(angle_starts_np),
        'lr_pairs': mx.array(lr_pairs_np),
        'lr_bounds': mx.array(lr_bounds_np),
        'lr_starts': mx.array(lr_starts_np),
        'total_pos_size': total_pos_size,
        'total_lbfgs_size': total_lbfgs_size,
    }


def metal_etk_lbfgs(
    pos,
    system,
    use_basic_knowledge=True,
    max_iters=300,
    grad_tol=None,
    tpm=DEFAULT_TPM,
    lbfgs_m=DEFAULT_LBFGS_M,
):
    """Run ETK L-BFGS minimization via Metal kernel with threadgroup parallelism.

    Args:
        pos: Initial flat positions, shape (n_atoms_total * dim,), float32.
        system: BatchedETKSystem with all terms.
        use_basic_knowledge: Include improper torsion terms.
        max_iters: Maximum L-BFGS iterations.
        grad_tol: Gradient convergence tolerance.
        tpm: Threads per molecule (must be power of 2).
        lbfgs_m: L-BFGS history depth.

    Returns:
        (final_pos, final_energies, statuses)
    """
    if grad_tol is None:
        grad_tol = DEFAULT_GRAD_TOL

    n_mols = system.n_mols
    inputs = _pack_etk_inputs(system, use_basic_knowledge, max_iters, grad_tol, lbfgs_m)
    total_pos_size = inputs['total_pos_size']
    total_lbfgs_size = inputs['total_lbfgs_size']

    kernel = mx.fast.metal_kernel(
        name="etk_lbfgs",
        input_names=[
            "pos", "atom_starts", "lbfgs_history_starts", "config",
            "torsion_quads", "torsion_fc", "torsion_signs_arr", "torsion_starts",
            "improper_quads", "improper_coeffs", "improper_starts",
            "dist12_pairs", "dist12_bounds", "dist12_starts",
            "dist13_pairs", "dist13_bounds", "dist13_starts",
            "angle_triples", "angle_bounds", "angle_starts",
            "lr_pairs", "lr_bounds", "lr_starts",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch",
            "work_lbfgs", "work_rho", "work_alpha",
        ],
        header=_MSL_HEADER,
        source=_MSL_SOURCE,
    )

    outputs = kernel(
        inputs=[
            pos,
            inputs['atom_starts'],
            inputs['lbfgs_history_starts'],
            inputs['config'],
            inputs['torsion_quads'],
            inputs['torsion_fc'],
            inputs['torsion_signs_arr'],
            inputs['torsion_starts'],
            inputs['improper_quads'],
            inputs['improper_coeffs'],
            inputs['improper_starts'],
            inputs['dist12_pairs'],
            inputs['dist12_bounds'],
            inputs['dist12_starts'],
            inputs['dist13_pairs'],
            inputs['dist13_bounds'],
            inputs['dist13_starts'],
            inputs['angle_triples'],
            inputs['angle_bounds'],
            inputs['angle_starts'],
            inputs['lr_pairs'],
            inputs['lr_bounds'],
            inputs['lr_starts'],
        ],
        output_shapes=[
            (total_pos_size,),
            (n_mols,),
            (n_mols,),
            (total_pos_size,),
            (total_pos_size,),
            (total_pos_size * 3,),
            (max(total_lbfgs_size, 1),),
            (max(n_mols * lbfgs_m, 1),),
            (max(n_mols * lbfgs_m, 1),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.float32,
        ],
        grid=(n_mols * tpm, 1, 1),
        threadgroup=(tpm, 1, 1),
        template=[
            ("TPM", tpm),
            ("LBFGS_M", lbfgs_m),
            ("total_pos_size", total_pos_size),
        ],
    )

    return outputs[0], outputs[1], outputs[2]
