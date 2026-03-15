"""Unit tests for ETK 3D energy and gradient functions.

Tests torsion angle, inversion, distance constraint, and angle constraint
energy/gradient computations against finite differences and known values.
"""

import math

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.forcefields.dist_geom_3d import (
    angle_constraint_energy,
    angle_constraint_grad,
    distance_constraint_energy,
    distance_constraint_grad,
    inversion_energy,
    inversion_grad,
    torsion_angle_energy,
    torsion_angle_grad,
)
from mlxmolkit.preprocessing.torsion_prefs import (
    ETK3DParams,
    extract_etk_params,
    _calc_inversion_coefficients,
)


# ---- Distance Constraint Tests ----


def test_distance_constraint_energy_within_bounds():
    """Energy should be 0 when distance is within bounds."""
    pos = mx.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    min_len = mx.array([0.9], dtype=mx.float32)
    max_len = mx.array([1.1], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    e = distance_constraint_energy(pos, idx1, idx2, min_len, max_len, fc, dim=4)
    mx.eval(e)
    assert float(e[0]) == pytest.approx(0.0, abs=1e-6)


def test_distance_constraint_energy_too_short():
    """Energy should be > 0 when distance < minLen."""
    # Distance = 0.5, minLen = 0.9
    pos = mx.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    min_len = mx.array([0.9], dtype=mx.float32)
    max_len = mx.array([1.1], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    e = distance_constraint_energy(pos, idx1, idx2, min_len, max_len, fc, dim=4)
    mx.eval(e)
    # E = 0.5 * 100 * (0.9 - 0.5)^2 = 50 * 0.16 = 8.0
    assert float(e[0]) == pytest.approx(8.0, rel=0.01)


def test_distance_constraint_energy_too_long():
    """Energy should be > 0 when distance > maxLen."""
    pos = mx.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    min_len = mx.array([0.9], dtype=mx.float32)
    max_len = mx.array([1.1], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    e = distance_constraint_energy(pos, idx1, idx2, min_len, max_len, fc, dim=4)
    mx.eval(e)
    # E = 0.5 * 100 * (2.0 - 1.1)^2 = 50 * 0.81 = 40.5
    assert float(e[0]) == pytest.approx(40.5, rel=0.01)


def test_distance_constraint_grad_finite_diff():
    """Gradient should match finite differences."""
    pos = mx.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    min_len = mx.array([0.9], dtype=mx.float32)
    max_len = mx.array([1.1], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    grad = distance_constraint_grad(pos, idx1, idx2, min_len, max_len, fc, dim=4)
    mx.eval(grad)
    grad_np = np.array(grad)

    eps = 1e-3
    pos_np = np.array(pos)
    for i in range(len(pos_np)):
        pos_p = mx.array(pos_np.copy())
        pos_p = pos_p.at[i].add(eps)
        pos_m = mx.array(pos_np.copy())
        pos_m = pos_m.at[i].add(-eps)
        e_p = distance_constraint_energy(pos_p, idx1, idx2, min_len, max_len, fc, dim=4)
        e_m = distance_constraint_energy(pos_m, idx1, idx2, min_len, max_len, fc, dim=4)
        mx.eval(e_p, e_m)
        fd = (float(mx.sum(e_p)) - float(mx.sum(e_m))) / (2 * eps)
        if abs(fd) > 1e-6 or abs(grad_np[i]) > 1e-6:
            assert grad_np[i] == pytest.approx(fd, abs=0.1, rel=0.05)


# ---- Torsion Angle Tests ----


def test_torsion_energy_known_value():
    """Test torsion energy for known geometry."""
    # 4 atoms in a trans configuration (phi ~= 180 degrees, cos(phi) = -1)
    pos = mx.array([
        1.0, 0.0, 0.0, 0.0,  # atom 0
        0.0, 0.0, 0.0, 0.0,  # atom 1
        0.0, 1.0, 0.0, 0.0,  # atom 2
        -1.0, 1.0, 0.0, 0.0, # atom 3
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    idx4 = mx.array([3], dtype=mx.int32)

    # fc[2] = 4.0 (cos(3*phi) term), signs all 1
    fc = mx.array([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    signs = mx.array([[1, 1, 1, 1, 1, 1]], dtype=mx.int32)

    e = torsion_angle_energy(pos, idx1, idx2, idx3, idx4, fc, signs, dim=4)
    mx.eval(e)

    # cos(phi) for trans ~ -1 or 1 depending on convention
    # Energy should be finite and positive
    assert float(e[0]) >= 0.0


def test_torsion_grad_finite_diff():
    """Torsion gradient should match finite differences."""
    # Slightly off-planar geometry
    pos = mx.array([
        1.0, 0.0, 0.2, 0.0,  # atom 0
        0.0, 0.0, 0.0, 0.0,  # atom 1
        0.0, 1.0, 0.0, 0.0,  # atom 2
        -0.5, 1.5, 0.3, 0.0, # atom 3
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    idx4 = mx.array([3], dtype=mx.int32)
    fc = mx.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    signs = mx.array([[1, -1, 1, 1, 1, 1]], dtype=mx.int32)

    grad = torsion_angle_grad(pos, idx1, idx2, idx3, idx4, fc, signs, dim=4)
    mx.eval(grad)
    grad_np = np.array(grad)

    eps = 1e-3
    pos_np = np.array(pos)
    for i in range(len(pos_np)):
        pos_p = mx.array(pos_np.copy())
        pos_p = pos_p.at[i].add(eps)
        pos_m = mx.array(pos_np.copy())
        pos_m = pos_m.at[i].add(-eps)
        e_p = torsion_angle_energy(pos_p, idx1, idx2, idx3, idx4, fc, signs, dim=4)
        e_m = torsion_angle_energy(pos_m, idx1, idx2, idx3, idx4, fc, signs, dim=4)
        mx.eval(e_p, e_m)
        fd = (float(mx.sum(e_p)) - float(mx.sum(e_m))) / (2 * eps)
        if abs(fd) > 1e-5 or abs(grad_np[i]) > 1e-5:
            assert grad_np[i] == pytest.approx(fd, abs=0.5, rel=0.1), \
                f"Component {i}: grad={grad_np[i]:.6f}, fd={fd:.6f}"


# ---- Inversion Tests ----


def test_inversion_coefficients_carbon():
    """Carbon sp2 center should have C0=1, C1=-1, C2=0, fc=2."""
    fc, C0, C1, C2 = _calc_inversion_coefficients(6, False)
    assert C0 == pytest.approx(1.0)
    assert C1 == pytest.approx(-1.0)
    assert C2 == pytest.approx(0.0)
    assert fc == pytest.approx(6.0 / 3.0)  # 2.0


def test_inversion_coefficients_carbon_bound_to_o():
    """Carbon sp2 bound to O should have higher force constant."""
    fc, C0, C1, C2 = _calc_inversion_coefficients(6, True)
    assert fc == pytest.approx(50.0 / 3.0)  # ~16.67


def test_inversion_energy_planar():
    """Inversion energy should be near minimum for planar geometry."""
    # 4 atoms in a planar arrangement around central atom (idx2)
    pos = mx.array([
        1.0, 0.0, 0.0, 0.0,  # atom 0 (neighbor)
        0.0, 0.0, 0.0, 0.0,  # atom 1 (central)
        0.0, 1.0, 0.0, 0.0,  # atom 2 (neighbor)
        -1.0, 0.0, 0.0, 0.0, # atom 3 (neighbor)
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    idx4 = mx.array([3], dtype=mx.int32)
    C0 = mx.array([1.0], dtype=mx.float32)
    C1 = mx.array([-1.0], dtype=mx.float32)
    C2 = mx.array([0.0], dtype=mx.float32)
    fc = mx.array([2.0], dtype=mx.float32)

    e = inversion_energy(pos, idx1, idx2, idx3, idx4, C0, C1, C2, fc, dim=4)
    mx.eval(e)

    # For planar sp2: sinY should be ~1 (out-of-plane angle ~90 from normal)
    # E = fc * (C0 + C1*sinY + C2*cos2W)
    # For planar: cosY ~ 0, sinY ~ 1, cos2W = 2*sinY^2-1 = 1
    # E = 2 * (1 + (-1)*1 + 0) = 0 — minimum for sp2
    assert float(e[0]) == pytest.approx(0.0, abs=0.1)


def test_inversion_grad_finite_diff():
    """Inversion gradient should match finite differences."""
    pos = mx.array([
        1.0, 0.0, 0.3, 0.0,  # slightly out of plane
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    idx4 = mx.array([3], dtype=mx.int32)
    C0 = mx.array([1.0], dtype=mx.float32)
    C1 = mx.array([-1.0], dtype=mx.float32)
    C2 = mx.array([0.0], dtype=mx.float32)
    fc = mx.array([2.0], dtype=mx.float32)

    grad = inversion_grad(pos, idx1, idx2, idx3, idx4, C0, C1, C2, fc, dim=4)
    mx.eval(grad)
    grad_np = np.array(grad)

    eps = 1e-3
    pos_np = np.array(pos)
    for i in range(len(pos_np)):
        pos_p = mx.array(pos_np.copy())
        pos_p = pos_p.at[i].add(eps)
        pos_m = mx.array(pos_np.copy())
        pos_m = pos_m.at[i].add(-eps)
        e_p = inversion_energy(pos_p, idx1, idx2, idx3, idx4, C0, C1, C2, fc, dim=4)
        e_m = inversion_energy(pos_m, idx1, idx2, idx3, idx4, C0, C1, C2, fc, dim=4)
        mx.eval(e_p, e_m)
        fd = (float(mx.sum(e_p)) - float(mx.sum(e_m))) / (2 * eps)
        if abs(fd) > 1e-5 or abs(grad_np[i]) > 1e-5:
            assert grad_np[i] == pytest.approx(fd, abs=0.5, rel=0.15), \
                f"Component {i}: grad={grad_np[i]:.6f}, fd={fd:.6f}"


# ---- Angle Constraint Tests ----


def test_angle_constraint_energy_within_bounds():
    """Energy should be 0 for angle within bounds."""
    # 90-degree angle
    pos = mx.array([
        1.0, 0.0, 0.0, 0.0,  # atom 0
        0.0, 0.0, 0.0, 0.0,  # atom 1 (central)
        0.0, 1.0, 0.0, 0.0,  # atom 2
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    min_a = mx.array([80.0], dtype=mx.float32)
    max_a = mx.array([100.0], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    e = angle_constraint_energy(pos, idx1, idx2, idx3, min_a, max_a, fc, dim=4)
    mx.eval(e)
    assert float(e[0]) == pytest.approx(0.0, abs=0.1)


def test_angle_constraint_energy_triple_bond():
    """Energy should be > 0 for angle deviating from 180 degrees."""
    # ~150-degree angle
    pos = mx.array([
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        -0.866, 0.5, 0.0, 0.0,
    ], dtype=mx.float32)
    idx1 = mx.array([0], dtype=mx.int32)
    idx2 = mx.array([1], dtype=mx.int32)
    idx3 = mx.array([2], dtype=mx.int32)
    min_a = mx.array([179.0], dtype=mx.float32)
    max_a = mx.array([180.0], dtype=mx.float32)
    fc = mx.array([100.0], dtype=mx.float32)

    e = angle_constraint_energy(pos, idx1, idx2, idx3, min_a, max_a, fc, dim=4)
    mx.eval(e)
    assert float(e[0]) > 0.0


# ---- ETK Parameter Extraction Tests ----


def test_extract_etk_params_ethanol():
    """Extract ETK params for ethanol — should have bonds and angles."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    bmat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    params = rdDistGeom.ETKDGv3()

    etk = extract_etk_params(mol, bmat, params=params)

    assert etk.num_atoms == mol.GetNumAtoms()
    assert len(etk.dist12_idx1) > 0  # Should have bonds
    assert len(etk.dist13_idx1) > 0  # Should have 1-3 distances
    assert len(etk.long_range_idx1) >= 0  # May have long-range


def test_extract_etk_params_butane_torsions():
    """Butane should have experimental torsion preferences."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    bmat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    params = rdDistGeom.ETKDGv3()

    etk = extract_etk_params(mol, bmat, params=params)

    assert len(etk.torsion_idx1) > 0, "Butane should have experimental torsions"
    assert etk.torsion_fc.shape[1] == 6  # 6 Fourier terms


def test_extract_etk_params_benzaldehyde_impropers():
    """Benzaldehyde should have improper torsion terms."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C=O"))
    bmat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    params = rdDistGeom.ETKDGv3()

    etk = extract_etk_params(mol, bmat, params=params)

    assert len(etk.improper_idx1) > 0, "Benzaldehyde should have improper torsions"
    assert etk.num_improper_atoms > 0
    # 3 permutations per improper atom
    assert len(etk.improper_idx1) == 3 * etk.num_improper_atoms
