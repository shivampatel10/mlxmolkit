"""Unit tests for mlxmolkit.forcefields.kernel_utils."""

import mlx.core as mx
import numpy as np
import numpy.testing as npt
import pytest

from mlxmolkit.forcefields.kernel_utils import (
    clamp,
    clip_to_one,
    cross_product,
    distance_squared,
    dot_product,
    normalize,
)


class TestDistanceSquared:
    def test_same_point(self):
        p = mx.array([1.0, 2.0, 3.0])
        result = distance_squared(p, p)
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_unit_distance(self):
        p1 = mx.array([0.0, 0.0, 0.0])
        p2 = mx.array([1.0, 0.0, 0.0])
        result = distance_squared(p1, p2)
        assert float(result) == pytest.approx(1.0, abs=1e-7)

    def test_3d_distance(self):
        p1 = mx.array([1.0, 2.0, 3.0])
        p2 = mx.array([4.0, 6.0, 3.0])
        expected = 9.0 + 16.0 + 0.0  # 25
        result = distance_squared(p1, p2)
        assert float(result) == pytest.approx(expected, abs=1e-5)

    def test_4d_distance(self):
        p1 = mx.array([0.0, 0.0, 0.0, 0.0])
        p2 = mx.array([1.0, 1.0, 1.0, 1.0])
        result = distance_squared(p1, p2)
        assert float(result) == pytest.approx(4.0, abs=1e-7)

    def test_batched(self):
        p1 = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        p2 = mx.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        result = distance_squared(p1, p2)
        expected = np.array([1.0, 1.0])
        npt.assert_allclose(np.array(result), expected, atol=1e-7)


class TestCrossProduct:
    def test_standard_basis(self):
        i = mx.array([1.0, 0.0, 0.0])
        j = mx.array([0.0, 1.0, 0.0])
        result = cross_product(i, j)
        expected = np.array([0.0, 0.0, 1.0])
        npt.assert_allclose(np.array(result), expected, atol=1e-7)

    def test_anticommutative(self):
        v1 = mx.array([1.0, 2.0, 3.0])
        v2 = mx.array([4.0, 5.0, 6.0])
        r1 = cross_product(v1, v2)
        r2 = cross_product(v2, v1)
        npt.assert_allclose(np.array(r1), -np.array(r2), atol=1e-6)

    def test_parallel_zero(self):
        v = mx.array([1.0, 2.0, 3.0])
        result = cross_product(v, v)
        npt.assert_allclose(np.array(result), [0.0, 0.0, 0.0], atol=1e-6)

    def test_batched(self):
        v1 = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v2 = mx.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = cross_product(v1, v2)
        expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        npt.assert_allclose(np.array(result), expected, atol=1e-7)


class TestDotProduct:
    def test_orthogonal(self):
        v1 = mx.array([1.0, 0.0, 0.0])
        v2 = mx.array([0.0, 1.0, 0.0])
        assert float(dot_product(v1, v2)) == pytest.approx(0.0, abs=1e-7)

    def test_parallel(self):
        v = mx.array([1.0, 2.0, 3.0])
        result = dot_product(v, v)
        assert float(result) == pytest.approx(14.0, abs=1e-6)

    def test_known_value(self):
        v1 = mx.array([1.0, 2.0, 3.0])
        v2 = mx.array([4.0, 5.0, 6.0])
        result = dot_product(v1, v2)
        assert float(result) == pytest.approx(32.0, abs=1e-5)


class TestNormalize:
    def test_unit_vector(self):
        v = mx.array([3.0, 4.0, 0.0])
        result = normalize(v)
        expected = np.array([0.6, 0.8, 0.0])
        npt.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_already_normalized(self):
        v = mx.array([0.0, 0.0, 1.0])
        result = normalize(v)
        npt.assert_allclose(np.array(result), [0.0, 0.0, 1.0], atol=1e-7)

    def test_result_has_unit_length(self):
        v = mx.array([1.0, 2.0, 3.0])
        result = normalize(v)
        length = float(mx.sqrt(mx.sum(result * result)))
        assert length == pytest.approx(1.0, abs=1e-6)


class TestClamp:
    def test_within_range(self):
        x = mx.array([0.5])
        result = clamp(x, 0.0, 1.0)
        assert float(result[0]) == pytest.approx(0.5, abs=1e-7)

    def test_below_min(self):
        x = mx.array([-1.0])
        result = clamp(x, 0.0, 1.0)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-7)

    def test_above_max(self):
        x = mx.array([2.0])
        result = clamp(x, 0.0, 1.0)
        assert float(result[0]) == pytest.approx(1.0, abs=1e-7)


class TestClipToOne:
    def test_clips_high(self):
        x = mx.array([1.5])
        result = clip_to_one(x)
        assert float(result[0]) == pytest.approx(1.0, abs=1e-7)

    def test_clips_low(self):
        x = mx.array([-1.5])
        result = clip_to_one(x)
        assert float(result[0]) == pytest.approx(-1.0, abs=1e-7)

    def test_within_range(self):
        x = mx.array([0.5])
        result = clip_to_one(x)
        assert float(result[0]) == pytest.approx(0.5, abs=1e-7)
