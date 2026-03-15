"""Utility functions for force field computations on MLX arrays.

All operations work on float32 MLX arrays.
"""

import mlx.core as mx


def distance_squared(p1: mx.array, p2: mx.array) -> mx.array:
    """Compute squared distance between position vectors.

    Args:
        p1: Positions of shape (N, dim) or (dim,).
        p2: Positions of shape (N, dim) or (dim,).

    Returns:
        Squared distances of shape (N,) or scalar.
    """
    diff = p1 - p2
    return mx.sum(diff * diff, axis=-1)


def cross_product(v1: mx.array, v2: mx.array) -> mx.array:
    """Compute cross product of 3D vectors.

    Args:
        v1: Vectors of shape (..., 3).
        v2: Vectors of shape (..., 3).

    Returns:
        Cross product of shape (..., 3).
    """
    x = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
    y = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
    z = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return mx.stack([x, y, z], axis=-1)


def dot_product(v1: mx.array, v2: mx.array) -> mx.array:
    """Compute dot product of vectors along last axis.

    Args:
        v1: Vectors of shape (..., D).
        v2: Vectors of shape (..., D).

    Returns:
        Dot products of shape (...).
    """
    return mx.sum(v1 * v2, axis=-1)


def normalize(v: mx.array) -> mx.array:
    """Normalize vectors along last axis.

    Args:
        v: Vectors of shape (..., D).

    Returns:
        Normalized vectors of shape (..., D).
    """
    norm = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True))
    return v / norm


def clamp(x: mx.array, min_val: float, max_val: float) -> mx.array:
    """Clamp values to [min_val, max_val].

    Args:
        x: Input array.
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        Clamped array.
    """
    return mx.maximum(min_val, mx.minimum(max_val, x))


def clip_to_one(x: mx.array) -> mx.array:
    """Clip values to [-1, 1].

    Args:
        x: Input array.

    Returns:
        Array with values clamped to [-1, 1].
    """
    return clamp(x, -1.0, 1.0)
