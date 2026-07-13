"""Independent metric implementations for the Lane 3 audit.

These functions intentionally do not call ``the_well`` metric code.  They match
the documented tensor convention: spatial dimensions precede the final field
or component dimension, and variance uses the sample convention (ddof=1), as
PyTorch ``std`` does by default in the audited release.
"""

from __future__ import annotations

import numpy as np


def _spatial_axes(array: np.ndarray, n_spatial_dims: int) -> tuple[int, ...]:
    if n_spatial_dims < 1:
        raise ValueError("n_spatial_dims must be positive")
    if array.ndim < n_spatial_dims + 1:
        raise ValueError("array must include spatial axes and a final field axis")
    return tuple(range(array.ndim - n_spatial_dims - 1, array.ndim - 1))


def spatial_mse(
    prediction: np.ndarray, target: np.ndarray, n_spatial_dims: int
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} != {target.shape}")
    return np.mean(
        np.square(prediction - target),
        axis=_spatial_axes(target, n_spatial_dims),
    )


def spatial_sample_variance(target: np.ndarray, n_spatial_dims: int) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    return np.var(
        target,
        axis=_spatial_axes(target, n_spatial_dims),
        ddof=1,
    )


def vrmse(
    prediction: np.ndarray,
    target: np.ndarray,
    n_spatial_dims: int,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    if eps < 0:
        raise ValueError("eps must be non-negative")
    mse = spatial_mse(prediction, target, n_spatial_dims)
    variance = spatial_sample_variance(target, n_spatial_dims)
    return np.sqrt(mse / (variance + eps))
