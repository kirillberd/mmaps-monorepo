from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalizes a 1D or 2D array along the last axis."""

    vec = vec.astype(np.float32, copy=False)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vec / norm
