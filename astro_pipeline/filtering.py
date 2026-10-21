"""Person 3: Gaussian smoothing, CPU reference and GPU port."""
from __future__ import annotations
import numpy as np


def _smooth_cpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Reference implementation: scipy.fft-based Gaussian smoothing.
    FREEZE this by Nov 3; the GPU version is validated against it."""
    raise NotImplementedError


def _smooth_gpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """cupy.fft-based version. Copy to device, filter, copy back as NumPy.
    Import cupy lazily here so CPU-only machines can still import this file.
    Target: match _smooth_cpu within ~1e-4 relative (float32)."""
    raise NotImplementedError


def smooth(data: np.ndarray, kernel_sigma: float, backend: str = "cpu"
           ) -> np.ndarray:
    """Dispatch on backend ("cpu" or "gpu"). Always returns a NumPy array."""
    if backend == "cpu":
        return _smooth_cpu(data, kernel_sigma)
    if backend == "gpu":
        return _smooth_gpu(data, kernel_sigma)
    raise ValueError(f"unknown backend {backend!r}")
