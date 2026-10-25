"""Samuel's matched SciPy/CuPy FFT Gaussian smoothing baseline."""
from __future__ import annotations
import math
import numpy as np


def _filter_input(data: np.ndarray, kernel_sigma: float
                  ) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate host data and construct the shared finite Gaussian kernel."""
    sigma = float(kernel_sigma)
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(4 * sigma):
        raise ValueError("kernel_sigma must be finite and positive")
    if np.iscomplexobj(data):
        raise ValueError("data must contain real pixel values")
    pixels = np.asarray(data, dtype=np.float32)
    if pixels.ndim != 2 or 0 in pixels.shape:
        raise ValueError("data must be a nonempty 2D image")
    if not np.isfinite(pixels).all():
        raise ValueError("data must be finite; handle NaN/inf during ingestion")

    radius = math.ceil(4 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    with np.errstate(over="ignore"):
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    kernel = np.outer(weights, weights).astype(np.float32)
    return pixels, kernel, radius


def _smooth_cpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Linear FFT convolution with a normalized, four-sigma Gaussian."""
    from scipy.signal import fftconvolve

    pixels, kernel, radius = _filter_input(data, kernel_sigma)
    # NumPy symmetric padding repeats the edge pixel, matching ndimage's
    # half-sample 'reflect' convention. Valid convolution restores input shape.
    padded = np.pad(pixels, radius, mode="symmetric")
    filtered = np.array(fftconvolve(padded, kernel, mode="valid"),
                        dtype=np.float32, order="C", copy=True)
    if not np.isfinite(filtered).all():
        raise FloatingPointError("FFT filtering produced nonfinite pixel values")
    return filtered


def _smooth_gpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Equivalent CuPy FFT convolution (cuFFT on NVIDIA), returning NumPy."""
    pixels, kernel, radius = _filter_input(data, kernel_sigma)
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as error:
        raise ImportError(
            "backend='gpu' requires a working CuPy installation matching the "
            "node's CUDA runtime; see docs/filtering.md"
        ) from error

    device_pixels = cp.asarray(pixels)
    device_kernel = cp.asarray(kernel)
    padded = cp.pad(device_pixels, radius, mode="symmetric")
    filtered = fftconvolve(padded, device_kernel, mode="valid")
    # asnumpy blocks for the result: callers receive completed host pixels.
    result = np.array(cp.asnumpy(filtered), dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise FloatingPointError("FFT filtering produced nonfinite pixel values")
    return result


def smooth(data: np.ndarray, kernel_sigma: float, backend: str = "cpu"
           ) -> np.ndarray:
    """Return an independent float32 image using the shared backend names.

    Inputs must be finite and nonempty. Both backends use radius ceil(4*sigma)
    and reflected edges. GPU errors propagate; there is no CPU fallback.
    """
    if backend == "cpu":
        return _smooth_cpu(data, kernel_sigma)
    if backend == "gpu":
        return _smooth_gpu(data, kernel_sigma)
    raise ValueError(f"unknown backend {backend!r}")
