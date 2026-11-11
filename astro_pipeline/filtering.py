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
    """Float64 FFT convolution with the shared four-sigma Gaussian kernel."""
    from scipy.signal import fftconvolve

    pixels, kernel, radius = _filter_input(data, kernel_sigma)
    # NumPy symmetric padding repeats the edge pixel, matching ndimage's
    # half-sample 'reflect' convention. Valid convolution restores input shape.
    # Match GPU arithmetic: bright-source roundoff must not change faint-pixel
    # threshold decisions when the image is processed in larger tiles.
    padded = np.pad(pixels.astype(np.float64), radius, mode="symmetric")
    filtered = np.array(fftconvolve(padded, kernel.astype(np.float64), mode="valid"),
                        dtype=np.float32, order="C", copy=True)
    if not np.isfinite(filtered).all():
        raise FloatingPointError("FFT filtering produced nonfinite pixel values")
    return filtered


def _smooth_gpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """CuPy/cuFFT convolution with float64 arithmetic and float32 output."""
    pixels, kernel, radius = _filter_input(data, kernel_sigma)
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve
    except ImportError as error:
        raise ImportError(
            "backend='gpu' requires a working CuPy installation matching the "
            "node's CUDA runtime"
        ) from error

    # Large float32 FFTs spread bright-source roundoff into faint pixels and
    # can move them across the detection cutoff. Widen on device after the
    # float32 upload; retain the shared kernel and round only the final image.
    device_pixels = cp.asarray(pixels).astype(cp.float64)
    device_kernel = cp.asarray(kernel, dtype=cp.float64)
    padded = cp.pad(device_pixels, radius, mode="symmetric")
    filtered = fftconvolve(padded, device_kernel, mode="valid").astype(cp.float32)
    # asnumpy blocks for the result: callers receive completed host pixels.
    result = np.array(cp.asnumpy(filtered), dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise FloatingPointError("FFT filtering produced nonfinite pixel values")
    return result


def smooth(data: np.ndarray, kernel_sigma: float, backend: str = "cpu", *,
           valid_mask: np.ndarray | None = None
           ) -> np.ndarray:
    """Return an independent float32 image using the shared backend names.

    Inputs must be finite and nonempty. Both backends use float64 transforms,
    radius ceil(4*sigma), and reflected edges, then return float32 pixels.
    GPU errors propagate; there is no CPU fallback.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"unknown backend {backend!r}")
    operation = _smooth_cpu if backend == "cpu" else _smooth_gpu
    if valid_mask is None:
        return operation(data, kernel_sigma)
    mask = np.asarray(valid_mask)
    if mask.dtype != np.bool_ or mask.shape != np.shape(data):
        raise ValueError("valid_mask must be a boolean array with the image shape")
    if mask.all():
        return operation(data, kernel_sigma)
    # Missing data do not count as zero observations. The denominator measures
    # the observed Gaussian weight; reject poorly supported centers, not stars.
    numerator = operation(np.where(mask, data, 0), kernel_sigma)
    denominator = operation(mask.astype(np.float32), kernel_sigma)
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=mask & (denominator >= 0.5))
    return result
