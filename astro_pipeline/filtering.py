"""Samuel's C++20/CUDA Gaussian filter behind the shared NumPy interface."""
from __future__ import annotations
from importlib import import_module
import numpy as np


def _smooth_native(data: np.ndarray, kernel_sigma: float,
                   backend: str) -> np.ndarray:
    try:
        native = import_module("astroproc")
    except ModuleNotFoundError as error:
        if error.name != "astroproc":
            raise
        raise ImportError(
            "Build the astroproc extension and add its build directory to "
            "PYTHONPATH; see docs/native_filtering.md."
        ) from error

    config = native.ProcessingConfig()
    config.gaussian_sigma = kernel_sigma
    result = native.filter_image(data, config=config, backend=backend)
    pixels = result["pixels"]
    # smooth() has no mask return value. Preserve invalid targets as NaN;
    # the native filter excludes them from neighboring weighted sums.
    pixels[result["invalid"] != 0] = np.nan
    return pixels


def _smooth_cpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Existing native separable Gaussian, radius ceil(3*sigma), reflect edges."""
    return _smooth_native(data, kernel_sigma, "cpu")


def _smooth_gpu(data: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Upload, run our horizontal/vertical CUDA kernels, download to NumPy.

    The call is synchronous. CUDA must be compiled in and available; failures
    propagate rather than silently running the CPU backend.
    """
    return _smooth_native(data, kernel_sigma, "cuda")


def smooth(data: np.ndarray, kernel_sigma: float, backend: str = "cpu"
           ) -> np.ndarray:
    """Dispatch shared backend names; return a new 2D float32 NumPy array.

    Humza supplies tile.data; tiles use halo >= ceil(4 * kernel_sigma).
    The native binding releases the GIL while computing. This function owns
    no worker pool and does not change tile coordinates or source records.
    """
    if backend == "cpu":
        return _smooth_cpu(data, kernel_sigma)
    if backend == "gpu":
        return _smooth_gpu(data, kernel_sigma)
    raise ValueError(f"unknown backend {backend!r}")
