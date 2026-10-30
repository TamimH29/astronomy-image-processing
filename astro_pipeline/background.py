"""Person 3: background and noise estimation."""
from __future__ import annotations
from importlib import import_module
import os
import numpy as np


def _image_pixels(img: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(img):
        raise ValueError("image must contain real pixel values")
    with np.errstate(over="raise", invalid="raise"):
        pixels = np.asarray(img, dtype=np.float32)
    if pixels.ndim != 2 or 0 in pixels.shape:
        raise ValueError("image must be a nonempty 2D array")
    if not np.isfinite(pixels).all():
        raise ValueError("image must be finite; handle NaN/inf during ingestion")
    return pixels


def _clipped_stats(values: np.ndarray) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64).ravel()
    for _ in range(5):
        center = float(np.median(samples))
        noise = float(1.4826 * np.median(np.abs(samples - center)))
        if noise == 0:
            return center, noise
        keep = np.abs(samples - center) <= 3 * noise
        if keep.all():
            return center, noise
        samples = samples[keep]
    center = float(np.median(samples))
    return center, float(1.4826 * np.median(np.abs(samples - center)))


def _axis_weights(centers: np.ndarray, length: int):
    if len(centers) == 1:
        indices = np.zeros(length, dtype=np.intp)
        return indices, indices, np.zeros(length)
    coordinates = np.arange(length, dtype=np.float64)
    lower = np.clip(np.searchsorted(centers, coordinates) - 1, 0, len(centers) - 2)
    upper = lower + 1
    weights = (coordinates - centers[lower]) / (centers[upper] - centers[lower])
    return lower, upper, weights


def estimate_background(img: np.ndarray, box: int = 64
                        ) -> tuple[np.ndarray, float]:
    """Sigma-clipped median on a coarse grid of `box`-pixel cells, interpolated
    to full size. Return (background_map, noise_sigma). Runs on the WHOLE
    image, before tiling (see conventions in interfaces.py).
    """
    pixels = _image_pixels(img)
    if isinstance(box, (bool, np.bool_)) or not isinstance(box, (int, np.integer)) or box < 1:
        raise ValueError("box must be a positive integer")
    box = int(box)
    backend = os.environ.get("ASTRO_BACKGROUND_BACKEND", "numpy")
    if backend == "numpy":
        return _estimate_background_numpy(pixels, box)
    if backend != "cpp":
        raise ValueError("ASTRO_BACKGROUND_BACKEND must be numpy or cpp")
    try:
        threads = int(os.environ.get("ASTRO_BACKGROUND_THREADS", "1"))
    except ValueError as error:
        raise ValueError("ASTRO_BACKGROUND_THREADS must be a positive integer") from error
    if threads < 1:
        raise ValueError("ASTRO_BACKGROUND_THREADS must be a positive integer")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    threads = min(threads, max(1, available))
    try:
        native = import_module("astro_pipeline._background_native")
    except ImportError as error:
        raise ImportError(
            "ASTRO_BACKGROUND_BACKEND=cpp requires the native background extension; "
            "build it with CMake for this Python interpreter"
        ) from error
    return native.estimate_background(pixels, min(box, np.iinfo(np.intp).max), threads)


def _estimate_background_numpy(pixels: np.ndarray, box: int) -> tuple[np.ndarray, float]:
    height, width = pixels.shape
    ys = list(range(0, height, box))
    xs = list(range(0, width, box))
    centers_y = np.array([(y + min(y + box, height) - 1) / 2 for y in ys])
    centers_x = np.array([(x + min(x + box, width) - 1) / 2 for x in xs])
    mesh = np.empty((len(ys), len(xs)), dtype=np.float64)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            mesh[iy, ix] = _clipped_stats(pixels[y:y + box, x:x + box])[0]

    x0, x1, tx = _axis_weights(centers_x, width)
    y0, y1, ty = _axis_weights(centers_y, height)
    rows = mesh[:, x0] * (1 - tx) + mesh[:, x1] * tx
    background = np.empty_like(pixels)
    with np.errstate(over="raise", invalid="raise"):
        for y in range(height):
            background[y] = rows[y0[y]] * (1 - ty[y]) + rows[y1[y]] * ty[y]
    residual = pixels.astype(np.float64) - background
    return background, _clipped_stats(residual)[1]
