"""Person 3: CPU background-map and pre-filter noise estimation.

Background clipping rejects statistical outliers; source detection applies a
separate threshold in detection.py using DetectionParams.threshold_sigma.
"""
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
    """Return a clipped median and robust noise estimate, in pixel-value units.

    Recompute center = median(samples) and noise = 1.4826 * MAD, where MAD is
    median(abs(samples - center)). The factor gives a Gaussian-consistent noise
    scale. Keep samples within center +/- 3 * noise for at most five clipping
    passes, then summarize the retained samples. This is a symmetric rejection
    rule, not a brightness cutoff for detecting objects.

    These fixed settings also live in native/background.cpp::clipped_stats;
    DetectionParams.threshold_sigma does not change them. A zero MAD returns
    zero noise: valid for constants, but also possible with repeated fill pixels.
    """
    samples = np.asarray(values, dtype=np.float64).ravel()
    for _ in range(5):
        center = float(np.median(samples))
        noise = float(1.4826 * np.median(np.abs(samples - center)))
        if noise == 0:
            return center, noise
        keep = np.abs(samples - center) <= 3 * noise  # Fixed background clipping threshold.
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
    """Return (float32 background_map, scalar pre-filter noise_sigma).

    On the whole image, clip each box-by-box cell with _clipped_stats and
    interpolate its median to full resolution; extrapolate at image edges.
    `box=64` controls this background grid, independently of Humza's filter tiles.
    Estimate noise with the same clipping rule on float64(image) - background,
    before filtering. Detection later compares the smoothed residual against
    DetectionParams.threshold_sigma * noise_sigma (default multiplier: 3).

    Both CPU backends use these definitions: ASTRO_BACKGROUND_BACKEND=numpy
    (default) or cpp; ASTRO_BACKGROUND_THREADS controls native workers only.
    There is currently no coverage mask: every finite pixel, including zero,
    contributes. Repeated padding values can therefore bias the map and collapse
    the noise estimate; a finite result alone does not validate a padded mosaic.
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
    # Preserve the reference arithmetic: widen before subtracting the float32
    # map. This whole-image noise is measured before Gaussian smoothing.
    residual = pixels.astype(np.float64) - background
    return background, _clipped_stats(residual)[1]
