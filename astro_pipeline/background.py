"""Person 3: background-map and pre-filter noise estimation.

Background clipping rejects statistical outliers; source detection applies a
separate threshold in detection.py using DetectionParams.threshold_sigma.
"""
from __future__ import annotations
from importlib import import_module
import os
from time import perf_counter
import numpy as np
from scipy.ndimage import binary_dilation
from . import _segmentation


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


def _clipped_stats(values: np.ndarray, *, _selection=None) -> tuple[float, float]:
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
    def median(values):
        started = perf_counter() if _selection is not None else 0
        result = float(np.median(values))
        if _selection is not None:
            _selection[0] += (perf_counter() - started) * 1000
        return result
    samples = np.asarray(values, dtype=np.float64).ravel()
    for _ in range(5):
        center = median(samples)
        noise = 1.4826 * median(np.abs(samples - center))
        if noise == 0:
            return center, noise
        keep = np.abs(samples - center) <= 3 * noise  # Fixed background clipping threshold.
        if keep.all():
            return center, noise
        samples = samples[keep]
    center = median(samples)
    return center, 1.4826 * median(np.abs(samples - center))


def _axis_weights(centers: np.ndarray, length: int):
    if len(centers) == 1:
        indices = np.zeros(length, dtype=np.intp)
        return indices, indices, np.zeros(length)
    coordinates = np.arange(length, dtype=np.float64)
    lower = np.clip(np.searchsorted(centers, coordinates) - 1, 0, len(centers) - 2)
    upper = lower + 1
    weights = (coordinates - centers[lower]) / (centers[upper] - centers[lower])
    return lower, upper, weights


def _valid_mask(shape, mask):
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError("valid_mask must be a boolean array with the image shape")
    return mask


def _native_threads(stage):
    name = f"ASTRO_{stage}_THREADS"
    try:
        threads = int(os.environ.get(name, "1"))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if threads < 1:
        raise ValueError(f"{name} must be a positive integer")
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return min(threads, max(1, available))


def _native_module(stage):
    try:
        native = import_module("astro_pipeline._background_native")
    except ImportError as error:
        raise ImportError(
            f"ASTRO_{stage}_BACKEND=cpp requires the native background extension; "
            "build it with CMake for this Python interpreter"
        ) from error
    if getattr(native, "API_VERSION", 0) != 2:
        raise ImportError("native CPU extension is stale; rebuild it with CMake for this Python interpreter")
    return native


def estimate_background(img: np.ndarray, box: int = 64, *, valid_mask=None,
                        refine: bool = True) -> tuple[np.ndarray, float]:
    """Return a float32 sky map and scalar pre-filter sky noise.

    Coverage and source exclusion are separate: invalid observations never enter
    statistics, while bright sources are excluded only from sky fitting. At most
    three fits use 3-sigma source seeds (5 pixels, circular dilation radius 8).
    DetectionParams.threshold_sigma does not change these background rules.
    `refine=False` disables source exclusion for ablation/legacy comparisons.
    ASTRO_BACKGROUND_BACKEND=numpy (default) or cpp selects matching algorithms.
    ASTRO_BACKGROUND_STATS_BACKEND=cpu (default) or gpu independently selects
    global bootstrap/noise statistics. GPU mode retains per-image device buffers;
    mesh fitting, source exclusion, and the shared return types stay on the CPU.
    """
    return _dispatch(img, box, valid_mask, refine, False)


def profile_background(img: np.ndarray, box: int = 64, *, valid_mask=None, refine=True):
    """Run the selected backend with stage timings and scientific intermediates."""
    return _dispatch(img, box, valid_mask, refine, True)


def _dispatch(img, box, valid_mask, refine, profile):
    pixels = _image_pixels(img)
    valid = _valid_mask(pixels.shape, valid_mask)
    if not valid.any():
        raise ValueError("image has no usable observations")
    if isinstance(box, (bool, np.bool_)) or not isinstance(box, (int, np.integer)) or box < 1:
        raise ValueError("box must be a positive integer")
    if not isinstance(refine, (bool, np.bool_)):
        raise ValueError("refine must be boolean")
    backend = os.environ.get("ASTRO_BACKGROUND_BACKEND", "numpy")
    if backend not in ("numpy", "cpp"):
        raise ValueError("ASTRO_BACKGROUND_BACKEND must be numpy or cpp")
    stats_backend = os.environ.get("ASTRO_BACKGROUND_STATS_BACKEND", "cpu")
    if stats_backend not in ("cpu", "gpu"):
        raise ValueError("ASTRO_BACKGROUND_STATS_BACKEND must be cpu or gpu")
    threads = _native_threads("BACKGROUND") if backend == "cpp" else 1
    native = _native_module("BACKGROUND") if backend == "cpp" else None
    if (stats_backend == "gpu" and native is not None
            and getattr(native, "GLOBAL_STATISTICS_API_VERSION", 0) != 1):
        raise ImportError("native extension lacks GPU statistics support; rebuild it with CMake")
    started = perf_counter()
    statistics = None
    if stats_backend == "gpu":
        from ._background_gpu import GpuBackgroundStatistics
        statistics = GpuBackgroundStatistics(pixels, valid, profile=profile)
    try:
        if native is None:
            result = _estimate_background_numpy(pixels, int(box), valid_mask=valid,
                                                refine=refine, profile=profile,
                                                statistics=statistics)
        else:
            operation = native.profile_background if profile else native.estimate_background
            options = dict(valid_mask=valid, refine=bool(refine))
            if statistics is not None:
                options["global_statistics"] = statistics
            result = operation(pixels, min(int(box), np.iinfo(np.intp).max), threads, **options)
    finally:
        if statistics is not None:
            statistics.close()
    if statistics is not None and profile:
        result["timings"].update(statistics.timings())
        # Include context construction, upload, callback marshaling, waits,
        # and profile exports rather than reporting just kernel intervals.
        result["timings"]["total_ms"] = (perf_counter() - started) * 1000
    return result


def _solve(matrix, rhs):
    """Small, pivoted float64 solve, mirrored literally in C++ (no BLAS)."""
    a, b = matrix.copy(), rhs.copy()
    size = len(b)
    tolerance = 1e-12 * float(np.max(np.abs(a)))
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(a[row, col]))
        if abs(a[pivot, col]) <= tolerance:
            return None
        a[[col, pivot]], b[[col, pivot]] = a[[pivot, col]], b[[pivot, col]]
        divisor = float(a[col, col])
        for j in range(size):
            a[col, j] /= divisor
        b[col] /= divisor
        for row in range(size):
            if row == col:
                continue
            factor = float(a[row, col])
            for j in range(size):
                a[row, j] -= factor * a[col, j]
            b[row] -= factor * b[col]
    return b


def _fill_mesh(mesh, centers_y, centers_x, box):
    measured = np.isfinite(mesh)
    iy, ix = np.nonzero(measured)
    if not len(iy):
        raise ValueError("insufficient background coverage: no usable mesh cells")
    values = mesh[measured].copy()
    points = list(zip(centers_x[ix], centers_y[iy]))
    axes = [axis for axis, centers in enumerate((centers_x, centers_y)) if len(centers) > 1]
    for y, x in zip(*np.nonzero(~measured)):
        delta = [((float(px) - centers_x[x]) / box, (float(py) - centers_y[y]) / box)
                 for px, py in points]
        distances = [dx * dx + dy * dy for dx, dy in delta]
        order = sorted(range(len(points)), key=lambda i: (distances[i], i))
        neighbors = min(8, len(order))
        while True:
            size = 1 + len(axes)
            matrix, rhs = np.zeros((size, size)), np.zeros(size)
            for i in order[:neighbors]:
                vector = [1.0] + [delta[i][axis] for axis in axes]
                weight = 1.0 / max(distances[i], 1e-12)
                for row in range(size):
                    rhs[row] += weight * vector[row] * values[i]
                    for col in range(size):
                        matrix[row, col] += weight * vector[row] * vector[col]
            coefficient = _solve(matrix, rhs)
            if coefficient is not None:
                mesh[y, x] = coefficient[0]
                break
            if neighbors == len(order):
                raise ValueError("insufficient background coverage: independent spatial support required")
            neighbors = min(2 * neighbors, len(order))
    return measured


def _fit_numpy(pixels, box, valid, source, timings, profile, statistics=None):
    height, width = pixels.shape
    ys = list(range(0, height, box))
    xs = list(range(0, width, box))
    centers_y = np.array([(y + min(y + box, height) - 1) / 2 for y in ys])
    centers_x = np.array([(x + min(x + box, width) - 1) / 2 for x in xs])
    started = perf_counter()
    mesh = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            coverage = valid[y:y + box, x:x + box]
            sky = coverage & ~source[y:y + box, x:x + box]
            # At most 10% missing coverage and 50% source exclusion. Partial
            # edge cells use their actual area, not a padded box-sized area.
            if (10 * int(coverage.sum()) >= 9 * coverage.size
                    and sky.any() and 2 * int(sky.sum()) >= int(coverage.sum())):
                mesh[iy, ix] = _clipped_stats(pixels[y:y + box, x:x + box][sky])[0]
    timings["mesh_ms"] += (perf_counter() - started) * 1000
    started = perf_counter()
    measured = _fill_mesh(mesh, centers_y, centers_x, box)

    x0, x1, tx = _axis_weights(centers_x, width)
    y0, y1, ty = _axis_weights(centers_y, height)
    rows = mesh[:, x0] * (1 - tx) + mesh[:, x1] * tx
    background = np.empty_like(pixels)
    with np.errstate(over="raise", invalid="raise"):
        for y in range(height):
            background[y] = rows[y0[y]] * (1 - ty[y]) + rows[y1[y]] * ty[y]
    background[~valid] = 0
    timings["interpolation_ms"] += (perf_counter() - started) * 1000
    started = perf_counter()
    # Preserve the reference arithmetic: widen before subtracting the float32
    # map. This whole-image noise is measured before Gaussian smoothing.
    sky = valid & ~source
    if not sky.any():
        raise ValueError("insufficient background coverage: no source-free observations")
    selection = [0.0] if profile and statistics is None else None
    if statistics is None:
        residual = pixels[sky].astype(np.float64) - background[sky]
        noise = _clipped_stats(residual, _selection=selection)[1]
    else:
        noise = statistics(background, source)[1]
    timings["noise_ms"] += (perf_counter() - started) * 1000
    if selection is not None:
        timings["global_median_ms"] += selection[0]
    return background, noise, mesh, measured


def _estimate_background_numpy(pixels, box, *, valid_mask=None, refine=True, profile=False,
                               statistics=None):
    started = perf_counter()
    valid = _valid_mask(pixels.shape, valid_mask)
    timings = dict(mesh_ms=0.0, interpolation_ms=0.0, noise_ms=0.0,
                   global_median_ms=0.0, source_mask_ms=0.0, effective_threads=1)
    mask_started = perf_counter()
    seed = np.zeros(pixels.shape, dtype=bool)
    if refine:
        center, noise = (_clipped_stats(pixels[valid]) if statistics is None
                         else statistics(None, None))
        floor = 8 * np.finfo(np.float32).eps * float(np.max(np.abs(pixels[valid].astype(np.float64) - center)))
        seed = _segmentation.source_seeds(pixels, valid, center + max(3 * noise, floor))
    yy, xx = np.mgrid[-8:9, -8:9]
    footprint = xx * xx + yy * yy <= 64
    timings["source_mask_ms"] += (perf_counter() - mask_started) * 1000
    for iteration in range(3 if refine else 1):
        mask_started = perf_counter()
        source = (binary_dilation(seed, structure=footprint) & valid) if seed.any() else seed.copy()
        timings["source_mask_ms"] += (perf_counter() - mask_started) * 1000
        level, noise, mesh, measured = _fit_numpy(pixels, box, valid, source, timings, profile,
                                                statistics=statistics)
        if not refine or iteration == 2:
            break
        mask_started = perf_counter()
        with np.errstate(over="raise", invalid="raise"):
            residual = np.where(valid, pixels - level, 0)
        floor = 8 * np.finfo(np.float32).eps * float(np.max(np.abs(residual)))
        updated = seed | _segmentation.source_seeds(residual, valid, max(3 * noise, floor))
        unchanged = np.array_equal(updated, seed)
        timings["source_mask_ms"] += (perf_counter() - mask_started) * 1000
        if unchanged:
            break
        seed = updated
    timings["refinement_passes"] = iteration + 1
    timings["total_ms"] = (perf_counter() - started) * 1000
    if profile:
        return dict(background=level, noise=noise, source_mask=source, mesh=mesh,
                    measured_mesh=measured, timings=timings)
    return level, noise
