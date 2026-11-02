"""Person 3: source detection and the single-process reference pipeline."""
from __future__ import annotations
import math
import os
from time import perf_counter
import numpy as np

from . import background, filtering, tiling, _segmentation
from .interfaces import ImageData, Tile, Source, DetectionParams


def _validate_params(params: DetectionParams) -> None:
    sigma = float(params.kernel_sigma)
    threshold = float(params.threshold_sigma)
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(4 * sigma):
        raise ValueError("kernel_sigma must be finite and positive")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold_sigma must be finite and positive")
    if (isinstance(params.min_area, (bool, np.bool_))
            or not isinstance(params.min_area, (int, np.integer)) or params.min_area < 1):
        raise ValueError("min_area must be a positive integer")
    if params.backend not in ("cpu", "gpu"):
        raise ValueError(f"unknown backend {params.backend!r}")
    if not isinstance(params.refine_background, (bool, np.bool_)) or not isinstance(params.deblend, (bool, np.bool_)):
        raise ValueError("refine_background and deblend must be boolean")
    separation = params.deblend_min_separation
    if isinstance(separation, (bool, np.bool_)) or not isinstance(separation, (int, np.integer)) or separation < 1:
        raise ValueError("deblend_min_separation must be a positive integer")
    prominence = float(params.deblend_prominence_sigma)
    if not math.isfinite(prominence) or prominence <= 0:
        raise ValueError("deblend_prominence_sigma must be finite and positive")


def _measure_sources(tile: Tile, filtered: np.ndarray, noise_sigma: float,
                     params: DetectionParams, *, profile=False, peak_image=None):
    started = perf_counter()
    valid = background._valid_mask(tile.data.shape, tile.valid_mask)
    floor = 8 * np.finfo(np.float32).eps * float(np.max(np.abs(np.where(valid, tile.data, 0))))
    # Source cutoff in background-subtracted pixel units. threshold_sigma is
    # configurable (default 3.0 in interfaces.py::DetectionParams); noise_sigma
    # comes from the UNFILTERED whole-image residual in process_image(). This is
    # separate from the fixed +/- 3-sigma clipping used to estimate background.
    threshold = float(params.threshold_sigma) * noise_sigma
    if noise_sigma == 0:
        # Numerical floor for truly noiseless inputs, not a recovered noise
        # estimate. It cannot correct a MAD collapsed by no-coverage padding.
        threshold = floor
    if not math.isfinite(threshold):
        raise ValueError("detection threshold must be finite")
    prominence = max(float(params.deblend_prominence_sigma) * noise_sigma, floor,
                     float(np.finfo(np.float32).tiny))
    backend = os.environ.get("ASTRO_DETECTION_BACKEND", "python")
    if backend == "python":
        timings = dict(effective_threads=1)
        centroid_peaks = []
        labels = _segmentation.segment(filtered, valid, threshold, int(params.min_area),
                                       bool(params.deblend), int(params.deblend_min_separation),
                                       prominence, timings if profile else None,
                                       peak_image=peak_image, centroid_peaks=centroid_peaks)
        measuring = perf_counter()
        # Approved point-location refinement: only split components use a local
        # peak-centered centroid. Flux, area, and boxes still use the full basin.
        rows = _segmentation.measure(tile.data, labels, centroid_peaks,
                                     radius=max(2, math.ceil(params.kernel_sigma)))
        timings["measurement_ms"] = (perf_counter() - measuring) * 1000
    elif backend == "cpp":
        threads = background._native_threads("DETECTION")
        native = background._native_module("DETECTION")
        result = native.detect_sources(tile.data, filtered, threshold, int(params.min_area),
                                       bool(params.deblend), int(params.deblend_min_separation),
                                       prominence, threads, valid_mask=valid, profile=profile,
                                       peak_image=peak_image,
                                       centroid_radius=max(2, math.ceil(params.kernel_sigma)))
        if profile:
            rows, labels, timings = result["rows"], result["labels"], result["timings"]
        else:
            rows = result
    else:
        raise ValueError("ASTRO_DETECTION_BACKEND must be python or cpp")
    sources = [Source(tile.image_id, 0, float(cx) + tile.x0, float(cy) + tile.y0,
                      int(x0) + tile.x0, int(y0) + tile.y0,
                      int(x1) + tile.x0, int(y1) + tile.y0, int(area), float(flux))
               for cx, cy, x0, y0, x1, y1, area, flux in rows]
    if profile:
        timings["total_ms"] = (perf_counter() - started) * 1000
        return dict(sources=sources, labels=labels, timings=timings)
    return sources


def detect(tile: Tile, noise_sigma: float, params: DetectionParams
           ) -> list[Source]:
    """Smooth the (already background-subtracted) tile, threshold at
    threshold_sigma * noise_sigma, label and optionally deblend regions,
    reject blobs smaller than min_area, measure
    centroid / bbox / area / flux, convert to FULL-IMAGE coordinates, and
    return ONLY sources whose centroid is inside tile.core.
    source_id may be 0 here; tiling.merge() reassigns it.
    Components extending beyond the supplied tile cannot be measured completely.
    """
    _validate_params(params)
    noise_sigma = float(noise_sigma)
    if not math.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    pixels = background._image_pixels(tile.data)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (tile.y0, tile.x0, *tile.core)):
        raise ValueError("tile coordinates must be integers")
    cy0, cx0, cy1, cx1 = tile.core
    height, width = pixels.shape
    if (tile.y0 < 0 or tile.x0 < 0
            or not tile.y0 <= cy0 < cy1 <= tile.y0 + height
            or not tile.x0 <= cx0 < cx1 <= tile.x0 + width):
        raise ValueError("tile core must lie inside its image window")
    mask = background._valid_mask(pixels.shape, tile.valid_mask)
    normalized = Tile(tile.image_id, tile.y0, tile.x0, pixels, tile.core, tile.halo, tile.valid_mask)
    filtered = filtering.smooth(pixels, params.kernel_sigma, params.backend,
                                **({"valid_mask": mask} if tile.valid_mask is not None else {}))
    peak_image = None
    if params.deblend and params.kernel_sigma > 1:
        peak_image = filtering.smooth(pixels, 1.0, params.backend,
                                      **({"valid_mask": mask} if tile.valid_mask is not None else {}))
    return [source for source in _measure_sources(normalized, filtered, noise_sigma, params, peak_image=peak_image)
            if cy0 <= source.y_centroid < cy1 and cx0 <= source.x_centroid < cx1]


def process_image(img: ImageData, params: DetectionParams,
                  tile_size: int | None = None) -> list[Source]:
    """Estimate background, filter Humza's tiles, reassemble cores, and detect
    complete sources globally. Assign sources to centroid-owning cores and use
    Humza's merge for final IDs. None tile_size requests one whole-image tile.
    """
    _validate_params(params)
    pixels = background._image_pixels(img.data)
    height, width = pixels.shape
    if tile_size is None:
        tile_size = max(height, width)
    if (isinstance(tile_size, (bool, np.bool_))
            or not isinstance(tile_size, (int, np.integer)) or tile_size < 1):
        raise ValueError("tile_size must be a positive integer or None")
    tile_size = int(tile_size)
    # Use the default 64-pixel background mesh before tiling or smoothing.
    # params.threshold_sigma affects source selection below, not this estimate.
    valid = background._valid_mask(pixels.shape, img.valid_mask)
    if not valid.any():
        raise ValueError("image has no usable observations")
    level, noise = background.estimate_background(pixels, valid_mask=img.valid_mask,
                                                  refine=params.refine_background)
    with np.errstate(over="raise", invalid="raise"):
        residual = np.where(valid, pixels - level, 0)
    halo = math.ceil(4 * float(params.kernel_sigma))
    tiles = tiling.make_tiles(ImageData(img.image_id, residual, img.header, img.valid_mask), tile_size, halo)
    filtered = np.empty_like(residual)
    # Parent detection retains the selected smoothing width. A sharper peak map
    # preserves close stars for deblending; its radius never exceeds the halo.
    peak_image = np.empty_like(residual) if params.deblend and params.kernel_sigma > 1 else None
    for tile in tiles:
        local = filtering.smooth(tile.data, params.kernel_sigma, params.backend,
                                 **({"valid_mask": tile.valid_mask} if tile.valid_mask is not None else {}))
        cy0, cx0, cy1, cx1 = tile.core
        filtered[cy0:cy1, cx0:cx1] = local[
            cy0 - tile.y0:cy1 - tile.y0, cx0 - tile.x0:cx1 - tile.x0]
        if peak_image is not None:
            sharp = filtering.smooth(tile.data, 1.0, params.backend,
                                     **({"valid_mask": tile.valid_mask} if tile.valid_mask is not None else {}))
            peak_image[cy0:cy1, cx0:cx1] = sharp[
                cy0 - tile.y0:cy1 - tile.y0, cx0 - tile.x0:cx1 - tile.x0]

    whole = Tile(img.image_id, 0, 0, residual, (0, 0, height, width), halo, img.valid_mask)
    sources = _measure_sources(whole, filtered, noise, params, peak_image=peak_image)
    per_tile = [[] for _ in tiles]
    columns = (width + tile_size - 1) // tile_size
    for source in sources:
        owner = (int(source.y_centroid // tile_size) * columns
                 + int(source.x_centroid // tile_size))
        per_tile[owner].append(source)
    return tiling.merge(img.image_id, per_tile, tiles)
