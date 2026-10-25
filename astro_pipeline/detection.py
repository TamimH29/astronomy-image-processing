"""Person 3: source detection and the single-process reference pipeline."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import find_objects, label

from . import background, filtering, tiling
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


def _measure_sources(tile: Tile, filtered: np.ndarray, noise_sigma: float,
                     params: DetectionParams) -> list[Source]:
    threshold = float(params.threshold_sigma) * noise_sigma
    if noise_sigma == 0:
        threshold = 8 * np.finfo(np.float32).eps * float(np.max(np.abs(tile.data)))
    if not math.isfinite(threshold):
        raise ValueError("detection threshold must be finite")
    labels, count = label(filtered > threshold, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    sources = []
    for component, window in enumerate(find_objects(labels, max_label=count), start=1):
        if window is None or areas[component] < params.min_area:
            continue
        sy, sx = window
        yy, xx = np.nonzero(labels[window] == component)
        values = tile.data[window][yy, xx].astype(np.float64)
        yy, xx = yy + sy.start, xx + sx.start
        weights = np.maximum(values, 0)
        weight = weights.sum(dtype=np.float64)
        if weight > 0:
            cx = float(np.sum(xx * weights, dtype=np.float64) / weight)
            cy = float(np.sum(yy * weights, dtype=np.float64) / weight)
        else:
            cx, cy = float(xx.mean()), float(yy.mean())
        sources.append(Source(
            image_id=tile.image_id, source_id=0,
            x_centroid=cx + tile.x0, y_centroid=cy + tile.y0,
            xmin=sx.start + tile.x0, ymin=sy.start + tile.y0,
            xmax=sx.stop - 1 + tile.x0, ymax=sy.stop - 1 + tile.y0,
            area_pix=int(areas[component]), flux=float(values.sum(dtype=np.float64)),
        ))
    return sources


def detect(tile: Tile, noise_sigma: float, params: DetectionParams
           ) -> list[Source]:
    """Smooth the (already background-subtracted) tile, threshold at
    threshold_sigma * noise_sigma, label connected regions
    (scipy.ndimage.label), reject blobs smaller than min_area, measure
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
    normalized = Tile(tile.image_id, tile.y0, tile.x0, pixels, tile.core, tile.halo)
    filtered = filtering.smooth(pixels, params.kernel_sigma, params.backend)
    return [source for source in _measure_sources(normalized, filtered, noise_sigma, params)
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
    level, noise = background.estimate_background(pixels)
    with np.errstate(over="raise", invalid="raise"):
        residual = pixels - level
    halo = math.ceil(4 * float(params.kernel_sigma))
    tiles = tiling.make_tiles(ImageData(img.image_id, residual, img.header), tile_size, halo)
    filtered = np.empty_like(residual)
    for tile in tiles:
        local = filtering.smooth(tile.data, params.kernel_sigma, params.backend)
        cy0, cx0, cy1, cx1 = tile.core
        filtered[cy0:cy1, cx0:cx1] = local[
            cy0 - tile.y0:cy1 - tile.y0, cx0 - tile.x0:cx1 - tile.x0]

    whole = Tile(img.image_id, 0, 0, residual, (0, 0, height, width), halo)
    sources = _measure_sources(whole, filtered, noise, params)
    per_tile = [[] for _ in tiles]
    columns = (width + tile_size - 1) // tile_size
    for source in sources:
        owner = (int(source.y_centroid // tile_size) * columns
                 + int(source.x_centroid // tile_size))
        per_tile[owner].append(source)
    return tiling.merge(img.image_id, per_tile, tiles)
