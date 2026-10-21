"""Person 3: source detection and the single-process reference pipeline."""
from __future__ import annotations
from .interfaces import ImageData, Tile, Source, DetectionParams


def detect(tile: Tile, noise_sigma: float, params: DetectionParams
           ) -> list[Source]:
    """Smooth the (already background-subtracted) tile, threshold at
    threshold_sigma * noise_sigma, label connected regions
    (scipy.ndimage.label), reject blobs smaller than min_area, measure
    centroid / bbox / area / flux, convert to FULL-IMAGE coordinates, and
    return ONLY sources whose centroid is inside tile.core.
    source_id may be 0 here; tiling.merge() reassigns it.
    """
    raise NotImplementedError


def process_image(img: ImageData, params: DetectionParams,
                  tile_size: int | None = None) -> list[Source]:
    """Reference pipeline for one image: estimate_background -> subtract ->
    make_tiles (or one whole-image tile when tile_size is None) -> detect per
    tile -> merge. This is the ground truth the parallel version must match.
    """
    raise NotImplementedError
