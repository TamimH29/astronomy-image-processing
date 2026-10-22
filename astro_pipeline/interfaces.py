"""
interfaces.py -- shared contract for the FITS source-detection pipeline.

This file holds ONLY types, constants, and conventions. Implementations live
in the modules listed below. Change this file only by team agreement: post in
the group chat and bump INTERFACE_VERSION.

MODULE MAP (who owns what)
  Person 1: io_fits.py (read_fits, write_catalog)
            synthetic.py (make_synthetic_image, save_synthetic_set)
            validate.py (score_catalog)
  Person 2: tiling.py (make_tiles, merge)
            parallel.py (run_batch)
            benchmark.py (benchmark)
            run_pipeline.py (command-line entry point)
  Person 3: background.py (estimate_background)
            filtering.py (smooth, CPU + GPU)
            detection.py (detect, process_image)

CONVENTIONS (read these first)
  * Images are 2D float32 NumPy arrays, shape (rows, cols) == (y, x).
  * Pixel coordinates are 0-indexed; (x, y) = (column, row); the CENTER of the
    top-left pixel is (0.0, 0.0).
  * All coordinates in Source records are FULL-IMAGE coordinates, never
    tile-local. Convert inside detect() before returning.
  * Background is estimated on the WHOLE image and subtracted BEFORE tiling,
    so tiles never disagree about the background level.
  * A source is "owned" by the tile whose CORE region contains its centroid.
    merge() keeps only owned sources, which prevents duplicates at tile edges.
  * halo >= ceil(4 * kernel_sigma) so filtering of the core is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

INTERFACE_VERSION = 2

# Fixed catalog schema (column order matters for CSV output).
CATALOG_COLUMNS = [
    "image_id", "source_id",
    "x_centroid", "y_centroid",
    "xmin", "ymin", "xmax", "ymax",   # inclusive bounding box, pixels
    "area_pix",                       # number of pixels above threshold
    "flux",                           # background-subtracted integrated flux
]


@dataclass
class ImageData:
    image_id: str                     # e.g. file stem
    data: np.ndarray                  # 2D float32
    header: dict = field(default_factory=dict)


@dataclass
class Tile:
    image_id: str
    y0: int                           # image coords of data[0, 0] (includes halo)
    x0: int
    data: np.ndarray                  # 2D float32, core + halo (clipped at edges)
    core: tuple[int, int, int, int]   # (cy0, cx0, cy1, cx1) owned region, image
                                      # coords, half-open [cy0, cy1) x [cx0, cx1)
    halo: int


@dataclass
class Source:
    image_id: str
    source_id: int                    # unique within image (assigned in merge)
    x_centroid: float
    y_centroid: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    area_pix: int
    flux: float

    def as_row(self) -> list:
        d = asdict(self)
        return [d[c] for c in CATALOG_COLUMNS]


@dataclass
class DetectionParams:
    kernel_sigma: float = 2.0         # Gaussian smoothing width (pixels)
    threshold_sigma: float = 3.0      # k in "k * noise above background"
    min_area: int = 5                 # reject smaller blobs
    backend: str = "cpu"              # "cpu" (SciPy) or "gpu" (CuPy)
