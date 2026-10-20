"""
interfaces.py -- shared contract for the FITS source-detection pipeline.

All three team members import from this file. Change it only by agreement
(post in the group chat, bump INTERFACE_VERSION).

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
from typing import Iterable, Sequence
import numpy as np

INTERFACE_VERSION = 1

# Fixed catalog schema (column order matters for CSV output).
CATALOG_COLUMNS = [
    "image_id", "source_id",
    "x_centroid", "y_centroid",
    "xmin", "ymin", "xmax", "ymax",   # inclusive bounding box, pixels
    "area_pix",                       # number of pixels above threshold
    "flux",                           # background-subtracted integrated flux
]


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Person 1: ingestion / output / synthetic data / validation
# --------------------------------------------------------------------------
def read_fits(path: str) -> ImageData:
    """Read the first image HDU (handle .fits.gz / .fits.bz2 and extension
    HDUs). Return float32 data. Replace NaN/inf with the image median."""
    raise NotImplementedError


def write_catalog(sources: Iterable[Source], path: str) -> None:
    """Write CATALOG_COLUMNS to CSV (or FITS table if path ends in .fits)."""
    raise NotImplementedError


def make_synthetic_image(shape: tuple[int, int], n_sources: int, seed: int
                         ) -> tuple[np.ndarray, list[dict]]:
    """Return (image, truth) where truth is a list of dicts with keys
    x, y, sigma, flux. Image = sources + smooth background gradient + noise."""
    raise NotImplementedError


def score_catalog(sources: Sequence[Source], truth: Sequence[dict],
                  match_radius: float = 3.0) -> dict:
    """Match detections to truth. Return dict with: completeness,
    false_positive_rate, median_pos_err_pix, median_flux_err_pct."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Person 2: tiling / parallelism / benchmarking
# --------------------------------------------------------------------------
def make_tiles(img: ImageData, tile_size: int, halo: int) -> list[Tile]:
    """Split into tile_size x tile_size cores, each padded by `halo` pixels
    (clipped at image borders). Cores must partition the image exactly."""
    raise NotImplementedError


def merge(image_id: str, per_tile: Iterable[Sequence[Source]],
          tiles: Sequence[Tile]) -> list[Source]:
    """Keep only sources whose centroid lies in the owning tile's core, then
    assign unique source_id values (sorted by y, then x for determinism)."""
    raise NotImplementedError


def run_batch(paths: Sequence[str], params: DetectionParams,
              n_workers: int, tile_size: int | None = None
              ) -> dict[str, list[Source]]:
    """Process many images with a worker pool (ProcessPoolExecutor).
    tile_size=None means whole-image parallelism only. Result must be
    identical for any n_workers."""
    raise NotImplementedError


def benchmark(paths: Sequence[str], params: DetectionParams,
              n_workers: int) -> dict[str, float]:
    """Return per-stage wall times in seconds: io, background, filter,
    detect, merge, total."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Person 3: background / filtering / detection / GPU
# --------------------------------------------------------------------------
def estimate_background(img: np.ndarray, box: int = 64
                        ) -> tuple[np.ndarray, float]:
    """Sigma-clipped median on a coarse grid, interpolated to full size.
    Return (background_map, noise_sigma)."""
    raise NotImplementedError


def smooth(data: np.ndarray, kernel_sigma: float, backend: str = "cpu"
           ) -> np.ndarray:
    """Gaussian smoothing via FFT. backend "cpu" uses scipy.fft, "gpu" uses
    cupy.fft. GPU output must match CPU within ~1e-4 relative (float32).
    Returns a NumPy array in both cases."""
    raise NotImplementedError


def detect(tile: Tile, noise_sigma: float, params: DetectionParams
           ) -> list[Source]:
    """Smooth, threshold at threshold_sigma * noise_sigma, label connected
    regions (scipy.ndimage.label), measure each, convert to full-image
    coords, and return ONLY sources whose centroid is inside tile.core.
    source_id may be 0 here; merge() reassigns it."""
    raise NotImplementedError


def process_image(img: ImageData, params: DetectionParams,
                  tile_size: int | None = None) -> list[Source]:
    """Reference single-process pipeline: background -> subtract -> tile ->
    detect -> merge. Used as the ground truth for the parallel version."""
    raise NotImplementedError
