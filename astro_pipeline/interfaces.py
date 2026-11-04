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
  * valid_mask=None means all observations are valid. False pixels have finite
    fillers, contribute no statistics/measurements, and retain their mask in halos.
    Internal source-exclusion masks affect background fitting only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, TypedDict
import numpy as np

INTERFACE_VERSION = 4

# Fixed catalog schema (column order matters for CSV output).
CATALOG_COLUMNS = [
    "image_id", "source_id",
    "x_centroid", "y_centroid",
    "xmin", "ymin", "xmax", "ymax",   # inclusive bounding box, pixels
    "area_pix",                       # number of pixels above threshold
    "flux",                           # background-subtracted integrated flux
]

# Timing keys are kept in report/CLI order. `total` is an end-to-end wall
# time and is not expected to equal the sum of stages when work overlaps.
BENCHMARK_STAGE_KEYS = (
    "io",
    "background",
    "tile",
    "filter",
    "detect",
    "merge",
    "write",
)
BENCHMARK_GPU_TIMING_KEYS = ("gpu_h2d", "gpu_compute", "gpu_d2h")
BENCHMARK_TIMING_KEYS = (
    *BENCHMARK_STAGE_KEYS,
    *BENCHMARK_GPU_TIMING_KEYS,
    "total",
)
BENCHMARK_SPEEDUP_KEYS = (*BENCHMARK_STAGE_KEYS, "total")

BenchmarkTier = Literal[
    "single_core_cpu",
    "multicore_cpu",
    "multicore_gpu",
]
BENCHMARK_TIER_KEYS: tuple[BenchmarkTier, ...] = (
    "single_core_cpu",
    "multicore_cpu",
    "multicore_gpu",
)
BENCHMARK_SPEEDUP_COMPARISONS = (
    "multicore_vs_single_core",
    "multicore_gpu_vs_multicore",
    "multicore_gpu_vs_single_core",
)


class BenchmarkTimings(TypedDict):
    """One benchmark configuration's median wall times, in seconds.

    GPU substage values are None for CPU-only runs. They are breakdowns of the
    filter stage, not additional durations to sum into total.
    """

    io: float
    background: float
    tile: float
    filter: float
    detect: float
    merge: float
    write: float
    gpu_h2d: float | None
    gpu_compute: float | None
    gpu_d2h: float | None
    total: float


@dataclass
class BenchmarkRun:
    """Median timings and execution metadata for one benchmark tier."""

    tier: BenchmarkTier
    node_type: str
    n_workers: int
    repeat_count: int
    median_timings_s: BenchmarkTimings
    gpu_warmup_s: float | None = None


@dataclass
class BenchmarkComparison:
    """Three-tier runs and per-stage speedups for a benchmark campaign.

    `speedups` is keyed by BENCHMARK_SPEEDUP_COMPARISONS, then by
    BENCHMARK_SPEEDUP_KEYS. Values are baseline seconds divided by comparison
    seconds: multicore/single-core, GPU/multicore, and GPU/single-core.
    Undefined ratios are represented by None.
    """

    runs: dict[BenchmarkTier, BenchmarkRun]
    speedups: dict[str, dict[str, float | None]]


@dataclass
class ImageData:
    image_id: str                     # e.g. file stem
    data: np.ndarray                  # 2D float32
    header: dict = field(default_factory=dict)
    valid_mask: np.ndarray | None = None  # True = observed; None = all valid


@dataclass
class Tile:
    image_id: str
    y0: int                           # image coords of data[0, 0] (includes halo)
    x0: int
    data: np.ndarray                  # 2D float32, core + halo (clipped at edges)
    core: tuple[int, int, int, int]   # (cy0, cx0, cy1, cx1) owned region, image
                                      # coords, half-open [cy0, cy1) x [cx0, cx1)
    halo: int
    valid_mask: np.ndarray | None = None  # includes the same halo as data


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
    refine_background: bool = True
    deblend: bool = True
    deblend_min_separation: int = 3    # pixels between retained peak markers
    deblend_prominence_sigma: float = 1.0  # peak-to-saddle height / pre-filter noise
