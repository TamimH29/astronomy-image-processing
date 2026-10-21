"""Person 2: worker pool for batches of images."""
from __future__ import annotations
from typing import Sequence
from .interfaces import DetectionParams, Source


def run_batch(paths: Sequence[str], params: DetectionParams,
              n_workers: int, tile_size: int | None = None
              ) -> dict[str, list[Source]]:
    """Process many images with a ProcessPoolExecutor (not threads: the GIL
    limits CPU-bound threads). tile_size=None means whole-image parallelism
    only. Result keyed by image_id and IDENTICAL for any n_workers.

    Each worker: read_fits -> detection.process_image -> list[Source].
    With params.backend == "gpu", watch how several workers share one GPU
    (initialize CuPy inside each worker, not in the parent).
    """
    raise NotImplementedError
