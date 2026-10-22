"""Person 2: per-stage timing harness."""
from __future__ import annotations
from typing import Sequence
from .interfaces import DetectionParams


def benchmark(paths: Sequence[str], params: DetectionParams,
              n_workers: int) -> dict[str, float]:
    """Return per-stage wall times in seconds: io, background, filter,
    detect, merge, total. Run CPU-only and GPU on the SAME node type with the
    same worker count so the 2x comparison is fair. Call cupy.cuda.Device()
    .synchronize() before stopping GPU timers.
    """
    raise NotImplementedError
