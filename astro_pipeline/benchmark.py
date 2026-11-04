"""Person 2: per-stage timing harness."""
from __future__ import annotations
from typing import Sequence
from .interfaces import BenchmarkTimings, DetectionParams


def benchmark(paths: Sequence[str], params: DetectionParams,
              n_workers: int, tile_size: int | None = None,
              repeats: int = 3) -> BenchmarkTimings:
    """Return median per-stage wall times in seconds for one configuration.

    The result uses all keys in BenchmarkTimings. GPU keys are None when
    params.backend is CPU. GPU transfer/compute times are substage breakdowns
    within `filter`; `total` is end-to-end time and can differ from the sum of
    stage times when stages overlap.

    Call this with backend='cpu', n_workers=1 for the single-core tier;
    backend='cpu' and the selected node-wide worker count for multicore CPU;
    and backend='gpu' with the same worker count/node type for multicore+GPU.
    Repeat each tier and report median timings. Exclude GPU warm-up from the
    samples or record it separately in BenchmarkRun.gpu_warmup_s. Synchronize
    the GPU before stopping GPU timers. The `write` stage uses a temporary
    catalog output so benchmark runs do not replace the user's requested file.
    """
    raise NotImplementedError
