"""Person 2: per-stage timing harness."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

from . import background, detection, filtering, io_fits, tiling
from .interfaces import (BENCHMARK_GPU_TIMING_KEYS, BENCHMARK_STAGE_KEYS,
                         BenchmarkTimings, DetectionParams)
from .parallel import _failure, _single_threaded_env

# (module, attribute, stage). detection.process_image calls these through
# their modules, so wrapping them inside each worker attributes time to a
# stage without editing anyone else's code.
_WRAPPED = (
    (background, "estimate_background", "background"),
    (tiling, "make_tiles", "tile"),
    (filtering, "smooth", "filter"),
    (detection, "_measure_sources", "detect"),
    (tiling, "merge", "merge"),
)


@contextlib.contextmanager
def _timed_stages(acc):
    saved = []
    for module, name, label in _WRAPPED:
        original = getattr(module, name)
        saved.append((module, name, original))

        def make(original=original, label=label):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    acc[label] += time.perf_counter() - start
            return wrapper

        setattr(module, name, make())
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def _sync_gpu() -> None:
    import cupy
    cupy.cuda.Device().synchronize()


def _timed_one(path: str, params: DetectionParams, tile_size: int | None,
               profile: bool):
    """Top-level (picklable) unit of work: returns (image_id, sources, stage
    seconds). With the resident GPU pipeline, per-stage numbers come only from
    its own profile mode (CUDA events); unprofiled passes measure wall time.
    """
    stages = defaultdict(float)
    start = time.perf_counter()
    image = io_fits.read_fits(path)
    stages["io"] = time.perf_counter() - start

    resident = params.pipeline_backend == "gpu"
    uses_gpu = resident or params.backend == "gpu"
    if resident and profile:
        from . import _pipeline_gpu
        pixels = background._image_pixels(image.data)
        valid = background._valid_mask(pixels.shape, image.valid_mask)
        size = tile_size if tile_size is not None else max(pixels.shape)
        result = _pipeline_gpu.process_image(image, pixels, valid, params,
                                             size, profile=True)
        sources, t = result["sources"], result["timings"]
        for stage in ("background", "filter", "detect"):
            stages[stage] = t.get(stage + "_ms", 0.0) / 1000
        stages["tile"] = t.get("tile_ms", 0.0) / 1000
        stages["merge"] = t.get("merge_ms", 0.0) / 1000
        for key in BENCHMARK_GPU_TIMING_KEYS:
            stages[key] = t.get(key + "_ms", 0.0) / 1000
    else:
        wrappers = contextlib.nullcontext() if resident else _timed_stages(stages)
        with wrappers:
            sources = detection.process_image(image, params, tile_size)
        if uses_gpu:
            _sync_gpu()          # stop timers only after queued GPU work ends
    return image.image_id, sources, dict(stages)


def _run_pass(paths, params, tile_size, pool, profile):
    """One whole batch. Returns (wall seconds, summed per-stage busy seconds).
    The catalog is written to a temporary file, never the user's output."""
    started = time.perf_counter()
    if pool is None:
        outcomes = []
        for path in paths:
            try:
                outcomes.append(_timed_one(path, params, tile_size, profile))
            except Exception as error:
                raise _failure(path, error) from error
    else:
        futures = [(p, pool.submit(_timed_one, p, params, tile_size, profile))
                   for p in paths]
        outcomes = []
        try:
            for path, future in futures:
                try:
                    outcomes.append(future.result())
                except Exception as error:
                    raise _failure(path, error) from error
        except BaseException:
            for _, future in futures:
                future.cancel()
            raise

    stages, results = defaultdict(float), {}
    for image_id, sources, timings in outcomes:
        if image_id in results:
            raise ValueError(f"duplicate image_id {image_id!r} in benchmark batch")
        results[image_id] = sources
        for key, value in timings.items():
            stages[key] += value

    with tempfile.TemporaryDirectory() as tmp:
        ordered = [s for image_id in sorted(results) for s in results[image_id]]
        write_start = time.perf_counter()
        io_fits.write_catalog(ordered, os.path.join(tmp, "benchmark_catalog.csv"))
        stages["write"] = time.perf_counter() - write_start
    return time.perf_counter() - started, dict(stages)


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

    How the numbers are defined (state this beside any table built from them):
      * Stage times are seconds of work summed over the whole batch and all
        workers (busy time), not per image and not wall time. With several
        workers their sum exceeds `total`.
      * `total` is wall-clock time for the batch: reads, processing, result
        collection, and the temporary catalog write. It excludes pool
        start-up and the untimed warm-up pass, which is printed to stderr.
      * With pipeline_backend='gpu', stage times and gpu_h2d/gpu_compute/
        gpu_d2h come from a separate profiled pass (CUDA events), and `total`
        from unprofiled passes, so profiling overhead never enters `total`.
      * With backend='gpu' and the staged pipeline, the transfers happen inside
        filtering._smooth_gpu and are included in `filter`; the three gpu_*
        substages are None because that function cannot be split from outside.
    """
    paths = [os.fspath(p) for p in paths]
    if not paths:
        raise ValueError("paths must not be empty")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate paths in batch")
    if (isinstance(n_workers, bool) or not isinstance(n_workers, int)
            or n_workers < 1):
        raise ValueError("n_workers must be a positive integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    resident = params.pipeline_backend == "gpu"
    workers = min(n_workers, len(paths))
    totals, samples = [], []
    with contextlib.ExitStack() as stack:
        pool = None
        if workers > 1:
            stack.enter_context(_single_threaded_env())
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")))
        warm = time.perf_counter()
        _run_pass(paths, params, tile_size, pool, profile=False)
        print(f"[benchmark] warm-up pass excluded from results: "
              f"{time.perf_counter() - warm:.2f} s", file=sys.stderr)
        for _ in range(repeats):
            wall, stages = _run_pass(paths, params, tile_size, pool, False)
            totals.append(wall)
            if resident:
                _, stages = _run_pass(paths, params, tile_size, pool, True)
            samples.append(stages)

    result = {key: statistics.median([s.get(key, 0.0) for s in samples])
              for key in BENCHMARK_STAGE_KEYS}
    for key in BENCHMARK_GPU_TIMING_KEYS:
        result[key] = (statistics.median([s.get(key, 0.0) for s in samples])
                       if resident else None)
    result["total"] = statistics.median(totals)
    return result