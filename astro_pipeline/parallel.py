"""Person 2: worker pool for batches of images."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

from . import detection, io_fits
from .interfaces import DetectionParams, Source

# One thread per worker process: the parallelism comes from the pool, so
# BLAS/OpenMP thread pools inside each worker would only oversubscribe cores.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _process_one(path: str, params: DetectionParams, tile_size: int | None
                 ) -> tuple[str, list[Source]]:
    """Top-level (picklable) unit of work: read one file, run the reference
    pipeline on it. CuPy is imported lazily inside filtering._smooth_gpu, so
    with backend="gpu" it is first imported inside the worker, not the parent.
    """
    image = io_fits.read_fits(path)
    return image.image_id, detection.process_image(image, params, tile_size)


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))      # respects Slurm/cgroup limits
    except AttributeError:                       # Windows / macOS
        return os.cpu_count() or 1


@contextlib.contextmanager
def _single_threaded_env():
    """Set thread-count variables while worker processes are spawned (they
    inherit the environment before importing NumPy), then restore them."""
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _failure(path: str, error: BaseException) -> RuntimeError:
    return RuntimeError(f"processing failed for {path!r}: "
                        f"{type(error).__name__}: {error}")


def _record(results: dict, origin: dict, image_id: str,
            sources: list[Source], path: str) -> None:
    if image_id in results:
        raise ValueError(f"duplicate image_id {image_id!r} from "
                         f"{origin[image_id]!r} and {path!r}")
    results[image_id] = sources
    origin[image_id] = path


def run_batch(paths: Sequence[str], params: DetectionParams,
              n_workers: int, tile_size: int | None = None
              ) -> dict[str, list[Source]]:
    """Process many images with a ProcessPoolExecutor (not threads: the GIL
    limits CPU-bound threads). tile_size=None means whole-image parallelism
    only. Result keyed by image_id (sorted) and IDENTICAL for any n_workers.

    Each worker: read_fits -> detection.process_image -> list[Source].
    With one worker (or one file) everything runs in this process, which is
    the baseline the multi-worker results are tested against. Workers use the
    "spawn" start method on every platform: it behaves the same on Windows
    and Linux and is safe with CUDA, which does not survive fork.

    Any failing file aborts the batch with a RuntimeError naming the path.
    """
    paths = [os.fspath(p) for p in paths]
    if (isinstance(n_workers, bool) or not isinstance(n_workers, int)
            or n_workers < 1):
        raise ValueError("n_workers must be a positive integer")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate paths in batch")

    results: dict[str, list[Source]] = {}
    origin: dict[str, str] = {}
    if not paths:
        return results

    if n_workers > _available_cpus():
        warnings.warn(f"n_workers={n_workers} exceeds the {_available_cpus()} "
                      "available CPUs; workers will compete for cores")
    workers = min(n_workers, len(paths))

    if workers == 1:
        for path in paths:
            try:
                image_id, sources = _process_one(path, params, tile_size)
            except Exception as error:
                raise _failure(path, error) from error
            _record(results, origin, image_id, sources, path)
    else:
        with _single_threaded_env(), ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_process_one, p, params, tile_size): p
                       for p in paths}
            try:
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        image_id, sources = future.result()
                    except Exception as error:
                        raise _failure(path, error) from error
                    _record(results, origin, image_id, sources, path)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    return dict(sorted(results.items()))