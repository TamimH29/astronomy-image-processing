"""Per-image GPU statistics for the CPU background-refinement loop.

Image/coverage buffers remain on the device across bootstrap and refinement.
Only updated sky maps/source masks are uploaded; explicit downloads contain
three scalars per clipping pass. CuPy compaction can also synchronize internally.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from importlib import import_module
from time import perf_counter

import numpy as np


def clipped_statistics_device(samples, scratch, keep_buffer, observe, *, timed=nullcontext):
    """Shared resident median/MAD operation for staged and full-GPU execution.

    `observe(center, noise, count)` reads only a scalar status packet. Buffers
    belong to the caller, so bootstrap and refinement reuse the same storage.
    """
    import cupy as cp
    if not samples.size:
        raise ValueError("insufficient background coverage: no source-free observations")
    passes = 0
    for iteration in range(6):
        with timed():
            center = cp.median(samples, overwrite_input=True)
            deviations = scratch[:samples.size]
            cp.subtract(samples, center, out=deviations)
            cp.abs(deviations, out=deviations)
            noise = cp.median(deviations, overwrite_input=True)*cp.float64(1.4826)
            keep = keep_buffer[:samples.size]
            # Selection rearranges deviations; rebuild correspondence to samples.
            cp.subtract(samples, center, out=deviations)
            cp.abs(deviations, out=deviations)
            cp.less_equal(deviations, 3*noise, out=keep)
            count = cp.count_nonzero(keep)
        noise_value, kept = observe(center, noise, count)
        if noise_value == 0 or kept == samples.size or iteration == 5:
            return center, noise, passes
        passes += 1
        with timed():
            samples = samples[keep]


class GpuBackgroundStatistics:
    _BUFFERS = ("pixels", "valid", "background", "source", "residual",
                "scratch", "keep", "packet")

    def __init__(self, pixels, valid, *, profile=False):
        started = perf_counter()
        try:
            cp = import_module("cupy")
        except ImportError as error:
            raise ImportError(
                "ASTRO_BACKGROUND_STATS_BACKEND=gpu requires CuPy matching CUDA"
            ) from error
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError(
                "ASTRO_BACKGROUND_STATS_BACKEND=gpu requires a working CUDA "
                "device in this process: no CUDA devices available"
            )
        self.device = cp.cuda.Device()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.cp, self.profile = cp, profile
        self.shape = pixels.shape
        self._spans = []
        self._metrics = dict(statistics_calls=0, statistics_clipping_passes=0,
                             statistics_explicit_h2d_bytes=0,
                             statistics_scalar_d2h_bytes=0,
                             statistics_host_wait_ms=0.0)
        self._closed = False
        # Reuse scratch within this image, then release this context's arena.
        # This avoids accumulating global-pool arenas for successive streams
        # and leaves the existing filtering allocator untouched.
        self._pool = cp.cuda.MemoryPool()
        self._host_uploads = [np.ascontiguousarray(pixels, dtype=np.float32),
                              np.ascontiguousarray(valid, dtype=np.bool_)]
        # Each context belongs to one call/process, including concurrent calls.
        with ExitStack() as cleanup:
            cleanup.callback(self.close)
            with self.device, self.stream, cp.cuda.using_allocator(self._pool.malloc):
                with self._timed("h2d"):
                    self.pixels = cp.asarray(self._host_uploads[0])
                    self.valid = cp.asarray(self._host_uploads[1])
                self._metrics["statistics_explicit_h2d_bytes"] += self.pixels.nbytes + self.valid.nbytes
                self.background = cp.empty(self.shape, dtype=cp.float32)
                self.source = cp.empty(self.shape, dtype=cp.uint8)
                self.residual = cp.empty(self.shape, dtype=cp.float64)
                self.scratch = cp.empty(pixels.size, dtype=cp.float64)
                self.keep = cp.empty(pixels.size, dtype=cp.bool_)
                self.packet = cp.empty(3, dtype=cp.float64)
                self._pinned = cp.cuda.alloc_pinned_memory(3 * np.dtype(np.float64).itemsize)
                self._host_packet = np.frombuffer(self._pinned, dtype=np.float64, count=3)
                self._downloaded = cp.cuda.Event()
            self._metrics["statistics_device_buffer_bytes"] = sum(
                getattr(self, name).nbytes for name in self._BUFFERS)
            self._metrics["statistics_setup_ms"] = (perf_counter() - started) * 1000
            cleanup.pop_all()

    @contextmanager
    def _timed(self, stage):
        if not self.profile:
            yield
            return
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record(self.stream)
        try:
            yield
        finally:
            end.record(self.stream)
            self._spans.append((stage, start, end))

    def _read_packet(self):
        # Pinned output permits one explicit async scalar download and one wait.
        with self._timed("d2h"):
            self.packet.get(stream=self.stream, out=self._host_packet, blocking=False)
        self._downloaded.record(self.stream)
        started = perf_counter()
        self._downloaded.synchronize()
        self._host_uploads.clear()
        self._metrics["statistics_host_wait_ms"] += (perf_counter() - started) * 1000
        self._metrics["statistics_scalar_d2h_bytes"] += self._host_packet.nbytes
        center, noise, kept = self._host_packet
        if not np.isfinite(center) or not np.isfinite(noise) or noise < 0:
            raise FloatingPointError("GPU background statistics produced nonfinite values")
        return float(center), float(noise), int(kept)

    def __call__(self, background=None, source=None):
        if self._closed:
            raise RuntimeError("GPU background statistics context is closed")
        if (background is None) != (source is None):
            raise ValueError("background and source mask must be supplied together")
        cp = self.cp
        with self.device, self.stream, cp.cuda.using_allocator(self._pool.malloc):
            if background is not None:
                if np.shape(background) != self.shape or np.shape(source) != self.shape:
                    raise ValueError("statistics inputs must match the image shape")
                level = np.ascontiguousarray(background, dtype=np.float32)
                excluded = np.ascontiguousarray(source, dtype=np.uint8)
                # Keep borrowed native/Python arrays alive until the queued
                # transfers finish, including if a later GPU operation raises.
                self._host_uploads.extend((level, excluded))
                with self._timed("h2d"):
                    self.background.set(level, stream=self.stream)
                    self.source.set(excluded, stream=self.stream)
                self._metrics["statistics_explicit_h2d_bytes"] += level.nbytes + excluded.nbytes
            with self._timed("compute"):
                if background is None:
                    samples = self.pixels[self.valid].astype(cp.float64)
                else:
                    # Widen BOTH float32 operands before subtraction, exactly as
                    # the reference; float32 subtraction followed by a cast loses bits.
                    cp.subtract(self.pixels, self.background, dtype=cp.float64,
                                out=self.residual)
                    samples = self.residual[self.valid & (self.source == 0)]
            if samples.size == 0:
                raise ValueError("insufficient background coverage: no source-free observations")
            self._metrics["statistics_calls"] += 1
            last = None
            def observe(center, noise, count):
                nonlocal last
                self.packet[0], self.packet[1], self.packet[2] = center, noise, count
                last = self._read_packet()
                return last[1:]
            _, _, passes = clipped_statistics_device(samples, self.scratch, self.keep,
                observe, timed=lambda: self._timed("compute"))
            self._metrics["statistics_clipping_passes"] += passes
            return last[:2]

    def timings(self):
        """Event intervals are stream elapsed times, not isolated kernel costs.

        Explicit byte counts exclude CuPy's internal scalar/shape transfers.
        The caller's total wall time includes allocation, launches, and all waits.
        """
        with self.device:
            self.stream.synchronize()
            measured = dict(self._metrics)
            for stage in ("h2d", "compute", "d2h"):
                measured[f"statistics_gpu_{stage}_ms"] = sum(
                    self.cp.cuda.get_elapsed_time(start, end)
                    for name, start, end in self._spans if name == stage)
            return measured

    def close(self):
        if not self._closed:
            with self.device:
                self.stream.synchronize()
                for name in self._BUFFERS:
                    setattr(self, name, None)
                self._pool.free_all_blocks()
            self._host_uploads.clear()
            self._closed = True
