"""Opt-in resident CUDA pipeline; host code only orchestrates device operations.

The public pipeline still returns Source records. Scientific arrays stay on the
current worker's GPU until then; profile exports are explicit and separate.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from . import tiling
from .interfaces import ImageData, Source


@lru_cache(maxsize=None)
def _kernels(device):
    import cupy as cp
    source = Path(__file__).resolve().parents[1] / "native" / "pipeline_gpu.cu"
    return cp.RawModule(code=source.read_text(), options=("--std=c++20", "--fmad=false"))


class Workspace:
    """Per-call arrays and timing state; no shared mutable image/stream state."""

    def __init__(self, pixels, valid, *, profile=False):
        import cupy as cp
        from cupyx.scipy import ndimage
        self.cp, self.ndimage = cp, ndimage
        self.profile = profile
        self.spans = []
        self.metrics = dict(explicit_h2d_bytes=pixels.nbytes + valid.nbytes,
                            catalog_d2h_bytes=0, scalar_d2h_bytes=0,
                            device_pool_peak_bytes=0)
        self.started = perf_counter()
        # CuPy can enqueue an asynchronous host copy. Retain temporary
        # contiguous views (including strided callers) until this call finishes.
        self._host_uploads = (np.ascontiguousarray(pixels), np.ascontiguousarray(valid))
        with self.stage("gpu_h2d"):
            self.pixels = cp.asarray(self._host_uploads[0])
            self.valid = cp.asarray(self._host_uploads[1])
        self.shape = pixels.shape
        self.all_valid = bool(valid.all())
        self.module = _kernels(cp.cuda.runtime.getDevice())

    @contextmanager
    def stage(self, name):
        if not self.profile:
            yield
            return
        cp = self.cp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.spans.append((name, start, end))
            self.metrics["device_pool_peak_bytes"] = max(
                self.metrics["device_pool_peak_bytes"], cp.get_default_memory_pool().total_bytes())

    def scalar(self, value):
        self.metrics["scalar_d2h_bytes"] += value.nbytes
        with self.stage("gpu_d2h"):
            return value.item()

    def launch(self, name, size, args):
        self.module.get_function(name)(((size + 255)//256,), (256,), args)

    def stats(self, samples):
        """Exact global clipped statistics, never chunk-median approximations."""
        cp = self.cp
        if samples.size == 0:
            raise ValueError("insufficient background coverage: no source-free observations")
        from ._background_gpu import clipped_statistics_device
        if not hasattr(self, "statistics_scratch"):
            self.statistics_scratch = cp.empty(self.pixels.size, cp.float64)
            self.statistics_keep = cp.empty(self.pixels.size, cp.bool_)
        def observe(center, noise, count):
            with self.stage("gpu_d2h"):
                packet = cp.stack((noise, count)).get()
            self.metrics["scalar_d2h_bytes"] += packet.nbytes
            return packet[0], int(packet[1])
        center, noise, _ = clipped_statistics_device(samples, self.statistics_scratch,
                                                     self.statistics_keep, observe)
        return center, noise

    def seeds(self, values, cutoff):
        cp = self.cp
        labels, count = self.ndimage.label(self.valid & (values.astype(cp.float64) > cutoff),
                                          structure=cp.ones((3, 3), cp.uint8))
        sizes = cp.bincount(labels.ravel(), minlength=count+1)
        keep = sizes >= 5
        keep[0] = False
        return keep[labels]

    def reconstruct(self, seed, mask, labels, mode):
        """Monotone 8-connected reconstruction, with no iteration truncation.

        mode 0: geodesic dilation; 1: geodesic erosion; 2: plateau distances.
        Active tiles and all image buffers remain on device. Only a convergence
        flag crosses to the host after a batch of launches.
        """
        cp = self.cp
        h, w = self.shape
        grid = ((w+31)//32, (h+7)//8)
        active = cp.ones(grid[::-1], cp.int32)
        other = seed.copy()
        kernel = self.module.get_function("reconstruct")
        while True:
            for _ in range(4):
                next_active = cp.zeros_like(active)
                kernel(grid, (32, 8), (seed, other, mask, labels, np.int32(h), np.int32(w),
                                      np.int32(mode), active, next_active))
                # Inactive tiles must retain their converged values. Updating
                # seed in place between launches is safe after the whole grid.
                cp.copyto(seed, other)
                active = next_active
            if not self.scalar(cp.any(active)):
                return seed

    def _mesh_medians(self, samples):
        cp = self.cp
        # Positive infinity represents missing slots, never an observation.
        # Row counts select exact odd/even medians despite unequal cell areas.
        def median(sorted_values, counts):
            lo = cp.maximum((counts-1)//2, 0)[:, None]
            hi = cp.maximum(counts//2, 0)[:, None]
            return ((cp.take_along_axis(sorted_values, lo, axis=1)[:, 0]
                     + cp.take_along_axis(sorted_values, hi, axis=1)[:, 0]) * .5)
        counts = cp.count_nonzero(cp.isfinite(samples), axis=1)
        result = cp.full(samples.shape[0], cp.nan, cp.float64)
        active = cp.flatnonzero(counts > 0)
        samples, counts = samples[active], counts[active]
        for iteration in range(6):
            if not active.size:
                break
            samples = cp.sort(samples, axis=1)
            center = median(samples, counts)
            deviations = cp.where(cp.isfinite(samples), cp.abs(samples-center[:, None]), cp.inf)
            noise = median(cp.sort(deviations, axis=1), counts)*1.4826
            if iteration == 5:
                result[active] = center
                break
            keep = cp.isfinite(samples) & ((noise[:, None] == 0) | (deviations <= 3*noise[:, None]))
            retained = cp.count_nonzero(keep, axis=1)
            finished = (noise == 0) | (retained == counts)
            result[active[finished]] = center[finished]
            # Retire converged cells from the batch. Continuing to sort those
            # rows wastes most of the mesh time on later clipping iterations.
            active = active[~finished]
            samples = cp.where(keep, samples, cp.inf)[~finished]
            counts = retained[~finished]
        return result

    def _fit(self, source, box):
        cp = self.cp
        h, w = self.shape
        ny, nx = (h+box-1)//box, (w+box-1)//box
        cy = (cp.arange(ny, dtype=cp.float64)*box + cp.minimum((cp.arange(ny)+1)*box, h)-1)*.5
        cx = (cp.arange(nx, dtype=cp.float64)*box + cp.minimum((cp.arange(nx)+1)*box, w)-1)*.5
        def pack(array, fill):
            return cp.pad(array, ((0, ny*box-h), (0, nx*box-w)), constant_values=fill).reshape(
                ny, box, nx, box).transpose(0, 2, 1, 3).reshape(ny*nx, box*box)
        with self.stage("background_mesh"):
            covered = pack(self.valid, False)
            sky = pack(self.valid & ~source, False)
            coverage, count = covered.sum(axis=1), sky.sum(axis=1)
            area = (cp.minimum(box, h-cp.arange(ny)*box)[:, None]
                    * cp.minimum(box, w-cp.arange(nx)*box)[None, :]).ravel()
            measured = (10*coverage >= 9*area) & (count > 0) & (2*count >= coverage)
            samples = cp.where(sky & measured[:, None], pack(self.pixels, 0).astype(cp.float64), cp.inf)
            mesh = self._mesh_medians(samples)
        with self.stage("background_interpolation"):
            accepted = int(self.scalar(cp.count_nonzero(measured)))
            if not accepted:
                raise ValueError("insufficient background coverage: no usable mesh cells")
            if accepted != ny*nx:
                yy, xx = cp.meshgrid(cy, cx, indexing="ij")
                distances = ((xx.ravel()[None, :]-xx.ravel()[:, None])/box)**2
                distances += ((yy.ravel()[None, :]-yy.ravel()[:, None])/box)**2
                order = cp.argsort(cp.where(measured[None, :], distances, cp.inf), axis=1).astype(cp.int32)
                error = cp.zeros((), cp.int32)
                self.launch("fill_mesh", ny*nx, (mesh, measured, order, cx, cy,
                            np.int32(nx), np.int32(ny), np.int32(accepted), np.int32(box), error))
                if self.scalar(error):
                    raise ValueError("insufficient background coverage: independent spatial support required")
            mesh = mesh.reshape(ny, nx)
            def weights(centers, length):
                if centers.size == 1:
                    zeros = cp.zeros(length, cp.int32)
                    return zeros, zeros, cp.zeros(length, cp.float64)
                coords = cp.arange(length, dtype=cp.float64)
                lo = cp.clip(cp.searchsorted(centers, coords)-1, 0, centers.size-2)
                return lo, lo+1, (coords-centers[lo])/(centers[lo+1]-centers[lo])
            x0, x1, tx = weights(cx, w)
            y0, y1, ty = weights(cy, h)
            rows = mesh[:, x0]*(1-tx) + mesh[:, x1]*tx
            level = cp.where(self.valid, (rows[y0]*(1-ty[:, None])+rows[y1]*ty[:, None]).astype(cp.float32), 0)
        with self.stage("background_noise"):
            sky = self.valid & ~source
            _, noise = self.stats((self.pixels.astype(cp.float64)-level)[sky])
        return level, noise, mesh, measured.reshape(ny, nx)

    def background(self, *, box=64, refine=True):
        cp = self.cp
        with self.stage("background"):
            with self.stage("background_source_mask"):
                seed = cp.zeros(self.shape, cp.bool_)
                if refine:
                    center, noise = self.stats(self.pixels[self.valid].astype(cp.float64))
                    floor = 8*np.finfo(np.float32).eps*cp.max(cp.abs(self.pixels[self.valid].astype(cp.float64)-center))
                    seed = self.seeds(self.pixels, center+cp.maximum(3*noise, floor))
                yy, xx = cp.mgrid[-8:9, -8:9]
                footprint = xx*xx+yy*yy <= 64
            for iteration in range(3 if refine else 1):
                with self.stage("background_source_mask"):
                    source = self.ndimage.binary_dilation(seed, structure=footprint) & self.valid
                level, noise, mesh, measured = self._fit(source, box)
                if not refine or iteration == 2:
                    break
                with self.stage("background_source_mask"):
                    residual = cp.where(self.valid, self.pixels-level, cp.float32(0))
                    floor = 8*np.finfo(np.float32).eps*cp.max(cp.abs(residual))
                    updated = seed | self.seeds(residual, cp.maximum(3*noise, floor))
                    if self.scalar(cp.all(updated == seed)):
                        break
                    seed = updated
            if not self.scalar(cp.all(cp.isfinite(level))) or not self.scalar(cp.isfinite(noise)):
                raise FloatingPointError("GPU background produced nonfinite values")
            self.level, self.noise = level, noise
            self.source_mask, self.mesh, self.measured_mesh = source, mesh, measured
            self.metrics["refinement_passes"] = iteration+1
            return level, noise

    def smooth(self, pixels, sigma, valid=None):
        cp = self.cp
        from cupyx.scipy.signal import fftconvolve
        radius = math.ceil(4*sigma)
        # Construct the same float32 finite kernel ON DEVICE, then widen both
        # operands for cuFFT. The host filtering wrapper is never called here.
        offsets = cp.arange(-radius, radius+1, dtype=cp.float64)
        weights = cp.exp(-.5*(offsets/sigma)**2)
        weights /= weights.sum()
        kernel = (weights[:, None]*weights[None, :]).astype(cp.float32).astype(cp.float64)
        def convolve(data):
            return fftconvolve(cp.pad(data.astype(cp.float64), radius, mode="symmetric"),
                               kernel, mode="valid").astype(cp.float32)
        if valid is None:
            return convolve(pixels)
        numerator = convolve(cp.where(valid, pixels, 0))
        denominator = convolve(valid.astype(cp.float32))
        supported = valid & (denominator >= .5)
        return cp.where(supported, numerator/cp.where(supported, denominator, 1), cp.float32(0))

    def _strongest(self, peaks, parents, count, candidates=None):
        cp = self.cp
        flat, lab = peaks.ravel(), parents.ravel()
        allowed = lab > 0 if candidates is None else (lab > 0) & candidates.ravel()
        pixels = cp.flatnonzero(allowed)
        maxima = cp.full(count+1, -cp.inf, cp.float32)
        cp.maximum.at(maxima, lab[pixels], flat[pixels])
        first = cp.full(count+1, flat.size, cp.int64)
        cp.minimum.at(first, lab[pixels], cp.where(flat[pixels] == maxima[lab[pixels]], pixels, flat.size))
        return first

    def markers(self, peaks, parents, count, prominence, separation):
        cp = self.cp
        with self.stage("deblend_markers"):
            values = peaks.astype(cp.float64)
            reconstructed = self.reconstruct(values-prominence, values, parents, 0)
            tolerance = 4*np.finfo(np.float64).eps*(cp.abs(values)+cp.abs(reconstructed)+prominence)
            maxima = (parents > 0) & (values-reconstructed >= prominence-tolerance)
            plateaus, n = self.ndimage.label(maxima, structure=cp.ones((3, 3), cp.uint8))
            pixels = cp.flatnonzero(maxima)
            # h-maxima plateaus can include lower shoulders. Choose their true
            # highest observed pixel, with raster order resolving a flat top.
            first = self._strongest(peaks, plateaus, n)
            strongest = self._strongest(peaks, parents, count)
            candidates = cp.zeros(peaks.size+1, cp.bool_)
            candidates[first[1:]] = True
            candidates[strongest[1:]] = True
            candidates = candidates[:-1].reshape(self.shape) & (parents > 0)
            pixels = cp.flatnonzero(candidates)
            order = cp.argsort(-peaks.ravel()[pixels])
            pixels = pixels[order]
            parent = parents.ravel()[pixels]
            order = cp.argsort(parent)
            pixels, parent = pixels[order], parent[order]
            starts = cp.searchsorted(parent, cp.arange(count+1)).astype(cp.int32)
            state = cp.zeros(pixels.size, cp.int32)
            winner = cp.empty_like(state)
            while pixels.size:
                args = (pixels, parent, starts, state, winner, np.int32(pixels.size),
                        np.int32(self.shape[1]), np.int32(separation))
                self.launch("choose_peaks", pixels.size, args)
                self.launch("suppress_peaks", pixels.size, args)
                if not self.scalar(cp.any(state == 0)):
                    break
            selected = cp.zeros(peaks.size, cp.bool_)
            selected[pixels[state == 1]] = True
            return selected.reshape(self.shape)

    def watershed(self, peaks, parents, count, markers, min_area):
        cp = self.cp
        h, w = self.shape
        while True:
            points = cp.flatnonzero(markers)
            n = points.size
            owner = cp.zeros(peaks.size, cp.int32)
            owner[points] = cp.arange(1, n+1, dtype=cp.int32)
            owner = owner.reshape(self.shape)
            marker_counts = cp.bincount(parents.ravel()[points], minlength=count+1)
            multiple = marker_counts[parents] > 1
            single_owner = cp.zeros(count+1, cp.int32)
            cp.maximum.at(single_owner, parents.ravel()[points], cp.arange(1, n+1, dtype=cp.int32))
            if self.scalar(cp.any(multiple)):
                with self.stage("deblend_watershed"):
                    active_parents = cp.where(multiple, parents, cp.int32(0))
                    mask = cp.where(multiple, -peaks.astype(cp.float64), cp.inf)
                    mask = cp.where(markers & multiple, -cp.inf, mask)
                    seed = cp.where(markers & multiple, -cp.inf, cp.inf)
                    height = self.reconstruct(seed, mask, active_parents, 1)
                    successor = cp.empty(peaks.size, cp.int32)
                    distance = cp.empty(self.shape, cp.float64)
                    self.launch("descent", peaks.size, (height, active_parents, markers, successor,
                                distance, np.int32(h), np.int32(w)))
                    distance = self.reconstruct(distance, height, active_parents, 2)
                    self.launch("plateau_paths", peaks.size, (height, distance, active_parents,
                                successor, np.int32(h), np.int32(w)))
                    scratch = cp.empty_like(successor)
                    # Descending (height, plateau distance) paths are acyclic.
                    # Doubling for ceil(log2(N)) steps must reach every root.
                    for _ in range(max(1, (peaks.size-1).bit_length())):
                        self.launch("jump", peaks.size, (successor, scratch, np.int32(peaks.size)))
                        successor, scratch = scratch, successor
                    labels = cp.where(multiple, owner.ravel()[successor].reshape(self.shape), single_owner[parents])
            else:
                labels = single_owner[parents]
            with self.stage("deblend_pruning"):
                areas = cp.bincount(labels.ravel(), minlength=n+1)
                keep = areas[1:] >= min_area
                if self.scalar(cp.all(keep)):
                    split = marker_counts[parents.ravel()[points]] > 1
                    return labels, points, split
                survivors = cp.bincount(parents.ravel()[points], weights=keep.astype(cp.int32), minlength=count+1)
                strongest = self._strongest(peaks, parents, count, markers)
                fallback = strongest[cp.flatnonzero((survivors == 0) & (marker_counts > 0))]
                markers = cp.zeros(peaks.size, cp.bool_)
                markers[points[keep]] = True
                markers[fallback] = True
                markers = markers.reshape(self.shape)

    def measure(self, residual, filtered, noise, params, peak_image=None):
        cp = self.cp
        with self.stage("detect"):
            floor = 8*np.finfo(np.float32).eps*cp.max(cp.abs(cp.where(self.valid, residual, 0)))
            threshold = cp.where(noise == 0, floor, params.threshold_sigma*noise)
            prominence = cp.maximum(cp.maximum(params.deblend_prominence_sigma*noise, floor), np.finfo(np.float32).tiny)
            with self.stage("label"):
                parents, count = self.ndimage.label(self.valid & (filtered.astype(cp.float64) > threshold),
                                                    structure=cp.ones((3, 3), cp.uint8))
                sizes = cp.bincount(parents.ravel(), minlength=count+1)
                keep = sizes >= params.min_area
                keep[0] = False
                parents = cp.where(keep[parents], parents, cp.int32(0))
            peaks = filtered if peak_image is None else peak_image
            with self.stage("deblend"):
                if params.deblend:
                    markers = self.markers(peaks, parents, count, prominence, params.deblend_min_separation)
                else:
                    strongest = self._strongest(peaks, parents, count)
                    markers = cp.zeros(peaks.size+1, cp.bool_)
                    markers[strongest[1:]] = True
                    markers = markers[:-1].reshape(self.shape)
                labels, points, split = self.watershed(peaks, parents, count, markers, params.min_area)
            with self.stage("measurement"):
                pixels = cp.flatnonzero(labels)
                order = cp.argsort(labels.ravel()[pixels])
                pixels = pixels[order]
                starts = cp.searchsorted(labels.ravel()[pixels], cp.arange(1, points.size+2)).astype(cp.int64)
                rows = cp.empty((points.size, 8), cp.float64)
                if points.size:
                    self.module.get_function("measure")((points.size,), (256,),
                        (residual, pixels, starts, points, split, rows, np.int32(self.shape[1]),
                         np.int32(max(2, math.ceil(params.kernel_sigma)))))
            self.labels = labels
            with self.stage("gpu_d2h"):
                result = rows.get()
                self.metrics["catalog_d2h_bytes"] += rows.nbytes
            return result

    def timings(self):
        cp = self.cp
        cp.cuda.get_current_stream().synchronize()
        times = {}
        for name, start, end in self.spans:
            times[name+"_ms"] = times.get(name+"_ms", 0.) + cp.cuda.get_elapsed_time(start, end)
        times["total_ms"] = (perf_counter()-self.started)*1000
        times["gpu_compute_ms"] = sum(times.get(s+"_ms", 0) for s in ("background", "filter", "detect"))-times.get("gpu_d2h_ms", 0)
        return {**times, **self.metrics}


def _sources(image_id, rows, x0=0, y0=0):
    return [Source(image_id, 0, float(cx)+x0, float(cy)+y0, int(xmin)+x0, int(ymin)+y0,
                   int(xmax)+x0, int(ymax)+y0, int(area), float(flux))
            for cx, cy, xmin, ymin, xmax, ymax, area, flux in rows]


def process_image(img, pixels, valid, params, tile_size, *, profile=False):
    workspace = Workspace(pixels, valid, profile=profile)
    cp = workspace.cp
    level, noise = workspace.background(refine=params.refine_background)
    residual = cp.where(workspace.valid, workspace.pixels-level, cp.float32(0))
    halo = math.ceil(4*params.kernel_sigma)
    # Reuse Humza's geometry without copying a residual or label image to host.
    # Host tile data are ignored; only shapes, origins, and ownership are used.
    t0 = perf_counter()
    tiles = tiling.make_tiles(ImageData(img.image_id, pixels, img.header, img.valid_mask), tile_size, halo)
    tile_ms = (perf_counter()-t0)*1000
    with workspace.stage("filter"):
        filtered = cp.empty_like(residual)
        peak_image = cp.empty_like(residual) if params.deblend and params.kernel_sigma > 1 else None
        for tile in tiles:
            h, w = tile.data.shape
            window = (slice(tile.y0, tile.y0+h), slice(tile.x0, tile.x0+w))
            mask = None if workspace.all_valid else workspace.valid[window]
            local = workspace.smooth(residual[window], params.kernel_sigma, mask)
            y0, x0, y1, x1 = tile.core
            core = (slice(y0-tile.y0, y1-tile.y0), slice(x0-tile.x0, x1-tile.x0))
            filtered[y0:y1, x0:x1] = local[core]
            if peak_image is not None:
                sharp = workspace.smooth(residual[window], 1., mask)
                peak_image[y0:y1, x0:x1] = sharp[core]
    rows = workspace.measure(residual, filtered, noise, params, peak_image)
    sources = _sources(img.image_id, rows)
    per_tile = [[] for _ in tiles]
    columns = (pixels.shape[1]+tile_size-1)//tile_size
    for source in sources:
        owner = int(source.y_centroid//tile_size)*columns+int(source.x_centroid//tile_size)
        per_tile[owner].append(source)
    t0 = perf_counter()
    sources = tiling.merge(img.image_id, per_tile, tiles)
    merge_ms = (perf_counter()-t0)*1000
    if profile:
        timings = workspace.timings()
        timings.update(tile_ms=tile_ms, merge_ms=merge_ms)
        return dict(sources=sources, timings=timings, workspace=workspace)
    return sources


def detect_tile(tile, valid, noise, params):
    workspace = Workspace(tile.data, valid)
    cp = workspace.cp
    mask = None if workspace.all_valid else workspace.valid
    filtered = workspace.smooth(workspace.pixels, params.kernel_sigma, mask)
    sharp = workspace.smooth(workspace.pixels, 1., mask) if params.deblend and params.kernel_sigma > 1 else None
    rows = workspace.measure(workspace.pixels, filtered, cp.asarray(noise), params, sharp)
    y0, x0, y1, x1 = tile.core
    return [s for s in _sources(tile.image_id, rows, tile.x0, tile.y0)
            if y0 <= s.y_centroid < y1 and x0 <= s.x_centroid < x1]
