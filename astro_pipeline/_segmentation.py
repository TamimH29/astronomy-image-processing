"""Deterministic Python reference for CPU segmentation and source measurements.

The C++ implementation follows the same neighbor, plateau, and queue ordering.
This module has no dependency on the optional native extension.
"""
from __future__ import annotations

import heapq
from time import perf_counter
import numpy as np
from scipy.ndimage import find_objects, label

CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def source_seeds(values, valid, cutoff):
    labels, count = label(valid & (values.astype(np.float64) > cutoff), structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= 5
    keep[0] = False
    return keep[labels]


def _neighbors(pixel, height, width):
    y, x = divmod(pixel, width)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and 0 <= y + dy < height and 0 <= x + dx < width:
                yield (y + dy) * width + x + dx


def _markers(image, region, prominence, separation):
    """Build the superlevel merge tree; a peak dies at its joining saddle.

    Equal-height peaks favor the smaller row-major index. Pixels on one flat
    maximum therefore produce one marker, while distinct equal peaks separated
    by a lower saddle remain distinct if their prominence is sufficient.
    """
    height, width = region.shape
    values = image.ravel()
    pixels = np.flatnonzero(region)
    order = pixels[np.lexsort((pixels, -values[pixels].astype(np.float64)))]
    parents = np.full(region.size, -1, dtype=np.int64)
    candidates = []

    def root(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]
            p = int(parents[p])
        return p

    for pixel in order:
        pixel = int(pixel)
        parents[pixel] = pixel
        for neighbor in _neighbors(pixel, height, width):
            if parents[neighbor] < 0:
                continue
            a, b = root(pixel), root(neighbor)
            if a == b:
                continue
            if values[a] < values[b] or (values[a] == values[b] and a > b):
                a, b = b, a
            if float(values[b]) - float(values[pixel]) >= prominence:
                candidates.append(b)
            parents[b] = a
    candidates.append(root(int(order[0])))  # keep the strongest even if marginal
    candidates.sort(key=lambda p: (-float(values[p]), p))
    kept = []
    for pixel in candidates:
        y, x = divmod(pixel, width)
        if all((y - q // width) ** 2 + (x - q % width) ** 2 >= separation ** 2
               for q in kept):
            kept.append(pixel)
    return sorted(kept)


def _watershed(image, region, markers):
    height, width = region.shape
    values, inside = image.ravel(), region.ravel()
    owners = np.zeros(region.size, dtype=np.int32)
    queue = []
    sequence = 0
    for owner, pixel in enumerate(markers, 1):
        owners[pixel] = owner
        heapq.heappush(queue, (-float(values[pixel]), sequence, pixel, owner, True))
        sequence += 1
    while queue:
        level, _, pixel, owner, seed = heapq.heappop(queue)
        if not seed:
            if owners[pixel]:
                continue
            owners[pixel] = owner
        for neighbor in _neighbors(pixel, height, width):
            if inside[neighbor] and not owners[neighbor]:
                # Path level never rises above its lowest saddle. FIFO settles
                # equal levels; a pixel is claimed only when its event is popped.
                heapq.heappush(queue, (max(level, -float(values[neighbor])),
                                      sequence, neighbor, owner, False))
                sequence += 1
    return owners.reshape(region.shape)


def segment(filtered, valid, threshold, min_area, deblend, separation, prominence,
            timings=None, *, peak_image=None, centroid_peaks=None):
    started = perf_counter()
    parents, count = label(valid & (filtered.astype(np.float64) > threshold), structure=CONNECTIVITY)
    sizes = np.bincount(parents.ravel(), minlength=count + 1)
    result = np.zeros_like(parents)
    next_label = 0
    if timings is not None:
        timings["label_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    for parent, window in enumerate(find_objects(parents), 1):
        if window is None or sizes[parent] < min_area:
            continue
        region = parents[window] == parent
        image = (filtered if peak_image is None else peak_image)[window]
        markers = _markers(image, region, prominence, separation) if deblend else []
        if len(markers) <= 1:
            children = region.astype(np.int32)
        else:
            while True:
                children = _watershed(image, region, markers)
                areas = np.bincount(children.ravel(), minlength=len(markers) + 1)
                retained = [p for i, p in enumerate(markers, 1) if areas[i] >= min_area]
                if len(retained) == len(markers):
                    break
                if not retained:
                    retained = [min(markers, key=lambda p: (-float(image.flat[p]), p))]
                markers = retained
                if len(markers) == 1:
                    children = region.astype(np.int32)
                    break
        target = result[window]
        positive = children > 0
        target[positive] = children[positive] + next_label
        if centroid_peaks is not None:
            sy, sx = window
            centroid_peaks.extend([(sy.start + p // image.shape[1], sx.start + p % image.shape[1])
                                   for p in markers] if int(children.max()) > 1 else [None])
        next_label += int(children.max())
    if timings is not None:
        timings["deblend_ms"] = (perf_counter() - started) * 1000
    return result


def measure(residual, labels, centroid_peaks=None, radius=2):
    rows = []
    for component, window in enumerate(find_objects(labels), 1):
        if window is None:
            continue
        sy, sx = window
        yy, xx = np.nonzero(labels[window] == component)
        values = residual[window][yy, xx].astype(np.float64)
        yy, xx = yy + sy.start, xx + sx.start
        selected = np.ones(len(values), dtype=bool)
        if centroid_peaks is not None and centroid_peaks[component - 1] is not None:
            py, px = centroid_peaks[component - 1]
            selected = (abs(yy - py) <= radius) & (abs(xx - px) <= radius)
        cyy, cxx, centered = yy[selected], xx[selected], values[selected]
        weights = np.maximum(centered, 0)
        total = weights.sum(dtype=np.float64)
        if total > 0:
            cx = float(np.sum(cxx * weights, dtype=np.float64) / total)
            cy = float(np.sum(cyy * weights, dtype=np.float64) / total)
        else:
            cx, cy = float(cxx.mean()), float(cyy.mean())
        rows.append((cx, cy, sx.start, sy.start, sx.stop - 1, sy.stop - 1,
                     len(values), float(values.sum(dtype=np.float64))))
    return rows
