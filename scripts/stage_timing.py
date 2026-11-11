"""Oct 31: where does the time go?  (Person 2)

Times CPU stages and the selected FFT filter, computes its share of total
runtime, and prints the Amdahl bound on GPU speedup: 1 / (1 - f).

Run from the repo root:
    python scripts/stage_timing.py
    python scripts/stage_timing.py --n-images 2 --repeats 1        # quick check
    python scripts/stage_timing.py --input path/to/real/fits_dir   # real files

Needs only astro_pipeline (interfaces, tiling, background, filtering,
detection) plus numpy/scipy/astropy. It generates its own test images, so it
does not depend on Person 1's synthetic.py. If io_fits.read_fits is
implemented it is used for the I/O timing; otherwise an astropy fallback is.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astro_pipeline import background, detection, filtering, tiling  # noqa: E402
from astro_pipeline.parallel import run_batch  # noqa: E402
from astro_pipeline.run_pipeline import collect_paths  # noqa: E402
from astro_pipeline.interfaces import DetectionParams, ImageData  # noqa: E402

# (module, attribute, stage label). These are called as module attributes
# inside detection.process_image, so patching the module works.
STAGES = [
    (background, "estimate_background", "background"),
    (tiling, "make_tiles", "tile"),
    (filtering, "smooth", "filter"),
    (detection, "_measure_sources", "detect"),
    (tiling, "merge", "merge"),
]


# ---------------------------------------------------------------- test data
def make_image(shape, n_sources, seed):
    """Gradient + Gaussian noise + Gaussian sources (float32)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 100 + 0.01 * yy + 0.005 * xx + rng.normal(0, 5, shape)
    for _ in range(n_sources):
        x, y = rng.uniform(10, w - 10), rng.uniform(10, h - 10)
        s, flux = rng.uniform(1.5, 3.5), rng.uniform(500, 5000)
        r = int(5 * s)
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, h)
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, w)
        py, px = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += (flux / (2 * np.pi * s * s)
                              * np.exp(-((px - x) ** 2 + (py - y) ** 2)
                                       / (2 * s * s)))
    return img.astype(np.float32)


def write_synthetic_files(out_dir, n_images, shape, n_sources):
    from astropy.io import fits
    paths = []
    for i in range(n_images):
        path = os.path.join(out_dir, f"timing_{i:03d}.fits")
        fits.PrimaryHDU(make_image(shape, n_sources, seed=i)).writeto(
            path, overwrite=True)
        paths.append(path)
    return paths


# ------------------------------------------------------------------- reader
def pick_reader(sample_path, mask_zero_border=False):
    """Use Person 1's read_fits if it works, else a plain astropy reader."""
    try:
        from astro_pipeline.io_fits import read_fits
        read_fits(sample_path, mask_zero_border=mask_zero_border)
        return "io_fits.read_fits", lambda path: read_fits(path, mask_zero_border=mask_zero_border)
    except (ImportError, NotImplementedError):
        pass
    from astropy.io import fits

    def fallback(path):
        data = np.nan_to_num(fits.getdata(path).astype(np.float32))
        stem = Path(path).name.split(".")[0]
        return ImageData(stem, np.ascontiguousarray(data, np.float32))

    return "astropy fallback (read_fits not implemented here)", fallback


# ------------------------------------------------------------------- timing
@contextlib.contextmanager
def instrumented(acc):
    originals = []
    for module, name, label in STAGES:
        orig = getattr(module, name)
        originals.append((module, name, orig))

        def make(orig=orig, label=label):
            def wrapper(*a, **k):
                t0 = time.perf_counter()
                try:
                    if label == "background":
                        result = background.profile_background(*a, **k)
                        for stage in ("mesh_ms", "interpolation_ms", "noise_ms", "source_mask_ms", "global_median_ms"):
                            acc["background_" + stage[:-3]] += result["timings"][stage] / 1000
                        for stage, value in result["timings"].items():
                            if stage.startswith("statistics_"):
                                if stage.endswith("_ms"):
                                    acc["background_" + stage[:-3]] += value / 1000
                                else:
                                    acc[stage] += value
                        return result["background"], result["noise"]
                    if label == "detect":
                        result = orig(*a, **k, profile=True)
                        for stage in ("label_ms", "deblend_ms", "measurement_ms"):
                            acc[stage[:-3]] += result["timings"][stage] / 1000
                        for stage in ("deblend_prepare_ms", "deblend_markers_ms",
                                      "deblend_watershed_ms", "deblend_pruning_ms"):
                            if stage in result["timings"]:
                                acc[stage[:-3]] += result["timings"][stage] / 1000
                        components = result.get("deblend_components", [])
                        if components:
                            acc["deblend_regions"] += len(components)
                            for counter in ("queue_pushes", "queue_pops", "passes"):
                                acc["deblend_" + counter] += sum(c[counter] for c in components)
                            largest = max(components, key=lambda c: c["total_ms"])
                            acc["deblend_largest_region"] += largest["total_ms"] / 1000
                            acc["deblend_largest_pixels"] += largest["pixels"]
                            acc["deblend_queue_peak"] += max(c["queue_peak"] for c in components)
                        return result["sources"]
                    return orig(*a, **k)
                finally:
                    acc[label] += time.perf_counter() - t0
            return wrapper

        setattr(module, name, make())
    try:
        yield
    finally:
        for module, name, orig in originals:
            setattr(module, name, orig)


def time_read(paths, reader):
    t0 = time.perf_counter()
    images = [reader(p) for p in paths]
    return time.perf_counter() - t0, images


def time_process(images, params, tile_size):
    # Measure production execution separately: diagnostic array exports are not
    # included in the reported unprofiled wall time.
    started = time.perf_counter()
    for image in images:
        detection.process_image(image, params, tile_size)
    unprofiled = time.perf_counter() - started
    if params.pipeline_backend == "gpu":
        from astro_pipeline._pipeline_gpu import process_image
        acc = defaultdict(float)
        started = time.perf_counter()
        for image in images:
            result = process_image(image, image.data,
                background._valid_mask(image.data.shape, image.valid_mask), params,
                tile_size or max(image.data.shape), profile=True)
            for key, value in result["timings"].items():
                acc[key[:-3] if key.endswith("_ms") else key] += value/1000 if key.endswith("_ms") else value
        acc["total"] = time.perf_counter()-started
        acc["unprofiled"] = unprofiled
        acc["other"] = acc["total"]-sum(acc[k] for k in ("background", "tile", "filter", "detect", "merge"))
        return dict(acc)
    acc = defaultdict(float)
    with instrumented(acc):
        t0 = time.perf_counter()
        for img in images:
            detection.process_image(img, params, tile_size)
        total = time.perf_counter() - t0
    out = {k: acc[k] for _, _, k in STAGES}
    out["other"] = total - sum(out.values())
    out["total"] = total
    out["unprofiled"] = unprofiled
    for key, value in acc.items():
        if key not in out:
            out[key] = value
    return out


def report(label, io_secs, stages, n_images, pipeline_backend="staged"):
    grand = io_secs + stages["total"]
    print(f"\n=== tile_size = {label} ===")
    print(f"{'stage':12s} {'ms/image':>10s} {'% of pipeline':>14s}")
    rows = [("io", io_secs)] + [(k, stages[k]) for _, _, k in STAGES]
    rows.append(("other", stages["other"]))
    for name, secs in rows:
        print(f"{name:12s} {1000 * secs / n_images:10.1f} "
              f"{100 * secs / grand:13.1f}%")
    print(f"{'total':12s} {1000 * grand / n_images:10.1f} {100.0:13.1f}%")
    print(f"unprofiled processing: {1000 * stages['unprofiled'] / n_images:.2f} ms/image")
    print("sub-stages (ms/image; CPU global median is a subset of background noise):")
    for name in ("background_mesh", "background_interpolation", "background_noise", "background_global_median",
                 "background_source_mask", "label", "deblend", "measurement"):
        print(f"  {name:28s} {1000 * stages.get(name, 0) / n_images:10.2f}")
    if "deblend_regions" in stages:
        print("deblending component work (summed durations can overlap across workers):")
        for name in ("prepare", "markers", "watershed", "pruning"):
            print(f"  {name:28s} {1000 * stages['deblend_' + name] / n_images:10.2f} ms/image")
        print(f"  {'largest region':28s} {1000 * stages['deblend_largest_region'] / n_images:10.2f} ms/image")
        for name in ("regions", "largest_pixels", "passes", "queue_pushes", "queue_pops", "queue_peak"):
            print(f"  {name:28s} {stages['deblend_' + name] / n_images:10.0f}/image")
    if "statistics_calls" in stages:
        print("GPU statistics (already included in background wall time):")
        for name in ("setup", "gpu_h2d", "gpu_compute", "gpu_d2h", "host_wait"):
            key = "background_statistics_" + name
            print(f"  {name:28s} {1000 * stages.get(key, 0) / n_images:10.2f} ms/image")
        for name in ("calls", "clipping_passes", "explicit_h2d_bytes",
                     "scalar_d2h_bytes", "device_buffer_bytes"):
            print(f"  {name:28s} {stages.get('statistics_' + name, 0) / n_images:10.0f}/image")
        print("  Event intervals and host waits overlap; do not sum them.")

    migrated = ("background", "filter", "detect") if pipeline_backend == "gpu" else ("filter",)
    f = sum(stages[k] for k in migrated) / grand
    print(f"\naccelerated-stage share f = {f:.3f} ({', '.join(migrated)})")
    print(f"Amdahl bound (infinitely fast accelerated stages): {1 / (1 - f):.2f}x")
    for s in (5, 10, 20):
        print(f"  stages {s:>2d}x faster -> overall {1 / ((1 - f) + f / s):.2f}x")
    print("2x target " + ("is reachable" if f > 0.5 else "is NOT reachable")
          + " by speeding up these stages alone"
          + ("" if f > 0.5 else " (f must exceed 0.5)"))
    if pipeline_backend == "gpu":
        print("resident GPU diagnostics (event intervals include stream waits):")
        for key in ("gpu_h2d", "gpu_compute", "gpu_d2h", "deblend_markers", "deblend_watershed", "deblend_pruning"):
            print(f"  {key:28s} {1000*stages.get(key, 0)/n_images:10.2f} ms/image")
        for key in ("explicit_h2d_bytes", "catalog_d2h_bytes", "scalar_d2h_bytes", "device_pool_peak_bytes"):
            print(f"  {key:28s} {stages.get(key, 0)/n_images:10.0f} bytes/image")


def parse_tile_sizes(items):
    return [None if s.lower() == "none" else int(s) for s in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", help="directory, FITS file, or glob of real inputs (optional)")
    ap.add_argument("--limit", type=int, default=10,
                    help="max real files to use with --input")
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--shape", type=int, nargs=2, default=[1489, 2048],
                    metavar=("ROWS", "COLS"), help="default is SDSS-like")
    ap.add_argument("--n-sources", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tile-sizes", nargs="+", default=["none", "512", "1024"],
                    help='e.g. none 512 1024 ("none" = whole image)')
    ap.add_argument("--kernel-sigma", type=float, default=2.0)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--pipeline-backend", choices=("staged", "gpu"), default="staged")
    ap.add_argument("--cpu-filter", choices=("fft", "separable"), default="fft",
                    help="benchmark the existing FFT baseline or a spatial separable CPU baseline")
    ap.add_argument("--results-dir", type=Path, default=Path("benchmarks/results/gpu-migration"))
    ap.add_argument("--mask-zero-border", action="store_true")
    ap.add_argument("--no-background-refinement", action="store_true")
    ap.add_argument("--no-deblend", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="worker count for the separate FITS-to-sources batch wall time")
    args = ap.parse_args()

    params = DetectionParams(kernel_sigma=args.kernel_sigma, backend=args.backend,
                             refine_background=not args.no_background_refinement,
                             deblend=not args.no_deblend, pipeline_backend=args.pipeline_backend)
    if args.cpu_filter == "separable":
        from scipy.ndimage import gaussian_filter
        def separable(data, sigma):
            pixels, _, radius = filtering._filter_input(data, sigma)
            return gaussian_filter(pixels.astype(np.float64), sigma, radius=radius,
                                   mode="reflect").astype(np.float32)
        filtering._smooth_cpu = separable

    args.results_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.results_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    saved = dict(pipeline_backend=args.pipeline_backend, filter_backend=args.backend,
                 cpu_filter=args.cpu_filter, repeats=args.repeats, workers=args.workers,
                 environment={key: value for key, value in os.environ.items() if key.startswith("ASTRO_")},
                 configurations=[])

    with contextlib.nullcontext(str(inputs_dir)) as tmp:
        if args.input:
            paths = collect_paths(args.input)
            paths = paths[:args.limit]
            if not paths:
                raise SystemExit(f"no FITS files in {args.input!r}")
            source = f"{len(paths)} real files from {args.input}"
        else:
            paths = write_synthetic_files(tmp, args.n_images,
                                          tuple(args.shape), args.n_sources)
            source = f"{len(paths)} generated images, shape {tuple(args.shape)}"

        reader_name, reader = pick_reader(paths[0], args.mask_zero_border)
        print(f"data:   {source}")
        print(f"reader: {reader_name}")
        print(f"cpus:   {os.cpu_count()}   repeats: {args.repeats} (median)")
        print(f"filter: {args.backend}; background: {os.environ.get('ASTRO_BACKGROUND_BACKEND', 'numpy')} "
              f"({os.environ.get('ASTRO_BACKGROUND_THREADS', '1')} threads); detection: "
              f"{os.environ.get('ASTRO_DETECTION_BACKEND', 'python')} "
              f"({os.environ.get('ASTRO_DETECTION_THREADS', '1')} threads)")
        print(f"global background statistics: {os.environ.get('ASTRO_BACKGROUND_STATS_BACKEND', 'cpu')}")

        time_read(paths, reader)                      # warm the file cache
        io_runs, images = [], None
        for _ in range(args.repeats):
            secs, images = time_read(paths, reader)
            io_runs.append(secs)
        io_secs = statistics.median(io_runs)

        warm_started = time.perf_counter()
        time_process(images[:1], params, None)        # warm up FFT / imports
        saved["initialization_and_warmup_s"] = time.perf_counter()-warm_started

        for tile_size in parse_tile_sizes(args.tile_sizes):
            runs = [time_process(images, params, tile_size)
                    for _ in range(args.repeats)]
            med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            report("none (whole image)" if tile_size is None else tile_size,
                   io_secs, med, len(images), args.pipeline_backend)
            batch = []
            from astro_pipeline.io_fits import write_catalog
            catalog_path = args.results_dir / f"catalog-{tile_size or 'whole'}.csv"
            for _ in range(args.repeats):
                started = time.perf_counter()
                results = run_batch(paths, params, args.workers, tile_size, mask_zero_border=args.mask_zero_border)
                write_catalog([source for image_id in sorted(results) for source in results[image_id]], str(catalog_path))
                batch.append(time.perf_counter() - started)
            print(f"FITS-to-written-catalog batch wall time ({args.workers} workers, includes I/O and pool startup): "
                  f"{statistics.median(batch):.3f} s")
            saved["configurations"].append(dict(tile_size=tile_size, inputs=paths, raw_stage_runs_s=runs,
                median_stages_s=med, raw_io_runs_s=io_runs, raw_fits_to_catalog_s=batch,
                median_fits_to_catalog_s=statistics.median(batch)))

    print(f"process peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB "
          "(includes resident inputs and profiling buffers)")
    if args.workers > 1:
        print(f"largest child peak RSS: {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.1f} MiB "
              "(not aggregate worker memory)")
    saved["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (args.results_dir / "timings.json").write_text(json.dumps(saved, indent=2)+"\n")

    print("\nNotes: The OS file cache is warm, so I/O is a "
          "lower bound;\nreal SDSS .bz2 files add decompression time. "
          "Re-run on a TACC CPU node for final numbers.")


if __name__ == "__main__":
    main()
