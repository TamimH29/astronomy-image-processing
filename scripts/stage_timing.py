"""Oct 31: where does the time go?  (Person 2)

Times each pipeline stage on the CPU, computes the filter's share of total
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
import glob
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astro_pipeline import background, detection, filtering, tiling  # noqa: E402
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
def pick_reader(sample_path):
    """Use Person 1's read_fits if it works, else a plain astropy reader."""
    try:
        from astro_pipeline.io_fits import read_fits
        read_fits(sample_path)
        return "io_fits.read_fits", read_fits
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
    acc = defaultdict(float)
    with instrumented(acc):
        t0 = time.perf_counter()
        for img in images:
            detection.process_image(img, params, tile_size)
        total = time.perf_counter() - t0
    out = {k: acc[k] for _, _, k in STAGES}
    out["other"] = total - sum(out.values())
    out["total"] = total
    return out


def report(label, io_secs, stages, n_images):
    grand = io_secs + stages["total"]
    print(f"\n=== tile_size = {label} ===")
    print(f"{'stage':12s} {'ms/image':>10s} {'% of pipeline':>14s}")
    rows = [("io", io_secs)] + [(k, stages[k]) for _, _, k in STAGES]
    rows.append(("other", stages["other"]))
    for name, secs in rows:
        print(f"{name:12s} {1000 * secs / n_images:10.1f} "
              f"{100 * secs / grand:13.1f}%")
    print(f"{'total':12s} {1000 * grand / n_images:10.1f} {100.0:13.1f}%")

    f = stages["filter"] / grand
    print(f"\nfilter share f = {f:.3f}")
    print(f"Amdahl bound (infinitely fast GPU filter): {1 / (1 - f):.2f}x")
    for s in (5, 10, 20):
        print(f"  filter {s:>2d}x faster -> overall {1 / ((1 - f) + f / s):.2f}x")
    print("2x target " + ("is reachable" if f > 0.5 else "is NOT reachable")
          + " by speeding up the filter alone"
          + ("" if f > 0.5 else " (f must exceed 0.5)"))


def parse_tile_sizes(items):
    return [None if s.lower() == "none" else int(s) for s in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", help="directory of real FITS files (optional)")
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
    args = ap.parse_args()

    params = DetectionParams(kernel_sigma=args.kernel_sigma, backend="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        if args.input:
            paths = sorted(p for pat in ("*.fits", "*.fits.gz", "*.fits.bz2",
                                         "*.fit")
                           for p in glob.glob(os.path.join(args.input, pat)))
            paths = paths[:args.limit]
            if not paths:
                raise SystemExit(f"no FITS files in {args.input!r}")
            source = f"{len(paths)} real files from {args.input}"
        else:
            paths = write_synthetic_files(tmp, args.n_images,
                                          tuple(args.shape), args.n_sources)
            source = f"{len(paths)} generated images, shape {tuple(args.shape)}"

        reader_name, reader = pick_reader(paths[0])
        print(f"data:   {source}")
        print(f"reader: {reader_name}")
        print(f"cpus:   {os.cpu_count()}   repeats: {args.repeats} (median)")

        time_read(paths, reader)                      # warm the file cache
        io_runs, images = [], None
        for _ in range(args.repeats):
            secs, images = time_read(paths, reader)
            io_runs.append(secs)
        io_secs = statistics.median(io_runs)

        time_process(images[:1], params, None)        # warm up FFT / imports

        for tile_size in parse_tile_sizes(args.tile_sizes):
            runs = [time_process(images, params, tile_size)
                    for _ in range(args.repeats)]
            med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            report("none (whole image)" if tile_size is None else tile_size,
                   io_secs, med, len(images))

    print("\nNotes: CPU backend only. The OS file cache is warm, so I/O is a "
          "lower bound;\nreal SDSS .bz2 files add decompression time. "
          "Re-run on a TACC CPU node for final numbers.")


if __name__ == "__main__":
    main()