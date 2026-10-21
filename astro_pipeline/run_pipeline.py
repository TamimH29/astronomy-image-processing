"""Command-line entry point (Person 2).

Example:
  python -m astro_pipeline.run_pipeline --input data/sdss --output out/catalog.csv \
      --workers 4 --tile-size 1024 --backend cpu
"""
from __future__ import annotations
import argparse
import glob
import os

from .interfaces import DetectionParams
from .io_fits import write_catalog
from .parallel import run_batch
from .benchmark import benchmark


def collect_paths(inp: str) -> list[str]:
    if os.path.isdir(inp):
        pats = ("*.fits", "*.fits.gz", "*.fits.bz2", "*.fit")
        paths = [p for pat in pats for p in glob.glob(os.path.join(inp, pat))]
    else:
        paths = glob.glob(inp)
    return sorted(paths)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="FITS source-detection pipeline")
    ap.add_argument("--input", required=True, help="directory or glob of FITS files")
    ap.add_argument("--output", required=True, help="catalog path (.csv or .fits)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tile-size", type=int, default=None,
                    help="omit for whole-image processing")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--kernel-sigma", type=float, default=2.0)
    ap.add_argument("--threshold-sigma", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=5)
    ap.add_argument("--benchmark", action="store_true",
                    help="print per-stage timings instead of writing a catalog")
    args = ap.parse_args(argv)

    paths = collect_paths(args.input)
    if not paths:
        raise SystemExit(f"no FITS files found for {args.input!r}")
    params = DetectionParams(kernel_sigma=args.kernel_sigma,
                             threshold_sigma=args.threshold_sigma,
                             min_area=args.min_area, backend=args.backend)

    if args.benchmark:
        for stage, secs in benchmark(paths, params, args.workers).items():
            print(f"{stage:12s} {secs:10.3f} s")
        return

    results = run_batch(paths, params, args.workers, args.tile_size)
    all_sources = [s for image_id in sorted(results) for s in results[image_id]]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_catalog(all_sources, args.output)
    print(f"{len(paths)} images, {len(all_sources)} sources -> {args.output}")


if __name__ == "__main__":
    main()
