"""Command-line entry point (Person 2).

Example:
  python -m astro_pipeline.run_pipeline --input data/sdss --output out/catalog.csv \
      --workers 4 --tile-size 1024 --backend cpu
"""
from __future__ import annotations
import argparse
import glob
import os

from .interfaces import BENCHMARK_TIMING_KEYS, DetectionParams
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
    ap.add_argument("--pipeline-backend", choices=["staged", "gpu"], default="staged",
                    help="gpu runs all numerical stages on device; --backend and stage environment selectors apply to staged mode")
    ap.add_argument("--kernel-sigma", type=float, default=2.0)
    ap.add_argument("--threshold-sigma", type=float, default=3.0)
    ap.add_argument("--min-area", type=int, default=5)
    ap.add_argument("--mask-zero-border", action="store_true", help="declare border-connected zeros to be missing coverage")
    ap.add_argument("--no-background-refinement", action="store_true", help="disable iterative source exclusion for comparisons")
    ap.add_argument("--no-deblend", action="store_true", help="keep one source per connected component")
    ap.add_argument("--deblend-min-separation", type=int, default=3, help="minimum distance between peak markers in pixels")
    ap.add_argument("--deblend-prominence-sigma", type=float, default=1.0, help="minimum peak-to-saddle height in pre-filter noise units")
    ap.add_argument("--benchmark", action="store_true",
                    help="print per-stage timings instead of writing a catalog")
    ap.add_argument("--benchmark-repeats", type=int, default=3,
                    help="number of timing runs used for the reported median")
    args = ap.parse_args(argv)

    paths = collect_paths(args.input)
    if not paths:
        raise SystemExit(f"no FITS files found for {args.input!r}")
    params = DetectionParams(kernel_sigma=args.kernel_sigma,
                             threshold_sigma=args.threshold_sigma,
                             min_area=args.min_area, backend=args.backend,
                             refine_background=not args.no_background_refinement,
                             deblend=not args.no_deblend,
                             deblend_min_separation=args.deblend_min_separation,
                             deblend_prominence_sigma=args.deblend_prominence_sigma,
                             pipeline_backend=args.pipeline_backend)

    if args.benchmark:
        timings = benchmark(
            paths,
            params,
            args.workers,
            tile_size=args.tile_size,
            repeats=args.benchmark_repeats,
        )
        for stage in BENCHMARK_TIMING_KEYS:
            seconds = timings[stage]
            display = "n/a" if seconds is None else f"{seconds:.3f} s"
            print(f"{stage:12s} {display:>12s}")
        return

    results = run_batch(paths, params, args.workers, args.tile_size,
                        mask_zero_border=args.mask_zero_border)
    all_sources = [s for image_id in sorted(results) for s in results[image_id]]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_catalog(all_sources, args.output)
    print(f"{len(paths)} images, {len(all_sources)} sources -> {args.output}")


if __name__ == "__main__":
    main()
