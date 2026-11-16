"""Time run_batch at several worker counts, inside one Python process.

Unlike Measure-Command on the whole command, this excludes interpreter start-up
and catalog writing, and it checks that every worker count returns identical
results (so it does not need write_catalog).

    python scripts/scaling_check.py --input data/timing
    python scripts/scaling_check.py --input data/timing --workers 1 2 4 8 --repeats 2
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astro_pipeline.interfaces import DetectionParams  # noqa: E402
from astro_pipeline.parallel import run_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of FITS files")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(p for pat in ("*.fits", "*.fits.gz", "*.fits.bz2", "*.fit")
                   for p in glob.glob(os.path.join(args.input, pat)))[:args.limit]
    if not paths:
        raise SystemExit(f"no FITS files in {args.input!r}")
    params = DetectionParams()
    print(f"{len(paths)} files, tile_size={args.tile_size}, "
          f"cpus={os.cpu_count()}, repeats={args.repeats} (median)\n")

    baseline = None
    base_time = None
    print(f"{'workers':>7s} {'seconds':>9s} {'s/image':>8s} "
          f"{'speedup':>8s} {'efficiency':>11s}  identical")
    for workers in args.workers:
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            result = run_batch(paths, params, workers, args.tile_size)
            times.append(time.perf_counter() - t0)
        secs = statistics.median(times)
        if baseline is None:
            baseline, base_time = result, secs
        speedup = base_time / secs
        same = result == baseline
        print(f"{workers:7d} {secs:9.2f} {secs / len(paths):8.2f} "
              f"{speedup:7.2f}x {100 * speedup * args.workers[0] / workers:10.0f}%"
              f"  {same}")
    print("\nTimes include process start-up for each batch. Efficiency = "
          "speedup / (workers / first workers value).")


if __name__ == "__main__":
    main()
