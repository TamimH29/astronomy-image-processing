"""Shared implementation for the one-tier accuracy comparison commands."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINAL_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(FINAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_SCRIPTS_DIR))

# Reuse the original accuracy script's input discovery and truth scoring so
# standalone tier jobs follow the same matching rules as the combined report.
from compare_accuracy_tiers import _collect_paths, _score_tier  # noqa: E402
from astro_pipeline.interfaces import DetectionParams  # noqa: E402
from astro_pipeline.io_fits import write_catalog  # noqa: E402
from astro_pipeline.parallel import run_batch  # noqa: E402


TIER_CONFIG = {
    "single_core_cpu": {"pipeline_backend": "staged", "single_core": True},
    "multicore_cpu": {"pipeline_backend": "staged", "single_core": False},
    "multicore_gpu": {"pipeline_backend": "gpu", "single_core": False},
}


def _available_workers() -> int:
    """Prefer the CPU allocation visible to the job scheduler/process."""
    slurm_count = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_count:
        try:
            return max(1, int(slurm_count))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def run_tier(tier: str) -> None:
    config = TIER_CONFIG[tier]
    parser = argparse.ArgumentParser(
        description=f"Run the {tier} tier and score its catalog against synthetic truth."
    )
    parser.add_argument("--input", required=True, help="FITS directory, file, or glob")
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="new or empty directory for this tier's output")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes; defaults to 1 or the job's available CPU count")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--kernel-sigma", type=float, default=2.0)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--mask-zero-border", action="store_true")
    parser.add_argument("--no-background-refinement", action="store_true")
    parser.add_argument("--no-deblend", action="store_true")
    args = parser.parse_args()

    if config["single_core"]:
        if args.workers not in (None, 1):
            parser.error("the single-core CPU script always uses exactly one worker")
        workers = 1
    else:
        workers = args.workers if args.workers is not None else _available_workers()
        if workers < 1:
            parser.error("--workers must be positive")
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be positive")
    if not math.isfinite(args.match_radius) or args.match_radius <= 0:
        parser.error("--match-radius must be positive and finite")

    paths = _collect_paths(args.input)
    if not paths:
        parser.error(f"no FITS files found for {args.input!r}")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        parser.error(f"results directory is not empty: {args.results_dir}; choose a new directory")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    params = DetectionParams(
        kernel_sigma=args.kernel_sigma,
        threshold_sigma=args.threshold_sigma,
        min_area=args.min_area,
        backend="cpu",
        refine_background=not args.no_background_refinement,
        deblend=not args.no_deblend,
        pipeline_backend=config["pipeline_backend"],
    )
    result = run_batch(paths, params, workers, args.tile_size,
                       mask_zero_border=args.mask_zero_border)
    flat = [source for image_id in sorted(result) for source in result[image_id]]
    catalog_path = args.results_dir / "catalog.csv"
    write_catalog(flat, str(catalog_path))
    score = _score_tier(paths, result, args.match_radius)

    report = {
        "tier": tier,
        "input": args.input,
        "input_count": len(paths),
        "workers_requested": workers,
        "workers_effective_max": min(workers, len(paths)),
        "pipeline_backend": config["pipeline_backend"],
        "tile_size": args.tile_size,
        "match_radius_pix": args.match_radius,
        "detection_params": asdict(params),
        "images_processed": len(result),
        "sources_detected": len(flat),
        "ground_truth_score": score,
        "truth_note": (
            "Truth is loaded from adjacent <image-stem>.truth.json files. "
            "Aggregate positional and flux errors are medians of per-image medians."
        ),
        "catalog": str(catalog_path),
    }
    report_path = args.results_dir / "accuracy.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if score.get("truth_available"):
        completeness = score.get("completeness")
        false_positive_rate = score.get("false_positive_rate")
        completeness_text = "n/a" if completeness is None else f"{completeness:.4f}"
        false_positive_text = (
            "n/a" if false_positive_rate is None else f"{false_positive_rate:.4f}"
        )
        print(f"{tier}: processed {len(paths)} images with {workers} worker(s), "
              f"detected {len(flat)} sources")
        print(f"completeness={completeness_text}; "
              f"false_positive_rate={false_positive_text}")
    else:
        print(f"{tier}: processed {len(paths)} images with {workers} worker(s); "
              "no adjacent truth JSON files were found")
    print(f"Wrote {catalog_path}")
    print(f"Wrote {report_path}")
