"""Compare catalog accuracy and CPU/GPU stage agreement across project tiers.

Runs the same inputs and detection settings through one-worker CPU, multicore
CPU, and multicore full-GPU pipeline configurations. If adjacent synthetic
truth files exist (*.truth.json), each tier is scored against them. A separate
single-image diagnostic pass compares intermediate CPU/GPU arrays and masks.

Example on Vista (from the repository root):
  python final_scripts/compare_accuracy_tiers.py \
      --input 'gen_fits/large/*.fits' --cpu-workers 20 --tile-size 1024 \
      --results-dir "$WORK/astro-test/accuracy-large"

The report is about correctness, not speed. Nsight and timing instrumentation
are intentionally not used here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from astro_pipeline import background, detection, filtering, io_fits, tiling
from astro_pipeline._pipeline_gpu import Workspace, _sources as gpu_sources
from astro_pipeline.interfaces import DetectionParams, ImageData, Tile
from astro_pipeline.parallel import run_batch
from astro_pipeline.validate import score_catalog
from astro_pipeline.io_fits import write_catalog


def _collect_paths(spec: str) -> list[str]:
    if os.path.isdir(spec):
        paths = []
        for pattern in ("*.fits", "*.fits.gz", "*.fits.bz2", "*.fit"):
            paths.extend(glob.glob(os.path.join(spec, pattern)))
        return sorted(set(paths))
    return sorted(glob.glob(spec))


def _truth_path(image_path: str) -> Path:
    path = Path(image_path)
    name = path.name
    for suffix in (".gz", ".bz2"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    return path.with_name(Path(name).stem + ".truth.json")


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _array_metrics(reference: np.ndarray, compared: np.ndarray,
                   relative_floor: float) -> dict:
    reference = np.asarray(reference)
    compared = np.asarray(compared)
    if reference.shape != compared.shape:
        raise ValueError(f"array shapes differ: {reference.shape} vs {compared.shape}")
    a, b = reference.astype(np.float64), compared.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return {"n_values": 0, "max_abs": None, "max_rel": None, "rms": None}
    delta = np.abs(a[finite] - b[finite])
    denom = np.maximum(np.abs(a[finite]), relative_floor)
    return {
        "n_values": int(delta.size),
        "max_abs": float(delta.max(initial=0.0)),
        "max_rel": float((delta / denom).max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def _mask_metrics(reference: np.ndarray, compared: np.ndarray) -> dict:
    a, b = np.asarray(reference, dtype=bool), np.asarray(compared, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    different = int(np.count_nonzero(a != b))
    union = int(np.count_nonzero(a | b))
    return {
        "pixels": int(a.size),
        "mismatch_pixels": different,
        "mismatch_fraction": different / a.size if a.size else 0.0,
        "iou": (int(np.count_nonzero(a & b)) / union) if union else 1.0,
    }


def _catalog_pair(reference, compared, radius: float) -> dict:
    candidates = []
    for i, left in enumerate(reference):
        for j, right in enumerate(compared):
            distance = math.hypot(left.x_centroid-right.x_centroid,
                                  left.y_centroid-right.y_centroid)
            if distance <= radius:
                candidates.append((distance, i, j))
    candidates.sort()
    used_left, used_right = set(), set()
    pos_errors, x_errors, y_errors = [], [], []
    flux_errors, flux_abs_errors = [], []
    exact_fields = 0
    geometry_matches = 0
    fields_to_check = ("xmin", "ymin", "xmax", "ymax", "area_pix")
    for distance, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        a, b = reference[i], compared[j]
        pos_errors.append(distance)
        x_errors.append(abs(a.x_centroid-b.x_centroid))
        y_errors.append(abs(a.y_centroid-b.y_centroid))
        flux_abs_errors.append(abs(a.flux-b.flux))
        denom = max(abs(a.flux), 1e-12)
        flux_errors.append(abs(a.flux-b.flux)/denom)
        geometry_matches += sum(getattr(a, key) == getattr(b, key) for key in fields_to_check)
        exact_fields += len(fields_to_check)
    return {
        "reference_sources": len(reference),
        "compared_sources": len(compared),
        "matched_sources": len(used_left),
        "only_in_reference": len(reference)-len(used_left),
        "only_in_compared": len(compared)-len(used_right),
        "median_position_difference_pix": statistics.median(pos_errors) if pos_errors else None,
        "max_position_difference_pix": max(pos_errors) if pos_errors else None,
        "max_abs_x_centroid_difference_pix": max(x_errors) if x_errors else None,
        "max_abs_y_centroid_difference_pix": max(y_errors) if y_errors else None,
        "max_abs_flux_difference": max(flux_abs_errors) if flux_abs_errors else None,
        "max_relative_flux_difference": max(flux_errors) if flux_errors else None,
        "median_flux_relative_difference": statistics.median(flux_errors) if flux_errors else None,
        "bbox_area_exact_agreement_fraction": geometry_matches/exact_fields if exact_fields else None,
    }


def _diagnose_stages(img: ImageData, params: DetectionParams,
                     tile_size: int, relative_floor: float) -> list[dict]:
    pixels = np.asarray(img.data, dtype=np.float32)
    valid = background._valid_mask(pixels.shape, img.valid_mask)
    halo = math.ceil(4 * params.kernel_sigma)

    # Lock the CPU stage baseline to the reference implementations for this
    # diagnostic pass, independent of the user's shell backend overrides.
    cpu_env = {
        "ASTRO_BACKGROUND_BACKEND": "numpy",
        "ASTRO_BACKGROUND_STATS_BACKEND": "cpu",
        "ASTRO_DETECTION_BACKEND": "python",
    }
    from unittest.mock import patch
    with patch.dict(os.environ, cpu_env):
        cpu_bg = background.profile_background(
            pixels, valid_mask=img.valid_mask, refine=params.refine_background)
        cpu_level = cpu_bg["background"]
        cpu_noise = float(cpu_bg["noise"])
        cpu_residual = np.where(valid, pixels-cpu_level, 0).astype(np.float32)
        cpu_filtered = filtering.smooth(cpu_residual, params.kernel_sigma, "cpu",
                                        valid_mask=valid if not valid.all() else None)
        cpu_peak = None
        if params.deblend and params.kernel_sigma > 1:
            cpu_peak = filtering.smooth(cpu_residual, 1.0, "cpu",
                                        valid_mask=valid if not valid.all() else None)
        cpu_tile = Tile(img.image_id, 0, 0, cpu_residual,
                        (0, 0, pixels.shape[0], pixels.shape[1]), halo, img.valid_mask)
        cpu_detection = detection._measure_sources(
            cpu_tile, cpu_filtered, cpu_noise, params, profile=True,
            peak_image=cpu_peak)

    # Run the GPU pipeline's corresponding full-image stages on the same input.
    # Its core-filtered result is equivalent to tile+halo filtering; this pass
    # also enables comparison of labels before final catalog merging.
    workspace = Workspace(pixels, valid, profile=False)
    try:
        gpu_level, gpu_noise = workspace.background(refine=params.refine_background)
        gpu_residual = workspace.cp.where(workspace.valid, workspace.pixels-gpu_level,
                                          workspace.cp.float32(0))
        gpu_valid = None if workspace.all_valid else workspace.valid
        gpu_filtered = workspace.smooth(gpu_residual, params.kernel_sigma, gpu_valid)
        gpu_peak = None
        if params.deblend and params.kernel_sigma > 1:
            gpu_peak = workspace.smooth(gpu_residual, 1.0, gpu_valid)
        gpu_rows = workspace.measure(gpu_residual, gpu_filtered, gpu_noise,
                                     params, gpu_peak)
        # Copy diagnostic arrays only after the GPU work has completed.
        gpu_level_np = workspace.cp.asnumpy(gpu_level)
        gpu_residual_np = workspace.cp.asnumpy(gpu_residual)
        gpu_filtered_np = workspace.cp.asnumpy(gpu_filtered)
        gpu_labels_np = workspace.cp.asnumpy(workspace.labels)
        gpu_noise_value = float(gpu_noise.item())
        gpu_source_rows = gpu_rows
    finally:
        del workspace

    cpu_sources = cpu_detection["sources"]
    cpu_labels = cpu_detection["labels"]
    gpu_source_list = gpu_sources(img.image_id, gpu_source_rows)
    # Source IDs are assigned by the same merge implementation used in normal
    # whole-image processing, preserving deterministic IDs for comparisons.
    cpu_sources = tiling.merge(img.image_id, [cpu_sources], [cpu_tile])
    gpu_height, gpu_width = pixels.shape
    gpu_tile = Tile(img.image_id, 0, 0, pixels,
                    (0, 0, gpu_height, gpu_width), halo, img.valid_mask)
    gpu_sources_list = tiling.merge(img.image_id, [gpu_source_list], [gpu_tile])

    output = []
    for name, a, b in (
        ("background", cpu_level, gpu_level_np),
        ("residual", cpu_residual, gpu_residual_np),
        ("filter", cpu_filtered, gpu_filtered_np),
        ("detection_labels_binary", cpu_labels != 0, gpu_labels_np != 0),
    ):
        if name == "detection_labels_binary":
            metrics = _mask_metrics(a, b)
        else:
            metrics = _array_metrics(a, b, relative_floor)
        output.append({"image_id": img.image_id, "stage": name, **metrics})
    output.append({"image_id": img.image_id, "stage": "background_noise",
                   **_array_metrics(np.asarray([cpu_noise]), np.asarray([gpu_noise_value]),
                                    relative_floor)})
    output.append({"image_id": img.image_id, "stage": "detection_catalog",
                   **_catalog_pair(cpu_sources, gpu_sources_list, 3.0)})
    return output


def _run_tier(paths, params: DetectionParams, workers: int, tile_size: int | None,
              mask_zero_border: bool):
    return run_batch(paths, params, workers, tile_size,
                     mask_zero_border=mask_zero_border)


def _truth_for_image(path: str):
    truth_path = _truth_path(path)
    if not truth_path.is_file():
        return None
    with truth_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("truth", payload)


def _score_tier(paths, catalog: dict, match_radius: float) -> dict:
    per_image = {}
    truth_images = 0
    true_total = detected_total = matched_total = false_total = 0
    position_medians, flux_medians = [], []
    for path in paths:
        truth = _truth_for_image(path)
        if truth is None:
            continue
        image_id = Path(path).name
        for suffix in (".gz", ".bz2"):
            if image_id.lower().endswith(suffix):
                image_id = image_id[:-len(suffix)]
                break
        image_id = Path(image_id).stem
        sources = catalog.get(image_id, [])
        score = score_catalog(sources, truth, match_radius=match_radius)
        per_image[image_id] = {"truth_count": len(truth), "detection_count": len(sources), **score}
        truth_images += 1
        true_total += len(truth)
        detected_total += len(sources)
        if score["completeness"] is not None:
            matched_total += round(score["completeness"] * len(truth))
        if score["false_positive_rate"] is not None:
            false_total += round(score["false_positive_rate"] * len(sources))
        if score["median_pos_err_pix"] is not None:
            position_medians.append(score["median_pos_err_pix"])
        if score["median_flux_err_pct"] is not None:
            flux_medians.append(score["median_flux_err_pct"])
    if not truth_images:
        return {"truth_available": False, "per_image": {}}
    return {
        "truth_available": True,
        "images_scored": truth_images,
        "truth_sources": true_total,
        "detections": detected_total,
        "matched_sources": matched_total,
        "completeness": matched_total / true_total if true_total else None,
        "false_positive_rate": false_total / detected_total if detected_total else None,
        "median_of_per_image_position_medians_pix": statistics.median(position_medians) if position_medians else None,
        "median_of_per_image_flux_medians_pct": statistics.median(flux_medians) if flux_medians else None,
        "per_image": per_image,
    }


def _catalog_comparison(reference: dict, compared: dict, radius: float) -> dict:
    by_image = {}
    for image_id in sorted(set(reference) | set(compared)):
        by_image[image_id] = _catalog_pair(reference.get(image_id, []),
                                           compared.get(image_id, []), radius)
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FITS directory, file, or glob")
    parser.add_argument("--cpu-workers", required=True, type=int,
                        help="worker count for multicore CPU and CPU+GPU tiers")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--relative-floor", type=float, default=1e-6,
                        help="denominator floor for per-pixel relative errors")
    parser.add_argument("--kernel-sigma", type=float, default=2.0)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--mask-zero-border", action="store_true")
    parser.add_argument("--no-background-refinement", action="store_true")
    parser.add_argument("--no-deblend", action="store_true")
    args = parser.parse_args()

    if args.cpu_workers < 2:
        parser.error("--cpu-workers must be at least 2 to compare the multicore tier")
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be positive")
    if not math.isfinite(args.match_radius) or args.match_radius <= 0:
        parser.error("--match-radius must be positive and finite")
    if not math.isfinite(args.relative_floor) or args.relative_floor <= 0:
        parser.error("--relative-floor must be positive and finite")
    paths = _collect_paths(args.input)
    if not paths:
        parser.error(f"no FITS files found for {args.input!r}")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        parser.error(f"results directory is not empty: {args.results_dir}; choose a new directory")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir = args.results_dir / "catalogs"
    catalog_dir.mkdir()

    tile_size = args.tile_size
    params_cpu = DetectionParams(
        kernel_sigma=args.kernel_sigma, threshold_sigma=args.threshold_sigma,
        min_area=args.min_area, backend="cpu",
        refine_background=not args.no_background_refinement,
        deblend=not args.no_deblend, pipeline_backend="staged")
    params_gpu = DetectionParams(
        kernel_sigma=args.kernel_sigma, threshold_sigma=args.threshold_sigma,
        min_area=args.min_area, backend="cpu",
        refine_background=not args.no_background_refinement,
        deblend=not args.no_deblend, pipeline_backend="gpu")

    tiers = [
        ("single_core_cpu", params_cpu, 1),
        ("multicore_cpu", params_cpu, args.cpu_workers),
        ("multicore_gpu", params_gpu, args.cpu_workers),
    ]
    catalogs = {}
    tier_summary = {}
    for name, params, workers in tiers:
        result = _run_tier(paths, params, workers, tile_size, args.mask_zero_border)
        catalogs[name] = result
        flat = [source for image_id in sorted(result) for source in result[image_id]]
        write_catalog(flat, str(catalog_dir / f"{name}.csv"))
        tier_summary[name] = {
            "workers_requested": workers,
            "workers_effective_max": min(workers, len(paths)),
            "images": len(result),
            "sources": len(flat),
            "ground_truth_score": _score_tier(paths, result, args.match_radius),
        }

    # Compare pixel-level stage intermediates for the one-worker CPU reference
    # and full GPU path. CPU multicore determinism is checked at catalog level
    # against the one-worker CPU output above.
    stage_rows = []
    for path in paths:
        img = io_fits.read_fits(path, mask_zero_border=args.mask_zero_border)
        stage_rows.extend(_diagnose_stages(
            img, params_cpu, tile_size or max(img.data.shape), args.relative_floor))

    with (args.results_dir / "stage_metrics.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in stage_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stage_rows)

    catalog_comparisons = {
        "single_core_cpu_vs_multicore_cpu": _catalog_comparison(
            catalogs["single_core_cpu"], catalogs["multicore_cpu"], args.match_radius),
        "multicore_cpu_vs_multicore_gpu": _catalog_comparison(
            catalogs["multicore_cpu"], catalogs["multicore_gpu"], args.match_radius),
        "single_core_cpu_vs_multicore_gpu": _catalog_comparison(
            catalogs["single_core_cpu"], catalogs["multicore_gpu"], args.match_radius),
    }
    report = {
        "input": args.input,
        "input_count": len(paths),
        "tile_size": tile_size,
        "match_radius_pix": args.match_radius,
        "relative_error_denominator_floor": args.relative_floor,
        "detection_params": asdict(params_cpu),
        "tiers": tier_summary,
        "catalog_comparisons": catalog_comparisons,
        "stage_comparisons": {
            "compared_tiers": "single_core_cpu vs full-GPU diagnostic pass",
            "details_csv": str(args.results_dir / "stage_metrics.csv"),
            "note": "Multicore CPU is compared to single-core CPU by final catalog; its per-image stage code is identical.",
        },
        "truth_note": (
            "Truth files are read from adjacent <image-stem>.truth.json files. "
            "Aggregate positional/flux error fields are medians of per-image medians."
        ),
    }
    report_path = args.results_dir / "accuracy_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_value) + "\n",
                           encoding="utf-8")
    print(f"Compared {len(paths)} images across three tiers.")
    for name, result in tier_summary.items():
        score = result["ground_truth_score"]
        label = (f"completeness={score['completeness']:.4f}, "
                 f"false_positive_rate={score['false_positive_rate']:.4f}"
                 if score.get("truth_available") and score.get("completeness") is not None
                 else "ground truth unavailable")
        print(f"{name:20s} {result['sources']:6d} sources; {label}")
    print(f"Wrote {report_path}")
    print(f"Wrote {args.results_dir / 'stage_metrics.csv'}")


if __name__ == "__main__":
    main()
