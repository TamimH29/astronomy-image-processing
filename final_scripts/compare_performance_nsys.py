"""Compare end-to-end pipeline performance across the three project tiers.

The script runs repeated unprofiled executions for comparable wall times, then
captures one Nsight Systems trace per tier for CUDA kernel, transfer, API, and
sampled GPU metrics. If Linux perf is available, it also records CPU counters.
Per-stage timings are collected separately by stage_timing.py and are explicitly
single-process diagnostics, not timings for the multicore worker pool.

Example on a Vista GH node (from the repository root):
  python final_scripts/compare_performance_nsys.py \
      --input 'gen_fits/large/*.fits' --cpu-workers 20 --tile-size 1024 \
      --repeats 5 --results-dir "$WORK/astro-test/nsys-large"

run_batch limits active workers to the number of input files. For a true
72-worker Vista comparison, provide at least 72 input images and request
--cpu-workers 72. Nsight collection is diagnostic; use median unprofiled wall
times as the headline end-to-end numbers.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _input_paths(spec: str) -> list[str]:
    if os.path.isdir(spec):
        paths = []
        for pattern in ("*.fits", "*.fits.gz", "*.fits.bz2", "*.fit"):
            paths.extend(glob.glob(os.path.join(spec, pattern)))
        return sorted(set(paths))
    return sorted(glob.glob(spec))


def _available_cpus() -> int:
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


def _pipeline_command(args: argparse.Namespace, workers: int, backend: str,
                      output: Path) -> list[str]:
    command = [
        sys.executable, "-m", "astro_pipeline.run_pipeline",
        "--input", args.input,
        "--output", str(output),
        "--workers", str(workers),
        "--backend", "cpu",
        "--pipeline-backend", backend,
    ]
    if args.tile_size is not None:
        command.extend(("--tile-size", str(args.tile_size)))
    command.extend(("--kernel-sigma", str(args.kernel_sigma),
                    "--threshold-sigma", str(args.threshold_sigma),
                    "--min-area", str(args.min_area)))
    if args.mask_zero_border:
        command.append("--mask-zero-border")
    if args.no_background_refinement:
        command.append("--no-background-refinement")
    if args.no_deblend:
        command.append("--no-deblend")
    return command


def _run_logged(command: list[str], log_path: Path, *, timed: bool) -> float | None:
    started = time.perf_counter() if timed else None
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started if started is not None else None


def _parse_perf_stat(path: Path, elapsed_s: float) -> dict:
    """Read perf's comma-separated output; absent/unsupported counters stay null."""
    values: dict[str, float | None] = {
        "cpu_task_clock_ms": None,
        "cpu_cores_busy_avg": None,
        "cpu_cycles": None,
        "cpu_instructions": None,
        "cpu_cache_misses": None,
    }
    names = {
        "task-clock": "cpu_task_clock_ms",
        "cycles": "cpu_cycles",
        "instructions": "cpu_instructions",
        "cache-misses": "cpu_cache_misses",
    }
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or fields[2] not in names:
            continue
        raw = fields[0].replace(" ", "")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
            continue
        value = float(raw)
        key = names[fields[2]]
        if key == "cpu_task_clock_ms":
            unit = fields[1].lower()
            value = value * 1000 if unit in {"sec", "seconds", "s"} else value
        values[key] = value
    if values["cpu_task_clock_ms"] is not None and elapsed_s > 0:
        values["cpu_cores_busy_avg"] = values["cpu_task_clock_ms"] / (elapsed_s * 1000)
    return values


def _run_perf(perf: str | None, command: list[str], tier_dir: Path,
              elapsed_s: float, workers: int) -> dict:
    metrics = {
        "perf_available": bool(perf),
        "cpu_task_clock_ms": None,
        "cpu_cores_busy_avg": None,
        "cpu_utilization_pct_of_workers": None,
        "cpu_cycles": None,
        "cpu_instructions": None,
        "cpu_cache_misses": None,
    }
    if not perf:
        return metrics
    output = tier_dir / "perf_stat.csv"
    proc = subprocess.run(
        [perf, "stat", "-x,", "--no-big-num", "-e",
         "task-clock,cycles,instructions,cache-misses", "-o", str(output), "--", *command],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode:
        (tier_dir / "perf_stat_error.log").write_text(proc.stderr, encoding="utf-8")
        metrics["perf_available"] = False
        return metrics
    metrics.update(_parse_perf_stat(output, elapsed_s))
    if metrics["cpu_cores_busy_avg"] is not None and workers > 0:
        metrics["cpu_utilization_pct_of_workers"] = min(
            100.0, metrics["cpu_cores_busy_avg"] / workers * 100.0)
    return metrics


def _gpu_sample_metrics(db: sqlite3.Connection) -> dict:
    """Summarize sampled SM activity from Nsight's generic GPU metric rows."""
    result = {"gpu_sm_active_avg_pct": None, "gpu_sm_active_max_pct": None,
              "gpu_metrics_samples": 0}
    tables = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    # Nsight versions have exported metrics either as GENERIC_EVENTS JSON or
    # normalized GPU_METRICS rows. Handle both schemas when present.
    if "GENERIC_EVENTS" in tables:
        columns = _table_columns(db, "GENERIC_EVENTS")
        if "data" in columns:
            values = []
            for (payload,) in db.execute('SELECT "data" FROM GENERIC_EVENTS'):
                try:
                    parsed = json.loads(payload)
                except (TypeError, json.JSONDecodeError):
                    continue
                for name, value in parsed.items():
                    if name.lower() in {"sm active", "sms active"}:
                        try:
                            values.append(float(value))
                        except (TypeError, ValueError):
                            pass
            if values:
                result.update(gpu_sm_active_avg_pct=statistics.mean(values),
                              gpu_sm_active_max_pct=max(values),
                              gpu_metrics_samples=len(values))
                return result
    if "GPU_METRICS" in tables:
        columns = _table_columns(db, "GPU_METRICS")
        metric_col = next((c for c in columns if c.lower() in {"sm active", "sms active"}), None)
        if metric_col:
            values = [float(row[0]) for row in db.execute(
                f'SELECT "{metric_col}" FROM GPU_METRICS WHERE "{metric_col}" IS NOT NULL')]
            if values:
                result.update(gpu_sm_active_avg_pct=statistics.mean(values),
                              gpu_sm_active_max_pct=max(values),
                              gpu_metrics_samples=len(values))
    return result


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def _sum_event_table(db: sqlite3.Connection, table: str) -> tuple[int, float, int | None, float | None]:
    """Return count, summed duration ms, optional byte count, union duration ms."""
    columns = _table_columns(db, table)
    if not {"start", "end"}.issubset(columns):
        return 0, 0.0, None, 0.0
    byte_column = "bytes" if "bytes" in columns else None
    if byte_column:
        count, duration_ns, byte_sum = db.execute(
            f'SELECT COUNT(*), COALESCE(SUM("end"-start),0), '
            f'COALESCE(SUM("{byte_column}"),0) FROM "{table}" '
            'WHERE "end" >= start'
        ).fetchone()
    else:
        count, duration_ns = db.execute(
            f'SELECT COUNT(*), COALESCE(SUM("end"-start),0) '
            f'FROM "{table}" WHERE "end" >= start'
        ).fetchone()
        byte_sum = None
    intervals = db.execute(
        f'SELECT start, "end" FROM "{table}" WHERE "end" > start ORDER BY start'
    )
    union_ns = 0
    start0 = end0 = None
    for start, end in intervals:
        if start0 is None:
            start0, end0 = start, end
        elif start <= end0:
            end0 = max(end0, end)
        else:
            union_ns += end0 - start0
            start0, end0 = start, end
    if start0 is not None:
        union_ns += end0 - start0
    return int(count), duration_ns / 1e6, byte_sum, union_ns / 1e6


def _nsys_metrics(nsys: str, report: Path) -> dict:
    export = subprocess.run(
        [nsys, "export", "--type=sqlite", str(report)],
        text=True, capture_output=True,
    )
    if export.returncode:
        raise RuntimeError(f"nsys export failed for {report}: {export.stderr.strip()}")
    db_path = report.with_suffix(".sqlite")
    if not db_path.exists():
        raise RuntimeError(f"Nsight export did not create expected database {db_path}")

    metrics = {
        "gpu_kernel_count": None,
        "gpu_kernel_sum_ms": None,
        "gpu_kernel_union_ms": None,
        "gpu_memcpy_count": None,
        "gpu_memcpy_sum_ms": None,
        "gpu_memcpy_bytes": None,
        "cuda_api_count": None,
        "cuda_api_sum_ms": None,
        "gpu_sm_active_avg_pct": None,
        "gpu_sm_active_max_pct": None,
        "gpu_metrics_samples": 0,
    }
    with sqlite3.connect(db_path) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        # Different Nsight versions may export either kernel table name.
        kernel_table = next((name for name in (
            "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
        ) if name in tables), None)
        if kernel_table:
            count, total, _, union = _sum_event_table(db, kernel_table)
            metrics.update(gpu_kernel_count=count, gpu_kernel_sum_ms=total,
                           gpu_kernel_union_ms=union)
        memcpy_table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        if memcpy_table in tables:
            count, total, byte_sum, _ = _sum_event_table(db, memcpy_table)
            metrics.update(gpu_memcpy_count=count, gpu_memcpy_sum_ms=total,
                           gpu_memcpy_bytes=byte_sum)
        runtime_table = "CUPTI_ACTIVITY_KIND_RUNTIME"
        if runtime_table in tables:
            count, total, _, _ = _sum_event_table(db, runtime_table)
            metrics.update(cuda_api_count=count, cuda_api_sum_ms=total)
        metrics.update(_gpu_sample_metrics(db))
    return metrics


def _run_stage_timing(args: argparse.Namespace, tier_dir: Path, pipeline_backend: str,
                      input_count: int) -> dict:
    """Collect one-process stage breakdown using the project's existing profiler."""
    timing_dir = tier_dir / "stage_timing_single_process"
    command = [
        sys.executable, "scripts/stage_timing.py", "--input", args.input,
        "--limit", str(input_count), "--repeats", str(args.repeats),
        "--tile-sizes", "none" if args.tile_size is None else str(args.tile_size),
        "--backend", "cpu", "--pipeline-backend", pipeline_backend,
        "--workers", "1", "--results-dir", str(timing_dir),
        "--kernel-sigma", str(args.kernel_sigma),
    ]
    if args.mask_zero_border:
        command.append("--mask-zero-border")
    if args.no_background_refinement:
        command.append("--no-background-refinement")
    if args.no_deblend:
        command.append("--no-deblend")
    with (tier_dir / "stage_timing.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    saved = json.loads((timing_dir / "timings.json").read_text(encoding="utf-8"))
    configurations = saved.get("configurations", [])
    return configurations[0].get("median_stages_s", {}) if configurations else {}


def _ratio(base: float | None, compared: float | None) -> float | None:
    if base is None or compared is None or compared <= 0:
        return None
    return base / compared


def main(selected_tier: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FITS directory, file, or glob")
    parser.add_argument("--cpu-workers", type=int, default=None,
                        help="worker count; standalone scripts default to the allocated CPU count")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5,
                        help="unprofiled repetitions per tier; report median wall time")
    parser.add_argument("--kernel-sigma", type=float, default=2.0)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--mask-zero-border", action="store_true")
    parser.add_argument("--no-background-refinement", action="store_true")
    parser.add_argument("--no-deblend", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.cpu_workers is not None and args.cpu_workers < 1:
        parser.error("--cpu-workers must be positive")
    if selected_tier is None and args.cpu_workers is None:
        parser.error("--cpu-workers is required when comparing all tiers")
    if selected_tier == "single_core_cpu" and args.cpu_workers not in (None, 1):
        parser.error("the single-core script always uses exactly one worker")
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be positive")
    paths = _input_paths(args.input)
    if not paths:
        parser.error(f"no FITS files found for {args.input!r}")
    nsys = shutil.which("nsys")
    if not nsys:
        parser.error("nsys not found; load/install Nsight Systems on the Vista compute node")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        parser.error(f"results directory is not empty: {args.results_dir}; choose a new directory")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cpu_workers = args.cpu_workers or _available_cpus()
    max_cpus = _available_cpus()
    # run_batch warns when workers exceed the CPU affinity, but its actual
    # process count is capped by the number of input files.
    effective_workers = min(cpu_workers, len(paths))
    if selected_tier is not None:
        active_workers = 1 if selected_tier == "single_core_cpu" else cpu_workers
        if active_workers > max_cpus:
            print(f"warning: requested {active_workers} workers but only {max_cpus} CPUs are available")
        if active_workers > len(paths):
            print(f"note: {len(paths)} files limit run_batch to at most {len(paths)} workers")
        print(f"inputs: {len(paths)}; tier: {selected_tier}; requested workers: {active_workers}; "
              f"expected effective workers: {min(active_workers, len(paths))}")
    else:
        if cpu_workers > max_cpus:
            print(f"warning: requested {cpu_workers} workers but only {max_cpus} CPUs are available")
        if cpu_workers > len(paths):
            print(f"note: {len(paths)} files limit run_batch to at most {len(paths)} workers")
        print(f"inputs: {len(paths)}; requested multicore workers: {cpu_workers}; "
              f"expected effective workers: {effective_workers}")
    nsys_version = subprocess.run(
        [nsys, "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    print(f"nsys: {nsys_version}")
    perf = shutil.which("perf")
    print(f"perf: {'available' if perf else 'not found (CPU counters will be empty)'}")

    if selected_tier is None:
        configurations = [
            ("single_core_cpu", 1, "staged"),
            ("multicore_cpu", cpu_workers, "staged"),
            ("multicore_gpu", cpu_workers, "gpu"),
        ]
    else:
        workers = 1 if selected_tier == "single_core_cpu" else cpu_workers
        backend = "gpu" if selected_tier == "multicore_gpu" else "staged"
        configurations = [(selected_tier, workers, backend)]
    summaries = []
    for tier, workers, pipeline_backend in configurations:
        tier_dir = args.results_dir if selected_tier else args.results_dir / tier
        tier_dir.mkdir(parents=True)
        samples = []
        perf_metrics = None
        for repeat in range(args.repeats):
            output = tier_dir / f"catalog_repeat_{repeat + 1}.csv"
            command = _pipeline_command(args, workers, pipeline_backend, output)
            elapsed = _run_logged(command, tier_dir / f"run_repeat_{repeat + 1}.log", timed=True)
            samples.append(elapsed)
        # One independent hardware-counter run; it is not used for headline timing.
        perf_metrics = _run_perf(perf, _pipeline_command(
            args, workers, pipeline_backend, tier_dir / "catalog_perf.csv"), tier_dir,
            statistics.median(samples), min(workers, len(paths)))

        profile_output = tier_dir / "catalog_profiled.csv"
        pipeline_command = _pipeline_command(args, workers, pipeline_backend, profile_output)
        report_base = tier_dir / "profile"
        collect_gpu_metrics = pipeline_backend == "gpu"
        gpu_metric_option = ("--gpu-metrics-devices=cuda-visible" if collect_gpu_metrics
                             else "--gpu-metrics-devices=none")
        profile_command = [
            nsys, "profile", "--trace=cuda,osrt", "--sample=none",
            "--cpuctxsw=process-tree", gpu_metric_option,
            "--force-overwrite=true", "-o", str(report_base),
            *pipeline_command,
        ]
        try:
            profile_wall = _run_logged(profile_command, tier_dir / "nsys_profile.log", timed=True)
            gpu_metrics_status = "collected" if collect_gpu_metrics else "not applicable to CPU tier"
        except subprocess.CalledProcessError:
            # GPU counter collection can be denied by node-level permissions or
            # conflicts with another counter client. Retain the CUDA trace.
            if not collect_gpu_metrics:
                raise
            profile_command[profile_command.index(gpu_metric_option)] = "--gpu-metrics-devices=none"
            profile_wall = _run_logged(profile_command, tier_dir / "nsys_profile.log", timed=True)
            gpu_metrics_status = "unavailable; retried without GPU metric sampling"
        report = report_base.with_suffix(".nsys-rep")
        if not report.exists():
            raise RuntimeError(f"Nsight Systems did not create expected report: {report}")
        metrics = _nsys_metrics(nsys, report)
        stage_timings = _run_stage_timing(args, tier_dir, pipeline_backend, len(paths))
        summaries.append({
            "tier": tier,
            "workers_requested": workers,
            "workers_effective": min(workers, len(paths)),
            "pipeline_backend": pipeline_backend,
            "repeats": args.repeats,
            "wall_samples_s": json.dumps(samples),
            "median_wall_s": statistics.median(samples),
            "profiled_wall_s_diagnostic_only": profile_wall,
            "nsys_report": str(report),
            "gpu_metrics_status": gpu_metrics_status,
            **perf_metrics,
            "stage_timings_single_process_s": json.dumps(stage_timings, sort_keys=True),
            **metrics,
        })

    fields = list(summaries[0])
    with (args.results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    comparison = {
        "tier_results": {row["tier"]: {
            "median_wall_s": row["median_wall_s"],
            "workers_requested": row["workers_requested"],
            "workers_effective": row["workers_effective"],
        } for row in summaries},
        "nsys_metrics_note": (
            "GPU kernel and memcpy sums aggregate activity durations across workers; "
            "they may exceed elapsed wall time when activities overlap. "
            "Profiled wall times include profiler overhead and are diagnostic only."
        ),
        "additional_metrics_note": (
            "CPU counters and task-clock come from a separate perf stat run when permitted; "
            "average busy cores = task-clock / elapsed wall time. GPU SM Active is sampled "
            "device-level utilization and may include other processes on a shared GPU. "
            "Per-stage timings are single-process diagnostics from stage_timing.py, not "
            "multicore worker-stage aggregates. Per-kernel occupancy requires Nsight Compute."
        ),
    }
    if selected_tier is None:
        by_tier = {row["tier"]: row for row in summaries}
        cpu1 = by_tier["single_core_cpu"]["median_wall_s"]
        cpum = by_tier["multicore_cpu"]["median_wall_s"]
        gpum = by_tier["multicore_gpu"]["median_wall_s"]
        comparison["speedups_from_unprofiled_median_wall_time"] = {
            "multicore_cpu_vs_single_core_cpu": _ratio(cpu1, cpum),
            "multicore_gpu_vs_multicore_cpu": _ratio(cpum, gpum),
            "multicore_gpu_vs_single_core_cpu": _ratio(cpu1, gpum),
        }
    else:
        comparison["note"] = (
            "This report contains one tier. Run the other standalone tier scripts "
            "with the same input and settings to compare their median wall times."
        )
    (args.results_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print("\nMedian unprofiled wall times and collected diagnostics:")
    for row in summaries:
        print(f"{row['tier']:20s} {row['median_wall_s']:9.3f} s  "
              f"kernels={row['gpu_kernel_count']} "
              f"kernel_sum={row['gpu_kernel_sum_ms']} ms "
              f"memcpy_sum={row['gpu_memcpy_sum_ms']} ms "
              f"CPU_util={row['cpu_utilization_pct_of_workers']}% "
              f"SM_active_avg={row['gpu_sm_active_avg_pct']}%")
    print(f"\nWrote {args.results_dir / 'summary.csv'} and "
          f"{args.results_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
