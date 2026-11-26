"""Focused checks for bounded Nsight Compute collection and honest coverage."""
import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from final_scripts import compare_performance_nsys as profiling


@pytest.mark.parametrize("selected_tier", [None, "multicore_gpu"])
@pytest.mark.parametrize("cpu_counters", [False, True])
def test_wall_clock_mode_needs_no_profilers_and_preserves_pipeline_settings(
        monkeypatch, tmp_path, selected_tier, cpu_counters):
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "compare_performance_nsys.py", "--wall-clock-only", "--input", "real-images",
        "--results-dir", str(output), "--repeats", "3", "--tile-size", "1024",
    ])
    if cpu_counters:
        sys.argv.append("--cpu-counters")
    monkeypatch.setattr(profiling, "_input_paths", lambda _: [f"frame-{i}.fits" for i in range(100)])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "72")

    def forbidden(*args, **kwargs):
        pytest.fail("Wall-clock mode attempted to discover or run diagnostic tools")

    monkeypatch.setattr(profiling.shutil, "which",
                        lambda name: "perf" if cpu_counters and name == "perf" else forbidden())
    monkeypatch.setattr(profiling.subprocess, "run", forbidden)
    for name in ("_run_perf", "_nsys_metrics", "_run_stage_timing", "_run_speed_of_light"):
        monkeypatch.setattr(profiling, name, forbidden)
    samples = {"single_core_cpu": [8., 4., 6.], "multicore_cpu": [2., 1., 3.],
               "multicore_gpu": [1., .5, .75]}
    calls = []
    counter_calls = []

    def timed(command, log_path, *, timed):
        tier = log_path.parent.name if selected_tier is None else selected_tier
        calls.append((tier, command))
        assert timed
        assert command[:3] == [sys.executable, "-m", "astro_pipeline.run_pipeline"]
        assert command[command.index("--workers")+1] == ("1" if tier == "single_core_cpu" else "72")
        assert command[command.index("--pipeline-backend")+1] == ("gpu" if tier == "multicore_gpu" else "staged")
        assert command[command.index("--tile-size")+1] == "1024"
        assert command[command.index("--backend")+1] == "cpu"
        assert command[command.index("--kernel-sigma")+1] == "2.0"
        assert command[command.index("--threshold-sigma")+1] == "3.0"
        return samples[tier].pop(0)

    monkeypatch.setattr(profiling, "_run_logged", timed)
    if cpu_counters:
        def counters(perf, command, tier_dir, workers):
            tier = tier_dir.name if selected_tier is None else selected_tier
            assert sum(t == tier for t, _ in calls) == 3
            assert Path(command[command.index("--output")+1]).name == "catalog_perf.csv"
            counter_calls.append(tier)
            return {"perf_available": True, "perf_wall_s": 99., "cpu_cycles": 1000.}
        monkeypatch.setattr(profiling, "_run_perf", counters)
    profiling.main(selected_tier)
    with (output / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["measurement"] == "wall_clock_only"
    assert comparison["input_count"] == 100
    assert comparison["tile_size"] == 1024
    assert comparison["repeats"] == 3
    assert comparison["cpu_counters"] == cpu_counters
    assert len(calls) == (9 if selected_tier is None else 3)
    assert float(rows[-1]["median_wall_s"]) == .75
    assert float(rows[-1]["throughput_images_s"]) == pytest.approx(100/.75)
    assert "profiled_wall_s_diagnostic_only" not in rows[-1]
    assert len(counter_calls) == ((3 if selected_tier is None else 1) if cpu_counters else 0)
    if cpu_counters:
        assert float(rows[-1]["perf_wall_s"]) == 99.
    if selected_tier is None:
        assert comparison["speedups_from_unprofiled_median_wall_time"] == {
            "multicore_cpu_vs_single_core_cpu": 3.,
            "multicore_gpu_vs_multicore_cpu": pytest.approx(2/.75),
            "multicore_gpu_vs_single_core_cpu": 8.,
        }


def test_perf_uses_its_own_elapsed_time_and_preserves_unsupported_counters(monkeypatch, tmp_path):
    def run(command, **kwargs):
        assert "context-switches,cpu-migrations,page-faults" in command[command.index("-e")+1]
        Path(command[command.index("-o")+1]).write_text(
            "400000,msec,task-clock,400000000000,100.00,10.000,CPUs utilized\n"
            "1000000,,cycles,400000000000,100.00,,\n"
            "1250000,,instructions,400000000000,100.00,,\n"
            "<not supported>,,cache-misses,0,100.00,,\n"
            "1234,,context-switches,400000000000,100.00,,\n"
            "12,,cpu-migrations,400000000000,100.00,,\n"
            "5678,,page-faults,400000000000,100.00,,\n")
        return subprocess.CompletedProcess(command, 0, stderr="")
    ticks = iter([1000., 1040.])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(profiling.subprocess, "run", run)
    result = profiling._run_perf("perf", ["python", "pipeline"], tmp_path, 16)
    assert result["perf_wall_s"] == 40.
    assert result["cpu_cores_busy_avg"] == 10.
    assert result["cpu_utilization_pct_of_workers"] == 62.5
    assert result["cpu_ipc"] == 1.25
    assert result["cpu_cache_misses"] is None
    assert result["cpu_context_switches"] == 1234
    assert result["cpu_migrations"] == 12
    assert result["cpu_page_faults"] == 5678


def test_systems_normalized_sm_metrics_use_metric_names_and_source_ids():
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
            CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS (typeId INTEGER, metricId INTEGER, value REAL);
            INSERT INTO TARGET_INFO_GPU_METRICS VALUES
                (1, 7, 'SMs Active [Throughput %]'), (1, 8, 'SM Issue [Throughput %]'),
                (2, 7, 'DRAM Read Bandwidth [Throughput %]');
            INSERT INTO GPU_METRICS VALUES (1, 7, 20), (1, 7, 80), (1, 8, 1), (2, 7, 99);
        """)
        assert profiling._gpu_sample_metrics(db) == {
            "gpu_sm_active_avg_pct": 50.0, "gpu_sm_active_max_pct": 80.0,
            "gpu_metrics_samples": 2}


@pytest.fixture
def trace(tmp_path):
    with sqlite3.connect(tmp_path / "profile.sqlite") as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, '_Z6kernelv'), (2, 'kernel()'),
                                         (3, '_Z3fftv'), (4, 'fft()');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                mangledName INTEGER, demangledName INTEGER,
                gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                staticSharedMemory INTEGER, dynamicSharedMemory INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (1, 2, 4, 1, 1, 256, 1, 1, 16400, 0),
                (1, 2, 8, 1, 1, 256, 1, 1, 16400, 0),
                (3, 4, 4, 1, 1, 256, 1, 1, 16400, 0);
        """)
    return tmp_path


def write_metrics(stream, symbols):
    metrics = ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
               "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
               "sm__warps_active.avg.pct_of_peak_sustained_active",
               "launch__registers_per_thread"]
    writer = csv.writer(stream)
    writer.writerow(["ID", "Kernel Name", "Grid Size", "Block Size",
                     "launch__shared_mem_per_block_static",
                     "launch__shared_mem_per_block_dynamic", *metrics])
    writer.writerow(["", "", "", "", "Kbyte/block", "byte/block", "%", "%", "%", "register/thread"])
    for index, symbol in enumerate(symbols):
        writer.writerow([index, symbol, "(4, 1, 1)", "(256, 1, 1)",
                         "16.400000", "0", "80.1", "11.0", "82.0", "48"])


@pytest.fixture
def args():
    return argparse.Namespace(input="real-images", tile_size=1024, kernel_sigma=2.0,
                              threshold_sigma=3.0, min_area=5, mask_zero_border=True,
                              no_background_refinement=False, no_deblend=False,
                              skip_speed_of_light=False)


def mock_profiler(monkeypatch, symbols, fail_page=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        page = command[command.index("--page") + 1] if "--page" in command else "collect"
        if page == fail_page:
            raise subprocess.CalledProcessError(7, command)
        if page == "collect":
            Path(command[command.index("--export") + 1]).write_bytes(b"report")
        elif page == "raw":
            write_metrics(kwargs["stdout"], symbols)
        else:
            kwargs["stdout"].write("Speed Of Light Throughput\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(profiling.subprocess, "run", run)
    return calls


def test_inventory_keeps_symbol_and_configuration_variants(trace):
    inventory = profiling._sol_inventory(trace / "profile.sqlite")
    assert set(inventory) == {"_Z6kernelv", "_Z3fftv"}
    assert inventory["_Z6kernelv"]["name"] == "kernel()"
    assert len(inventory["_Z6kernelv"]["configurations"]) == 2


def test_samples_require_metrics_and_convert_shared_memory_units(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as stream:
        write_metrics(stream, ["_Z6kernelv"])
    assert profiling._sol_samples(path)["_Z6kernelv"]["configurations"] == {
        (4, 1, 1, 256, 1, 1, 16400, 0)}
    path.write_text(path.read_text().replace("80.1", "n/a"))
    with pytest.raises(ValueError, match="Missing Nsight Compute metric"):
        profiling._sol_samples(path)


@pytest.mark.parametrize("symbols, expected", [
    (["_Z6kernelv", "_Z3fftv"], "complete"),
    (["_Z6kernelv"], "incomplete"),
])
def test_bounded_collection_and_coverage(monkeypatch, trace, args, symbols, expected):
    calls = mock_profiler(monkeypatch, symbols)
    result = profiling._run_speed_of_light("ncu", args, trace, 16)
    assert result["status"] == expected
    assert result["benchmark_workers"] == 16
    assert result["profiling_workers"] == 1
    command = calls[0][0]
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--kernel-id") + 1] == ":::1"
    assert command[command.index("--kernel-name-base") + 1] == "mangled"
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--section"] == [
        "SpeedOfLight", "Occupancy", "LaunchStats"]
    assert command[command.index("--tile-size") + 1] == "1024"
    assert "--mask-zero-border" in command
    assert len(calls) == 3
    assert all("timeout" not in kwargs for _, kwargs in calls)
    kernel = next(item for item in result["kernels"] if item["symbol"] == "_Z6kernelv")
    assert len(kernel["sampled_configurations"]) == 1
    assert len(kernel["unmeasured_configurations"]) == 1
    saved = json.loads((trace / "speed-of-light/coverage.json").read_text())
    assert saved["status"] == expected
    assert (trace / "speed-of-light/speed_of_light.txt").is_file()


@pytest.mark.parametrize("page, count", [("collect", 1), ("raw", 2), ("details", 3)])
def test_failure_is_recorded_without_retry(monkeypatch, trace, args, page, count):
    calls = mock_profiler(monkeypatch, ["_Z6kernelv", "_Z3fftv"], fail_page=page)
    result = profiling._run_speed_of_light("ncu", args, trace, 16)
    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    assert len(calls) == count
    if page != "collect":
        assert (trace / "speed-of-light/profile.ncu-rep").read_bytes() == b"report"


@pytest.mark.parametrize("skip, expected", [(True, "skipped"), (False, "failed")])
def test_skip_and_missing_tool_are_explicit(monkeypatch, trace, args, skip, expected):
    calls = mock_profiler(monkeypatch, [])
    args.skip_speed_of_light = skip
    assert profiling._run_speed_of_light(None, args, trace, 16)["status"] == expected
    assert calls == []


def test_existing_reports_are_preserved(monkeypatch, trace, args):
    output = trace / "speed-of-light"
    output.mkdir()
    original = output / "profile.ncu-rep"
    original.write_bytes(b"original")
    mock_profiler(monkeypatch, ["_Z6kernelv", "_Z3fftv"])
    result = profiling._run_speed_of_light("ncu", args, trace, 16)
    assert result["status"] == "complete"
    assert Path(result["output_dir"]) != output
    assert original.read_bytes() == b"original"
