"""Focused checks for bounded Nsight Compute collection and honest coverage."""
import argparse
import csv
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from final_scripts import compare_performance_nsys as profiling


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
