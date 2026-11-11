"""Global-statistics handoff tests and real-CUDA scientific regressions."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from astro_pipeline import background, detection, _background_gpu
from astro_pipeline.interfaces import DetectionParams, ImageData


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("ASTRO_BACKGROUND_BACKEND", "ASTRO_BACKGROUND_STATS_BACKEND",
                 "ASTRO_BACKGROUND_THREADS", "ASTRO_DETECTION_BACKEND",
                 "ASTRO_DETECTION_THREADS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def native():
    module = pytest.importorskip("astro_pipeline._background_native")
    assert module.GLOBAL_STATISTICS_API_VERSION == 1, "rebuild the native extension"
    return module


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable in this session: {error}")
    if not count:
        pytest.skip("no CUDA device in this session")
    return cp


def scene(seed=13):
    yy, xx = np.mgrid[:129, :193]
    pixels = (100 + .04 * xx + .03 * yy
              + np.random.default_rng(seed).normal(0, 2, xx.shape)
              + 80 * np.exp(-((xx-88)**2 + (yy-64)**2) / 4.5)).astype(np.float32)
    valid = np.ones(pixels.shape, bool)
    valid[:, :32] = False
    valid[80:83, 120:124] = False
    pixels[~valid] = 0
    return pixels, valid


class StatisticsOracle:
    """Independent CPU provider tests the native handoff without simulating CUDA."""
    def __init__(self, pixels, valid, *, profile=False):
        self.pixels, self.valid = pixels, valid
        self.calls, self.retained_views = [], []
        self.closed = False

    def __call__(self, level, source):
        self.calls.append(level is None)
        if level is None:
            return background._clipped_stats(self.pixels[self.valid])
        self.retained_views.append((level, source))
        sky = self.valid & (source == 0)
        values = self.pixels[sky].astype(np.float64) - level[sky]
        return background._clipped_stats(values)

    def close(self):
        self.closed = True

    def timings(self):
        return dict(statistics_calls=len(self.calls))


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("threads", [1, 4, 32])
def test_native_provider_matches_reference_and_owns_callback_views(native, refine, threads):
    pixels, valid = scene()
    provider = StatisticsOracle(pixels, valid)
    expected = native.profile_background(pixels, 32, threads, valid_mask=valid, refine=refine)
    actual = native.profile_background(pixels, 32, threads, valid_mask=valid, refine=refine,
                                       global_statistics=provider)
    for key in ("background", "source_mask", "mesh", "measured_mesh"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["noise"] == expected["noise"]
    assert provider.calls == ([True] if refine else []) + [False] * actual["timings"]["refinement_passes"]
    assert actual["timings"]["global_median_ms"] == 0
    # Retained views remain valid after native source-mask buffers are destroyed.
    for level, source in provider.retained_views:
        assert not level.flags.writeable and not source.flags.writeable
        assert np.isfinite(level).all() and np.isin(source, (0, 1)).all()


def test_native_provider_concurrent_calls_are_isolated(native):
    def process(seed):
        pixels, valid = scene(seed)
        provider = StatisticsOracle(pixels, valid)
        expected = native.estimate_background(pixels, 32, 4, valid_mask=valid)
        actual = native.estimate_background(pixels, 32, 4, valid_mask=valid,
                                            global_statistics=provider)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(process, range(4)))


@pytest.mark.parametrize("result", [(np.nan, 1), (0, np.inf), (0, -1)])
def test_invalid_provider_statistics_fail_and_workers_recover(native, result):
    pixels, _ = scene()
    with pytest.raises(ValueError, match="finite center.*noise"):
        native.estimate_background(pixels, 32, 4, global_statistics=lambda *args: result)
    level, noise = native.estimate_background(np.ones((17, 19), np.float32), 8, 4)
    assert noise == 0 and np.all(level == 1)


def test_provider_exception_crosses_released_gil_and_joins_workers(native):
    def fail(*args):
        raise RuntimeError("statistics provider failed")
    with pytest.raises(RuntimeError, match="statistics provider failed"):
        native.estimate_background(np.ones((33, 35), np.float32), 16, 4,
                                     global_statistics=fail)
    with pytest.raises(TypeError, match="callable"):
        native.estimate_background(np.ones((3, 4), np.float32), global_statistics=123)


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
def test_opt_in_dispatch_creates_one_provider_and_closes_it(monkeypatch, request, backend):
    if backend == "cpp":
        request.getfixturevalue("native")
    pixels, valid = scene()
    expected = background.profile_background(pixels, 32, valid_mask=valid)
    created = []
    def factory(*args, **kwargs):
        context = StatisticsOracle(*args, **kwargs)
        created.append(context)
        return context
    monkeypatch.setattr(_background_gpu, "GpuBackgroundStatistics", factory)
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", backend)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    actual = background.profile_background(pixels, 32, valid_mask=valid)
    np.testing.assert_array_equal(actual["background"], expected["background"])
    assert actual["noise"] == expected["noise"]
    assert len(created) == 1 and created[0].closed
    assert actual["timings"]["statistics_calls"] == len(created[0].calls)


def test_dispatch_closes_provider_after_background_failure(monkeypatch):
    class Failing(StatisticsOracle):
        def __call__(self, *args):
            raise RuntimeError("provider failure")
    created = []
    def factory(*args, **kwargs):
        context = Failing(*args, **kwargs)
        created.append(context)
        return context
    monkeypatch.setattr(_background_gpu, "GpuBackgroundStatistics", factory)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    with pytest.raises(RuntimeError, match="provider failure"):
        background.estimate_background(np.ones((17, 19), np.float32))
    assert created[0].closed


def test_default_cpu_statistics_never_construct_gpu_context(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CPU statistics must stay independent of CUDA")
    monkeypatch.setattr(_background_gpu, "GpuBackgroundStatistics", fail)
    background.estimate_background(np.ones((17, 19), np.float32))


def test_invalid_statistics_selection_and_stale_native_extension(monkeypatch):
    pixels = np.ones((17, 19), np.float32)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "unknown")
    with pytest.raises(ValueError, match="STATS_BACKEND"):
        background.estimate_background(pixels)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setattr(background, "import_module", lambda _: SimpleNamespace(API_VERSION=2))
    with pytest.raises(ImportError, match="lacks GPU statistics support.*rebuild"):
        background.estimate_background(pixels)


def test_explicit_gpu_request_reports_missing_cupy_without_fallback(monkeypatch):
    def missing(_):
        raise ModuleNotFoundError("missing CuPy")
    monkeypatch.setattr(_background_gpu, "import_module", missing)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    with pytest.raises(ImportError, match="STATS_BACKEND=gpu requires CuPy"):
        background.estimate_background(np.ones((17, 19), np.float32))


def test_explicit_gpu_request_reports_no_device(monkeypatch):
    unavailable = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))
    monkeypatch.setattr(_background_gpu, "import_module", lambda _: unavailable)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    with pytest.raises(RuntimeError, match="working CUDA device.*no CUDA devices"):
        background.estimate_background(np.ones((17, 19), np.float32))


@pytest.mark.parametrize("values", [[0], [7, 7, 7, 7], [0, 1, 2, 3, 4, 100],
                                    [0, 0, 0, 0, 100], [-7, -2, 1, 3, 8, 400, 900]])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e6])
def test_cuda_global_statistics_match_frozen_definition(cuda, values, scale):
    pixels = (np.array([values], np.float32) * scale).astype(np.float32)
    valid = np.ones(pixels.shape, bool)
    context = _background_gpu.GpuBackgroundStatistics(pixels, valid, profile=True)
    try:
        assert context() == pytest.approx(background._clipped_stats(pixels), rel=1e-12, abs=1e-20)
        assert context(np.zeros_like(pixels), np.zeros_like(valid)) == pytest.approx(
            background._clipped_stats(pixels.astype(np.float64)), rel=1e-12, abs=1e-20)
    finally:
        context.close()


@pytest.mark.parametrize("backend", ["numpy", "cpp"])
@pytest.mark.parametrize("refine", [False, True])
def test_cuda_refinement_maps_masks_noise_and_transfer_accounting(cuda, request, monkeypatch, backend, refine):
    if backend == "cpp":
        request.getfixturevalue("native")
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", backend)
    pixels, valid = scene()
    expected = background.profile_background(pixels, 32, valid_mask=valid, refine=refine)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    actual = background.profile_background(pixels, 32, valid_mask=valid, refine=refine)
    for key in ("background", "source_mask", "mesh", "measured_mesh"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["noise"] == pytest.approx(expected["noise"], rel=1e-12)
    if refine:
        yy, xx = np.indices(pixels.shape)
        sky = 100 + .04 * xx + .03 * yy
        assert np.sqrt(np.mean((actual["background"][valid] - sky[valid]) ** 2)) < .5
        assert actual["noise"] == pytest.approx(2, rel=.05)
    timing = actual["timings"]
    fits = timing["refinement_passes"]
    assert timing["statistics_calls"] == fits + int(refine)
    assert timing["statistics_explicit_h2d_bytes"] == 5 * pixels.size * (1 + fits)
    assert 0 < timing["statistics_scalar_d2h_bytes"] <= 24 * 6 * timing["statistics_calls"]
    for stage in ("h2d", "compute", "d2h"):
        assert timing[f"statistics_gpu_{stage}_ms"] >= 0


def test_cuda_noise_widens_before_subtracting_and_reuses_buffers(cuda):
    pixels = np.random.default_rng(18).uniform(-1e8, 1e8, (51, 73)).astype(np.float32)
    level = np.random.default_rng(19).uniform(-1e8, 1e8, pixels.shape).astype(np.float32)
    valid = np.ones_like(pixels, bool)
    context = _background_gpu.GpuBackgroundStatistics(pixels, valid)
    pointers = [array.data.ptr for array in (context.pixels, context.background, context.source,
                                            context.residual, context.scratch, context.keep)]
    try:
        for offset in (0, 4, 8):
            shifted = level + np.float32(offset)
            expected = background._clipped_stats(pixels.astype(np.float64)-shifted)
            actual = context(shifted, np.zeros_like(valid))
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
            assert pointers == [array.data.ptr for array in (context.pixels, context.background,
                                context.source, context.residual, context.scratch, context.keep)]
        with pytest.raises(ValueError, match="no source-free observations"):
            context(level, valid)
    finally:
        context.close()


def test_cuda_native_calls_and_catalogs_are_isolated(cuda, native, monkeypatch):
    images = []
    for seed in range(3):
        pixels, valid = scene(seed)
        images.append(ImageData(str(seed), pixels, valid_mask=valid))
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "4")
    expected = [detection.process_image(image, DetectionParams(), 64) for image in images]
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda image: detection.process_image(image, DetectionParams(), 64), images))
    assert actual == expected


def test_cuda_statistics_runs_through_fits_process_pool(cuda, native, monkeypatch, tmp_path):
    from astropy.io import fits
    from astro_pipeline.parallel import run_batch
    paths = []
    for seed in range(3):
        pixels, _ = scene(seed)
        path = tmp_path / f"gpu_stats_{seed}.fits"
        fits.PrimaryHDU(pixels).writeto(path)
        paths.append(str(path))
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", "cpp")
    expected = run_batch(paths, DetectionParams(), 1, 64, mask_zero_border=True)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    for workers in (1, 2):
        assert run_batch(paths, DetectionParams(), workers, 64, mask_zero_border=True) == expected


@pytest.mark.parametrize("filename", ["255wmos.fits", "555wmos.fits"])
@pytest.mark.parametrize("tile_size", [256, 1024, None], ids=["tiles256", "tiles1024", "whole"])
@pytest.mark.parametrize("filter_backend", ["cpu", "gpu"])
def test_cuda_real_fits_catalog_matches_native_cpu(cuda, native, monkeypatch, filename,
                                                 tile_size, filter_backend):
    from pathlib import Path
    from astro_pipeline import io_fits
    path = Path(__file__).resolve().parents[1] / "fits" / filename
    image = io_fits.read_fits(str(path), mask_zero_border=True)
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "4")
    expected = detection.process_image(image, DetectionParams(), 256)
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "gpu")
    actual = detection.process_image(image, DetectionParams(backend=filter_backend), tile_size)
    assert actual == expected
