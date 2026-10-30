from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astro_pipeline import background, detection, io_fits
from astro_pipeline.interfaces import DetectionParams


@pytest.fixture(autouse=True)
def default_backend(monkeypatch):
    monkeypatch.delenv("ASTRO_BACKGROUND_BACKEND", raising=False)
    monkeypatch.delenv("ASTRO_BACKGROUND_THREADS", raising=False)


@pytest.fixture
def native():
    return pytest.importorskip("astro_pipeline._background_native",
                               reason="native background extension is not built")


@pytest.mark.parametrize("shape,box", [((1, 1), 1), ((1, 17), 4), ((13, 1), 4),
                                       ((17, 19), 1), ((11, 12), 3),
                                       ((65, 97), 16), ((65, 97), 64),
                                       ((131, 147), 64), ((13, 7), 64)])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e6])
def test_native_exact_reference_and_thread_count_parity(native, shape, box, scale):
    rng = np.random.default_rng(7)
    yy, xx = np.indices(shape)
    data = ((100 + 0.02 * xx + 0.03 * yy + rng.normal(0, 2, shape)) * scale).astype(np.float32)
    data.flat[-1] += 50 * scale
    original = data.copy()
    expected, expected_noise = background.estimate_background(data, box)
    for threads in (1, 2, 4):
        actual, noise = native.estimate_background(data, box, threads)
        np.testing.assert_array_equal(actual, expected)
        assert noise == expected_noise
        assert actual.dtype == np.float32
        assert not np.shares_memory(actual, data)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("value", [0.0, 7.0, -4.0])
def test_constant_and_zero_noise(native, value):
    data = np.full((65, 97), value, np.float32)
    for threads in (1, 4):
        level, noise = native.estimate_background(data, 16, threads)
        np.testing.assert_array_equal(level, data)
        assert noise == 0


def test_reference_clipping_is_frozen(native):
    data = np.array([[0, 1, 2], [3, 4, 100]], np.float32)
    for estimate in (background.estimate_background, native.estimate_background):
        level, noise = estimate(data, 64)
        np.testing.assert_array_equal(level, np.full(data.shape, 2, np.float32))
        assert noise == 1.4826


@pytest.mark.parametrize("shape", [(91, 117), (1, 117), (91, 1)])
def test_gradient_extrapolation_and_singleton_mesh_axes(native, shape):
    yy, xx = np.indices(shape)
    data = (10 + yy * 0.125 + xx * 0.25).astype(np.float32)
    level, noise = native.estimate_background(data, 32, 4)
    np.testing.assert_array_equal(level, data)
    assert noise == 0


def test_strided_float64_and_readonly_inputs(native):
    data = np.random.default_rng(13).normal(100, 2, (67, 122))[::-1, ::2]
    data.setflags(write=False)
    expected, expected_noise = background.estimate_background(data, 16)
    actual, noise = native.estimate_background(data, 16, 4)
    np.testing.assert_array_equal(actual, expected)
    assert noise == expected_noise
    assert not data.flags.writeable


def test_float64_noise_residual_is_not_rounded_to_float32(native):
    # Opposite-sign float32 inputs can lose low bits if subtraction is done
    # before widening. The reference widens each input first.
    data = np.random.default_rng(18).uniform(-1e8, 1e8, (51, 73)).astype(np.float32)
    expected, expected_noise = background.estimate_background(data, 13)
    actual, noise = native.estimate_background(data, 13, 4)
    np.testing.assert_array_equal(actual, expected)
    assert noise == expected_noise


def test_concurrent_native_calls_have_independent_buffers(native):
    images = [np.random.default_rng(seed).normal(100, 2, (97, 131)).astype(np.float32)
              for seed in range(8)]
    expected = [background.estimate_background(image, 16) for image in images]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda image: native.estimate_background(image, 16, 4), images))
    for (level, noise), (expected_level, expected_noise) in zip(actual, expected):
        np.testing.assert_array_equal(level, expected_level)
        assert noise == expected_noise


def test_profiling_reports_global_selection_as_subset_of_noise(native):
    data = np.random.default_rng(14).normal(100, 2, (131, 147)).astype(np.float32)
    result = native.profile_background(data, 16, 4)
    expected, noise = background.estimate_background(data, 16)
    np.testing.assert_array_equal(result["background"], expected)
    assert result["noise"] == noise
    timings = result["timings"]
    assert 1 <= timings["effective_threads"] <= 4
    assert 0 <= timings["global_median_ms"] <= timings["noise_ms"]
    assert timings["mesh_ms"] + timings["interpolation_ms"] + timings["noise_ms"] <= timings["total_ms"]


def test_numpy_is_default_and_does_not_import_native(monkeypatch):
    def unexpected_import(name):
        raise AssertionError("default path must not import the native extension")
    monkeypatch.setattr(background, "import_module", unexpected_import)
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "irrelevant-for-numpy")
    level, noise = background.estimate_background(np.ones((9, 13), np.float32))
    np.testing.assert_equal(level, 1)
    assert noise == 0


def test_cpp_dispatch_defaults_to_one_thread_per_process(monkeypatch):
    calls = []
    def estimate(data, box, threads):
        calls.append((box, threads))
        return np.zeros_like(data), 0.0
    monkeypatch.setattr(background, "import_module", lambda name: SimpleNamespace(estimate_background=estimate))
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    background.estimate_background(np.ones((9, 13), np.float32))
    assert calls == [(64, 1)]
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "4")
    background.estimate_background(np.ones((9, 13), np.float32), box=16)
    assert calls[-1] == (16, 4)


def test_explicit_cpp_request_never_silently_falls_back(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(background, "import_module", missing)
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    with pytest.raises(ImportError, match="build it with CMake"):
        background.estimate_background(np.ones((9, 13), np.float32))


def test_cpp_dispatch_respects_process_cpu_affinity(monkeypatch):
    calls = []
    def estimate(data, box, threads):
        calls.append(threads)
        return np.zeros_like(data), 0.0
    monkeypatch.setattr(background, "import_module", lambda name: SimpleNamespace(estimate_background=estimate))
    monkeypatch.setattr(background.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "100000000000000000000")
    background.estimate_background(np.zeros((8, 8), np.float32))
    assert calls == [2]


def test_arbitrarily_large_box_still_means_one_cell(native, monkeypatch):
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    expected, expected_noise = background.estimate_background(data, 10**30)
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    actual, noise = background.estimate_background(data, 10**30)
    np.testing.assert_array_equal(actual, expected)
    assert noise == expected_noise


@pytest.mark.parametrize("threads", ["0", "-1", "1.5", "invalid"])
def test_invalid_thread_configuration(monkeypatch, threads):
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", threads)
    with pytest.raises(ValueError, match="ASTRO_BACKGROUND_THREADS"):
        background.estimate_background(np.zeros((8, 8), np.float32))


def test_invalid_backend_configuration(monkeypatch):
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cuda")
    with pytest.raises(ValueError, match="ASTRO_BACKGROUND_BACKEND"):
        background.estimate_background(np.zeros((8, 8), np.float32))


@pytest.mark.parametrize("kwargs", [dict(box=0), dict(box=-1), dict(threads=0), dict(threads=-1)])
def test_native_validates_positive_settings(native, kwargs):
    with pytest.raises(ValueError):
        native.estimate_background(np.zeros((8, 8), np.float32), **kwargs)


@pytest.mark.parametrize("data", [np.zeros(3), np.zeros((0, 3)),
                                 np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
def test_native_validates_images(native, data):
    with pytest.raises(ValueError):
        native.estimate_background(data, threads=4)


def test_worker_failure_joins_threads_and_preserves_later_calls(native):
    maximum = np.finfo(np.float32).max
    data = np.array([[-maximum, -maximum, maximum, maximum]], np.float32)
    with pytest.raises(FloatingPointError):
        native.estimate_background(data, 2, 4)
    level, noise = native.estimate_background(np.ones((9, 13), np.float32), 4, 4)
    np.testing.assert_equal(level, 1)
    assert noise == 0


def test_near_float32_limit_uses_reference_rounding(native):
    maximum = np.float32(np.finfo(np.float32).max)
    previous = np.nextafter(maximum, np.float32(0))
    data = np.array([[maximum, maximum, previous, previous]], np.float32)
    expected, expected_noise = background.estimate_background(data, 2)
    actual, noise = native.estimate_background(data, 2, 4)
    np.testing.assert_array_equal(actual, expected)
    assert noise == expected_noise


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_medium_fits_catalogs_match_numpy(native, monkeypatch, threads):
    paths = sorted((Path(__file__).resolve().parents[1] / "gen_fits/medium").glob("*.fits"))
    assert len(paths) == 10
    params = DetectionParams()
    for path in paths:
        image = io_fits.read_fits(path)
        monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "numpy")
        expected = detection.process_image(image, params, 64)
        monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
        monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", str(threads))
        assert detection.process_image(image, params, 64) == expected
