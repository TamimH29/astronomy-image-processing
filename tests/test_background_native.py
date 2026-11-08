from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from astro_pipeline import background, detection, io_fits
from astro_pipeline.interfaces import DetectionParams, ImageData


@pytest.fixture(autouse=True)
def default_backend(monkeypatch):
    monkeypatch.delenv("ASTRO_BACKGROUND_BACKEND", raising=False)
    monkeypatch.delenv("ASTRO_BACKGROUND_THREADS", raising=False)
    monkeypatch.delenv("ASTRO_DETECTION_BACKEND", raising=False)
    monkeypatch.delenv("ASTRO_DETECTION_THREADS", raising=False)


@pytest.fixture
def native():
    module = pytest.importorskip("astro_pipeline._background_native",
                                reason="native background extension is not built")
    assert getattr(module, "API_VERSION", 0) == 2, "rebuild the stale native CPU extension"
    return module


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
    def estimate(data, box, threads, **kwargs):
        calls.append((box, threads))
        return np.zeros_like(data), 0.0
    monkeypatch.setattr(background, "import_module", lambda name: SimpleNamespace(API_VERSION=2, estimate_background=estimate))
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
    def estimate(data, box, threads, **kwargs):
        calls.append(threads)
        return np.zeros_like(data), 0.0
    monkeypatch.setattr(background, "import_module", lambda name: SimpleNamespace(API_VERSION=2, estimate_background=estimate))
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


@pytest.fixture(params=[("numpy", 1), ("cpp", 1), ("cpp", 2), ("cpp", 4)],
                ids=["numpy", "cpp-1", "cpp-2", "cpp-4"])
def accuracy_backend(request, monkeypatch):
    backend, threads = request.param
    if backend == "cpp":
        request.getfixturevalue("native")
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", backend)
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", str(threads))


def _injected_sky(shape):
    # Analytic truth from synthetic.py, independent of either estimator.
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return 100.0 + 8.0 * xx / max(width - 1, 1) + 5.0 * yy / max(height - 1, 1)


def test_medium_fits_background_accuracy_against_injected_sky(accuracy_backend):
    """Backend agreement alone cannot establish sky or noise accuracy."""
    paths = sorted((Path(__file__).resolve().parents[1] / "gen_fits/medium").glob("*.fits"))
    assert len(paths) == 10
    for path in paths:
        pixels = io_fits.read_fits(str(path)).data
        truth = json.loads(path.with_suffix(".truth.json").read_text())["truth"]
        sky = _injected_sky(pixels.shape)
        clean = np.ones(pixels.shape, dtype=bool)
        for source in truth:
            radius = int(np.ceil(4 * source["sigma"]))
            x, y = int(np.floor(source["x"])), int(np.floor(source["y"]))
            clean[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = False

        level, noise = background.estimate_background(pixels)
        rmse = float(np.sqrt(np.mean((level.astype(np.float64) - sky) ** 2)))
        residual = pixels[clean].astype(np.float64) - level[clean]
        assert rmse < 0.15, (path.name, rmse)
        assert abs(noise - 2.0) < 0.1, (path.name, noise)
        assert abs(float(np.median(residual))) < 0.1, path.name
        assert abs(float(residual.std()) - 2.0) < 0.1, path.name


def test_medium_fits_source_accuracy_against_injected_truth(accuracy_backend):
    """Guard measured baseline accuracy, rather than only comparing backends.

    These are regression limits for the current default settings, not final
    project acceptance targets. Match one-to-one within three pixels, minimizing
    distance after maximizing the number of valid pairs. Flux error compares
    catalog flux with the full injected flux, including missed faint wings.
    """
    paths = sorted((Path(__file__).resolve().parents[1] / "gen_fits/medium").glob("*.fits"))
    assert len(paths) == 10
    truth_count, detected_count, matched_count = 0, 0, 0
    position_errors, flux_errors = [], []
    for path in paths:
        truth = json.loads(path.with_suffix(".truth.json").read_text())["truth"]
        image = io_fits.read_fits(str(path))
        sources = detection.process_image(image, DetectionParams(), 64)
        truth_count += len(truth)
        detected_count += len(sources)
        if not sources:
            continue
        measured = np.array([[source.x_centroid, source.y_centroid] for source in sources])
        expected = np.array([[source["x"], source["y"]] for source in truth])
        distances = np.linalg.norm(measured[:, None] - expected[None], axis=2)
        ii, jj = linear_sum_assignment(np.where(distances <= 3, distances, 1e6))
        for i, j in zip(ii, jj):
            if distances[i, j] <= 3:
                matched_count += 1
                position_errors.append(float(distances[i, j]))
                flux_errors.append(abs(sources[i].flux - truth[j]["flux"]) / truth[j]["flux"])

    assert truth_count == 250
    assert matched_count >= 230, matched_count
    assert detected_count - matched_count <= 2, (detected_count, matched_count)
    assert float(np.median(position_errors)) < 0.1
    assert float(np.quantile(position_errors, 0.95)) < 0.3
    assert float(np.median(flux_errors)) < 0.15


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="default 64-pixel mesh has one cell on a 64x64 gradient image")
def test_tiny_fits_background_tracks_known_gradient(accuracy_backend):
    path = Path(__file__).resolve().parents[1] / "gen_fits/tiny/tiny_000.fits"
    pixels = io_fits.read_fits(str(path)).data
    level, noise = background.estimate_background(pixels)
    rmse = float(np.sqrt(np.mean((level.astype(np.float64) - _injected_sky(pixels.shape)) ** 2)))
    assert rmse < 0.5, rmse
    assert abs(noise - 2.0) < 0.1, noise


def test_known_sky_with_real_mosaic_coverage(accuracy_backend):
    """Padding must not bias the known sky in the covered part of an image.

    The shared zero footprint is used only as an explicit experimental coverage
    mask; this test does not assert that zero-valued pixels are always invalid.
    """
    path = Path(__file__).resolve().parents[1] / "fits/255wmos.fits"
    footprint = io_fits.read_fits(str(path)).data
    yy = np.arange(256) * footprint.shape[0] // 256
    xx = np.arange(256) * footprint.shape[1] // 256
    valid = footprint[yy[:, None], xx[None, :]] != 0
    sky = _injected_sky(valid.shape)
    pixels = (sky + np.random.default_rng(37931).normal(0, 2, valid.shape)).astype(np.float32)
    pixels[~valid] = 0
    level, noise = background.estimate_background(pixels, valid_mask=valid)
    rmse = float(np.sqrt(np.mean((level[valid].astype(np.float64) - sky[valid]) ** 2)))
    assert rmse < 0.5, rmse
    assert abs(noise - 2.0) < 0.1, noise


@pytest.mark.parametrize("background_backend,detector_backend", [
    ("numpy", "python"), ("cpp", "python"), ("numpy", "cpp"), ("cpp", "cpp"), ("resident", "resident")])
@pytest.mark.parametrize("filename,required", [("Vcomb.fits", 20), ("green.fits", 19),
                                               ("255wmos.fits", 20), ("555wmos.fits", 20)])
@pytest.mark.parametrize("filter_backend", ["cpu", "gpu"])
def test_real_fits_injected_star_recovery(request, monkeypatch, filename, required,
                                         background_backend, detector_backend, filter_backend):
    """Frozen additions: the five diagnosed mosaic failures must all recover.

    The peak ratio is based on neighboring-pixel differences, not calibrated
    global detection significance. The separate isolated control requires 20/20.
    """
    if "cpp" in (background_backend, detector_backend):
        request.getfixturevalue("native")
    if filter_backend == "gpu":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "numpy" if background_backend == "resident" else background_backend)
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", "python" if detector_backend == "resident" else detector_backend)
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "4")
    monkeypatch.setenv("ASTRO_DETECTION_THREADS", "4")
    fixture = Path(__file__).parent / "fixtures/real_star_injections.json"
    truth = json.loads(fixture.read_text())[filename]
    image = io_fits.read_fits(str(Path(__file__).resolve().parents[1] / "fits" / filename),
                             mask_zero_border=filename in ("255wmos.fits", "555wmos.fits"))
    pixels = image.data.copy()
    for source in truth:
        x, y = source["x"], source["y"]
        ix, iy = int(np.floor(x)), int(np.floor(y))
        yy, xx = np.mgrid[iy-6:iy+7, ix-6:ix+7]
        model = np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 1.5**2))
        model *= source["flux"] / model.sum()
        pixels[iy-6:iy+7, ix-6:ix+7] += model.astype(np.float32)
    sources = detection.process_image(ImageData(image.image_id, pixels, image.header, image.valid_mask),
                                      DetectionParams(backend=filter_backend,
                                          pipeline_backend="gpu" if background_backend == "resident" else "staged"), 256)
    measured = np.array([[source.x_centroid, source.y_centroid] for source in sources])
    expected = np.array([[source["x"], source["y"]] for source in truth])
    distances = np.linalg.norm(measured[:, None] - expected[None], axis=2)
    ii, jj = linear_sum_assignment(np.where(distances <= 3, distances, 1e6))
    strong = [(i, j) for i, j in zip(ii, jj)
              if distances[i, j] <= 3 and truth[j]["peak_local_noise_units"] == 20]
    assert len(strong) >= required, (filename, len(strong))
    assert float(np.median([distances[i, j] for i, j in strong])) < 0.5
    assert float(np.median([abs(sources[i].flux - truth[j]["flux"]) / truth[j]["flux"]
                            for i, j in strong])) < 0.4
    if filename == "green.fits":
        # This remaining addition has fewer than min_area pixels above the
        # unchanged global cutoff. Its neighbor-difference peak ratio is not
        # the significance used by the detector; it is not a deblending miss.
        from astro_pipeline import filtering
        level, noise = background.estimate_background(pixels)
        filtered = filtering.smooth(pixels - level, 2)
        x, y = int(truth[8]["x"]), int(truth[8]["y"])
        assert np.count_nonzero(filtered[y-4:y+5, x-4:x+5] > 3 * noise) < 5
