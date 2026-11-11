"""Validate the FFT filter independently and through Humza's tile interface."""
import math
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from astro_pipeline.filtering import smooth
from astro_pipeline.interfaces import ImageData
from astro_pipeline.tiling import make_tiles


@pytest.fixture(scope="module")
def cupy_device():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA devices available")
    return cp


@pytest.fixture(params=["cpu", "gpu"])
def backend(request):
    if request.param == "gpu":
        request.getfixturevalue("cupy_device")
    return request.param


def _spatial_reference(data, sigma):
    return gaussian_filter(np.asarray(data, dtype=np.float64), sigma,
                           radius=math.ceil(4 * sigma), mode="reflect")


def _assert_close(result, expected, data, rtol=1e-4):
    # Relative error alone is undefined at zeros and unstable near cancellation.
    scale = float(np.max(np.abs(data)))
    np.testing.assert_allclose(result, expected, rtol=rtol,
                               atol=max(scale * 1e-6, np.finfo(np.float32).tiny))


@pytest.mark.parametrize("sigma", [0.4, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("shape", [(1, 1), (1, 29), (23, 1), (37, 51)])
def test_fft_matches_independent_spatial_reference(backend, sigma, shape):
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    original = data.copy()
    filtered = smooth(data, sigma, backend)
    assert filtered.shape == data.shape
    assert filtered.dtype == np.float32
    assert filtered.flags.c_contiguous
    assert not np.shares_memory(filtered, data)
    _assert_close(filtered, _spatial_reference(data, sigma), data, rtol=2e-5)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("value", [0.0, 7.0, -3.0])
def test_constant_image_has_no_dark_borders(backend, value):
    data = np.full((17, 23), value, dtype=np.float32)
    _assert_close(smooth(data, 2.0, backend), data, data)


@pytest.mark.parametrize("location", [(32, 32), (0, 0)])
def test_impulse_alignment_and_reflected_edges(backend, location):
    data = np.zeros((65, 65), np.float32)
    data[location] = 1.0
    filtered = smooth(data, 2.0, backend)
    _assert_close(filtered, _spatial_reference(data, 2.0), data)
    assert np.sum(filtered, dtype=np.float64) == pytest.approx(1.0, abs=5e-6)
    if location == (0, 0):
        # A circular convolution would incorrectly wrap signal to this corner.
        assert abs(filtered[-1, -1]) < 1e-6


@pytest.mark.parametrize("sigma", [0.4, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("shape,tile_size", [((83, 119), 32), ((13, 19), 6),
                                           ((1, 31), 7), ((31, 1), 7)])
def test_humza_filtered_cores_match_whole_image(backend, sigma, shape, tile_size):
    data = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    whole = smooth(data, sigma, backend)
    assembled = np.empty_like(data)
    for tile in make_tiles(ImageData("fft-integration", data), tile_size,
                           math.ceil(4 * sigma)):
        filtered = smooth(tile.data, sigma, backend)
        cy0, cx0, cy1, cx1 = tile.core
        assembled[cy0:cy1, cx0:cx1] = filtered[
            cy0 - tile.y0:cy1 - tile.y0, cx0 - tile.x0:cx1 - tile.x0]
    _assert_close(assembled, whole, data, rtol=2e-5)


def test_noncontiguous_float64_conversion(backend):
    data = np.random.default_rng(9).normal(size=(17, 38))[:, ::2]
    original = data.copy()
    filtered = smooth(data, 1.3, backend)
    assert filtered.dtype == np.float32
    assert filtered.flags.c_contiguous
    _assert_close(filtered, _spatial_reference(data.astype(np.float32), 1.3), data)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e6])
def test_gpu_matches_cpu_across_intensity_scales(cupy_device, scale):
    data = (np.random.default_rng(11).normal(size=(67, 93)) * scale).astype(np.float32)
    _assert_close(smooth(data, 2.0, "gpu"), smooth(data, 2.0, "cpu"), data)


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@pytest.mark.parametrize("masked", [False, True])
def test_large_fft_preserves_faint_pixels_near_bright_source(backend, sigma, masked):
    data = np.random.default_rng(379).normal(1, .1, (1037, 1049)).astype(np.float32)
    data[114, 118] = 1e6
    valid = np.ones(data.shape, dtype=bool)
    if masked:
        valid[:, :21] = False
        valid[491:503, 751:763] = False
    result = smooth(data, sigma, backend, **({"valid_mask": valid} if masked else {}))

    # Direct float64 sums at faint pixels provide an independent reference.
    # An input-wide tolerance based on the bright peak hides FFT leakage here.
    radius = math.ceil(4 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    kernel = np.outer(weights, weights).astype(np.float32).astype(np.float64)
    pixels = np.pad(np.where(valid, data, 0).astype(np.float64), radius, mode="symmetric")
    mask = np.pad(valid.astype(np.float64), radius, mode="symmetric")
    for y, x in ((0, 23), (28, 44), (487, 754), (520, 777), (1036, 1048)):
        window = np.s_[y:y+2*radius+1, x:x+2*radius+1]
        numerator = np.sum(pixels[window] * kernel, dtype=np.float64)
        denominator = np.sum(mask[window] * kernel, dtype=np.float64) if masked else 1.0
        expected = numerator / denominator if valid[y, x] and denominator >= .5 else 0.0
        assert result[y, x] == pytest.approx(expected, rel=0, abs=2e-7)


@pytest.mark.parametrize("backend_name", ["cpu", "gpu"])
@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_sigma(backend_name, sigma):
    with pytest.raises(ValueError, match="kernel_sigma"):
        smooth(np.zeros((8, 9), np.float32), sigma, backend_name)


@pytest.mark.parametrize("shape", [(8,), (2, 3, 4), (0, 8), (8, 0)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError, match="nonempty 2D"):
        smooth(np.zeros(shape, np.float32), 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pixels_require_ingestion_cleanup(value):
    data = np.zeros((8, 9), np.float32)
    data[3, 4] = value
    with pytest.raises(ValueError, match="ingestion"):
        smooth(data, 1.0)


def test_complex_input_is_rejected():
    with pytest.raises(ValueError, match="real pixel"):
        smooth(np.zeros((8, 9), np.complex64), 1.0)


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        smooth(np.zeros((8, 9), np.float32), 1.0, "cuda")


def test_missing_cupy_does_not_affect_cpu_or_silently_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    data = np.ones((8, 9), np.float32)
    _assert_close(smooth(data, 1.0, "cpu"), data, data)
    with pytest.raises(ImportError, match="requires a working CuPy"):
        smooth(data, 1.0, "gpu")
