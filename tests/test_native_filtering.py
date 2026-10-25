"""Exercise Samuel's real native filter through Humza's public interfaces."""
from concurrent.futures import ThreadPoolExecutor
import math

import astroproc
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from astro_pipeline import filtering
from astro_pipeline.filtering import smooth
from astro_pipeline.interfaces import ImageData
from astro_pipeline.tiling import make_tiles


@pytest.fixture(params=["cpu", "gpu"])
def backend(request):
    if request.param == "gpu":
        available = astroproc.cuda_availability()
        if not available["available"]:
            pytest.skip(available["reason"])
    return request.param


def _radius(sigma):
    # ProcessingConfig stores the user value as float32.
    return math.ceil(3 * float(np.float32(sigma)))


def _oracle(data, sigma):
    data = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(data)
    options = dict(sigma=float(np.float32(sigma)), mode="reflect",
                   radius=_radius(sigma))
    values = gaussian_filter(np.where(valid, data, 0).astype(np.float64), **options)
    weights = gaussian_filter(valid.astype(np.float64), **options)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    np.divide(values, weights, out=out, where=valid & (weights > 0))
    return out


@pytest.mark.parametrize("sigma", [0.4, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("shape", [(1, 1), (1, 27), (23, 1), (29, 41)])
def test_filter_matches_independent_oracle(backend, sigma, shape):
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    original = data.copy()
    result = smooth(data, sigma, backend)
    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert not np.shares_memory(result, data)
    np.testing.assert_array_equal(data, original)
    np.testing.assert_allclose(result, _oracle(data, sigma), rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("sigma", [0.4, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("shape,tile_size", [((83, 119), 32), ((13, 19), 6),
                                           ((1, 31), 7), ((31, 1), 7)])
def test_humza_tile_cores_match_native_whole_image(backend, sigma, shape, tile_size):
    data = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    # Invalid pixels at or near tile boundaries exercise masked normalization.
    data.flat[data.size // 2] = np.nan
    original = data.copy()
    whole = smooth(data, sigma, backend)
    assembled = np.full(shape, np.nan, dtype=np.float32)
    tiles = make_tiles(ImageData("native-integration", data), tile_size,
                       math.ceil(4 * sigma))
    for tile in tiles:
        filtered = smooth(tile.data, sigma, backend)
        cy0, cx0, cy1, cx1 = tile.core
        assembled[cy0:cy1, cx0:cx1] = filtered[
            cy0 - tile.y0:cy1 - tile.y0, cx0 - tile.x0:cx1 - tile.x0]
    np.testing.assert_allclose(assembled, whole, rtol=1e-5, atol=1e-6,
                               equal_nan=True)
    np.testing.assert_array_equal(data, original)


def test_masked_noncontiguous_float64_input(backend):
    data = np.full((17, 38), 4.0, dtype=np.float64)[:, ::2]
    data[3, 4] = np.nan
    data[5, 7] = np.inf
    result = smooth(data, 2.0, backend)
    np.testing.assert_allclose(result, _oracle(data, 2.0), rtol=1e-5,
                               atol=1e-6, equal_nan=True)
    assert np.isnan(result[3, 4]) and np.isnan(result[5, 7])
    assert np.isinf(data[5, 7])


def test_cpu_calls_from_multiple_threads():
    data = np.random.default_rng(12).normal(size=(57, 63)).astype(np.float32)
    expected = smooth(data, 2.0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: smooth(data, 2.0), range(8)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf"), 1025.0])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        smooth(np.zeros((8, 9), np.float32), sigma)


@pytest.mark.parametrize("shape", [(8,), (2, 3, 4), (0, 8), (8, 0)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        smooth(np.zeros(shape, np.float32), 1.0)


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        smooth(np.zeros((8, 9), np.float32), 1.0, "cuda")


def test_missing_extension_has_build_instructions(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'astroproc'", name=name)

    monkeypatch.setattr(filtering, "import_module", missing)
    with pytest.raises(ImportError, match="docs/native_filtering.md"):
        smooth(np.zeros((8, 9), np.float32), 1.0)


def test_requested_gpu_never_silently_falls_back():
    available = astroproc.cuda_availability()
    data = np.random.default_rng(19).normal(size=(47, 53)).astype(np.float32)
    if available["available"]:
        data[8, 12] = np.nan
        np.testing.assert_allclose(smooth(data, 2.0, "gpu"), smooth(data, 2.0, "cpu"),
                                   rtol=1e-4, atol=1e-6, equal_nan=True)
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            smooth(data, 2.0, "gpu")
