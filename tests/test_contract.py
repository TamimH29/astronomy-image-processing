from dataclasses import replace

import numpy as np
import pytest

from astro_pipeline import background, detection, filtering, tiling
from astro_pipeline.interfaces import CATALOG_COLUMNS, Source, DetectionParams, ImageData, Tile


def test_as_row_matches_schema():
    s = Source("img", 1, 10.5, 20.5, 8, 18, 13, 23, 40, 123.4)
    row = s.as_row()
    assert len(row) == len(CATALOG_COLUMNS)
    assert row[0] == "img" and row[-1] == 123.4


def test_default_params():
    p = DetectionParams()
    assert p.backend == "cpu" and p.threshold_sigma == 3.0


def _scene():
    pixels = np.random.default_rng(42).normal(10, 0.2, (192, 224)).astype(np.float32)
    # Long sources exceed the halo; one crosses a four-core intersection.
    for y0, y1, x0, x1, value in [
        (28, 36, 40, 88, 15), (120, 137, 120, 137, 20),
        (45, 115, 10, 18, 12), (0, 5, 180, 190, 18),
    ]:
        pixels[y0:y1, x0:x1] += value
    return ImageData("synthetic", pixels, {"BUNIT": "counts"})


def _assert_catalogs_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a.image_id, a.source_id, a.xmin, a.ymin, a.xmax, a.ymax, a.area_pix) == (
            b.image_id, b.source_id, b.xmin, b.ymin, b.xmax, b.ymax, b.area_pix)
        np.testing.assert_allclose([a.x_centroid, a.y_centroid, a.flux],
                                   [b.x_centroid, b.y_centroid, b.flux],
                                   rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("tile_size", [16, 32, 64, 97, 512])
def test_tiled_equals_whole_image(tile_size):
    image = _scene()
    params = DetectionParams()
    whole = detection.process_image(image, params)
    tiled = detection.process_image(image, params, tile_size)
    assert len(whole) == 4
    _assert_catalogs_equal(tiled, whole)


@pytest.mark.skip(reason="Person 2: 1, 2, 4 workers give identical catalogs")
def test_worker_count_invariance():
    ...


def test_gpu_matches_cpu():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if count == 0:
        pytest.skip("No CUDA devices available")
    image = _scene()
    params = DetectionParams()
    cpu = detection.process_image(image, params, 64)
    gpu = detection.process_image(image, replace(params, backend="gpu"), 64)
    _assert_catalogs_equal(gpu, cpu)


@pytest.mark.skip(reason="Person 1: synthetic recovery meets accuracy targets")
def test_synthetic_recovery():
    ...


@pytest.mark.parametrize("shape,box", [((1, 1), 64), ((8, 11), 64), ((65, 97), 32)])
def test_constant_background_and_zero_noise(shape, box):
    pixels = np.full(shape, 7, np.float32)
    original = pixels.copy()
    level, noise = background.estimate_background(pixels, box)
    assert level.shape == shape and level.dtype == np.float32
    np.testing.assert_array_equal(level, pixels)
    np.testing.assert_array_equal(pixels, original)
    assert noise == 0


@pytest.mark.parametrize("shape", [(91, 117), (1, 117), (91, 1)])
def test_background_gradient_partial_cells_and_singleton_axes(shape):
    yy, xx = np.indices(shape)
    pixels = (10 + yy * 0.125 + xx * 0.25).astype(np.float32)
    level, noise = background.estimate_background(pixels, 32)
    np.testing.assert_allclose(level, pixels, rtol=0, atol=1e-6)
    assert noise < 1e-6


def test_background_rejects_bright_outliers():
    pixels = np.full((80, 95), 7, np.float32)
    pixels[10:13, 15:18] = 1e6
    pixels[-1, -1] = 1e6
    level, noise = background.estimate_background(pixels, 32)
    np.testing.assert_equal(level, 7)
    assert noise == 0


def test_background_returns_prefilter_noise():
    pixels = np.random.default_rng(12).normal(10, 2.5, (200, 220)).astype(np.float32)
    level, noise = background.estimate_background(pixels)
    assert abs(float(np.median(level)) - 10) < 0.15
    assert noise == pytest.approx(2.5, rel=0.1)


def test_background_converts_strided_input_without_mutating_it():
    pixels = np.full((9, 22), 3.0, np.float64)[:, ::2]
    level, noise = background.estimate_background(pixels)
    assert level.dtype == np.float32 and noise == 0
    np.testing.assert_equal(level, 3)
    np.testing.assert_equal(pixels, 3)
    assert not np.shares_memory(level, pixels)


def test_detect_connectivity_min_area_coordinates_and_measurements():
    pixels = np.zeros((16, 16), np.float32)
    pixels[2, 2] = 10  # rejected singleton
    pixels[8, 8], pixels[9, 9] = 10, 20  # eight-connected, not four-connected
    tile = Tile("offset", 200, 100, pixels, (200, 100, 216, 116), 1)
    sources = detection.detect(tile, 1, DetectionParams(kernel_sigma=0.1, min_area=2))
    assert len(sources) == 1
    source = sources[0]
    assert source.image_id == "offset" and source.source_id == 0
    assert (source.xmin, source.ymin, source.xmax, source.ymax) == (108, 208, 109, 209)
    assert source.area_pix == 2 and source.flux == 30
    assert source.x_centroid == pytest.approx(100 + 26 / 3)
    assert source.y_centroid == pytest.approx(200 + 26 / 3)


def test_detect_centroid_ownership_is_half_open():
    pixels = np.zeros((12, 12), np.float32)
    pixels[5, 4:7] = 10
    left = Tile("owned", 200, 100, pixels, (200, 100, 212, 105), 1)
    right = replace(left, core=(200, 105, 212, 112))
    params = DetectionParams(kernel_sigma=0.1, min_area=1)
    assert detection.detect(left, 1, params) == []
    assert len(detection.detect(right, 1, params)) == 1


def test_measurement_signed_flux_and_zero_positive_weight_centroid():
    pixels = np.array([[-2, -1], [0, 0]], np.float32)
    tile = Tile("signed", 0, 0, pixels, (0, 0, 2, 2), 0)
    filtered = np.array([[5, 5], [0, 0]], np.float32)
    sources = detection._measure_sources(tile, filtered, 1, DetectionParams(min_area=1))
    assert len(sources) == 1
    assert sources[0].flux == -3
    assert (sources[0].x_centroid, sources[0].y_centroid) == (0.5, 0)
    assert sources[0].area_pix == 2


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e6])
def test_zero_noise_floor_preserves_sources_across_units(scale):
    pixels = np.zeros((32, 32), np.float32)
    pixels[13:16, 13:16] = 10 * scale
    tile = Tile("units", 0, 0, pixels, (0, 0, 32, 32), 4)
    sources = detection.detect(tile, 0, DetectionParams(kernel_sigma=1))
    assert len(sources) == 1
    assert sources[0].flux == pytest.approx(90 * scale, rel=1e-6)
    assert sources[0].x_centroid == pytest.approx(14)
    assert sources[0].y_centroid == pytest.approx(14)


@pytest.mark.parametrize("value", [0, 7])
def test_process_image_constant_is_empty(value):
    image = ImageData("empty", np.full((80, 95), value, np.float32))
    assert detection.process_image(image, DetectionParams(), 32) == []


def test_process_image_recovers_known_source_locations_and_flux():
    sources = detection.process_image(_scene(), DetectionParams(), 64)
    expected = [(184.5, 2, 900), (63.5, 31.5, 5760),
                (13.5, 79.5, 6720), (128, 128, 5780)]
    assert len(sources) == len(expected)
    for source, (cx, cy, flux) in zip(sources, expected):
        assert source.x_centroid == pytest.approx(cx, abs=0.15)
        assert source.y_centroid == pytest.approx(cy, abs=0.15)
        assert source.flux == pytest.approx(flux, rel=0.03)


def test_process_image_calls_real_humza_functions_and_preserves_inputs(monkeypatch):
    image = _scene()
    original = image.data.copy()
    params = DetectionParams()
    real_background = background.estimate_background
    real_tiles, real_merge = tiling.make_tiles, tiling.merge
    real_smooth, real_measure = filtering.smooth, detection._measure_sources
    calls = {"background": 0, "filter": 0, "measure": 0, "merge": 0}
    recorded = {}

    def estimate(data, box=64):
        calls["background"] += 1
        result = real_background(data, box)
        recorded["noise"] = result[1]
        return result

    def tiles(img, tile_size, halo):
        assert halo == 8 and tile_size == 64
        result = real_tiles(img, tile_size, halo)
        recorded["tile_count"] = len(result)
        return result

    def smooth(data, sigma, backend):
        calls["filter"] += 1
        return real_smooth(data, sigma, backend)

    def measure(tile, filtered, noise, config):
        calls["measure"] += 1
        assert tile.data.shape == image.data.shape == filtered.shape
        assert noise == recorded["noise"]
        return real_measure(tile, filtered, noise, config)

    def merge(image_id, per_tile, tile_list):
        calls["merge"] += 1
        assert sum(map(len, per_tile)) == 4
        for tile, source_list in zip(tile_list, per_tile):
            cy0, cx0, cy1, cx1 = tile.core
            for source in source_list:
                assert source.source_id == 0
                assert cy0 <= source.y_centroid < cy1
                assert cx0 <= source.x_centroid < cx1
        return real_merge(image_id, per_tile, tile_list)

    monkeypatch.setattr(background, "estimate_background", estimate)
    monkeypatch.setattr(tiling, "make_tiles", tiles)
    monkeypatch.setattr(tiling, "merge", merge)
    monkeypatch.setattr(filtering, "smooth", smooth)
    monkeypatch.setattr(detection, "_measure_sources", measure)
    sources = detection.process_image(image, params, 64)
    assert calls == {"background": 1, "filter": recorded["tile_count"], "measure": 1, "merge": 1}
    assert [s.source_id for s in sources] == [1, 2, 3, 4]
    np.testing.assert_array_equal(image.data, original)
    assert image.header == {"BUNIT": "counts"}
    assert params == DetectionParams()


def test_process_image_is_deterministic():
    image = _scene()
    params = DetectionParams()
    assert detection.process_image(image, params, 64) == detection.process_image(image, params, 64)


@pytest.mark.parametrize("box", [0, -1, 2.5, True])
def test_invalid_background_box(box):
    with pytest.raises(ValueError, match="box"):
        background.estimate_background(np.zeros((8, 8)), box)


@pytest.mark.parametrize("pixels", [np.zeros(3), np.empty((0, 3)),
                                    np.zeros((2, 2), np.complex64),
                                    np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
def test_invalid_image_inputs(pixels):
    with pytest.raises(ValueError):
        background.estimate_background(pixels)
    with pytest.raises(ValueError):
        detection.process_image(ImageData("invalid", pixels), DetectionParams())


@pytest.mark.parametrize("tile_size", [0, -1, 1.5, True])
def test_invalid_process_tile_size(tile_size):
    with pytest.raises(ValueError, match="tile_size"):
        detection.process_image(ImageData("invalid", np.zeros((8, 8))),
                                DetectionParams(), tile_size)


@pytest.mark.parametrize("changes", [dict(kernel_sigma=0), dict(kernel_sigma=np.nan),
                                     dict(threshold_sigma=0), dict(threshold_sigma=np.inf),
                                     dict(min_area=0), dict(min_area=2.5),
                                     dict(backend="cuda")])
def test_invalid_detection_params(changes):
    params = replace(DetectionParams(), **changes)
    pixels = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        detection.process_image(ImageData("invalid", pixels), params)
    with pytest.raises(ValueError):
        detection.detect(Tile("invalid", 0, 0, pixels, (0, 0, 8, 8), 0), 1, params)


@pytest.mark.parametrize("noise", [-1, np.nan, np.inf])
def test_invalid_detection_noise(noise):
    tile = Tile("invalid", 0, 0, np.zeros((8, 8)), (0, 0, 8, 8), 0)
    with pytest.raises(ValueError, match="noise_sigma"):
        detection.detect(tile, noise, DetectionParams())


def test_invalid_tile_core():
    tile = Tile("invalid", 0, 0, np.zeros((8, 8)), (0, 0, 9, 8), 0)
    with pytest.raises(ValueError, match="tile core"):
        detection.detect(tile, 1, DetectionParams())


def test_process_gpu_request_does_not_fallback(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="requires a working CuPy"):
        detection.process_image(ImageData("gpu", np.zeros((8, 8))),
                                DetectionParams(backend="gpu"))
