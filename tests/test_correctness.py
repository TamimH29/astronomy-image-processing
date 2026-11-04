"""Independent scientific controls and CPU-backend integration regressions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from astro_pipeline import background, detection, filtering, io_fits, tiling, _segmentation
from astro_pipeline.interfaces import DetectionParams, ImageData, Tile, INTERFACE_VERSION

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for stage in ("BACKGROUND", "DETECTION"):
        monkeypatch.delenv(f"ASTRO_{stage}_BACKEND", raising=False)
        monkeypatch.delenv(f"ASTRO_{stage}_THREADS", raising=False)


@pytest.fixture(params=[("numpy", "python"), ("cpp", "python"),
                        ("numpy", "cpp"), ("cpp", "cpp")])
def cpu_backends(request, monkeypatch):
    bg, detector = request.param
    if "cpp" in request.param:
        native = pytest.importorskip("astro_pipeline._background_native")
        assert native.API_VERSION == 2, "rebuild the stale native extension"
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", bg)
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", detector)
    monkeypatch.setenv("ASTRO_BACKGROUND_THREADS", "2")
    monkeypatch.setenv("ASTRO_DETECTION_THREADS", "2")
    return request.param


def test_mask_contract_is_appended_and_tiles_preserve_it():
    assert INTERFACE_VERSION == 4
    pixels = np.ones((17, 23), np.float32)
    valid = pixels.astype(bool)
    valid[7:12, 8:13] = False
    image = ImageData("mask", pixels, {}, valid)
    assert ImageData("old", pixels).valid_mask is None
    for tile in tiling.make_tiles(image, 8, 3):
        h, w = tile.data.shape
        np.testing.assert_array_equal(tile.valid_mask, valid[tile.y0:tile.y0+h, tile.x0:tile.x0+w])
        assert not np.shares_memory(tile.valid_mask, valid)


@pytest.mark.parametrize("suffix", [".fits", ".fits.gz", ".fits.bz2"])
def test_reader_retains_invalidity_and_border_option(tmp_path, suffix):
    data = np.full((20, 21), 7, np.float32)
    data[0, :6] = 0
    data[1:4, 3] = 0
    data[10, 10] = 0  # valid interior zero must survive the optional border rule
    data[7, 7], data[8, 8] = np.nan, np.inf
    path = tmp_path / ("input" + suffix)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, name="SCI")]).writeto(path)
    ordinary = io_fits.read_fits(str(path))
    assert ordinary.image_id == "input"
    assert ordinary.data.dtype == np.float32 and np.isfinite(ordinary.data).all()
    assert ordinary.valid_mask[0, 0] and ordinary.valid_mask[10, 10]
    assert not ordinary.valid_mask[7, 7] and not ordinary.valid_mask[8, 8]
    padded = io_fits.read_fits(str(path), mask_zero_border=True)
    assert not padded.valid_mask[3, 3] and padded.valid_mask[10, 10]
    assert padded.data[7, 7] == 0


def test_reader_uses_four_connected_border_zeros(tmp_path):
    data = np.ones((4, 4), np.float32)
    data[0, 0] = data[1, 1] = 0
    path = tmp_path / "diagonal.fits"
    fits.PrimaryHDU(data).writeto(path)
    image = io_fits.read_fits(str(path), mask_zero_border=True)
    assert not image.valid_mask[0, 0] and image.valid_mask[1, 1]


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_reader_rejects_no_observations(tmp_path, value):
    path = tmp_path / "empty.fits"
    fits.PrimaryHDU(np.full((10, 10), value, np.float32)).writeto(path)
    with pytest.raises(ValueError, match="no (usable observations|finite pixels)"):
        io_fits.read_fits(str(path), mask_zero_border=True)


@pytest.mark.parametrize("path", sorted((ROOT / "fits").glob("*.fits")), ids=lambda p: p.name)
def test_all_supplied_fits_headers_pixels_and_cube_rejection(path):
    before = sha256(path.read_bytes()).digest()
    with fits.open(path, memmap=False) as hdus:
        raw = hdus[0].data
        if raw.ndim != 2:
            with pytest.raises(ValueError, match="unsupported image shapes.*cubes"):
                io_fits.read_fits(str(path))
            return
        expected = np.asarray(raw, dtype=np.float32).copy()
    image = io_fits.read_fits(str(path))
    expected[~np.isfinite(expected)] = 0
    np.testing.assert_array_equal(image.data, expected)
    assert sha256(path.read_bytes()).digest() == before
    if "SKEW" in image.header:
        assert isinstance(image.header["SKEW"], str)


def _known_scene():
    yy, xx = np.mgrid[:256, :256]
    sky = 100 + 8 * xx / 255 + 5 * yy / 255 + 2 * (xx / 255) ** 2
    valid = np.ones(sky.shape, dtype=bool)
    valid[:64, :128] = False
    valid[80:100, 110:130] = False
    data = (sky + np.random.default_rng(31).normal(0, 2, sky.shape)).astype(np.float32)
    data[~valid] = 0
    return sky, data, valid


def test_masked_known_sky_and_blank_detection(cpu_backends):
    sky, data, valid = _known_scene()
    level, noise = background.estimate_background(data, valid_mask=valid)
    assert np.sqrt(np.mean((level[valid] - sky[valid]) ** 2)) < 0.5
    assert noise == pytest.approx(2, rel=0.05)
    assert np.all(level[~valid] == 0)
    assert detection.process_image(ImageData("blank", data, {}, valid), DetectionParams(), 64) == []


def test_bright_extended_source_is_not_sky(cpu_backends):
    yy, xx = np.mgrid[:384, :384]
    sky = 100 + .01 * xx + .02 * yy
    disk = np.hypot(xx - 220, yy - 190) < 62
    data = (sky + np.random.default_rng(3).normal(0, 2, sky.shape) + 1000 * disk).astype(np.float32)
    result = background.profile_background(data)
    assert np.sqrt(np.mean((result["background"] - sky) ** 2)) < 0.5
    assert result["noise"] == pytest.approx(2, rel=0.05)
    assert result["source_mask"][190, 220]
    assert abs(float(result["background"][190, 220]) - sky[190, 220]) < .5


def test_insufficient_coverage_and_bad_mask(cpu_backends):
    data = np.ones((128, 128), np.float32)
    valid = np.zeros_like(data, dtype=bool)
    with pytest.raises(ValueError, match="no usable observations"):
        background.estimate_background(data, valid_mask=valid)
    valid[::2] = True
    with pytest.raises(ValueError, match="insufficient background coverage"):
        background.estimate_background(data, valid_mask=valid)
    with pytest.raises(ValueError, match="boolean"):
        background.estimate_background(data, valid_mask=np.ones_like(data))


def test_twenty_isolated_sources_on_real_coverage(cpu_backends):
    footprint = io_fits.read_fits(str(ROOT / "fits/255wmos.fits"), mask_zero_border=True).valid_mask
    indices = np.arange(512) * 1600 // 512
    valid = footprint[indices[:, None], indices[None, :]]
    distance = distance_transform_edt(valid)
    yy, xx = np.mgrid[:512, :512]
    rng = np.random.default_rng(37931)
    data = 100 + 8 * xx / 511 + 5 * yy / 511 + rng.normal(0, 2, valid.shape)
    truth = []
    for _ in range(10000):
        if len(truth) == 20:
            break
        x, y = rng.uniform(30, 482, 2)
        if distance[int(y), int(x)] <= 25 or any(np.hypot(x-qx, y-qy) < 35 for qx, qy in truth):
            continue
        truth.append((x, y))
        data += 40 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 1.5**2))
    assert len(truth) == 20
    data[~valid] = 0
    image = ImageData("isolated", data.astype(np.float32), {}, valid)
    sources = detection.process_image(image, DetectionParams(), 64)
    whole = detection.process_image(image, DetectionParams())
    assert len(whole) == len(sources)
    for tiled_source, whole_source in zip(sources, whole):
        assert (tiled_source.xmin, tiled_source.ymin, tiled_source.xmax, tiled_source.ymax,
                tiled_source.area_pix) == (whole_source.xmin, whole_source.ymin,
                                           whole_source.xmax, whole_source.ymax, whole_source.area_pix)
        np.testing.assert_allclose((tiled_source.x_centroid, tiled_source.y_centroid, tiled_source.flux),
                                   (whole_source.x_centroid, whole_source.y_centroid, whole_source.flux), rtol=1e-5)
    positions = np.array([[s.x_centroid, s.y_centroid] for s in sources])
    distances = np.linalg.norm(positions[:, None] - np.asarray(truth)[None], axis=2)
    ii, jj = linear_sum_assignment(distances)
    assert len(ii) == 20 and np.all(distances[ii, jj] < 0.5)
    assert len(sources) == 20


@pytest.mark.parametrize("separation,expected", [(2, 1), (8, 2), (12, 2)])
def test_resolved_and_unresolved_blends(cpu_backends, separation, expected):
    yy, xx = np.mgrid[:96, :96]
    residual = (40 * np.exp(-((xx-42)**2 + (yy-48)**2) / 4.5)
                + 40 * np.exp(-((xx-42-separation)**2 + (yy-48)**2) / 4.5)).astype(np.float32)
    tile = Tile("pair", 0, 0, residual, (0, 0, 96, 96), 8)
    result = detection._measure_sources(tile, filtering.smooth(residual, 2), 2,
                                         DetectionParams(), profile=True)
    assert len(result["sources"]) == expected
    assert sum(s.area_pix for s in result["sources"]) == np.count_nonzero(result["labels"])
    assert sum(s.flux for s in result["sources"]) == pytest.approx(float(residual[result["labels"] > 0].sum(dtype=np.float64)), rel=1e-12)


def test_plateau_and_masked_label_pixels(cpu_backends):
    raw = np.zeros((23, 25), np.float32)
    raw[5:18, 5:20] = 8
    valid = np.ones_like(raw, dtype=bool)
    valid[9:12, 10:13] = False
    tile = Tile("plateau", 0, 0, raw, (0, 0, 23, 25), 8, valid)
    result = detection._measure_sources(tile, raw, 1, DetectionParams(), profile=True)
    assert len(result["sources"]) == 1
    assert np.all(result["labels"][~valid] == 0)


def test_native_segmentation_matches_reference_and_threads():
    native = pytest.importorskip("astro_pipeline._background_native")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        raw = rng.normal(4, 3, (29, 31)).astype(np.float32)
        valid = rng.random(raw.shape) > .1
        peaks = []
        expected = _segmentation.segment(raw, valid, 3, 5, True, 3, 1, centroid_peaks=peaks)
        rows = _segmentation.measure(raw, expected, peaks)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda threads: native.detect_sources(raw, raw, 3, 5, True, 3, 1,
                                                                          threads, valid_mask=valid, profile=True), (1, 2, 4)))
        for result in results:
            np.testing.assert_array_equal(result["labels"], expected)
            np.testing.assert_allclose(result["rows"], rows, rtol=1e-12, atol=1e-12)


def test_stale_native_and_invalid_detector_configuration(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(background, "import_module", lambda name: SimpleNamespace(API_VERSION=1))
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "cpp")
    with pytest.raises(ImportError, match="stale"):
        background.estimate_background(np.ones((8, 8), np.float32))
    monkeypatch.setenv("ASTRO_DETECTION_BACKEND", "other")
    tile = Tile("invalid", 0, 0, np.zeros((8, 8), np.float32), (0, 0, 8, 8), 0)
    with pytest.raises(ValueError, match="ASTRO_DETECTION_BACKEND"):
        detection._measure_sources(tile, tile.data, 1, DetectionParams())


@pytest.mark.parametrize("changes", [dict(refine_background=1), dict(deblend="yes"),
                                    dict(deblend_min_separation=0), dict(deblend_prominence_sigma=np.nan)])
def test_invalid_new_parameters(changes):
    with pytest.raises(ValueError):
        detection.process_image(ImageData("bad", np.zeros((8, 8), np.float32)), replace(DetectionParams(), **changes))


def test_masked_fits_worker_pool(cpu_backends, tmp_path):
    from astro_pipeline.parallel import run_batch
    yy, xx = np.mgrid[:128, :160]
    paths = []
    for seed in range(4):
        data = (100 + .02 * xx + .03 * yy + np.random.default_rng(seed).normal(0, 2, xx.shape)
                + 40 * np.exp(-((xx-88-seed)**2 + (yy-64)**2) / 4.5)).astype(np.float32)
        data[:, :32] = 0
        path = tmp_path / f"masked_{seed}.fits"
        fits.PrimaryHDU(data).writeto(path)
        paths.append(str(path))
    params = DetectionParams()
    expected = {Path(path).stem: detection.process_image(io_fits.read_fits(path, mask_zero_border=True), params, 32)
                for path in paths}
    assert all(len(sources) == 1 for sources in expected.values())
    for workers in (1, 2, 4):
        assert run_batch(paths, params, workers, 32, mask_zero_border=True) == expected


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("sigma", [.4, 1, 2])
def test_masked_filter_independent_spatial_reference(backend, sigma):
    from scipy.ndimage import gaussian_filter
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            pytest.skip(f"CUDA unavailable in this session: {error}")
        if count == 0:
            pytest.skip("no CUDA device in this session")
    data = np.random.default_rng(15).normal(10, 2, (71, 77)).astype(np.float32)
    valid = np.ones_like(data, dtype=bool)
    valid[:24, :35] = False
    valid[33:38, 25:41] = False
    radius = int(np.ceil(4*sigma))
    numerator = gaussian_filter(np.where(valid, data, 0).astype(np.float64), sigma, radius=radius, mode="reflect")
    denominator = gaussian_filter(valid.astype(np.float64), sigma, radius=radius, mode="reflect")
    expected = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=expected, where=valid & (denominator >= .5))
    actual = filtering.smooth(data, sigma, backend, valid_mask=valid)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert np.all(actual[~valid] == 0)
