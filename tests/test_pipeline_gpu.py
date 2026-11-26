"""Resident GPU checks against observed sky and independently injected sources."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from astro_pipeline import background, detection, io_fits
from astro_pipeline.interfaces import DetectionParams, ImageData, Tile
from astro_pipeline._pipeline_gpu import Workspace

ROOT = Path(__file__).resolve().parents[1]
PARAMS = DetectionParams(pipeline_backend="gpu")


@pytest.fixture(autouse=True)
def gpu():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device required")


@pytest.mark.parametrize("width", [1, 2, 17, 64, 4096])
def test_mesh_clipping_exact_medians_and_padding(width):
    rng = np.random.default_rng(379)
    rows = rng.normal(100, 2, (41, width))
    for i in range(len(rows)):
        retained = (i*127) % (width+1)
        rows[i, retained:] = np.inf
    rows[1] = 7
    if width >= 2:
        rows[2, :2] = [-1e12, 1e12]
        rows[3, -1] = 1e12
    if width >= 17:
        rows[4, :17] = np.nextafter(np.arange(17, dtype=np.float64), np.inf)
        rows[5, :17] = [0]*8 + [1]*8 + [1000]
    expected = np.array([background._clipped_stats(row[np.isfinite(row)])[0]
                         if np.isfinite(row).any() else np.nan for row in rows])
    work = Workspace(np.zeros((1, 1), np.float32), np.ones((1, 1), bool))
    actual = work._mesh_medians(work.cp.asarray(rows)).get()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("shape", [(1, 131), (131, 1), (17, 67)])
def test_reconstruction_matches_global_fixed_point(mode, shape):
    yy, xx = np.indices(shape)
    labels = np.ones(shape, np.int32)
    labels[(yy % 7 == 3) & (xx % 11 == 5)] = 0
    if min(shape) > 1:
        labels[shape[0]//3:2*shape[0]//3, shape[1]//3:shape[1]//2] = 2
    mask = np.ones(shape, np.float64) if mode == 0 else np.zeros(shape, np.float64)
    seed = np.zeros(shape, np.float64) if mode == 0 else np.full(shape, np.inf)
    seed[0, 0] = 1 if mode == 0 else 0
    seed[labels == 0] = 17
    expected = seed.copy()
    for _ in range(seed.size + 1):
        updated = expected.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                y = slice(max(0, -dy), min(shape[0], shape[0] - dy))
                x = slice(max(0, -dx), min(shape[1], shape[1] - dx))
                ny = slice(max(0, dy), min(shape[0], shape[0] + dy))
                nx = slice(max(0, dx), min(shape[1], shape[1] + dx))
                matching = (labels[y, x] != 0) & (labels[y, x] == labels[ny, nx])
                neighbor = expected[ny, nx]
                if mode == 2:
                    matching &= mask[y, x] == mask[ny, nx]
                    neighbor = neighbor + 1
                operation = np.maximum if mode == 0 else np.minimum
                updated[y, x] = np.where(matching, operation(updated[y, x], neighbor), updated[y, x])
        if mode != 2:
            operation = np.minimum if mode == 0 else np.maximum
            updated = np.where(labels != 0, operation(updated, mask), updated)
        if np.array_equal(updated, expected):
            break
        expected = updated
    else:
        pytest.fail("reference reconstruction did not converge")
    work = Workspace(np.zeros(shape, np.float32), np.ones(shape, bool))
    actual = work.reconstruct(work.cp.asarray(seed), work.cp.asarray(mask), work.cp.asarray(labels), mode)
    np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("sigma", [1., 2.])
def test_workspace_filter_preserves_faint_pixels_and_cached_spectra(masked, sigma):
    import math
    shape = (1037, 1049)
    pixels = np.random.default_rng(379).normal(1, .1, shape).astype(np.float32)
    pixels[114, 118] = 1e6
    valid = np.ones(shape, bool)
    if masked:
        valid[:, :21] = False
        valid[491:503, 751:763] = False
    work = Workspace(pixels, valid)
    offsets = np.arange(-math.ceil(4*sigma), math.ceil(4*sigma)+1, dtype=np.float64)
    weights = np.exp(-.5*(offsets/sigma)**2)
    weights /= weights.sum()
    kernel = np.outer(weights, weights).astype(np.float32).astype(np.float64)
    radius = math.ceil(4*sigma)
    coverage = np.pad(valid.astype(np.float64), radius, mode="symmetric")
    for shift in (0., 3.):
        data = pixels+shift
        actual = work.smooth(work.cp.asarray(data), sigma, work.valid if masked else None).get()
        padded = np.pad(np.where(valid, data, 0).astype(np.float64), radius, mode="symmetric")
        for y, x in ((0, 23), (28, 44), (487, 754), (520, 777), (1036, 1048)):
            window = np.s_[y:y+2*radius+1, x:x+2*radius+1]
            numerator = np.sum(padded[window]*kernel, dtype=np.float64)
            denominator = np.sum(coverage[window]*kernel, dtype=np.float64) if masked else 1.
            expected = numerator/denominator if valid[y, x] and denominator >= .5 else 0.
            assert actual[y, x] == pytest.approx(expected, rel=0, abs=5e-7)
        assert len(work._gaussian_spectra) == 1


@pytest.mark.parametrize("extended", [False, True])
def test_known_sky_noise_and_blank(extended):
    yy, xx = np.mgrid[:384, :384]
    sky = 100+.01*xx+.02*yy
    data = sky+np.random.default_rng(3).normal(0, 2, sky.shape)
    valid = np.ones(sky.shape, bool)
    valid[:64, :128] = False
    valid[80:100, 110:130] = False
    if extended:
        data += 1000*(np.hypot(xx-220, yy-190) < 62)
    data = data.astype(np.float32)
    data[~valid] = 1e6
    work = Workspace(data, valid)
    level, noise = work.background()
    level = level.get()
    assert np.sqrt(np.mean((level[valid]-sky[valid])**2)) < .5
    assert noise.item() == pytest.approx(2, rel=.05)
    assert np.all(level[~valid] == 0)
    if extended:
        assert work.source_mask[190, 220].item()
        assert abs(level[190, 220]-sky[190, 220]) < .5
    else:
        assert detection.process_image(ImageData("blank", data, {}, valid), PARAMS, 64) == []


@pytest.mark.parametrize("shape", [(1, 131), (131, 1), (133, 145)])
@pytest.mark.parametrize("scale", [1e-8, 1., 1e6])
def test_background_statistical_precision(shape, scale, monkeypatch):
    # Paired arithmetic check supplements the independent known-sky checks;
    # it protects the specified float64 residual/float32 background definition.
    monkeypatch.setenv("ASTRO_BACKGROUND_BACKEND", "numpy")
    monkeypatch.setenv("ASTRO_BACKGROUND_STATS_BACKEND", "cpu")
    data = ((100+np.random.default_rng(23).normal(0, 2, shape))*scale).astype(np.float32)
    valid = np.ones(shape, bool)
    data = data[:, ::-1]
    work = Workspace(data, valid)
    level, noise = work.background()
    expected, expected_noise = background.estimate_background(data)
    np.testing.assert_allclose(level.get(), expected, rtol=1e-7, atol=1e-12*scale)
    assert noise.item() == pytest.approx(expected_noise, rel=1e-5, abs=1e-14*scale)


@pytest.mark.parametrize("value", [0., 7., -4.])
def test_constant_sky_and_no_sources(value):
    data = np.full((65, 97), value, np.float32)
    work = Workspace(data, np.ones(data.shape, bool))
    level, noise = work.background()
    np.testing.assert_array_equal(level.get(), data)
    assert noise.item() == 0
    assert detection.process_image(ImageData("constant", data), PARAMS) == []


def test_odd_even_clipping_and_insufficient_sky():
    data = np.array([[0, 1, 2], [3, 4, 100]], np.float32)
    work = Workspace(data, np.ones(data.shape, bool))
    level, noise = work.background(refine=False)
    np.testing.assert_array_equal(level.get(), np.full(data.shape, 2, np.float32))
    assert noise.item() == 1.4826
    data = np.ones((128, 128), np.float32)
    valid = np.zeros(data.shape, bool)
    valid[::2] = True
    with pytest.raises(ValueError, match="insufficient background coverage"):
        Workspace(data, valid).background()


def test_medium_sources_against_injected_truth():
    # Preserve existing scientific accuracy on the supplied medium set. These
    # checks use injected positions/brightness, not the CPU catalog as truth.
    matched, detected, positions, flux_errors = 0, 0, [], []
    paths = sorted((ROOT/"gen_fits/medium").glob("*.fits"))
    assert len(paths) == 10
    for path in paths:
        truth = json.loads(path.with_suffix(".truth.json").read_text())["truth"]
        image = io_fits.read_fits(path)
        yy, xx = np.mgrid[:image.data.shape[0], :image.data.shape[1]]
        sky = 100+8*xx/(image.data.shape[1]-1)+5*yy/(image.data.shape[0]-1)
        work = Workspace(image.data, np.ones(image.data.shape, bool))
        level, noise = work.background()
        assert np.sqrt(np.mean((level.get()-sky)**2)) < .15
        assert noise.item() == pytest.approx(2, abs=.1)
        sources = detection.process_image(image, PARAMS, 64)
        detected += len(sources)
        measured = np.array([[s.x_centroid, s.y_centroid] for s in sources])
        expected = np.array([[s["x"], s["y"]] for s in truth])
        distances = np.linalg.norm(measured[:, None]-expected[None], axis=2)
        ii, jj = linear_sum_assignment(np.where(distances <= 3, distances, 1e6))
        for i, j in zip(ii, jj):
            if distances[i, j] <= 3:
                matched += 1
                positions.append(distances[i, j])
                flux_errors.append(abs(sources[i].flux-truth[j]["flux"])/truth[j]["flux"])
    assert matched >= 230 and detected-matched <= 2
    assert np.median(positions) < .1 and np.quantile(positions, .95) < .3
    assert np.median(flux_errors) < .15


@pytest.mark.parametrize("separation,expected", [(2, 1), (8, 2), (12, 2)])
def test_resolved_unresolved_stars_and_conserved_measurements(separation, expected):
    yy, xx = np.mgrid[:96, :96]
    residual = (40*np.exp(-((xx-42)**2+(yy-48)**2)/4.5)
                +40*np.exp(-((xx-42-separation)**2+(yy-48)**2)/4.5)).astype(np.float32)
    work = Workspace(residual, np.ones(residual.shape, bool))
    rows = work.measure(work.pixels, work.smooth(work.pixels, 2), work.cp.asarray(2.), PARAMS,
                        work.smooth(work.pixels, 1))
    assert len(rows) == expected
    centers = sorted(rows[:, 0])
    np.testing.assert_allclose(centers, [42+separation/2] if expected == 1 else [42, 42+separation], atol=.5, rtol=0)
    assert np.all(abs(rows[:, 1]-48) < .5)
    selected = work.labels.get() > 0
    assert sum(rows[:, 6]) == selected.sum()
    assert sum(rows[:, 7]) == pytest.approx(residual[selected].sum(dtype=np.float64), rel=1e-12)


def test_twenty_isolated_stars_with_real_coverage():
    footprint = io_fits.read_fits(str(ROOT/"fits/255wmos.fits"), mask_zero_border=True).valid_mask
    indices = np.arange(512)*1600//512
    valid = footprint[indices[:, None], indices[None, :]]
    distance = distance_transform_edt(valid)
    yy, xx = np.mgrid[:512, :512]
    rng = np.random.default_rng(37931)
    data = 100+8*xx/511+5*yy/511+rng.normal(0, 2, valid.shape)
    truth = []
    for _ in range(10000):
        if len(truth) == 20:
            break
        x, y = rng.uniform(30, 482, 2)
        if distance[int(y), int(x)] <= 25 or any(np.hypot(x-qx, y-qy) < 35 for qx, qy in truth):
            continue
        truth.append((x, y))
        data += 40*np.exp(-((xx-x)**2+(yy-y)**2)/4.5)
    assert len(truth) == 20
    data[~valid] = 0
    image = ImageData("isolated", data.astype(np.float32), {}, valid)
    tiled = detection.process_image(image, PARAMS, 64)
    whole = detection.process_image(image, PARAMS)
    assert len(tiled) == len(whole) == 20
    for sources in (tiled, whole):
        measured = np.array([[s.x_centroid, s.y_centroid] for s in sources])
        distances = np.linalg.norm(measured[:, None]-np.asarray(truth)[None], axis=2)
        ii, jj = linear_sum_assignment(distances)
        assert np.all(distances[ii, jj] < .5)
    for a, b in zip(tiled, whole):
        assert (a.xmin, a.ymin, a.xmax, a.ymax, a.area_pix) == (b.xmin, b.ymin, b.xmax, b.ymax, b.area_pix)
        np.testing.assert_allclose((a.x_centroid, a.y_centroid, a.flux), (b.x_centroid, b.y_centroid, b.flux), rtol=1e-5)


def test_concurrent_calls_and_process_workers(tmp_path):
    from astro_pipeline.parallel import run_batch
    yy, xx = np.mgrid[:128, :160]
    paths = []
    for seed in range(4):
        data = (100+.02*xx+.03*yy+np.random.default_rng(seed).normal(0, 2, xx.shape)
                +40*np.exp(-((xx-88-seed)**2+(yy-64)**2)/4.5)).astype(np.float32)
        data[:, :32] = 0
        path = tmp_path/f"star_{seed}.fits"
        fits.PrimaryHDU(data).writeto(path)
        paths.append(str(path))
    expected = run_batch(paths, PARAMS, 1, 32, mask_zero_border=True)
    for seed, sources in enumerate(expected.values()):
        assert len(sources) == 1
        assert abs(sources[0].x_centroid-88-seed) < .5
        assert abs(sources[0].y_centroid-64) < .5
    for workers in (2, 4):
        assert run_batch(paths, PARAMS, workers, 32, mask_zero_border=True) == expected
    def observe(path):
        return detection.process_image(io_fits.read_fits(path, mask_zero_border=True), PARAMS, 32)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(observe, paths))
    assert results == list(expected.values())
