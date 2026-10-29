"""Person 2: run_batch tests (worker-count invariance and error handling)."""
import numpy as np
import pytest
from astropy.io import fits

from astro_pipeline import detection, io_fits, parallel
from astro_pipeline.interfaces import DetectionParams
from astro_pipeline.parallel import run_batch


def _write_image(path, seed, shape=(96, 128), n_sources=4):
    rng = np.random.default_rng(seed)
    h, w = shape
    img = 100 + 0.02 * np.arange(w)[None, :] + rng.normal(0, 1.0, shape)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_sources):
        x, y = rng.uniform(15, w - 15), rng.uniform(15, h - 15)
        img += 60 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
    fits.PrimaryHDU(img.astype(np.float32)).writeto(str(path), overwrite=True)


@pytest.fixture(scope="module")
def fits_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch")
    paths = []
    for i in range(6):
        p = d / f"img_{i:02d}.fits"
        _write_image(p, seed=i)
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("tile_size", [None, 40])
def test_worker_count_invariance(fits_files, tile_size):
    params = DetectionParams()
    baseline = run_batch(fits_files, params, 1, tile_size)
    # guard against a vacuous test: the images must contain detections
    assert sum(len(v) for v in baseline.values()) > 0
    for workers in (2, 4):
        assert run_batch(fits_files, params, workers, tile_size) == baseline


def test_matches_direct_process_image(fits_files):
    params = DetectionParams()
    result = run_batch(fits_files, params, 1, 40)
    for path in fits_files:
        image = io_fits.read_fits(path)
        assert result[image.image_id] == detection.process_image(
            image, params, 40)


def test_keys_sorted_and_input_order_irrelevant(fits_files):
    params = DetectionParams()
    forward = run_batch(fits_files, params, 2)
    backward = run_batch(list(reversed(fits_files)), params, 2)
    assert list(forward) == sorted(forward)
    assert forward == backward


def test_empty_batch():
    assert run_batch([], DetectionParams(), 4) == {}


@pytest.mark.parametrize("workers", [1, 2])
def test_bad_file_names_the_path(fits_files, tmp_path, workers):
    bad = tmp_path / "broken.fits"
    bad.write_bytes(b"this is not a FITS file")
    with pytest.raises(RuntimeError, match="broken.fits"):
        run_batch(fits_files[:2] + [str(bad)], DetectionParams(), workers)


@pytest.mark.parametrize("workers", [1, 2])
def test_duplicate_image_id_raises(tmp_path, workers):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    paths = []
    for sub in ("a", "b"):
        p = tmp_path / sub / "same.fits"
        _write_image(p, seed=1)
        paths.append(str(p))
    with pytest.raises(ValueError, match="duplicate image_id"):
        run_batch(paths, DetectionParams(), workers)


def test_duplicate_paths_raise(fits_files):
    with pytest.raises(ValueError, match="duplicate paths"):
        run_batch([fits_files[0], fits_files[0]], DetectionParams(), 1)


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_worker_count(fits_files, workers):
    with pytest.raises(ValueError, match="n_workers"):
        run_batch(fits_files, DetectionParams(), workers)


def test_thread_env_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    import os
    with parallel._single_threaded_env():
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in os.environ
