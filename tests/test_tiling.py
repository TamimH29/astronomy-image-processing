import numpy as np
import pytest
import math
from astro_pipeline.interfaces import ImageData
from astro_pipeline.tiling import make_tiles
from scipy.ndimage import gaussian_filter

def _img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return ImageData("t", rng.random((h, w), dtype=np.float32))

SHAPES = [(1000, 1300), (512, 512), (100, 70), (513, 1025), (1, 1)]
SETTINGS = [(512, 8), (512, 0), (64, 16), (1024, 8), (7, 3)]

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tile_size,halo", SETTINGS)
def test_cores_partition_image(shape, tile_size, halo):
    img = _img(*shape)
    count = np.zeros(shape, int)
    for t in make_tiles(img, tile_size, halo):
        cy0, cx0, cy1, cx1 = t.core
        count[cy0:cy1, cx0:cx1] += 1
    assert (count == 1).all()

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tile_size,halo", SETTINGS)
def test_reassemble_cores_matches_image(shape, tile_size, halo):
    img = _img(*shape)
    out = np.full(shape, np.nan, np.float32)
    for t in make_tiles(img, tile_size, halo):
        cy0, cx0, cy1, cx1 = t.core
        out[cy0:cy1, cx0:cx1] = t.data[cy0 - t.y0:cy1 - t.y0,
                                       cx0 - t.x0:cx1 - t.x0]
    np.testing.assert_array_equal(out, img.data)

@pytest.mark.parametrize("tile_size,halo", SETTINGS)
def test_halo_window_is_correct(tile_size, halo):
    img = _img(1000, 1300)
    H, W = img.data.shape
    for t in make_tiles(img, tile_size, halo):
        cy0, cx0, cy1, cx1 = t.core
        h, w = t.data.shape
        # window is the core expanded by halo, clipped to the image
        assert t.y0 == max(cy0 - halo, 0)
        assert t.x0 == max(cx0 - halo, 0)
        assert t.y0 + h == min(cy1 + halo, H)
        assert t.x0 + w == min(cx1 + halo, W)
        # pixels are the right pixels, not just the right shape
        np.testing.assert_array_equal(
            t.data, img.data[t.y0:t.y0 + h, t.x0:t.x0 + w])

def test_tile_properties():
    img = _img(1000, 1300)
    for t in make_tiles(img, 512, 8):
        assert t.data.dtype == np.float32
        assert t.data.flags["C_CONTIGUOUS"]
        assert t.halo == 8
        assert t.image_id == "t"

def test_tiles_are_copies():
    img = _img(100, 100)
    original = img.data.copy()
    for t in make_tiles(img, 50, 4):
        t.data[:] = -1.0
    np.testing.assert_array_equal(img.data, original)

def test_row_major_order_and_count():
    tiles = make_tiles(_img(1000, 1300), 512, 8)
    assert len(tiles) == 6
    origins = [(t.core[0], t.core[1]) for t in tiles]
    assert origins == sorted(origins)       # row-major == sorted by (y, x)

def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_tiles(_img(10, 10), 0, 1)
    with pytest.raises(ValueError):
        make_tiles(_img(10, 10), 4, -1)
    with pytest.raises(ValueError):
        make_tiles(ImageData("t", np.zeros(10, np.float32)), 4, 1)

@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.5])
@pytest.mark.parametrize("shape,tile_size",
                         [((200, 300), 64), ((130, 97), 50), ((64, 64), 128)])
def test_halo_gives_exact_filtering(sigma, shape, tile_size):
    img = _img(*shape)
    halo = math.ceil(4 * sigma)
    whole = gaussian_filter(img.data, sigma, mode="reflect", truncate=4.0)
    for t in make_tiles(img, tile_size, halo):
        local = gaussian_filter(t.data, sigma, mode="reflect", truncate=4.0)
        cy0, cx0, cy1, cx1 = t.core
        got = local[cy0 - t.y0:cy1 - t.y0, cx0 - t.x0:cx1 - t.x0]
        np.testing.assert_allclose(got, whole[cy0:cy1, cx0:cx1],
                                   rtol=1e-5, atol=1e-6)


def test_too_small_halo_is_detected():
    # Sanity check that the test above can fail: with no halo, tile
    # interiors differ from the whole-image result.
    img = _img(200, 300)
    whole = gaussian_filter(img.data, 2.0, mode="reflect", truncate=4.0)
    worst = 0.0
    for t in make_tiles(img, 64, 0):
        local = gaussian_filter(t.data, 2.0, mode="reflect", truncate=4.0)
        cy0, cx0, cy1, cx1 = t.core
        diff = np.abs(local[cy0 - t.y0:cy1 - t.y0, cx0 - t.x0:cx1 - t.x0]
                      - whole[cy0:cy1, cx0:cx1]).max()
        worst = max(worst, diff)
    assert worst > 1e-3