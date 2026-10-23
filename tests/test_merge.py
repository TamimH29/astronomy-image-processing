import random

import numpy as np
import pytest

from astro_pipeline.interfaces import ImageData, Source
from astro_pipeline.tiling import make_tiles, merge

H, W = 1000, 1300


def _tiles(halo=8):
    return make_tiles(ImageData("t", np.zeros((H, W), np.float32)), 512, halo)


def _src(x, y, **kw):
    d = dict(image_id="t", source_id=0, x_centroid=x, y_centroid=y,
             xmin=int(x) - 1, ymin=int(y) - 1, xmax=int(x) + 1,
             ymax=int(y) + 1, area_pix=9, flux=100.0)
    d.update(kw)
    return Source(**d)


def _visible(tile, srcs):
    """Sources whose centroid falls inside the tile's halo-inclusive window,
    i.e. everything a tile could see (no ownership filtering)."""
    h, w = tile.data.shape
    return [s for s in srcs
            if tile.y0 <= s.y_centroid < tile.y0 + h
            and tile.x0 <= s.x_centroid < tile.x0 + w]


def test_keeps_owned_drops_unowned():
    tiles = _tiles()                       # tile 0 core is (0, 0, 512, 512)
    owned, outside = _src(100, 100), _src(520, 100)
    out = merge("t", [[owned, outside]] + [[] for _ in tiles[1:]], tiles)
    assert [(s.x_centroid, s.y_centroid) for s in out] == [(100, 100)]


def test_boundary_is_half_open():
    tiles = _tiles()                       # tile 0 above, tile 3 below y=512
    s = _src(100, 512.0)
    empty = [[] for _ in tiles]
    upper = [list(e) for e in empty]; upper[0] = [s]
    lower = [list(e) for e in empty]; lower[3] = [s]
    assert merge("t", upper, tiles) == []              # 512.0 not in [0, 512)
    assert len(merge("t", lower, tiles)) == 1          # 512.0 in [512, 1000)

    s2 = _src(100, 511.999)
    upper[0] = [s2]
    assert len(merge("t", upper, tiles)) == 1

    # same rule for columns: x = 512.0 belongs to column 1
    right = [list(e) for e in empty]; right[1] = [_src(512.0, 100)]
    left = [list(e) for e in empty]; left[0] = [_src(512.0, 100)]
    assert len(merge("t", right, tiles)) == 1
    assert merge("t", left, tiles) == []


def test_duplicate_across_tiles_kept_once():
    tiles = _tiles()
    s = _src(100, 510.0)                   # visible to tile 0 and tile 3 halos
    per_tile = [_visible(t, [s]) for t in tiles]
    assert sum(len(p) for p in per_tile) >= 2
    assert len(merge("t", per_tile, tiles)) == 1


def test_every_source_kept_exactly_once():
    rng = random.Random(0)
    tiles = _tiles()
    srcs = [_src(rng.uniform(0, W - 1), rng.uniform(0, H - 1))
            for _ in range(200)]
    # deliberately on/next to borders (cores split at 512, 1024 / 512)
    for x, y in [(512.0, 100), (1024.0, 100), (100, 512.0), (511.999, 511.999),
                 (504.0, 504.0), (519.99, 519.99), (1023.99, 999.5), (0, 0)]:
        srcs.append(_src(x, y))
    per_tile = [_visible(t, srcs) for t in tiles]
    out = merge("t", per_tile, tiles)
    assert len(out) == len(srcs)
    assert (sorted((s.x_centroid, s.y_centroid) for s in out)
            == sorted((s.x_centroid, s.y_centroid) for s in srcs))


def test_ids_unique_start_at_one_and_sorted():
    tiles = _tiles()
    srcs = [_src(300, 700), _src(50, 20), _src(900, 20), _src(10, 700)]
    out = merge("t", [_visible(t, srcs) for t in tiles], tiles)
    assert [s.source_id for s in out] == [1, 2, 3, 4]
    assert [(s.y_centroid, s.x_centroid) for s in out] == [
        (20, 50), (20, 900), (700, 10), (700, 300)]


def test_empty():
    tiles = _tiles()
    assert merge("t", [[] for _ in tiles], tiles) == []
    assert merge("t", [], []) == []


def test_order_independent():
    rng = random.Random(1)
    tiles = _tiles()
    srcs = [_src(rng.uniform(0, W - 1), rng.uniform(0, H - 1))
            for _ in range(100)]
    per_tile = [_visible(t, srcs) for t in tiles]
    baseline = merge("t", per_tile, tiles)

    pairs = list(zip(tiles, per_tile))
    rng.shuffle(pairs)
    t2, p2 = zip(*pairs)
    assert merge("t", p2, t2) == baseline

    # order of sources within each tile's list must not matter either
    p3 = [list(reversed(p)) for p in per_tile]
    assert merge("t", p3, tiles) == baseline


def test_accepts_generator():
    tiles = _tiles()
    s = _src(100, 100)
    gen = ([s] if i == 0 else [] for i in range(len(tiles)))
    assert len(merge("t", gen, tiles)) == 1


def test_length_mismatch_raises():
    tiles = _tiles()
    with pytest.raises(ValueError):
        merge("t", [[]], tiles)


def test_sets_image_id_and_does_not_mutate_input():
    tiles = _tiles()
    s = _src(100, 100, image_id="wrong", source_id=99)
    out = merge("right", [[s]] + [[] for _ in tiles[1:]], tiles)
    assert out[0].image_id == "right" and out[0].source_id == 1
    assert s.image_id == "wrong" and s.source_id == 99