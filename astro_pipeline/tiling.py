"""Person 2: tiling and merging."""
from __future__ import annotations
from dataclasses import replace
from typing import Iterable, Sequence
import numpy as np
from .interfaces import ImageData, Tile, Source


def make_tiles(img: ImageData, tile_size: int, halo: int) -> list[Tile]:
    """Split into tile_size x tile_size cores, each padded by `halo` pixels
    (clipped at image borders). Cores must partition the image exactly:
    every pixel belongs to exactly one core.
    """
    data = img.data
    if data.ndim != 2:
        raise ValueError("expected a 2D image")
    if tile_size < 1 or halo < 0:
        raise ValueError("need tile_size >= 1 and halo >= 0")
    H, W = data.shape

    tiles = []
    for cy0 in range(0, H, tile_size):                 # row-major order
        cy1 = min(cy0 + tile_size, H)                  # clip last core
        for cx0 in range(0, W, tile_size):
            cx1 = min(cx0 + tile_size, W)

            y0, y1 = max(cy0 - halo, 0), min(cy1 + halo, H)   # clipped halo
            x0, x1 = max(cx0 - halo, 0), min(cx1 + halo, W)

            tiles.append(Tile(
                image_id=img.image_id,
                y0=y0, x0=x0,
                data=np.array(data[y0:y1, x0:x1], dtype=np.float32, order="C"),
                core=(cy0, cx0, cy1, cx1),
                halo=halo,
            ))
    return tiles

def _owns(tile: Tile, x: float, y: float) -> bool:
    """Ownership rule: centroid inside the tile's half-open core."""
    cy0, cx0, cy1, cx1 = tile.core
    return cy0 <= y < cy1 and cx0 <= x < cx1


def merge(image_id: str, per_tile: Iterable[Sequence[Source]],
          tiles: Sequence[Tile]) -> list[Source]:
    """Keep only sources whose centroid lies in the owning tile's core, then
    assign unique source_id values (sorted by y, then x for determinism).
    source_id starts at 1.
    """
    per_tile = list(per_tile)               # may be a generator
    if len(per_tile) != len(tiles):
        raise ValueError(
            f"got {len(per_tile)} source lists for {len(tiles)} tiles")

    kept = [s for tile, sources in zip(tiles, per_tile)
            for s in sources if _owns(tile, s.x_centroid, s.y_centroid)]

    kept.sort(key=lambda s: (s.y_centroid, s.x_centroid,
                             s.xmin, s.ymin, s.area_pix, s.flux))
    return [replace(s, image_id=image_id, source_id=i)
            for i, s in enumerate(kept, start=1)]