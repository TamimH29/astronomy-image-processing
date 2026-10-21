"""Person 2: tiling and merging."""
from __future__ import annotations
from typing import Iterable, Sequence
from .interfaces import ImageData, Tile, Source


def make_tiles(img: ImageData, tile_size: int, halo: int) -> list[Tile]:
    """Split into tile_size x tile_size cores, each padded by `halo` pixels
    (clipped at image borders). Cores must partition the image exactly:
    every pixel belongs to exactly one core.
    """
    raise NotImplementedError


def merge(image_id: str, per_tile: Iterable[Sequence[Source]],
          tiles: Sequence[Tile]) -> list[Source]:
    """Keep only sources whose centroid lies in the owning tile's core, then
    assign unique source_id values (sorted by y, then x for determinism).
    Test: tiled result must equal whole-image result.
    """
    raise NotImplementedError
