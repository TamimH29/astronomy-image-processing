"""Person 1: FITS ingestion and catalog output."""
from __future__ import annotations
from typing import Iterable
from .interfaces import ImageData, Source, CATALOG_COLUMNS  # noqa: F401


def read_fits(path: str) -> ImageData:
    """Read the first image HDU (handle .fits.gz / .fits.bz2 and extension
    HDUs). Return float32 data. Replace NaN/inf with the image median.
    image_id = file stem.

    TODO: astropy.io.fits.open; pick first HDU with 2D data; keep header dict.
    """
    raise NotImplementedError


def write_catalog(sources: Iterable[Source], path: str) -> None:
    """Write CATALOG_COLUMNS to CSV (or a FITS table if path ends in .fits).
    Rows sorted by (image_id, source_id) so output is deterministic.
    """
    raise NotImplementedError
