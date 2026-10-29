"""Person 1: FITS ingestion and catalog output."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from astropy.io import fits
from .interfaces import ImageData, Source, CATALOG_COLUMNS  # noqa: F401


def read_fits(path: str) -> ImageData:
    """Read the first image HDU (handle .fits.gz / .fits.bz2 and extension
    HDUs). Return float32 data. Replace NaN/inf with the image median.
    image_id = file stem.

    The image_id is the filename stem with optional FITS compression and
    extension suffixes removed. Raises ValueError if the file has no usable
    2D image HDU or contains no finite image pixels.
    """
    fits_path = Path(path)
    image_id_name = fits_path.name
    for compression_suffix in (".gz", ".bz2"):
        if image_id_name.lower().endswith(compression_suffix):
            image_id_name = image_id_name[:-len(compression_suffix)]
            break
    image_id = Path(image_id_name).stem

    image = None
    header: dict = {}
    with fits.open(fits_path, memmap=False) as hdus:
        for hdu in hdus:
            if not isinstance(
                hdu,
                (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU),
            ):
                continue
            hdu_data = hdu.data
            if hdu_data is None or hdu_data.ndim != 2 or hdu_data.size == 0:
                continue

            # Copy out of the HDU before its file is closed. Astropy applies
            # FITS scaling (BSCALE/BZERO) when accessing hdu.data.
            image = np.array(hdu_data, dtype=np.float32, copy=True)
            header = dict(hdu.header)
            break

    if image is None:
        raise ValueError(f"No non-empty 2D image HDU found in {fits_path}")

    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError(f"Image HDU in {fits_path} contains no finite pixels")
    if not finite.all():
        image[~finite] = np.median(image[finite])

    return ImageData(image_id=image_id, data=image, header=header)


def write_catalog(sources: Iterable[Source], path: str) -> None:
    """Write CATALOG_COLUMNS to CSV (or a FITS table if path ends in .fits).
    Rows sorted by (image_id, source_id) so output is deterministic.
    """
    raise NotImplementedError
