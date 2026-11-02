"""Person 1: FITS ingestion and catalog output."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .interfaces import ImageData, Source, CATALOG_COLUMNS  # noqa: F401


def read_fits(path: str, *, mask_zero_border: bool = False) -> ImageData:
    """Read the first image HDU (handle .fits.gz / .fits.bz2 and extension
    HDUs). Return float32 data and preserve invalidity before filling NaN/inf.
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
    unsupported_shapes = []
    with fits.open(fits_path, memmap=False) as hdus:
        for hdu in hdus:
            if not isinstance(
                hdu,
                (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU),
            ):
                continue
            hdu_data = hdu.data
            if hdu_data is None or hdu_data.size == 0:
                continue
            if hdu_data.ndim != 2:
                unsupported_shapes.append(hdu_data.shape)
                continue

            # Copy out of the HDU before its file is closed. Astropy applies
            # FITS scaling (BSCALE/BZERO) when accessing hdu.data.
            image = np.array(hdu_data, dtype=np.float32, copy=True)
            # Repair fixable metadata in memory, retaining Astropy's warnings.
            # Opening readonly and never flushing leaves the source file intact.
            hdu.verify("fix")
            header = dict(hdu.header)
            break

    if image is None:
        detail = f"; unsupported image shapes {unsupported_shapes}; cubes require explicit 2D selection" if unsupported_shapes else ""
        raise ValueError(f"No non-empty 2D image HDU found in {fits_path}{detail}")

    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError(f"Image HDU in {fits_path} contains no finite pixels")
    if mask_zero_border:
        from scipy.ndimage import binary_propagation
        zero = image == 0
        seed = np.zeros_like(zero)
        seed[0], seed[-1] = zero[0], zero[-1]
        seed[:, 0], seed[:, -1] = zero[:, 0], zero[:, -1]
        # Default cross connectivity is four-neighbor connectivity.
        finite &= ~binary_propagation(seed, mask=zero)
    if not finite.any():
        raise ValueError(f"Image HDU in {fits_path} contains no usable observations")
    image[~finite] = 0

    return ImageData(image_id=image_id, data=image, header=header,
                     valid_mask=None if finite.all() else finite)


def write_catalog(sources: Iterable[Source], path: str) -> None:
    """Write CATALOG_COLUMNS to CSV (or a FITS table if path ends in .fits).
    Rows sorted by (image_id, source_id) so output is deterministic.
    """
    output_path = Path(path)
    ordered_sources = sorted(
        sources,
        key=lambda source: (source.image_id, source.source_id),
    )
    rows = [source.as_row() for source in ordered_sources]

    if output_path.suffix.lower() == ".fits":
        image_id_width = max(
            (len(source.image_id) for source in ordered_sources),
            default=1,
        )
        column_dtypes = [
            f"U{image_id_width}",
            np.int64,
            np.float64,
            np.float64,
            np.int64,
            np.int64,
            np.int64,
            np.int64,
            np.int64,
            np.float64,
        ]
        table = Table(rows=rows, names=CATALOG_COLUMNS, dtype=column_dtypes)
        table.write(output_path, format="fits", overwrite=True)
        return

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CATALOG_COLUMNS)
        writer.writerows(rows)
