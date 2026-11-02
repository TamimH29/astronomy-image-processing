import csv

import numpy as np
from astropy.table import Table

from astro_pipeline.interfaces import CATALOG_COLUMNS, Source
from astro_pipeline.io_fits import write_catalog


def _source(image_id: str, source_id: int, offset: float = 0.0) -> Source:
    return Source(
        image_id=image_id,
        source_id=source_id,
        x_centroid=10.25 + offset,
        y_centroid=20.75 + offset,
        xmin=8,
        ymin=18,
        xmax=13,
        ymax=23,
        area_pix=24,
        flux=123.5 + offset,
    )


def test_write_catalog_csv_uses_schema_and_deterministic_sort(tmp_path):
    path = tmp_path / "catalog.csv"
    sources = [
        _source("field-b", 2, 2.0),
        _source("field-a", 2, 1.0),
        _source("field-a", 1),
    ]

    write_catalog(iter(sources), str(path))

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == CATALOG_COLUMNS
    assert [(row[0], int(row[1])) for row in rows[1:]] == [
        ("field-a", 1),
        ("field-a", 2),
        ("field-b", 2),
    ]
    assert float(rows[1][-1]) == 123.5


def test_write_catalog_empty_csv_still_writes_header(tmp_path):
    path = tmp_path / "empty.csv"

    write_catalog([], str(path))

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [CATALOG_COLUMNS]


def test_write_catalog_fits_round_trips_sorted_typed_rows(tmp_path):
    path = tmp_path / "catalog.fits"
    sources = [_source("field-b", 1), _source("field-a", 3, 3.0)]

    write_catalog(sources, str(path))
    table = Table.read(path, format="fits")

    assert table.colnames == CATALOG_COLUMNS
    assert list(zip(table["image_id"], table["source_id"])) == [
        ("field-a", 3),
        ("field-b", 1),
    ]
    assert np.issubdtype(table["source_id"].dtype, np.integer)
    assert np.issubdtype(table["x_centroid"].dtype, np.floating)
    assert np.issubdtype(table["flux"].dtype, np.floating)
    assert np.isclose(table["flux"][0], 126.5)


def test_write_catalog_empty_fits_preserves_schema(tmp_path):
    path = tmp_path / "empty.fits"

    write_catalog([], str(path))
    table = Table.read(path, format="fits")

    assert table.colnames == CATALOG_COLUMNS
    assert len(table) == 0
