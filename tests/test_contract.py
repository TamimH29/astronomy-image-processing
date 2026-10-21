import pytest
from astro_pipeline.interfaces import CATALOG_COLUMNS, Source, DetectionParams


def test_as_row_matches_schema():
    s = Source("img", 1, 10.5, 20.5, 8, 18, 13, 23, 40, 123.4)
    row = s.as_row()
    assert len(row) == len(CATALOG_COLUMNS)
    assert row[0] == "img" and row[-1] == 123.4


def test_default_params():
    p = DetectionParams()
    assert p.backend == "cpu" and p.threshold_sigma == 3.0


# Placeholders: replace with real tests as each piece lands.
@pytest.mark.skip(reason="Person 2: tiled result equals whole-image result")
def test_tiled_equals_whole_image():
    ...


@pytest.mark.skip(reason="Person 2: 1, 2, 4 workers give identical catalogs")
def test_worker_count_invariance():
    ...


@pytest.mark.skip(reason="Person 3: GPU smooth matches CPU within tolerance")
def test_gpu_matches_cpu():
    ...


@pytest.mark.skip(reason="Person 1: synthetic recovery meets accuracy targets")
def test_synthetic_recovery():
    ...
