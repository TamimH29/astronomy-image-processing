import math

import pytest

from astro_pipeline.interfaces import Source
from astro_pipeline.validate import score_catalog


def _source(x: float, y: float, flux: float = 100.0, source_id: int = 1) -> Source:
    return Source(
        image_id="synthetic",
        source_id=source_id,
        x_centroid=x,
        y_centroid=y,
        xmin=0,
        ymin=0,
        xmax=1,
        ymax=1,
        area_pix=4,
        flux=flux,
    )


def test_score_catalog_reports_partial_recovery_and_median_errors():
    sources = [
        _source(1.0, 1.0, flux=110.0, source_id=1),
        _source(10.0, 10.0, flux=200.0, source_id=2),
        _source(40.0, 40.0, flux=500.0, source_id=3),
    ]
    truth = [
        {"x": 0.0, "y": 0.0, "flux": 100.0},
        {"x": 10.0, "y": 10.0, "flux": 200.0},
        {"x": 80.0, "y": 80.0, "flux": 700.0},
    ]

    scores = score_catalog(sources, truth, match_radius=2.0)

    assert scores["completeness"] == pytest.approx(2 / 3)
    assert scores["false_positive_rate"] == pytest.approx(1 / 3)
    assert scores["median_pos_err_pix"] == pytest.approx(math.sqrt(2) / 2)
    assert scores["median_flux_err_pct"] == pytest.approx(5.0)


def test_score_catalog_includes_matches_exactly_on_radius():
    source = _source(3.0, 4.0)
    truth = [{"x": 0.0, "y": 0.0, "flux": 100.0}]

    scores = score_catalog([source], truth, match_radius=5.0)

    assert scores["completeness"] == 1.0
    assert scores["false_positive_rate"] == 0.0
    assert scores["median_pos_err_pix"] == 5.0


def test_score_catalog_uses_each_truth_at_most_once_and_picks_nearest():
    sources = [
        _source(0.5, 0.0, source_id=1),
        _source(0.25, 0.0, source_id=2),
    ]
    truth = [{"x": 0.0, "y": 0.0, "flux": 100.0}]

    scores = score_catalog(sources, truth, match_radius=1.0)

    assert scores["completeness"] == 1.0
    assert scores["false_positive_rate"] == 0.5
    assert scores["median_pos_err_pix"] == 0.25


def test_score_catalog_returns_none_for_undefined_empty_metrics():
    empty = score_catalog([], [])
    only_truth = score_catalog([], [{"x": 1.0, "y": 2.0, "flux": 3.0}])
    only_detections = score_catalog([_source(1.0, 2.0)], [])

    assert empty == {
        "completeness": None,
        "false_positive_rate": None,
        "median_pos_err_pix": None,
        "median_flux_err_pct": None,
    }
    assert only_truth["completeness"] == 0.0
    assert only_truth["false_positive_rate"] is None
    assert only_truth["median_pos_err_pix"] is None
    assert only_detections["completeness"] is None
    assert only_detections["false_positive_rate"] == 1.0


def test_score_catalog_rejects_invalid_radius():
    with pytest.raises(ValueError, match="match_radius"):
        score_catalog([], [], match_radius=0.0)
