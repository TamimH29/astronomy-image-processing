"""Person 1: compare detections against truth."""
from __future__ import annotations
import math
from statistics import median
from typing import Sequence
from .interfaces import Source


def score_catalog(sources: Sequence[Source], truth: Sequence[dict],
                  match_radius: float = 3.0) -> dict:
    """Match detections to truth within match_radius pixels (nearest unmatched
    neighbor). Return dict with: completeness, false_positive_rate,
    median_pos_err_pix, median_flux_err_pct.

    The false-positive rate is the fraction of detections left unmatched.
    Metrics whose denominator is zero, and median errors with no matches,
    are returned as None.
    """
    try:
        radius = float(match_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("match_radius must be a positive finite number") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("match_radius must be a positive finite number")

    source_values: list[tuple[float, float, float]] = []
    for index, source in enumerate(sources):
        try:
            x = float(source.x_centroid)
            y = float(source.y_centroid)
            flux = float(source.flux)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"source {index} has invalid centroid or flux") from exc
        if not all(math.isfinite(value) for value in (x, y, flux)):
            raise ValueError(f"source {index} has non-finite centroid or flux")
        source_values.append((x, y, flux))

    truth_values: list[tuple[float, float, float]] = []
    for index, row in enumerate(truth):
        try:
            x = float(row["x"])
            y = float(row["y"])
            flux = float(row["flux"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"truth row {index} must contain numeric x, y, and flux values"
            ) from exc
        if not all(math.isfinite(value) for value in (x, y, flux)):
            raise ValueError(f"truth row {index} has non-finite x, y, or flux")
        truth_values.append((x, y, flux))

    # Sort all candidate pairs by separation, then by input order to make ties
    # deterministic. Accepting only unmatched endpoints enforces one-to-one
    # matching, so neither a detection nor a truth object can be reused.
    candidates: list[tuple[float, int, int]] = []
    for source_index, (source_x, source_y, _) in enumerate(source_values):
        for truth_index, (truth_x, truth_y, _) in enumerate(truth_values):
            distance = math.hypot(source_x - truth_x, source_y - truth_y)
            if distance <= radius:
                candidates.append((distance, source_index, truth_index))
    candidates.sort()

    matched_sources: set[int] = set()
    matched_truth: set[int] = set()
    position_errors: list[float] = []
    flux_errors: list[float] = []
    for distance, source_index, truth_index in candidates:
        if source_index in matched_sources or truth_index in matched_truth:
            continue
        matched_sources.add(source_index)
        matched_truth.add(truth_index)
        position_errors.append(distance)

        measured_flux = source_values[source_index][2]
        true_flux = truth_values[truth_index][2]
        if true_flux != 0:
            flux_errors.append(
                abs(measured_flux - true_flux) / abs(true_flux) * 100.0
            )

    true_positive_count = len(matched_sources)
    false_positive_count = len(source_values) - true_positive_count
    return {
        "completeness": (
            true_positive_count / len(truth_values) if truth_values else None
        ),
        "false_positive_rate": (
            false_positive_count / len(source_values) if source_values else None
        ),
        "median_pos_err_pix": median(position_errors) if position_errors else None,
        "median_flux_err_pct": median(flux_errors) if flux_errors else None,
    }
