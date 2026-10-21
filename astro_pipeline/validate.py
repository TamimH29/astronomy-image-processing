"""Person 1: compare detections against truth."""
from __future__ import annotations
from typing import Sequence
from .interfaces import Source


def score_catalog(sources: Sequence[Source], truth: Sequence[dict],
                  match_radius: float = 3.0) -> dict:
    """Match detections to truth within match_radius pixels (nearest unmatched
    neighbor). Return dict with: completeness, false_positive_rate,
    median_pos_err_pix, median_flux_err_pct.
    """
    raise NotImplementedError
