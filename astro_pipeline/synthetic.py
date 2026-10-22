"""Person 1: synthetic test images with known truth."""
from __future__ import annotations
import numpy as np


def make_synthetic_image(shape: tuple[int, int], n_sources: int, seed: int
                         ) -> tuple[np.ndarray, list[dict]]:
    """Return (image, truth). truth is a list of dicts with keys
    x, y, sigma, flux (full-image coordinates, same conventions as
    interfaces.py). Image = Gaussian sources + smooth background gradient +
    Gaussian read noise (+ optional Poisson noise). Deterministic in `seed`.
    """
    raise NotImplementedError


def save_synthetic_set(out_dir: str, name: str, shape: tuple[int, int],
                       n_images: int, n_sources: int, seed: int) -> None:
    """Write n_images FITS files plus one truth JSON per image into out_dir.
    Planned sets: tiny (unit tests), medium (correctness), large (benchmark).
    """
    raise NotImplementedError
