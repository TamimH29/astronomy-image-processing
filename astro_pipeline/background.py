"""Person 3: background and noise estimation."""
from __future__ import annotations
import numpy as np


def estimate_background(img: np.ndarray, box: int = 64
                        ) -> tuple[np.ndarray, float]:
    """Sigma-clipped median on a coarse grid of `box`-pixel cells, interpolated
    to full size. Return (background_map, noise_sigma). Runs on the WHOLE
    image, before tiling (see conventions in interfaces.py).
    """
    raise NotImplementedError
