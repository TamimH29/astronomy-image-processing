"""Person 1: synthetic test images with known truth."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _write_fits_image(path: Path, image: np.ndarray, seed: int,
                      n_sources: int) -> None:
    """Write a 2D float32 primary image HDU without optional dependencies."""
    height, width = image.shape

    def card(key: str, value: str, comment: str = "") -> bytes:
        text = f"{key:<8}= {value:>20}"
        if comment:
            text += f" / {comment}"
        return text[:80].ljust(80).encode("ascii")

    cards = [
        card("SIMPLE", "T", "conforms to FITS standard"),
        card("BITPIX", "-32", "32-bit floating point pixels"),
        card("NAXIS", "2", "number of data axes"),
        card("NAXIS1", str(width), "image width"),
        card("NAXIS2", str(height), "image height"),
        card("EXTEND", "T"),
        card("RNGSEED", str(seed), "synthetic image random seed"),
        card("NSOURCE", str(n_sources), "number of injected sources"),
        b"END".ljust(80),
    ]
    header = b"".join(cards)
    header += b" " * ((-len(header)) % 2880)

    # FITS stores the first axis (x) as the fastest-changing axis and uses
    # big-endian bytes. A C-order (rows, columns) NumPy array has that layout.
    data = np.asarray(image, dtype=">f4").tobytes(order="C")
    data += b"\0" * ((-len(data)) % 2880)
    path.write_bytes(header + data)


def make_synthetic_image(shape: tuple[int, int], n_sources: int, seed: int
                         ) -> tuple[np.ndarray, list[dict]]:
    """Return (image, truth). truth is a list of dicts with keys
    x, y, sigma, flux (full-image coordinates, same conventions as
    interfaces.py). Image = Gaussian sources + smooth background gradient +
    Gaussian read noise (+ optional Poisson noise). Deterministic in `seed`.
    """
    if len(shape) != 2:
        raise ValueError("shape must be (rows, columns)")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if n_sources < 0:
        raise ValueError("n_sources must be non-negative")

    rng = np.random.default_rng(seed)

    # A gently varying sky level plus read noise gives the detector a realistic
    # background to estimate without making the test image hard to reproduce.
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    x_fraction = x_grid / max(width - 1, 1)
    y_fraction = y_grid / max(height - 1, 1)
    background = 100.0 + 8.0 * x_fraction + 5.0 * y_fraction
    image = background + rng.normal(0.0, 2.0, size=shape)

    truth: list[dict] = []
    for source_id in range(n_sources):
        sigma = float(rng.uniform(1.0, 2.5))
        # Keep each source at least four sigma from the image edge so its
        # integrated flux is represented in the image rather than clipped.
        margin = 4.0 * sigma
        if width <= 2.0 * margin or height <= 2.0 * margin:
            raise ValueError(
                "image is too small for the generated source profile; "
                "use dimensions larger than 8 * the maximum source sigma"
            )
        x = float(rng.uniform(margin, width - margin))
        y = float(rng.uniform(margin, height - margin))
        flux = float(rng.uniform(250.0, 1200.0))

        # Normalize the sampled Gaussian so `flux` is the discrete sum of the
        # injected source, matching the background-subtracted catalog flux
        # convention. Four-sigma support keeps work local and captures >99.9%
        # of the profile; normalization compensates for the finite grid.
        radius = int(np.ceil(4.0 * sigma))
        x0, x1 = max(0, int(np.floor(x)) - radius), min(width, int(np.floor(x)) + radius + 1)
        y0, y1 = max(0, int(np.floor(y)) - radius), min(height, int(np.floor(y)) + radius + 1)
        dx = x_grid[y0:y1, x0:x1] - x
        dy = y_grid[y0:y1, x0:x1] - y
        profile = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        image[y0:y1, x0:x1] += profile * (flux / profile.sum())

        truth.append({"x": x, "y": y, "sigma": sigma, "flux": flux})

    return image.astype(np.float32), truth


def save_synthetic_set(out_dir: str, name: str, shape: tuple[int, int],
                       n_images: int, n_sources: int, seed: int) -> None:
    """Write n_images FITS files plus one truth JSON per image into out_dir.
    Planned sets: tiny (unit tests), medium (correctness), large (benchmark).
    """
    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty filename stem")
    if n_images <= 0:
        raise ValueError("n_images must be positive")
    if n_sources < 0:
        raise ValueError("n_sources must be non-negative")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_index in range(n_images):
        image_seed = seed + image_index
        image, truth = make_synthetic_image(shape, n_sources, image_seed)
        stem = f"{name}_{image_index:03d}"
        fits_path = output_dir / f"{stem}.fits"
        truth_path = output_dir / f"{stem}.truth.json"

        _write_fits_image(fits_path, image, image_seed, len(truth))
        truth_path.write_text(
            json.dumps(
                {
                    "image": fits_path.name,
                    "shape": list(shape),
                    "seed": image_seed,
                    "n_sources": len(truth),
                    "truth": truth,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def make_morphology_image(
    shape: tuple[int, int],
    seed: int,
    n_stars: int = 26,
    n_galaxies: int = 3,
    n_nebulae: int = 2,
    n_irregular: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """Create an image with stars, spiral galaxies, nebulae, and irregulars.

    Returned truth rows include the object kind, full-image centroid, and
    integrated injected flux. Galaxy/nebula/irregular profiles are procedural
    stress-test shapes, not physical forward models.
    """
    if len(shape) != 2:
        raise ValueError("shape must be (rows, columns)")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    counts = {
        "star": n_stars,
        "galaxy": n_galaxies,
        "nebula": n_nebulae,
        "irregular": n_irregular,
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError("object counts must be non-negative")

    margin = 90
    min_separation = 180.0
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("morphology images must be larger than 180x180 pixels")

    rng = np.random.default_rng(seed)
    yy = np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1)
    xx = np.arange(width, dtype=np.float32)[None, :] / max(width - 1, 1)
    image = (100.0 + 8.0 * xx + 5.0 * yy).astype(np.float32)
    image += rng.normal(0.0, 2.0, size=shape).astype(np.float32)

    # Spread source centers across the field. This leaves room for the broad
    # galaxy and nebula profiles while keeping the injected objects distinct.
    centers: list[tuple[float, float]] = []
    total_objects = sum(counts.values())
    attempts = 0
    while len(centers) < total_objects:
        attempts += 1
        if attempts > max(10000, total_objects * 1000):
            raise ValueError("could not place all objects with the chosen separation")
        x = float(rng.uniform(margin, width - margin))
        y = float(rng.uniform(margin, height - margin))
        if all((x - ox) ** 2 + (y - oy) ** 2 >= min_separation**2
               for ox, oy in centers):
            centers.append((x, y))

    object_specs = [
        kind
        for kind, count in counts.items()
        for _ in range(count)
    ]
    rng.shuffle(object_specs)
    truth: list[dict] = []

    for source_id, (kind, (x, y)) in enumerate(zip(object_specs, centers), start=1):
        params: dict = {}
        shift_x = x - np.floor(x)
        shift_y = y - np.floor(y)
        if kind == "star":
            sigma = float(rng.uniform(1.3, 2.4))
            radius = int(np.ceil(4.5 * sigma))
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            dx, dy = np.meshgrid(axis - shift_x, axis - shift_y)
            profile = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            flux = float(rng.uniform(400.0, 2200.0))
            params = {"sigma": sigma}
        elif kind == "galaxy":
            scale = float(rng.uniform(9.0, 16.0))
            axis_ratio = float(rng.uniform(0.45, 0.85))
            angle = float(rng.uniform(0.0, np.pi))
            radius = int(np.ceil(4.5 * scale))
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            dx, dy = np.meshgrid(axis - shift_x, axis - shift_y)
            major = dx * np.cos(angle) + dy * np.sin(angle)
            minor = -dx * np.sin(angle) + dy * np.cos(angle)
            elliptical_r = np.sqrt(major**2 + (minor / axis_ratio) ** 2)
            theta = np.arctan2(minor / axis_ratio, major)
            arms = 1.0 + 0.32 * np.cos(2.0 * (theta - 0.20 * elliptical_r / scale))
            disk = np.exp(-elliptical_r / scale) * arms
            bulge = 0.28 * np.exp(-elliptical_r**2 / (2.0 * (0.22 * scale) ** 2))
            profile = np.where(elliptical_r <= 4.5 * scale, disk + bulge, 0.0)
            flux = float(rng.uniform(7000.0, 18000.0))
            params = {
                "scale_radius_pix": scale,
                "axis_ratio": axis_ratio,
                "orientation_deg": float(np.degrees(angle)),
                "spiral_arms": 2,
            }
        elif kind == "nebula":
            scale_x = float(rng.uniform(24.0, 34.0))
            scale_y = float(rng.uniform(17.0, 27.0))
            angle = float(rng.uniform(0.0, np.pi))
            radius = 78
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            dx, dy = np.meshgrid(axis - shift_x, axis - shift_y)
            major = dx * np.cos(angle) + dy * np.sin(angle)
            minor = -dx * np.sin(angle) + dy * np.cos(angle)
            profile = 0.22 * np.exp(-0.5 * ((major / scale_x) ** 2 + (minor / scale_y) ** 2))
            # A broken, lumpy shell with several bright knots evokes the
            # diffuse cloud and dark-lane structure in the M42 reference.
            radial = np.sqrt((major / (0.72 * scale_x)) ** 2 + (minor / (0.72 * scale_y)) ** 2)
            theta = np.arctan2(minor / scale_y, major / scale_x)
            shell = np.exp(-0.5 * ((radial - 1.0) / 0.18) ** 2)
            profile += 0.20 * shell * (0.65 + 0.35 * np.cos(3.0 * theta + 0.7))
            knot_count = int(rng.integers(4, 8))
            for _ in range(knot_count):
                knot_x = float(rng.uniform(-0.7, 0.7) * scale_x)
                knot_y = float(rng.uniform(-0.7, 0.7) * scale_y)
                knot_sigma = float(rng.uniform(5.0, 11.0))
                knot_flux = float(rng.uniform(0.3, 1.0))
                profile += knot_flux * np.exp(
                    -((major - knot_x) ** 2 + (minor - knot_y) ** 2)
                    / (2.0 * knot_sigma**2)
                )
            # A soft dust lane adds a dark, curved break through the cloud.
            lane = np.exp(-0.5 * ((minor - 0.18 * major + 0.003 * major**2) / 3.2) ** 2)
            profile *= 1.0 - 0.38 * lane * np.exp(-0.5 * (major / 35.0) ** 2)
            flux = float(rng.uniform(18000.0, 42000.0))
            params = {
                "major_scale_pix": scale_x,
                "minor_scale_pix": scale_y,
                "orientation_deg": float(np.degrees(angle)),
                "subclumps": knot_count,
                "dust_lane": True,
            }
        else:  # irregular
            radius = 72
            axis = np.arange(-radius, radius + 1, dtype=np.float32)
            dx, dy = np.meshgrid(axis - shift_x, axis - shift_y)
            profile = 0.16 * np.exp(-0.5 * ((dx / 34.0) ** 2 + (dy / 25.0) ** 2))
            knot_count = int(rng.integers(4, 9))
            for _ in range(knot_count):
                knot_x = float(rng.uniform(-28.0, 28.0))
                knot_y = float(rng.uniform(-24.0, 24.0))
                sigma_x = float(rng.uniform(4.0, 10.0))
                sigma_y = float(rng.uniform(4.0, 13.0))
                knot_angle = float(rng.uniform(0.0, np.pi))
                u = dx * np.cos(knot_angle) + dy * np.sin(knot_angle) - knot_x
                v = -dx * np.sin(knot_angle) + dy * np.cos(knot_angle) - knot_y
                profile += float(rng.uniform(0.35, 1.0)) * np.exp(
                    -0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2)
                )
            # Add faint bridges between knots so each irregular remains a
            # connected, asymmetric object rather than a collection of stars.
            bridge = np.exp(-0.5 * ((dy - 0.35 * dx + 0.0008 * dx**2) / 5.5) ** 2)
            profile += 0.08 * bridge * np.exp(-0.5 * (dx / 42.0) ** 2)
            flux = float(rng.uniform(6000.0, 16000.0))
            params = {"subclumps": knot_count, "profile": "asymmetric_clumps_and_filament"}

        # Center each finite profile at the recorded sub-pixel centroid and
        # normalize its pixel sum to the truth flux.
        size = profile.shape[0]
        profile_sum = float(profile.sum())
        profile *= flux / profile_sum

        local_axis = np.arange(-radius, radius + 1, dtype=np.float32)
        centroid_dx = float((profile * (local_axis[None, :] - shift_x)).sum() / flux)
        centroid_dy = float((profile * (local_axis[:, None] - shift_y)).sum() / flux)

        center_col, center_row = int(np.floor(x)), int(np.floor(y))
        x0, y0 = center_col - radius, center_row - radius
        x1, y1 = x0 + size, y0 + size
        image[y0:y1, x0:x1] += profile.astype(np.float32)
        truth.append({
            "source_id": source_id,
            "kind": kind,
            "x": x + centroid_dx,
            "y": y + centroid_dy,
            "anchor_x": x,
            "anchor_y": y,
            "flux": flux,
            "bbox_x_min": x0,
            "bbox_y_min": y0,
            "bbox_x_max": x1 - 1,
            "bbox_y_max": y1 - 1,
            **params,
        })

    return image.astype(np.float32), truth


def save_morphology_set(
    out_dir: str,
    name: str,
    shape: tuple[int, int],
    n_images: int,
    seed: int,
    n_stars: int = 26,
    n_galaxies: int = 3,
    n_nebulae: int = 2,
    n_irregular: int = 3,
) -> None:
    """Write morphology-rich FITS images and per-image truth JSON files."""
    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty filename stem")
    if n_images <= 0:
        raise ValueError("n_images must be positive")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_index in range(n_images):
        image_seed = seed + image_index
        image, truth = make_morphology_image(
            shape,
            image_seed,
            n_stars=n_stars,
            n_galaxies=n_galaxies,
            n_nebulae=n_nebulae,
            n_irregular=n_irregular,
        )
        stem = f"{name}_{image_index:03d}"
        fits_path = output_dir / f"{stem}.fits"
        truth_path = output_dir / f"{stem}.truth.json"
        _write_fits_image(fits_path, image, image_seed, len(truth))
        truth_path.write_text(
            json.dumps(
                {
                    "image": fits_path.name,
                    "shape": list(shape),
                    "seed": image_seed,
                    "n_sources": len(truth),
                    "counts_by_kind": {
                        "star": n_stars,
                        "galaxy": n_galaxies,
                        "nebula": n_nebulae,
                        "irregular": n_irregular,
                    },
                    "truth": truth,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
