# Morphology-rich large synthetic FITS set

This stress-test set contains four 2048 x 2048 float32 FITS images. Each image has 34 injected objects: 26 point-like stars, 3 spiral-like galaxies, 2 diffuse nebulae, and 3 clumpy irregular objects. Seeds run from 74001 through 74004. Each FITS image has a matching `.truth.json` file with object kind, flux, centroid, bounding box, and morphology parameters.

The shapes are procedural approximations for source-detection stress testing, not calibrated astrophysical simulations. The galaxy model uses an inclined disk, bulge, and two-arm modulation; nebulae combine diffuse clouds, shells, bright knots, and a dust lane; irregulars combine asymmetric clumps with faint filamentary bridges.

The morphology choices were informed by the local references: `fits/green.fits` for spiral structure, `fits/m42_40min_red.fits` for diffuse nebular texture, and `fits/Vcomb.fits` for the surrounding star field. The reference data themselves are not copied into these generated images.

Regenerate with `save_morphology_set` from `astro_pipeline.synthetic`, using `shape=(2048, 2048)`, `n_images=4`, `seed=74001`, `n_stars=26`, `n_galaxies=3`, `n_nebulae=2`, and `n_irregular=3`.
