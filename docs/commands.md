# ECE 379K FITS source-detection pipeline

- `astro_pipeline/interfaces.py`: shared types, schema, conventions (the contract)
- One module per owner; several pipeline stages still contain stubs.
- Implemented: Humza's tiler/merge and Samuel's SciPy/CuPy FFT Gaussian filter.
- Filtering setup and behavior: [filtering.md](filtering.md).
- CLI entry point (requires remaining pipeline stubs to be implemented):
  `python -m astro_pipeline.run_pipeline --input DIR --output catalog.csv`
- Test: `python -m pytest tests/`
- Change `interfaces.py` only by team agreement and bump `INTERFACE_VERSION`.
