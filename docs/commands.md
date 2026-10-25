# ECE 379K FITS source-detection pipeline

- `astro_pipeline/interfaces.py`: shared types, schema, conventions (the contract)
- One module per owner; shared contracts stay in `interfaces.py`.
- Samuel's native CPU/CUDA smoothing is connected through `filtering.smooth`;
  Humza's tile construction and source-list merging are implemented in `tiling.py`.
  Other pipeline modules still contain stubs.
- Native build and filtering tests: [native_filtering.md](native_filtering.md).
- CLI entry point (still depends on pipeline stubs):
  `python -m astro_pipeline.run_pipeline --input DIR --output catalog.csv`
- Test after building the extension:
  `PYTHONPATH=build-cpu:. python -m pytest tests/`
- Change `interfaces.py` only by team agreement and bump `INTERFACE_VERSION`.
