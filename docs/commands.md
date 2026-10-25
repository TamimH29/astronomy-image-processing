# ECE 379K FITS source-detection pipeline

- `astro_pipeline/interfaces.py`: shared types, schema, conventions (the contract)
- One module per owner; each function is a stub with a docstring saying what it must do
- Run: `python -m astro_pipeline.run_pipeline --input DIR --output catalog.csv`
- Test: `pytest tests/`
- Change `interfaces.py` only by team agreement and bump `INTERFACE_VERSION`.
