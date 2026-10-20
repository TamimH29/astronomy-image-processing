# ECE 379K Astronomy Pipeline: Checklist (Oct 20 - Nov 24, 2026)

**Person 1** = ingestion, output, synthetic data, validation
**Person 2** = tiling, parallelism, merge, benchmarking
**Person 3** = background, filtering, detection, GPU

## Milestones

| Date | Milestone |
|---|---|
| Oct 22 | Interfaces agreed, repo set up, GPU allocation requested, synthetic data ready |
| Oct 27 | Single-threaded baseline: one FITS in, catalog out |
| Nov 3 | Concurrent CPU pipeline (2-4 workers); CPU reference frozen |
| Nov 12 | GPU filter matches CPU reference (go/no-go on Nov 10) |
| Nov 19 | Full pipeline on 500 images; benchmarks and accuracy tables done |
| Nov 24 | Final report submitted |

## Everyone (Oct 20-22)
- [ ] Review and agree on `interfaces.py` (conventions, schema, signatures)
- [ ] Create shared repo with `tests/` folder and the agreed layout
- [ ] Request Vista GH GPU allocation (about 20-30 hours) and confirm CPU node access
- [ ] Confirm Python environment on TACC: astropy, NumPy, SciPy; CuPy/CUDA module for later

## Person 1: ingestion, data, validation
- [ ] (Oct 20-22) `make_synthetic_image` + FITS output + truth table
- [ ] (Oct 22) Generate three synthetic sets: tiny, medium, large
- [ ] (Oct 20-22) Start SDSS r-band download (500 images, about 6 GB) in background
- [ ] (Oct 27) `read_fits` (compressed files, extension HDUs, NaN handling)
- [ ] (Oct 27) `write_catalog` with fixed schema
- [ ] (Oct 27) `score_catalog`: completeness, false positives, position error, flux error %
- [ ] (Nov 3) Integrate ingestion with Person 2's tiling and worker pool; test on real SDSS images
- [ ] (Nov 3) Review Person 3's detection code (backup owner)
- [ ] (Nov 12) CPU-vs-GPU comparison script (max abs/rel difference per stage)
- [ ] (Nov 12) Accuracy regression run on every merged change
- [ ] (Nov 19) Full 500-image validation; accuracy tables and plots
- [ ] (Nov 24) Report: data, validation, and results sections

## Person 2: tiling, parallelism, benchmarking
- [ ] (Oct 22) Tiling prototype on random arrays
- [ ] (Oct 27) `make_tiles` with halo, cores partition the image exactly
- [ ] (Oct 27) `merge` with centroid-ownership rule and deterministic source_id
- [ ] (Oct 27) Test: tiled result equals whole-image result; no duplicate or lost edge sources
- [ ] (Nov 3) `run_batch` with `ProcessPoolExecutor`: whole-image parallelism first, then tiles
- [ ] (Nov 3) Test: 1, 2, and 4 workers give identical catalogs
- [ ] (Oct 31) Measure I/O share of runtime (is 2x speedup realistic?)
- [ ] (Oct 31) Decide: keep tiling or fall back to whole images
- [ ] (Nov 12) GPU batching: group tiles, overlap reads with compute, minimize transfers
- [ ] (Nov 19) `benchmark` harness with per-stage timings, CPU-only vs GPU at same worker count
- [ ] (Nov 24) Report: parallelism and performance sections

## Person 3: filtering, detection, GPU
- [ ] (Oct 22) Prototype filter and detector on tiny synthetic set
- [ ] (Oct 27) `estimate_background` (sigma-clipped, gridded, interpolated)
- [ ] (Oct 27) `smooth` CPU (SciPy FFT Gaussian)
- [ ] (Oct 27) `detect`: threshold, label, centroid, bbox, area, flux
- [ ] (Nov 3) Integrate detection into Person 2's pipeline
- [ ] (Nov 3) Tune threshold and kernel width on medium synthetic set
- [ ] (Nov 3) **Freeze CPU reference**
- [ ] (Nov 12) `smooth(backend="gpu")` with CuPy; match CPU within about 1e-4 relative
- [ ] (Nov 10) Go/no-go on GPU path; fall back to CPU filter if not matching
- [ ] (Nov 19) Profile and optimize GPU path (likely transfers, not the FFT)
- [ ] (Nov 24) Report: algorithm and GPU sections

## Final week (Nov 20-24)
- [ ] Document limitations (tiling boundary effects, threshold sensitivity, GPU speedup vs I/O)
- [ ] Combine report sections, check figures and tables
- [ ] Re-run full pipeline once from a clean checkout
- [ ] Submit

## Risk watch
- [ ] Oct 31: is I/O dominating runtime? If yes, adjust the 2x target discussion early
- [ ] Nov 4: GPU allocation granted and CuPy imports on the GPU node
- [ ] Nov 10: GPU match go/no-go
