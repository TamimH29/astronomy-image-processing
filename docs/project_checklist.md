# ECE 379K Astronomy Pipeline: Checklist (Oct 20 - Nov 24, 2026)

**Version 2 (Oct 29).** Items marked **[NEW]** were added in this version. Items marked **[CHANGED]** existed before but were modified (what changed is stated in the item). Unmarked items are unchanged.

**Person 1** = ingestion, output, synthetic data, validation
**Person 2** = tiling, parallelism, merge, benchmarking
**Person 3** = background, filtering, detection, GPU

## What changed in this version and why

The project now reports **speedup by stage** and compares **three tiers**: (1) one CPU core, (2) all CPU cores on a node, (3) all CPU cores plus the GPU. Success shifts from "hit 2x" to "measure and explain the speedup". Consequences:

- Per-stage timers exist from the start, so the benchmark harness moves from Nov 19 to Nov 3.
- The Oct 31 I/O measurement becomes a core result: a per-stage time profile on real SDSS files.
- A core-scaling sweep and an Amdahl's-law analysis (predicted vs measured speedup) are added.
- The final benchmark jobs are submitted by Nov 17 so queue waits don't threaten the Nov 24 deadline.

## Milestones

| Date | Milestone |
|---|---|
| Oct 22 | Interfaces agreed, repo set up, GPU allocation requested, synthetic data ready |
| Oct 27 | Single-threaded baseline: one FITS in, catalog out |
| **[NEW]** Oct 31 | Per-stage time profile of the CPU baseline on real SDSS files; speedup definition agreed with team and instructor/TA notified |
| **[CHANGED]** Nov 3 | Concurrent CPU pipeline; CPU reference frozen. **Added:** benchmark harness with per-stage timers running on the CPU baseline |
| Nov 12 | GPU filter matches CPU reference (go/no-go on Nov 10). **[NEW]** GPU timing keys (transfer vs compute) added to the harness |
| **[NEW]** Nov 17 | Final benchmark jobs submitted (three tiers plus scaling sweep) |
| **[CHANGED]** Nov 19 | Full pipeline on 500 images; accuracy tables done. **Added:** per-stage timing table, scaling plot, three-tier comparison, Amdahl analysis |
| Nov 24 | Final report submitted |

## Everyone (Oct 20-22)
- [ ] Review and agree on `interfaces.py` (conventions, schema, signatures)
- [ ] Create shared repo with `tests/` folder and the agreed layout
- [ ] Request Lonestar6 GPU allocation (about 20-30 hours) and confirm CPU node access
- [ ] Confirm Python environment on TACC: astropy, NumPy, SciPy; CuPy/CUDA module for later

### New shared items
- [ ] **[NEW]** (Oct 30) Agree on the benchmark design: the three tiers, which node type is used for tiers 2 and 3 (same node type for both), and that each measurement is repeated and the median reported
- [ ] **[NEW]** (Oct 31) Agree what "CPU-only version" means in the report (one core, all cores, or both) and tell the instructor/TA that the performance goal is being reframed from a pass/fail 2x to a per-stage characterization
- [ ] **[NEW]** (Oct 30) Update `interfaces.py` to `INTERFACE_VERSION = 3` with the expanded benchmark keys (see Person 2)

## Person 1: ingestion, data, validation
- [ ] (Oct 20-22) `make_synthetic_image` + FITS output + truth table
- [ ] (Oct 22) Generate three synthetic sets: tiny, medium, large
- [ ] (Oct 20-22) Start SDSS r-band download (500 images, about 6 GB) in background
- [ ] (Oct 27) `read_fits` (compressed files, extension HDUs, NaN handling)
- [ ] (Oct 27) `write_catalog` with fixed schema
- [ ] (Oct 27) `score_catalog`: completeness, false positives, position error, flux error %
- [ ] **[NEW]** (Oct 30) Make an uncompressed copy of a subset (about 20 SDSS images) in node-local `/tmp`; record compressed vs uncompressed read times so Person 2 can quantify decompression cost
- [ ] (Nov 3) Integrate ingestion with Person 2's tiling and worker pool; test on real SDSS images
- [ ] (Nov 3) Review Person 3's detection code (backup owner)
- [ ] (Nov 12) CPU-vs-GPU comparison script (max abs/rel difference per stage)
- [ ] (Nov 12) Accuracy regression run on every merged change
- [ ] **[CHANGED]** (Nov 19) Full 500-image validation; accuracy tables and plots. **Added:** confirm all three tiers produce the same catalog (within the GPU tolerance), so speedups are not bought with accuracy
- [ ] **[CHANGED]** (Nov 24) Report: data, validation, and results sections. **Added:** a table showing accuracy is equivalent across the three tiers

## Person 2: tiling, parallelism, benchmarking
- [ ] (Oct 22) Tiling prototype on random arrays
- [ ] (Oct 27) `make_tiles` with halo, cores partition the image exactly
- [ ] (Oct 27) `merge` with centroid-ownership rule and deterministic source_id
- [ ] (Oct 27) Test: tiled result equals whole-image result; no duplicate or lost edge sources
- [ ] **[NEW]** (Oct 30) Shared timing helper (a small context manager used around every stage) and expanded `benchmark()` keys: `io, background, tile, filter, detect, merge, write, gpu_h2d, gpu_compute, gpu_d2h` (GPU timers need a `synchronize` before stopping)
- [ ] **[CHANGED]** (Oct 31) Measure the per-stage share of runtime on real SDSS files. **Was:** only the I/O share. **Now:** the full per-stage profile, which is a core result of the project
- [ ] (Oct 31) Decide: keep tiling or fall back to whole images
- [ ] **[CHANGED]** (Nov 3) `run_batch` with `ProcessPoolExecutor`: whole-image parallelism first, then tiles. **Added:** the benchmark harness now runs on the CPU baseline (was Nov 19)
- [ ] **[CHANGED]** (Nov 3) Test: 1, 2, and 4 workers give identical catalogs. **Added:** also test the full core count of the node
- [ ] **[NEW]** (Nov 3) Core-scaling sweep mode: time per stage for 1, 2, 4, 8, 16, ... workers on a CPU node; produce the table and plot
- [ ] (Nov 12) GPU batching: group tiles, overlap reads with compute, minimize transfers
- [ ] **[NEW]** (Nov 12) Handle GPU warm-up: first CuPy call includes one-time setup (context, FFT plan); exclude it from timings or report it separately
- [ ] **[NEW]** (Nov 12) Test several workers sharing one GPU, with and without NVIDIA MPS; choose a worker cap for GPU runs
- [ ] **[NEW]** (Nov 17) Submit the final benchmark jobs: tier 1 (one core), tier 2 (all cores), tier 3 (all cores + GPU), same node type for tiers 2 and 3, each repeated, median reported
- [ ] **[CHANGED]** (Nov 19) Benchmark results complete. **Was:** harness built here. **Now:** harness already exists, this is the finished three-tier comparison
- [ ] **[NEW]** (Nov 19) Amdahl analysis: from the per-stage times, predict the speedup of each tier, then compare to the measured speedup
- [ ] **[CHANGED]** (Nov 24) Report: parallelism and performance sections. **Added:** per-stage timing table, core-scaling plot, three-tier comparison (tier 3 vs tier 2 isolates the GPU), Amdahl predicted vs measured

## Person 3: filtering, detection, GPU
- [ ] (Oct 22) Prototype filter and detector on tiny synthetic set
- [ ] (Oct 27) `estimate_background` (sigma-clipped, gridded, interpolated). **[NEW]** Vectorize it (no Python loop over boxes) and time it on its own
- [ ] (Oct 27) `smooth` CPU (SciPy FFT Gaussian)
- [ ] **[NEW]** (Nov 3) Time the CPU filter two ways, FFT and separable `gaussian_filter`, at the sigma you use; report both so a reader can see the baseline is fair
- [ ] (Oct 27) `detect`: threshold, label, centroid, bbox, area, flux
- [ ] **[NEW]** (Nov 3) Time `detect` on its own; note that `scipy.ndimage.label` is single-threaded and that this stage stays on the CPU
- [ ] (Nov 3) Integrate detection into Person 2's pipeline
- [ ] (Nov 3) Tune threshold and kernel width on medium synthetic set
- [ ] (Nov 3) **Freeze CPU reference**
- [ ] (Nov 12) `smooth(backend="gpu")` with CuPy; match CPU within about 1e-4 relative
- [ ] (Nov 10) Go/no-go on GPU path; fall back to CPU filter if not matching
- [ ] **[NEW]** (Nov 10, optional) Multi-scale smoothing (several kernel widths) only if it improves detection on the synthetic set; it raises filter cost, so do not add it just to inflate the benchmark
- [ ] **[CHANGED]** (Nov 19) Profile and optimize GPU path. **Added:** report transfer time and compute time separately, since transfers are the likely bottleneck
- [ ] **[NEW]** (stretch, only if ahead of schedule) GPU connected-component labeling (`cupyx.scipy.ndimage.label`) to move detect off the CPU
- [ ] **[CHANGED]** (Nov 24) Report: algorithm and GPU sections. **Added:** the FFT vs separable CPU baseline timings and the transfer vs compute breakdown

## Final week (Nov 20-24)
- [ ] **[CHANGED]** Document limitations (tiling boundary effects, threshold sensitivity). **Replaced** "GPU speedup vs I/O" with: the speedup ceiling set by the stages that stay on the CPU, and the I/O share
- [ ] **[NEW]** State in the report what "CPU-only" means and which node type each tier ran on
- [ ] **[NEW]** Report figures: per-stage timing table, core-scaling plot, three-tier comparison, Amdahl predicted vs measured
- [ ] Combine report sections, check figures and tables
- [ ] Re-run full pipeline once from a clean checkout
- [ ] Submit

## Risk watch
- [ ] **[CHANGED]** Oct 31: per-stage profile on real SDSS files done. **Was:** "is I/O dominating?" If I/O dominates, the GPU can't reach 2x, so frame results as characterization (see shared items)
- [ ] Nov 4: GPU allocation granted and CuPy imports on the GPU node
- [ ] Nov 10: GPU match go/no-go
- [ ] **[NEW]** Nov 17: final benchmark jobs submitted. If a GPU queue wait exceeds a day, fall back to fewer repeats or a smaller batch (and state this in the report)
