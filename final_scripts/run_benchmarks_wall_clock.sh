#!/usr/bin/env bash
# Compare the three production tiers using unprofiled wall-clock time only.
set -eo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/100_image_real/sdss_frames}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/benchmarks/results/wall-clock-$(date +%Y%m%d-%H%M%S)-$$}"
REPEATS="${REPEATS:-5}"

source "$PROJECT_DIR/.tools/miniforge3/etc/profile.d/conda.sh"
conda activate "$PROJECT_DIR/.conda-env"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$PROJECT_DIR/benchmarks/results/local-cuda12/cupy-cache}"

python -u final_scripts/compare_performance_nsys.py --wall-clock-only \
    --input "$INPUT_DIR" --tile-size 1024 --repeats "$REPEATS" \
    --results-dir "$RESULTS_DIR"
