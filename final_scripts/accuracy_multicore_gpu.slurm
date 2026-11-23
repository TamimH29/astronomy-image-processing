#!/bin/bash
# Submit from the repository root:
#   sbatch final_scripts/accuracy_multicore_gpu.slurm
# Workers follow SLURM_CPUS_PER_TASK. Override paths with --export=ALL,INPUT_DIR=...,RESULTS_DIR=...
# If your TACC account requires an explicit allocation, add its #SBATCH -A directive.
# For a short development run, change -p gh to -p gh-dev and shorten -t.
# Vista GH jobs use the node's GPU without --gres/--gpus-per-task.
#SBATCH -J astro_acc_gpu
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH -p gh
#SBATCH -A TRA25006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 04:00:00

set -eo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
: "${WORK:?WORK is not set; use a TACC shell with WORK configured}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/100_image_synth}"
RESULTS_DIR="${RESULTS_DIR:-$WORK/astro-test/accuracy-multicore-gpu-${SLURM_JOB_ID:-manual}}"

[[ -d "$INPUT_DIR" ]] || { echo "Input directory not found: $INPUT_DIR" >&2; exit 1; }
INPUT_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name '*.fits' -o -name '*.fits.gz' -o -name '*.fits.bz2' -o -name '*.fit' \) | wc -l)
[[ "$INPUT_COUNT" -gt 0 ]] || { echo "No FITS images found in $INPUT_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$RESULTS_DIR")"

module reset
source "$PROJECT_DIR/final_scripts/setup_env.sh"
# Load Python after setup_env selects the compiler/CUDA module versions it needs.
module load python3
source "$WORK/venvs/astro-pipeline/bin/activate"
module list 2>&1
python -c 'import numpy, scipy, astropy; import cupy as cp; print("CuPy", cp.__version__, "GPU count:", cp.cuda.runtime.getDeviceCount()); assert cp.cuda.runtime.getDeviceCount() > 0'
nvidia-smi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python final_scripts/compare_accuracy_gpu.py \
    --input "$INPUT_DIR" --tile-size 1024 --results-dir "$RESULTS_DIR"
