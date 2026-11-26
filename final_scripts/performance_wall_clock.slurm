#!/bin/bash
# Submit from the repository root: sbatch final_scripts/performance_wall_clock.slurm
# INPUT_DIR, RESULTS_DIR, and REPEATS can be overridden through --export.
# All three tiers run sequentially on the same GH node; workers follow the CPU allocation.
#SBATCH -J astro_wall_clock
#SBATCH -o slurm-%x_%j.out
#SBATCH -e slurm-%x_%j.err
#SBATCH -p gh
#SBATCH -A TRA25006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 08:00:00

set -eo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
: "${WORK:?WORK is not set; use a TACC shell with WORK configured}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/100_image_real/sdss_frames}"
RESULTS_DIR="${RESULTS_DIR:-$WORK/astro-test/wall-clock-${SLURM_JOB_ID:-manual}}"
REPEATS="${REPEATS:-5}"

module reset
source "$PROJECT_DIR/final_scripts/setup_env.sh"
module load python3
source "$WORK/venvs/astro-pipeline/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -u final_scripts/compare_performance_nsys.py --wall-clock-only \
    --input "$INPUT_DIR" --tile-size 1024 --repeats "$REPEATS" \
    --results-dir "$RESULTS_DIR"
