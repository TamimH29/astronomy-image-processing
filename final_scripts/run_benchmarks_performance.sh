#!/usr/bin/env bash
# Run the existing performance jobs from a WSL Bash shell.
# INPUT_DIR, RESULTS_DIR, and REPEATS have the same role as in the Slurm jobs.
set -eo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/100_image_real/sdss_frames}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/benchmarks/results/benchmark-performance-$(date +%Y%m%d-%H%M%S)-$$}"
REPEATS="${REPEATS:-5}"

# Activate the desired Python/CUDA environment before running this script.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$PROJECT_DIR/benchmarks/results/local-cuda12/cupy-cache}"

[[ -d "$INPUT_DIR" ]] || { echo "Input directory not found: $INPUT_DIR" >&2; exit 1; }
nsys --version
python -c 'import numpy, scipy, astropy; import cupy as cp; print("CuPy", cp.__version__, "GPU count:", cp.cuda.runtime.getDeviceCount()); assert cp.cuda.runtime.getDeviceCount() > 0'
mkdir -p "$RESULTS_DIR"

for tier in one_core all_cores_cpu gpu; do
    python -u "final_scripts/compare_performance_${tier}.py" \
        --input "$INPUT_DIR" --tile-size 1024 --repeats "$REPEATS" \
        --results-dir "$RESULTS_DIR/performance-$tier" \
        2>&1 | tee "$RESULTS_DIR/performance-$tier.log"
done

python - "$RESULTS_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('final_scripts').resolve()))
from compare_performance_nsys import _ratio

root = Path(sys.argv[1])
rows = []
for tier in ('one_core', 'all_cores_cpu', 'gpu'):
    with (root / f'performance-{tier}' / 'summary.csv').open() as stream:
        rows.extend(csv.DictReader(stream))
with (root / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
seconds = {row['tier']: float(row['median_wall_s']) for row in rows}
comparison = {
    'median_wall_s': seconds,
    'speedups_from_unprofiled_median_wall_time': {
        'multicore_cpu_vs_single_core_cpu': _ratio(seconds['single_core_cpu'], seconds['multicore_cpu']),
        'multicore_gpu_vs_multicore_cpu': _ratio(seconds['multicore_cpu'], seconds['multicore_gpu']),
        'multicore_gpu_vs_single_core_cpu': _ratio(seconds['single_core_cpu'], seconds['multicore_gpu']),
    },
}
(root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison, indent=2))
PY

echo "Performance results: $RESULTS_DIR"
echo "Nsight reports: $RESULTS_DIR/performance-*/profile.nsys-rep"
echo "Speed of Light report: $RESULTS_DIR/performance-gpu/speed-of-light/profile.ncu-rep"
echo "Speed of Light text: $RESULTS_DIR/performance-gpu/speed-of-light/speed_of_light.txt"
echo "Speed of Light coverage/status: $RESULTS_DIR/performance-gpu/speed-of-light/coverage.txt"
