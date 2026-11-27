#!/usr/bin/env bash
# Unprofiled CPU and CPU+GPU scaling, with separate CPU-counter passes.
set -eo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/100_image_real/sdss_frames}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/benchmarks/results/worker-sweep-$(date +%Y%m%d-%H%M%S)-$$}"
REPEATS="${REPEATS:-3}"

# Activate the desired Python/CUDA environment before running this script.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$PROJECT_DIR/benchmarks/results/local-cuda12/cupy-cache}"
mkdir -p "$RESULTS_DIR"
echo "Worker sweep results: $RESULTS_DIR"

for tier in all_cores_cpu gpu; do
    for workers in 8 16 24 32; do
        case_dir="$RESULTS_DIR/$tier-$workers"
        if python -u "final_scripts/compare_performance_${tier}.py" \
            --wall-clock-only --cpu-counters --cpu-workers "$workers" \
            --input "$INPUT_DIR" --tile-size 1024 --repeats "$REPEATS" \
            --results-dir "$case_dir" 2>&1 | tee "$RESULTS_DIR/$tier-$workers.log"; then
            echo 0 > "$RESULTS_DIR/$tier-$workers.exit-code"
        else
            case_status=("${PIPESTATUS[@]}")
            case_exit_code=${case_status[0]}
            if [[ "$case_exit_code" == 0 ]]; then case_exit_code=${case_status[1]}; fi
            echo "$case_exit_code" > "$RESULTS_DIR/$tier-$workers.exit-code"
            echo "$tier with $workers workers failed; logs retained."
        fi
    done
done

python - "$RESULTS_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for tier in ('all_cores_cpu', 'gpu'):
    for workers in (8, 16, 24, 32):
        case_dir = root / f'{tier}-{workers}'
        code = int((root / f'{tier}-{workers}.exit-code').read_text())
        row = {'configuration': tier, 'workers_requested': workers, 'exit_code': code,
               'status': 'complete' if code == 0 else 'failed'}
        if code == 0:
            with (case_dir / 'summary.csv').open() as stream:
                row.update(next(csv.DictReader(stream)))
            row['workers_requested'] = int(row['workers_requested'])
            row['workers_effective'] = int(row['workers_effective'])
            row['repeats'] = int(row['repeats'])
            row['perf_available'] = row['perf_available'] == 'True'
            row['wall_samples_s'] = json.loads(row['wall_samples_s'])
            for key in ('median_wall_s', 'throughput_images_s', 'perf_wall_s',
                        'cpu_task_clock_ms', 'cpu_cores_busy_avg', 'cpu_ipc',
                        'cpu_cycles', 'cpu_instructions', 'cpu_cache_misses',
                        'cpu_context_switches', 'cpu_migrations', 'cpu_page_faults'):
                row[key] = float(row[key]) if row.get(key) else None
        rows.append(row)
for tier in ('all_cores_cpu', 'gpu'):
    reference = next(r.get('median_wall_s') for r in rows
                     if r['configuration'] == tier and int(r['workers_requested']) == 16)
    for row in rows:
        if row['configuration'] == tier and reference and row.get('median_wall_s'):
            row['speedup_vs_16_workers'] = reference / row['median_wall_s']
fields = list(dict.fromkeys(key for row in rows for key in row))
with (root / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(root / 'comparison.json').write_text(json.dumps({'results': rows}, indent=2) + '\n')
for row in rows:
    if row['status'] == 'complete':
        print(f"{row['configuration']:14s} workers={int(row['workers_requested']):2d} "
              f"median={row['median_wall_s']:.3f} s "
              f"throughput={row['throughput_images_s']:.3f} images/s "
              f"vs16={row.get('speedup_vs_16_workers', float('nan')):.3f}x")
    else:
        print(f"{row['configuration']} workers={row['workers_requested']}: FAILED")
print(f'Results: {root}')
PY
