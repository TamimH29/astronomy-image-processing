"""Measure performance for the multicore+GPU tier.

Workers default to SLURM_CPUS_PER_TASK or the process CPU affinity.
Example:
  python final_scripts/compare_performance_gpu.py --input '100_image_synth/*.fits' \
      --tile-size 1024 --repeats 5 \
      --results-dir "$WORK/astro-test/perf-gpu"
"""
from compare_performance_nsys import main


if __name__ == "__main__":
    main("multicore_gpu")
