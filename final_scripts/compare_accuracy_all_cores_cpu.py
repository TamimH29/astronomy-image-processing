"""Run ground-truth accuracy scoring with all CPUs allocated to the job.

Example (workers default to SLURM_CPUS_PER_TASK or the process CPU affinity):
  python final_scripts/compare_accuracy_all_cores_cpu.py --input '100_image_synth/*.fits' \
      --tile-size 1024 --results-dir "$WORK/astro-test/accuracy-all-cpu"
"""
from accuracy_tier_common import run_tier


if __name__ == "__main__":
    run_tier("multicore_cpu")
