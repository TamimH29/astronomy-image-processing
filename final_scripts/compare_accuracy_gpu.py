"""Run ground-truth accuracy scoring through the full-GPU pipeline.

This is the multicore+GPU tier; workers default to the CPUs allocated to the job.
Example:
  python final_scripts/compare_accuracy_gpu.py --input '100_image_synth/*.fits' \
      --tile-size 1024 --results-dir "$WORK/astro-test/accuracy-gpu"
"""
from accuracy_tier_common import run_tier


if __name__ == "__main__":
    run_tier("multicore_gpu")
