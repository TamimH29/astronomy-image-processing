"""Run ground-truth accuracy scoring with one CPU worker.

Example:
  python final_scripts/compare_accuracy_one_core.py --input '100_image_synth/*.fits' \
      --tile-size 1024 --results-dir "$WORK/astro-test/accuracy-one-core"
"""
from accuracy_tier_common import run_tier


if __name__ == "__main__":
    run_tier("single_core_cpu")
