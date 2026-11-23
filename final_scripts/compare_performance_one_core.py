"""Measure performance for the single-core CPU tier.

Example:
  python final_scripts/compare_performance_one_core.py --input '100_image_synth/*.fits' \
      --tile-size 1024 --repeats 5 \
      --results-dir "$WORK/astro-test/perf-one-core"
"""
from compare_performance_nsys import main


if __name__ == "__main__":
    main("single_core_cpu")
