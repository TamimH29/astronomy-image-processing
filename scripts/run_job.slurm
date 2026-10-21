#!/bin/bash
# Template: fill in the <PLACEHOLDERS>, check queue names with `qlimits`.
#SBATCH -J astro_pipeline
#SBATCH -o logs/%x_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -p <QUEUE>            # e.g. a GPU dev queue while testing
#SBATCH -A <PROJECT>          # only needed if you have several projects
# Do NOT request or exclude specific nodes. On Vista, do NOT use --gres or
# --gpus-per-task (jobs are rejected); a GH node has one GPU anyway.

module reset
# module load <python / cuda modules you settled on>
# source <path to your venv>/bin/activate

# Stage input to node-local /tmp so shared-filesystem reads stay out of timings.
mkdir -p /tmp/sdss_r logs
cp "$WORK/sdss_r/"*.fits* /tmp/sdss_r/

python -m astro_pipeline.run_pipeline \
    --input /tmp/sdss_r \
    --output "$SCRATCH/astro_out/catalog.csv" \
    --workers 4 --backend gpu --tile-size 1024
# For timings instead of a catalog, add --benchmark (and run once with
# --backend cpu on the same node type for the baseline).
