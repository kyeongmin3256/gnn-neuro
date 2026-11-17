#!/bin/bash
# Submit full train + shared-split chain after an existing job (e.g. pruned train).
# Usage:
#   bash scripts/submit_shared_after.sh 79755
set -euo pipefail

AFTER_JOB="${1:?usage: bash scripts/submit_shared_after.sh <jobid>}"

cd /local/scratch/kkim636/gnn-neuro

FULL=$(sbatch --dependency=afterok:"$AFTER_JOB" --parsable \
  --job-name=gnn-train-full \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,OUT=outputs/runs_full \
  scripts/job_train.sh)
echo "full train = $FULL"

# afterany: shared prepare still runs if full train fails/times out.
PREP=$(sbatch --dependency=afterany:"$FULL" --parsable scripts/job_prepare_comparison.sh)
echo "shared prepare = $PREP"

PRUNED=$(sbatch --dependency=afterok:"$PREP" --parsable \
  --job-name=gnn-train-pruned-shared \
  --export=ALL,GRAPH=outputs/graph_pruned/kg_index.pt,OUT=outputs/runs_pruned_shared \
  scripts/job_train.sh)
echo "shared pruned train = $PRUNED"

FULL_SHARED=$(sbatch --dependency=afterok:"$PRUNED" --parsable \
  --job-name=gnn-train-full-shared \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,OUT=outputs/runs_full_shared \
  scripts/job_train.sh)
echo "shared full train = $FULL_SHARED"

squeue -u kkim636
