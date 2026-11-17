#!/bin/bash
# Submit shared-split prepare + pruned/full train after an existing job.
# Usage:
#   bash scripts/submit_shared_after.sh 79756
set -euo pipefail

AFTER_JOB="${1:?usage: bash scripts/submit_shared_after.sh <jobid>}"

cd /local/scratch/kkim636/gnn-neuro

# afterany: still run shared-split even if the previous full train failed/timed out.
PREP=$(sbatch --dependency=afterany:"$AFTER_JOB" --parsable scripts/job_prepare_comparison.sh)
echo "shared prepare = $PREP"

PRUNED=$(sbatch --dependency=afterok:"$PREP" --parsable \
  --job-name=gnn-train-pruned-shared \
  --export=ALL,GRAPH=outputs/graph_pruned/kg_index.pt,OUT=outputs/runs_pruned_shared \
  scripts/job_train.sh)
echo "shared pruned train = $PRUNED"

FULL=$(sbatch --dependency=afterok:"$PRUNED" --parsable \
  --job-name=gnn-train-full-shared \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,OUT=outputs/runs_full_shared \
  scripts/job_train.sh)
echo "shared full train = $FULL"

squeue -u kkim636
