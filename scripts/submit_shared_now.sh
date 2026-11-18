#!/bin/bash
# Remaining fair-comparison chain (no prior-job dependency):
#   prepare shared split → pruned train → pruned 79-eval
#   → full train → full 79-eval
#
# Usage:
#   bash scripts/submit_shared_now.sh
set -euo pipefail

cd /local/scratch/kkim636/gnn-neuro

PREP=$(sbatch --parsable scripts/job_prepare_comparison.sh)
echo "shared prepare = $PREP"

PRUNED=$(sbatch --dependency=afterok:"$PREP" --parsable \
  --job-name=gnn-train-pruned-shared \
  --export=ALL,GRAPH=outputs/graph_pruned/kg_index.pt,OUT=outputs/runs_pruned_shared \
  scripts/job_train.sh)
echo "shared pruned train = $PRUNED"

EVAL_PRUNED=$(sbatch --dependency=afterok:"$PRUNED" --parsable \
  --job-name=gnn-eval-pruned-shared79 \
  --export=ALL,GRAPH=outputs/graph_pruned/kg_index.pt,RUNS=outputs/runs_pruned_shared,OUT_NAME=metrics_INDICATION_seed_diseases.json \
  scripts/job_eval_only.sh)
echo "eval pruned-shared 79 = $EVAL_PRUNED"

FULL=$(sbatch --dependency=afterok:"$EVAL_PRUNED" --parsable \
  --job-name=gnn-train-full-shared \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,OUT=outputs/runs_full_shared \
  scripts/job_train.sh)
echo "shared full train = $FULL"

EVAL_FULL=$(sbatch --dependency=afterok:"$FULL" --parsable \
  --job-name=gnn-eval-full-shared79 \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,RUNS=outputs/runs_full_shared,OUT_NAME=metrics_INDICATION_seed_diseases.json \
  scripts/job_eval_only.sh)
echo "eval full-shared 79 = $EVAL_FULL"

squeue -u kkim636
