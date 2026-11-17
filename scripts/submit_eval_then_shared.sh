#!/bin/bash
# After full train finishes:
#   1) 79-disease eval on pruned runs
#   2) 79-disease eval on full runs
#   3) shared-split prepare + pruned/full retrain
#
# Usage:
#   bash scripts/submit_eval_then_shared.sh 79763
set -euo pipefail

AFTER_JOB="${1:?usage: bash scripts/submit_eval_then_shared.sh <full-train-jobid>}"

cd /local/scratch/kkim636/gnn-neuro

EVAL_PRUNED=$(sbatch --dependency=afterok:"$AFTER_JOB" --parsable \
  --job-name=gnn-eval-pruned79 \
  --export=ALL,GRAPH=outputs/graph/kg_index.pt,RUNS=outputs/runs,OUT_NAME=metrics_INDICATION_seed_diseases.json \
  scripts/job_eval_only.sh)
echo "eval pruned 79-dis = $EVAL_PRUNED"

EVAL_FULL=$(sbatch --dependency=afterok:"$EVAL_PRUNED" --parsable \
  --job-name=gnn-eval-full79 \
  --export=ALL,GRAPH=outputs/graph_full/kg_index.pt,RUNS=outputs/runs_full,OUT_NAME=metrics_INDICATION_seed_diseases.json \
  scripts/job_eval_only.sh)
echo "eval full 79-dis = $EVAL_FULL"

PREP=$(sbatch --dependency=afterok:"$EVAL_FULL" --parsable scripts/job_prepare_comparison.sh)
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
