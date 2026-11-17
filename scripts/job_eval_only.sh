#!/bin/bash
#SBATCH --job-name=gnn-eval-only
#SBATCH --partition=hopperhour
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.out
#SBATCH --error=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.err

set -euo pipefail
source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/kkim636/envs/gnn-neuro
cd /local/scratch/kkim636/gnn-neuro
mkdir -p outputs/logs

GRAPH="${GRAPH:-outputs/graph/kg_index.pt}"
RUNS="${RUNS:-outputs/runs}"
OUT_NAME="${OUT_NAME:-metrics_INDICATION_seed_diseases.json}"

python scripts/eval_only.py \
  --config configs/default.yaml \
  --graph "$GRAPH" \
  --runs "$RUNS" \
  --relation INDICATION \
  --model both \
  --out-name "$OUT_NAME"

echo done
