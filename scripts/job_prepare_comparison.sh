#!/bin/bash
#SBATCH --job-name=gnn-prepare-both
#SBATCH --partition=hopperweek
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.out
#SBATCH --error=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.err

set -euo pipefail
source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/kkim636/envs/gnn-neuro
cd /local/scratch/kkim636/gnn-neuro
mkdir -p outputs/logs

python scripts/prepare_comparison.py --config configs/default.yaml \
  --out-full outputs/graph_full \
  --out-pruned outputs/graph_pruned

echo done
