#!/bin/bash
#SBATCH --job-name=gnn-prepare-full
#SBATCH --partition=hopperday
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.out
#SBATCH --error=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.err

set -euo pipefail
source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/kkim636/envs/gnn-neuro
cd /local/scratch/kkim636/gnn-neuro
mkdir -p outputs/logs

# Does NOT touch outputs/graph used by the current pruned training job.
python scripts/prepare_graph.py --config configs/full.yaml --out outputs/graph_full

echo done
