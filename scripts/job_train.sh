#!/bin/bash
#SBATCH --job-name=gnn-neuro-train
#SBATCH --partition=hopperweek
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.out
#SBATCH --error=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.err

set -euo pipefail

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/kkim636/envs/gnn-neuro

cd /local/scratch/kkim636/gnn-neuro
mkdir -p outputs/logs

GRAPH="${GRAPH:-outputs/graph_pruned/kg_index.pt}"
OUT="${OUT:-outputs/runs_pruned}"
RELATION="${RELATION:-INDICATION}"
MODEL="${MODEL:-both}"

mkdir -p "$OUT"

echo "host=$(hostname) job=$SLURM_JOB_ID graph=$GRAPH out=$OUT"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python scripts/train.py \
  --config configs/default.yaml \
  --graph "$GRAPH" \
  --model "$MODEL" \
  --relation "$RELATION" \
  --out "$OUT"

echo done
