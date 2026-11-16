#!/bin/bash
#SBATCH --job-name=gnn-neuro-train
#SBATCH --partition=hopperday
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.out
#SBATCH --error=/local/scratch/kkim636/gnn-neuro/outputs/logs/%x_%j.err

set -euo pipefail

source /local/scratch/kkim636/miniconda3/etc/profile.d/conda.sh
conda activate /local/scratch/kkim636/envs/gnn-neuro

cd /local/scratch/kkim636/gnn-neuro
mkdir -p outputs/logs outputs/runs

echo "host=$(hostname) job=$SLURM_JOB_ID cuda_check:"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Relation can be overridden: sbatch --export=ALL,RELATION=CONTRAINDICATION scripts/job_train.sh
RELATION="${RELATION:-INDICATION}"
MODEL="${MODEL:-both}"

python scripts/train.py --model "$MODEL" --relation "$RELATION" --out outputs/runs

echo "done"
