# gnn-neuro

OptimusKG baselines for neurological drug repurposing.

**Models:** RotatE vs R-GCN  
**Targets:** drug–disease `INDICATION` / `CONTRAINDICATION`  
**Metrics:** disease-centric AUPRC, MRR, Hits@K (filtered)

## Design choices (TxGNN-aligned)

Earlier runs produced near-chance disease→drug AUPRC (~0.04) mainly because:

1. Positives are very sparse relative to the drug candidate set, while AUPRC was estimated from a tiny random negative subsample
2. Pruning, split construction, and message-passing edge masks were loosely coupled
3. Restricting candidates to drugs that already have an indication can inflate metrics but excludes true repurposing candidates

Defaults in this repo:

- **Candidates = all `DRG` nodes in the working graph** (same as TxGNN). Do not limit ranking to drugs that already have an indication
- **Supervision = known `INDICATION` / `CONTRAINDICATION` edges**
- **Evaluation = per-disease ranking over all drugs**, then macro AUPRC / MRR / Hits@K (filtered)
- **Pruning seeds = 6 neuroinflammation BPOs + 10 neurodegenerative disease roots (+ descendants) + BPO-linked / seed genes**

Recommended prune setting from the comparison table: **BPO+gene+disease, k=1** with traversal types `GEN/DRG/DIS/BPO/PWY`.

## Setup

```bash
cd ~/Projects/gnn-neuro
python3 -m pip install -e .
```

Place OptimusKG LCC parquets under `data/raw/` (or point `configs/default.yaml` at your paths):

- `data/raw/largest_connected_component_nodes.parquet`
- `data/raw/largest_connected_component_edges.parquet`

## Pipeline

```bash
# 1) Load OptimusKG + seed prune + split
python3 scripts/prepare_graph.py --config configs/default.yaml

# 2) Optional: pruning comparison table
python3 scripts/compare_pruning.py --config configs/default.yaml

# 3) Train baselines
python3 scripts/train.py --model both --relation INDICATION
python3 scripts/train.py --model both --relation CONTRAINDICATION
```

Artifacts:

- `outputs/graph/` — pruned nodes/edges, splits, `kg_index.pt`
- `outputs/runs/` — checkpoints + `metrics_*.json`
- `outputs/pruning_comparison.json` — seed × k comparison

## Seeds

See `configs/seeds.yaml` and `data/seeds/`.
