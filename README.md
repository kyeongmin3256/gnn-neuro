# gnn-neuro

OptimusKG 기반 neurological drug repurposing 베이스라인.

**비교 모델:** RotatE vs R-GCN  
**예측:** drug–disease `INDICATION` / `CONTRAINDICATION`  
**평가:** disease-centric AUPRC, MRR, Hits@K (filtered)

## 설계 결정 (TxGNN 정렬)

이전 실험에서 disease→drugs AUPRC가 ~0.04로 chance에 가까운 이유가:

1. 후보 약물 대비 positive가 매우 sparse한데, 작은 random negative subsample로 AUPRC를 측정함
2. pruning / split / message-passing mask가 약하게 묶여 있었음
3. “indication 있는 약물만” 쓰면 메트릭은 예뻐질 수 있지만, 재창출 후보를 원천 배제함

이번 코드의 기본값:

- **후보 약물 = 그래프에 있는 모든 DRG** (TxGNN과 동일). indication이 이미 있는 약물만으로 후보를 제한하지 않음
- **지도 신호** = 알려진 INDICATION / CONTRAINDICATION 엣지
- **평가** = disease마다 전체 약물 랭킹 후 macro AUPRC / MRR / Hits@K (filtered)
- **pruning seed** = 슬라이드의 neuroinflammation BPO 6개 + neurodegenerative disease 10개(+descendants) + BPO-linked genes

## Setup

```bash
cd ~/Projects/gnn-neuro
python -m pip install -e .
```

## Pipeline

```bash
# 1) OptimusKG load + seed prune + split
python scripts/prepare_graph.py --config configs/default.yaml

# 2) Train baselines
python scripts/train.py --model both --relation INDICATION
python scripts/train.py --model both --relation CONTRAINDICATION
```

Artifacts:

- `outputs/graph/` — pruned nodes/edges, splits, `kg_index.pt`
- `outputs/runs/` — checkpoints + `metrics_*.json`

## Seeds

See `configs/seeds.yaml` and `data/seeds/`.
