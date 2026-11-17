#!/usr/bin/env python3
"""Re-evaluate saved RotatE / R-GCN weights without retraining.

drug→disease ranking uses neurodegenerative seed diseases (~79), not all DIS nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_neuro.data.seeds import load_seed_config, resolve_seed_ids
from gnn_neuro.data.tensorize import KnowledgeGraphIndex, relation_edge_index
from gnn_neuro.models.rgcn import RGCNLinkPredictor
from gnn_neuro.models.rotate import RotatE
from gnn_neuro.train.train_rgcn import _eval_rgcn
from gnn_neuro.train.train_rotate import _eval_rotate


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_kg(path: Path) -> KnowledgeGraphIndex:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return KnowledgeGraphIndex(
        entity2id=blob["entity2id"],
        id2entity=blob["id2entity"],
        relation2id=blob["relation2id"],
        id2relation=blob["id2relation"],
        node_labels=blob["node_labels"],
        drug_ids=blob["drug_ids"],
        disease_ids=blob["disease_ids"],
        triples=blob["triples"],
        clinical=blob["clinical"],
    )


def seed_disease_entity_ids(kg: KnowledgeGraphIndex, cfg: dict) -> list[int]:
    seed_cfg = load_seed_config(ROOT / cfg["seeds_config"])
    seeds = resolve_seed_ids(seed_cfg, ROOT)
    ids = [kg.entity2id[d] for d in sorted(seeds["diseases"]) if d in kg.entity2id]
    print(f"drug→disease candidates restricted to {len(ids)} seed diseases")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default="outputs/graph/kg_index.pt")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--relation", default="INDICATION")
    parser.add_argument("--model", choices=["rotate", "rgcn", "both"], default="both")
    parser.add_argument("--out-name", default="metrics_INDICATION_seed_diseases.json")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    device = _device(cfg.get("device", "auto"))
    kg = load_kg(ROOT / args.graph)
    disease_cands = seed_disease_entity_ids(kg, cfg)
    runs = ROOT / args.runs
    results: dict = {}

    if args.model in ("rotate", "both"):
        mcfg = cfg["model"]["rotate"]
        dim = int(cfg["model"].get("dim", mcfg.get("dim", 128)))
        model = RotatE(kg.num_entities, kg.num_relations, dim=dim, margin=float(mcfg.get("margin", 9.0)))
        ckpt = runs / f"rotate_{args.relation}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to(device)
        metrics = _eval_rotate(
            model, kg, relation=args.relation, device=device, disease_candidate_ids=disease_cands
        )
        results["rotate"] = metrics
        print("RotatE:", json.dumps(metrics, indent=2))

    if args.model in ("rgcn", "both"):
        mcfg = cfg["model"]["rgcn"]
        dim = int(mcfg.get("hidden_dim", cfg["model"].get("dim", 128)))
        model = RGCNLinkPredictor(
            kg.num_entities,
            kg.num_relations,
            dim=dim,
            num_layers=int(mcfg.get("num_layers", 2)),
            dropout=float(mcfg.get("dropout", 0.2)),
        )
        ckpt = runs / f"rgcn_{args.relation}.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.to(device)
        triples = kg.triples.to(device)
        edge_index_per_rel = [e.to(device) for e in relation_edge_index(triples, kg.num_relations)]
        metrics = _eval_rgcn(
            model,
            kg,
            edge_index_per_rel,
            relation=args.relation,
            device=device,
            disease_candidate_ids=disease_cands,
        )
        results["rgcn"] = metrics
        print("R-GCN:", json.dumps(metrics, indent=2))

    out = runs / args.out_name
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
