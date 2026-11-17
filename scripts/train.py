#!/usr/bin/env python3
"""Train RotatE and/or R-GCN baselines and write metrics JSON."""

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
from gnn_neuro.data.tensorize import KnowledgeGraphIndex
from gnn_neuro.train.train_rgcn import train_rgcn
from gnn_neuro.train.train_rotate import train_rotate


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
    """Restrict drug→disease ranking to neurodegenerative seed diseases (~79)."""
    seed_cfg = load_seed_config(ROOT / cfg["seeds_config"])
    seeds = resolve_seed_ids(seed_cfg, ROOT)
    ids = [kg.entity2id[d] for d in sorted(seeds["diseases"]) if d in kg.entity2id]
    print(f"drug→disease candidates restricted to {len(ids)} seed diseases")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--graph", default="outputs/graph/kg_index.pt")
    parser.add_argument("--model", choices=["rotate", "rgcn", "both"], default="both")
    parser.add_argument("--relation", default="INDICATION")
    parser.add_argument("--out", default="outputs/runs")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    device = _device(cfg.get("device", "auto"))
    kg = load_kg(ROOT / args.graph)
    disease_cands = seed_disease_entity_ids(kg, cfg)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    results = {}
    if args.model in ("rotate", "both"):
        mcfg = cfg["model"]["rotate"]
        final_rotate = out_dir / f"rotate_{args.relation}.pt"
        model, info = train_rotate(
            kg,
            dim=int(cfg["model"].get("dim", mcfg.get("dim", 128))),
            margin=float(mcfg.get("margin", 9.0)),
            lr=float(mcfg.get("lr", 5e-4)),
            batch_size=int(mcfg.get("batch_size", 2048)),
            num_neg=int(mcfg.get("num_neg", 64)),
            epochs=int(mcfg.get("epochs", 50)),
            device=device,
            train_all_relations=bool(mcfg.get("train_all_relations", True)),
            relation=args.relation,
            ckpt_dir=ckpt_dir,
            save_every=int(args.save_every),
            resume=resume,
            final_path=final_rotate,
            disease_candidate_ids=disease_cands,
        )
        results["rotate"] = info["metrics"]
        results["rotate_train_loss"] = info["train_loss"][-1] if info["train_loss"] else None
        print("RotatE:", json.dumps(info["metrics"], indent=2))

    if args.model in ("rgcn", "both"):
        mcfg = cfg["model"]["rgcn"]
        final_rgcn = out_dir / f"rgcn_{args.relation}.pt"
        model, info = train_rgcn(
            kg,
            dim=int(mcfg.get("hidden_dim", cfg["model"].get("dim", 128))),
            num_layers=int(mcfg.get("num_layers", 2)),
            dropout=float(mcfg.get("dropout", 0.2)),
            lr=float(mcfg.get("lr", 1e-3)),
            batch_size=int(mcfg.get("batch_size", 512)),
            num_neg=int(mcfg.get("num_neg", 32)),
            epochs=int(mcfg.get("epochs", 40)),
            device=device,
            relation=args.relation,
            ckpt_dir=ckpt_dir,
            save_every=int(args.save_every),
            resume=resume,
            final_path=final_rgcn,
            disease_candidate_ids=disease_cands,
        )
        results["rgcn"] = info["metrics"]
        results["rgcn_train_loss"] = info["train_loss"][-1] if info["train_loss"] else None
        print("R-GCN:", json.dumps(info["metrics"], indent=2))

    (out_dir / f"metrics_{args.relation}.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_dir / f'metrics_{args.relation}.json'}")


if __name__ == "__main__":
    main()
