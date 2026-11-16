#!/usr/bin/env python3
"""Load OptimusKG, prune with neurodegenerative seeds, write artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_neuro.data.load_optimuskg import load_optimuskg, summarize_graph
from gnn_neuro.data.prune import prune_subgraph
from gnn_neuro.data.seeds import load_seed_config, resolve_seed_ids
from gnn_neuro.data.splits import split_drug_disease_edges
from gnn_neuro.data.tensorize import build_kg_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs/graph")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    seed_cfg = load_seed_config(ROOT / cfg["seeds_config"])
    seeds = resolve_seed_ids(seed_cfg, ROOT)

    print("Loading OptimusKG...")
    nodes, edges = load_optimuskg(
        lcc=cfg["kg"].get("lcc", True),
        nodes_path=cfg["kg"].get("nodes_path"),
        edges_path=cfg["kg"].get("edges_path"),
        force_download=args.force_download,
    )
    print("Full graph:", summarize_graph(nodes, edges))

    if cfg.get("graph_mode", "pruned") == "pruned":
        prune_cfg = seed_cfg.get("prune", {})
        nodes, edges, stats = prune_subgraph(
            nodes,
            edges,
            bpo_ids=seeds["bpo"],
            disease_ids=seeds["diseases"],
            gene_ids=seeds["gene"],
            k=int(prune_cfg.get("k", 2)),
            type_allow=prune_cfg.get("type_allow"),
            expand_disease_descendants=True,
        )
        print("Pruned graph stats:", json.dumps(stats, indent=2))
    else:
        stats = summarize_graph(nodes, edges)

    split_cfg = cfg["split"]
    disease_filter = seeds["diseases"] if split_cfg.get("restrict_diseases_to_seeds", True) else None
    split = split_drug_disease_edges(
        nodes,
        edges,
        relations=tuple(split_cfg.get("relations", ["INDICATION", "CONTRAINDICATION"])),
        disease_ids=disease_filter,
        ratios=tuple(split_cfg.get("ratios", [0.8, 0.1, 0.1])),
        seed=int(split_cfg.get("seed", 42)),
        mask_eval_edges=bool(split_cfg.get("mask_eval_edges", True)),
    )
    print(
        f"Split sizes: train={split.train.height}, val={split.val.height}, test={split.test.height}"
    )
    n_drugs = nodes.filter(pl.col("label") == "DRG").height
    print(f"Candidates: ALL drugs in graph (n={n_drugs}) — TxGNN-aligned")

    kg = build_kg_index(
        nodes,
        split.graph_edges,
        {"train": split.train, "val": split.val, "test": split.test},
        clinical_relations=tuple(split_cfg.get("relations", ["INDICATION", "CONTRAINDICATION"])),
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(out / "nodes.parquet")
    edges.write_parquet(out / "edges.parquet")
    split.graph_edges.write_parquet(out / "graph_edges.parquet")
    split.train.write_parquet(out / "split_train.parquet")
    split.val.write_parquet(out / "split_val.parquet")
    split.test.write_parquet(out / "split_test.parquet")
    torch_path = out / "kg_index.pt"
    import torch

    torch.save(
        {
            "entity2id": kg.entity2id,
            "id2entity": kg.id2entity,
            "relation2id": kg.relation2id,
            "id2relation": kg.id2relation,
            "node_labels": kg.node_labels,
            "drug_ids": kg.drug_ids,
            "disease_ids": kg.disease_ids,
            "triples": kg.triples,
            "clinical": kg.clinical,
            "stats": stats,
            "seed_counts": {k: len(v) for k, v in seeds.items()},
        },
        torch_path,
    )
    (out / "stats.json").write_text(json.dumps({"prune": stats, "seed_counts": {k: len(v) for k, v in seeds.items()}}, indent=2))
    print(f"Wrote artifacts to {out}")


if __name__ == "__main__":
    main()
