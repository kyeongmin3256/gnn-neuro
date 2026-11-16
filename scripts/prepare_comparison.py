#!/usr/bin/env python3
"""Build full and pruned graphs that share the same clinical split.

This is the fair setup for the paper question: does domain-informed pruning
help relative to the full OptimusKG under identical train/val/test edges?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_neuro.data.load_optimuskg import load_optimuskg, summarize_graph
from gnn_neuro.data.prune import prune_subgraph
from gnn_neuro.data.seeds import load_seed_config, resolve_seed_ids
from gnn_neuro.data.splits import apply_clinical_split, split_drug_disease_edges
from gnn_neuro.data.tensorize import build_kg_index


def _write_graph(
    out: Path,
    *,
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    split,
    stats: dict,
    seeds: dict,
    clinical_relations: tuple[str, ...],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    kg = build_kg_index(
        nodes,
        split.graph_edges,
        {"train": split.train, "val": split.val, "test": split.test},
        clinical_relations=clinical_relations,
    )
    nodes.write_parquet(out / "nodes.parquet")
    edges.write_parquet(out / "edges.parquet")
    split.graph_edges.write_parquet(out / "graph_edges.parquet")
    split.train.write_parquet(out / "split_train.parquet")
    split.val.write_parquet(out / "split_val.parquet")
    split.test.write_parquet(out / "split_test.parquet")
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
        out / "kg_index.pt",
    )
    (out / "stats.json").write_text(
        json.dumps(
            {
                "stats": stats,
                "seed_counts": {k: len(v) for k, v in seeds.items()},
                "split": {
                    "train": split.train.height,
                    "val": split.val.height,
                    "test": split.test.height,
                },
            },
            indent=2,
        )
    )
    print(
        f"Wrote {out} | nodes={nodes.height} edges={edges.height} "
        f"split={split.train.height}/{split.val.height}/{split.test.height} "
        f"drugs={len(kg.drug_ids)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-full", default="outputs/graph_full")
    parser.add_argument("--out-pruned", default="outputs/graph_pruned")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    seed_cfg = load_seed_config(ROOT / cfg["seeds_config"])
    seeds = resolve_seed_ids(seed_cfg, ROOT)
    split_cfg = cfg["split"]
    clinical_relations = tuple(split_cfg.get("relations", ["INDICATION", "CONTRAINDICATION"]))

    print("Loading OptimusKG...")
    full_nodes, full_edges = load_optimuskg(
        lcc=cfg["kg"].get("lcc", True),
        nodes_path=cfg["kg"].get("nodes_path"),
        edges_path=cfg["kg"].get("edges_path"),
        force_download=args.force_download,
    )
    print("Full graph:", summarize_graph(full_nodes, full_edges))

    disease_filter = seeds["diseases"] if split_cfg.get("restrict_diseases_to_seeds", True) else None
    # Shared clinical split is defined on the FULL graph.
    shared = split_drug_disease_edges(
        full_nodes,
        full_edges,
        relations=clinical_relations,
        disease_ids=disease_filter,
        ratios=tuple(split_cfg.get("ratios", [0.8, 0.1, 0.1])),
        seed=int(split_cfg.get("seed", 42)),
        mask_eval_edges=bool(split_cfg.get("mask_eval_edges", True)),
    )
    print(
        f"Shared clinical split: train={shared.train.height}, "
        f"val={shared.val.height}, test={shared.test.height}"
    )

    # Save shared split tables once for auditability.
    split_dir = ROOT / "outputs" / "shared_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    shared.train.write_parquet(split_dir / "split_train.parquet")
    shared.val.write_parquet(split_dir / "split_val.parquet")
    shared.test.write_parquet(split_dir / "split_test.parquet")

    _write_graph(
        ROOT / args.out_full,
        nodes=full_nodes,
        edges=full_edges,
        split=shared,
        stats=summarize_graph(full_nodes, full_edges),
        seeds=seeds,
        clinical_relations=clinical_relations,
    )

    prune_cfg = seed_cfg.get("prune", {})
    pruned_nodes, pruned_edges, prune_stats = prune_subgraph(
        full_nodes,
        full_edges,
        bpo_ids=seeds["bpo"],
        disease_ids=seeds["diseases"],
        gene_ids=seeds["gene"],
        k=int(prune_cfg.get("k", 1)),
        type_allow=prune_cfg.get("type_allow"),
        expand_disease_descendants=True,
    )
    print("Pruned stats:", json.dumps(prune_stats, indent=2))
    keep = set(pruned_nodes["id"].to_list())
    pruned_split = apply_clinical_split(
        pruned_edges,
        shared.train,
        shared.val,
        shared.test,
        keep_nodes=keep,
        mask_eval_edges=bool(split_cfg.get("mask_eval_edges", True)),
    )
    print(
        "Pruned split coverage: "
        f"train={pruned_split.train.height}/{shared.train.height}, "
        f"val={pruned_split.val.height}/{shared.val.height}, "
        f"test={pruned_split.test.height}/{shared.test.height}"
    )
    _write_graph(
        ROOT / args.out_pruned,
        nodes=pruned_nodes,
        edges=pruned_edges,
        split=pruned_split,
        stats=prune_stats,
        seeds=seeds,
        clinical_relations=clinical_relations,
    )
    print("Done. Full and pruned graphs share the same clinical split seed.")


if __name__ == "__main__":
    main()
