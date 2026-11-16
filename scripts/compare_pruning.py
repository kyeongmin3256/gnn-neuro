#!/usr/bin/env python3
"""Pruning comparison table: seed set x k (nodes/edges/indication/test coverage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnn_neuro.data.load_optimuskg import load_optimuskg
from gnn_neuro.data.prune import _expand_disease_descendants, _genes_from_bpo, prune_subgraph
from gnn_neuro.data.seeds import load_seed_config, resolve_seed_ids


def clinical_indication_edges(
    edges: pl.DataFrame,
    drugs: set[str],
    diseases: set[str],
) -> pl.DataFrame:
    clin = edges.filter(pl.col("relation") == "INDICATION")
    rows = []
    for frm, to in zip(clin["from"].to_list(), clin["to"].to_list()):
        if frm in drugs and to in diseases:
            rows.append({"drug": frm, "disease": to})
        elif to in drugs and frm in diseases:
            rows.append({"drug": to, "disease": frm})
    if not rows:
        return pl.DataFrame(schema={"drug": pl.Utf8, "disease": pl.Utf8})
    return pl.DataFrame(rows).unique()


def make_test_edges(ind: pl.DataFrame, *, frac: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(ind.height)
    rng.shuffle(idx)
    n_test = max(1, int(round(frac * ind.height)))
    return ind[idx[:n_test].tolist()]


def coverage(test: pl.DataFrame, keep: set[str]) -> tuple[int, int]:
    if test.height == 0:
        return 0, 0
    hit = 0
    for d, dis in zip(test["drug"].to_list(), test["disease"].to_list()):
        if d in keep and dis in keep:
            hit += 1
    return hit, test.height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs/pruning_comparison.json")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    seed_cfg = load_seed_config(ROOT / cfg["seeds_config"])
    seeds = resolve_seed_ids(seed_cfg, ROOT)
    prune_cfg = seed_cfg.get("prune", {})
    type_allow = prune_cfg.get(
        "type_allow",
        ["GEN", "DRG", "DIS", "BPO", "PWY"],
    )

    print("Loading graph...")
    nodes, edges = load_optimuskg(
        lcc=cfg["kg"].get("lcc", True),
        nodes_path=cfg["kg"].get("nodes_path"),
        edges_path=cfg["kg"].get("edges_path"),
    )
    n_nodes = nodes.height
    n_edges = edges.height

    drugs = set(nodes.filter(pl.col("label") == "DRG")["id"].to_list())
    diseases = _expand_disease_descendants(edges, set(seeds["diseases"]))
    diseases &= set(nodes["id"].to_list())
    genes = set(seeds["gene"]) | _genes_from_bpo(edges, seeds["bpo"])
    genes &= set(nodes["id"].to_list())
    bpo = seeds["bpo"] & set(nodes["id"].to_list())

    # Fixed test pool: INDICATION edges on neurodegenerative disease seeds.
    ind_neuro = clinical_indication_edges(edges, drugs, diseases)
    test = make_test_edges(ind_neuro, frac=args.test_frac, seed=args.seed)
    print(
        f"Neuro INDICATION edges={ind_neuro.height}, "
        f"held-out test={test.height} (frac={args.test_frac})"
    )
    print(f"type_allow={type_allow}")
    print(f"seeds: BPO={len(bpo)}, gene={len(genes)}, disease={len(diseases)}")

    settings = [
        ("BPO+gene", bpo | genes, [1, 2]),
        ("BPO+gene+disease", bpo | genes | diseases, [1, 2, 3]),
    ]

    rows = []
    print()
    header = (
        f"{'Seeds':<22}{'k':>3}{'Nodes':>10}{'Edges':>10}"
        f"{'Indication':>12}{'Test covered':>16}"
    )
    print(header)
    print("-" * len(header))

    for name, _core, ks in settings:
        use_disease = "disease" in name
        for k in ks:
            _pn, pe, stats = prune_subgraph(
                nodes,
                edges,
                bpo_ids=bpo,
                disease_ids=diseases if use_disease else set(),
                gene_ids=genes,
                k=k,
                type_allow=type_allow,
                expand_disease_descendants=False,  # already expanded above
            )
            keep = set(_pn["id"].to_list())
            # If BPO+gene only, force disease seeds out of core by passing empty disease_ids;
            # but genes/bpo still expand. Good.
            hit, total = coverage(test, keep)
            row = {
                "seeds": name,
                "k": k,
                "nodes": stats["n_nodes"],
                "nodes_pct": 100.0 * stats["n_nodes"] / n_nodes,
                "edges": stats["n_edges"],
                "edges_pct": 100.0 * stats["n_edges"] / n_edges,
                "indication_kept": stats["n_indication"],
                "test_covered": hit,
                "test_total": total,
                "n_core": stats["n_core"],
                "n_drugs": stats["n_drugs"],
            }
            rows.append(row)
            print(
                f"{name:<22}{k:>3}"
                f"{row['nodes_pct']:>9.1f}%"
                f"{row['edges_pct']:>9.1f}%"
                f"{row['indication_kept']:>12,}"
                f"{hit:>7}/{total:<7}"
            )

    # Simple recommendation: maximize test coverage / edges retained (efficiency).
    def score(r: dict) -> tuple[float, float, float]:
        cov = r["test_covered"] / max(r["test_total"], 1)
        # Prefer higher coverage, then fewer edges, then fewer nodes.
        return (cov, -r["edges_pct"], -r["nodes_pct"])

    best = max(rows, key=score)
    out = {
        "n_nodes_full": n_nodes,
        "n_edges_full": n_edges,
        "type_allow": type_allow,
        "test_frac": args.test_frac,
        "test_total": int(test.height),
        "neuro_indication_total": int(ind_neuro.height),
        "rows": rows,
        "recommended": best,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(
        "Recommended:",
        f"{best['seeds']} k={best['k']} "
        f"(test {best['test_covered']}/{best['test_total']}, "
        f"nodes {best['nodes_pct']:.1f}%, edges {best['edges_pct']:.1f}%)",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
