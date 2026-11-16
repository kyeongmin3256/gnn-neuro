"""Seed-centered k-hop pruning of OptimusKG."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import polars as pl


def _expand_disease_descendants(
    edges: pl.DataFrame,
    roots: set[str],
) -> set[str]:
    """Follow DIS-DIS PARENT edges where from=parent, to=child."""
    dis_parent = edges.filter(
        (pl.col("label") == "DIS-DIS") & (pl.col("relation") == "PARENT")
    )
    if dis_parent.is_empty():
        return set(roots)

    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in zip(dis_parent["from"].to_list(), dis_parent["to"].to_list()):
        children[parent].append(child)

    out = set(roots)
    q = deque(roots)
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v not in out:
                out.add(v)
                q.append(v)
    return out


def _genes_from_bpo(edges: pl.DataFrame, bpo_ids: set[str]) -> set[str]:
    """Genes linked to seed BPOs via BPO-GEN edges."""
    if not bpo_ids:
        return set()
    bpo_list = list(bpo_ids)
    bpo_gen = edges.filter(pl.col("label") == "BPO-GEN")
    hit = bpo_gen.filter(pl.col("from").is_in(bpo_list) | pl.col("to").is_in(bpo_list))
    genes: set[str] = set()
    for frm, to in zip(hit["from"].to_list(), hit["to"].to_list()):
        if frm in bpo_ids:
            genes.add(to)
        if to in bpo_ids:
            genes.add(frm)
    return genes


def prune_subgraph(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    bpo_ids: set[str],
    disease_ids: set[str],
    gene_ids: set[str] | None = None,
    k: int = 2,
    type_allow: Iterable[str] | None = None,
    expand_disease_descendants: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Extract a k-hop subgraph around neurodegenerative + BPO/gene seeds.

    Core seeds = BPO ∪ diseases ∪ genes (genes include BPO-derived + optional CSV).
    """
    type_allow = set(
        type_allow or {"GEN", "DRG", "DIS", "BPO", "PWY", "PHE", "MFN", "CCO", "ANA"}
    )

    node_ids = set(nodes["id"].to_list())
    type_df = nodes.select(["id", "label"])
    allowed_nodes = set(
        type_df.filter(pl.col("label").is_in(list(type_allow)))["id"].to_list()
    )

    diseases = set(disease_ids)
    if expand_disease_descendants:
        diseases = _expand_disease_descendants(edges, diseases)

    genes = set(gene_ids or set()) | _genes_from_bpo(edges, bpo_ids)
    genes &= node_ids
    bpo_present = bpo_ids & node_ids
    diseases &= node_ids

    core = (bpo_present | diseases | genes) & allowed_nodes
    keep = set(core)

    # Restrict traversal edges to endpoints in type_allow for speed/memory.
    trav = edges.filter(
        pl.col("from").is_in(list(allowed_nodes)) & pl.col("to").is_in(list(allowed_nodes))
    ).select(["from", "to"])

    for _ in range(k):
        if not keep:
            break
        keep_list = list(keep)
        nbr = trav.filter(pl.col("from").is_in(keep_list) | pl.col("to").is_in(keep_list))
        new_ids = set(nbr["from"].to_list()) | set(nbr["to"].to_list())
        new_ids &= allowed_nodes
        grown = new_ids - keep
        keep |= new_ids
        if not grown:
            break

    keep_list = list(keep)
    pruned_nodes = nodes.filter(pl.col("id").is_in(keep_list))
    pruned_edges = edges.filter(pl.col("from").is_in(keep_list) & pl.col("to").is_in(keep_list))

    stats = {
        "n_bpo_seeds": len(bpo_present),
        "n_disease_seeds": len(diseases),
        "n_gene_seeds": len(genes),
        "n_core": len(core),
        "k": k,
        "n_nodes": pruned_nodes.height,
        "n_edges": pruned_edges.height,
        "n_indication": pruned_edges.filter(pl.col("relation") == "INDICATION").height,
        "n_contraindication": pruned_edges.filter(pl.col("relation") == "CONTRAINDICATION").height,
        "n_drugs": pruned_nodes.filter(pl.col("label") == "DRG").height,
        "n_diseases_in_graph": pruned_nodes.filter(pl.col("label") == "DIS").height,
    }
    return pruned_nodes, pruned_edges, stats
