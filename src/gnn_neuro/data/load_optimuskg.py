"""Load OptimusKG via the official client or local parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def load_optimuskg(
    *,
    lcc: bool = True,
    nodes_path: str | Path | None = None,
    edges_path: str | Path | None = None,
    force_download: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return `(nodes, edges)` with columns `id,label` and `from,to,label,relation`.

    Prefer explicit parquet paths when provided; otherwise use `optimuskg.load_graph`.
    """
    if nodes_path is not None and edges_path is not None:
        nodes = pl.read_parquet(nodes_path)
        edges = pl.read_parquet(edges_path)
    else:
        import optimuskg

        nodes, edges = optimuskg.load_graph(lcc=lcc, force=force_download)

    nodes = _standardize_nodes(nodes)
    edges = _standardize_edges(edges)
    return nodes, edges


def _standardize_nodes(nodes: pl.DataFrame) -> pl.DataFrame:
    cols = {c.lower(): c for c in nodes.columns}
    id_col = cols.get("id")
    label_col = cols.get("label")
    if id_col is None or label_col is None:
        raise ValueError(f"nodes missing id/label columns: {nodes.columns}")

    out = nodes.select(
        pl.col(id_col).cast(pl.Utf8).str.replace_all(":", "_").alias("id"),
        pl.col(label_col).cast(pl.Utf8).alias("label"),
    ).unique(subset=["id"])
    return out


def _standardize_edges(edges: pl.DataFrame) -> pl.DataFrame:
    cols = {c.lower(): c for c in edges.columns}
    # OptimusKG uses `from` / `to`; some dumps use source/target.
    frm = cols.get("from") or cols.get("source") or cols.get("x_id")
    to = cols.get("to") or cols.get("target") or cols.get("y_id")
    label = cols.get("label")
    relation = cols.get("relation") or cols.get("display_relation")
    if frm is None or to is None:
        raise ValueError(f"edges missing from/to columns: {edges.columns}")
    if label is None:
        raise ValueError(f"edges missing label column: {edges.columns}")
    if relation is None:
        relation = label

    out = edges.select(
        pl.col(frm).cast(pl.Utf8).str.replace_all(":", "_").alias("from"),
        pl.col(to).cast(pl.Utf8).str.replace_all(":", "_").alias("to"),
        pl.col(label).cast(pl.Utf8).alias("label"),
        pl.col(relation).cast(pl.Utf8).alias("relation"),
    )
    # Drop self-loops / nulls.
    out = out.filter(
        pl.col("from").is_not_null()
        & pl.col("to").is_not_null()
        & (pl.col("from") != pl.col("to"))
    )
    return out


def summarize_graph(nodes: pl.DataFrame, edges: pl.DataFrame) -> dict[str, Any]:
    return {
        "n_nodes": nodes.height,
        "n_edges": edges.height,
        "node_types": nodes.group_by("label").len().sort("len", descending=True).to_dicts(),
        "relations": edges.group_by("relation").len().sort("len", descending=True).to_dicts(),
        "edge_labels": edges.group_by("label").len().sort("len", descending=True).to_dicts(),
    }
