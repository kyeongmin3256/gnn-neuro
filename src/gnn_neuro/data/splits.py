"""Train/val/test splits for drug–disease clinical edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


CLINICAL_RELATIONS = ("INDICATION", "CONTRAINDICATION")


@dataclass
class EdgeSplit:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    # Full working-graph edges with val/test clinical edges removed (for MP / KGE).
    graph_edges: pl.DataFrame


def split_drug_disease_edges(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    relations: tuple[str, ...] = CLINICAL_RELATIONS,
    disease_ids: set[str] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    mask_eval_edges: bool = True,
) -> EdgeSplit:
    """Random-link holdout on drug–disease clinical relations (TxGNN-style).

    Only DRG–DIS edges with `relation` in `relations` are split. If `disease_ids`
    is set, only edges touching those diseases are used for supervision/eval.
    """
    drugs = set(nodes.filter(pl.col("label") == "DRG")["id"].to_list())
    diseases = set(nodes.filter(pl.col("label") == "DIS")["id"].to_list())
    if disease_ids is not None:
        diseases &= set(disease_ids)

    clinical = edges.filter(pl.col("relation").is_in(list(relations)))
    # Orient as (drug, disease) regardless of edge direction in the dump.
    rows = []
    for frm, to, label, rel in zip(
        clinical["from"].to_list(),
        clinical["to"].to_list(),
        clinical["label"].to_list(),
        clinical["relation"].to_list(),
    ):
        if frm in drugs and to in diseases:
            rows.append({"drug": frm, "disease": to, "label": label, "relation": rel})
        elif to in drugs and frm in diseases:
            rows.append({"drug": to, "disease": frm, "label": label, "relation": rel})

    if not rows:
        empty = pl.DataFrame(
            schema={
                "drug": pl.Utf8,
                "disease": pl.Utf8,
                "label": pl.Utf8,
                "relation": pl.Utf8,
            }
        )
        return EdgeSplit(train=empty, val=empty, test=empty, graph_edges=edges)

    dd = pl.DataFrame(rows).unique(subset=["drug", "disease", "relation"])
    rng = np.random.default_rng(seed)
    idx = np.arange(dd.height)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train = dd[train_idx.tolist()]
    val = dd[val_idx.tolist()]
    test = dd[test_idx.tolist()]

    graph_edges = edges
    if mask_eval_edges:
        holdout = pl.concat([val, test])
        # Build a set of (drug, disease, relation) to drop from the raw edge list.
        hold = {
            (r["drug"], r["disease"], r["relation"]) for r in holdout.to_dicts()
        }
        keep_mask = []
        for frm, to, rel in zip(edges["from"].to_list(), edges["to"].to_list(), edges["relation"].to_list()):
            a, b = frm, to
            drop = ((a, b, rel) in hold) or ((b, a, rel) in hold)
            keep_mask.append(not drop)
        graph_edges = edges.filter(pl.Series(keep_mask))

    return EdgeSplit(train=train, val=val, test=test, graph_edges=graph_edges)


def apply_clinical_split(
    edges: pl.DataFrame,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    *,
    keep_nodes: set[str] | None = None,
    mask_eval_edges: bool = True,
) -> EdgeSplit:
    """Reuse a shared clinical split on a (possibly pruned) edge set.

    If `keep_nodes` is set, clinical rows whose drug/disease fall outside the
    node set are dropped. This keeps full vs pruned comparisons aligned.
    """

    def _filter(df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df
        if keep_nodes is None:
            return df
        return df.filter(
            pl.col("drug").is_in(list(keep_nodes)) & pl.col("disease").is_in(list(keep_nodes))
        )

    train_f, val_f, test_f = _filter(train), _filter(val), _filter(test)
    graph_edges = edges
    if mask_eval_edges:
        holdout = pl.concat([val_f, test_f]) if (val_f.height + test_f.height) else val_f
        hold = {(r["drug"], r["disease"], r["relation"]) for r in holdout.to_dicts()}
        keep_mask = []
        for frm, to, rel in zip(
            edges["from"].to_list(), edges["to"].to_list(), edges["relation"].to_list()
        ):
            drop = ((frm, to, rel) in hold) or ((to, frm, rel) in hold)
            keep_mask.append(not drop)
        graph_edges = edges.filter(pl.Series(keep_mask))
    return EdgeSplit(train=train_f, val=val_f, test=test_f, graph_edges=graph_edges)
