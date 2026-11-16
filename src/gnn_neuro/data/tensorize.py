"""Map string KG ids to contiguous integer indices for torch models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch


@dataclass
class KnowledgeGraphIndex:
    entity2id: dict[str, int]
    id2entity: list[str]
    relation2id: dict[str, int]
    id2relation: list[str]
    node_labels: dict[str, str]
    drug_ids: list[int]
    disease_ids: list[int]
    # triples as LongTensor [E, 3] = (h, r, t) over graph_edges
    triples: torch.Tensor
    # clinical supervision triples by split / relation name
    clinical: dict[str, dict[str, torch.Tensor]]

    @property
    def num_entities(self) -> int:
        return len(self.id2entity)

    @property
    def num_relations(self) -> int:
        return len(self.id2relation)

    def entity_index(self, name: str) -> int:
        return self.entity2id[name]


def build_kg_index(
    nodes: pl.DataFrame,
    graph_edges: pl.DataFrame,
    split_tables: dict[str, pl.DataFrame],
    *,
    clinical_relations: tuple[str, ...] = ("INDICATION", "CONTRAINDICATION"),
) -> KnowledgeGraphIndex:
    entities = nodes["id"].to_list()
    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = entities
    node_labels = dict(zip(nodes["id"].to_list(), nodes["label"].to_list()))

    rels = sorted(set(graph_edges["relation"].to_list()) | set(clinical_relations))
    relation2id = {r: i for i, r in enumerate(rels)}
    id2relation = rels

    trip = []
    for frm, to, rel in zip(
        graph_edges["from"].to_list(),
        graph_edges["to"].to_list(),
        graph_edges["relation"].to_list(),
    ):
        if frm not in entity2id or to not in entity2id:
            continue
        trip.append((entity2id[frm], relation2id[rel], entity2id[to]))
    triples = torch.tensor(trip, dtype=torch.long) if trip else torch.zeros((0, 3), dtype=torch.long)

    drug_ids = [entity2id[e] for e, lab in node_labels.items() if lab == "DRG" and e in entity2id]
    disease_ids = [entity2id[e] for e, lab in node_labels.items() if lab == "DIS" and e in entity2id]

    clinical: dict[str, dict[str, torch.Tensor]] = {}
    for split_name, df in split_tables.items():
        clinical[split_name] = {}
        for rel in clinical_relations:
            sub = df.filter(pl.col("relation") == rel) if df.height else df
            rows = []
            if sub.height:
                for drug, disease in zip(sub["drug"].to_list(), sub["disease"].to_list()):
                    if drug in entity2id and disease in entity2id:
                        rows.append((entity2id[drug], relation2id[rel], entity2id[disease]))
            clinical[split_name][rel] = (
                torch.tensor(rows, dtype=torch.long)
                if rows
                else torch.zeros((0, 3), dtype=torch.long)
            )

    return KnowledgeGraphIndex(
        entity2id=entity2id,
        id2entity=id2entity,
        relation2id=relation2id,
        id2relation=id2relation,
        node_labels=node_labels,
        drug_ids=drug_ids,
        disease_ids=disease_ids,
        triples=triples,
        clinical=clinical,
    )


def relation_edge_index(triples: torch.Tensor, num_relations: int) -> list[torch.Tensor]:
    """Return list of COO edge_index [2, E_r] per relation (for R-GCN)."""
    out: list[torch.Tensor] = []
    for r in range(num_relations):
        mask = triples[:, 1] == r
        edges = triples[mask][:, [0, 2]].t().contiguous()  # [2, E]
        out.append(edges)
    return out
