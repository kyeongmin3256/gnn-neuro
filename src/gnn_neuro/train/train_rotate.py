"""Train RotatE on a knowledge graph."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gnn_neuro.eval.metrics import evaluate_disease_to_drugs, evaluate_drug_to_diseases
from gnn_neuro.models.rotate import RotatE
from gnn_neuro.train.negatives import corrupt_head, corrupt_tail


def train_rotate(
    kg,
    *,
    dim: int = 128,
    margin: float = 9.0,
    lr: float = 5e-4,
    batch_size: int = 2048,
    num_neg: int = 64,
    epochs: int = 50,
    device: torch.device,
    train_all_relations: bool = True,
    relation: str = "INDICATION",
) -> tuple[RotatE, dict[str, Any]]:
    model = RotatE(kg.num_entities, kg.num_relations, dim=dim, margin=margin).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if train_all_relations:
        train_triples = kg.triples
    else:
        train_triples = kg.clinical["train"][relation]
        # Also include other graph triples for structure? keep clinical-only if False.
        if train_triples.numel() == 0:
            train_triples = kg.triples

    if train_triples.numel() == 0:
        raise RuntimeError("No training triples for RotatE.")

    loader = DataLoader(
        TensorDataset(train_triples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n = 0
        for (batch,) in tqdm(loader, desc=f"RotatE epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            neg_t = corrupt_tail(batch, kg.num_entities, num_neg)
            neg_h = corrupt_head(batch, kg.num_entities, num_neg)
            neg = torch.cat([neg_t, neg_h], dim=0)
            loss = model.loss(batch, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.size(0)
            n += batch.size(0)
        history.append(total / max(n, 1))

    metrics = _eval_rotate(model, kg, relation=relation, device=device)
    return model, {"train_loss": history, "metrics": metrics}


@torch.no_grad()
def _eval_rotate(model: RotatE, kg, *, relation: str, device: torch.device) -> dict[str, Any]:
    model.eval()
    test = kg.clinical["test"][relation].to(device)
    train = kg.clinical["train"][relation]
    val = kg.clinical["val"][relation]
    filt = torch.cat([train, val, test], dim=0).to(device) if train.numel() else test

    def score_fn(h, r, t):
        return model.score(h, r, t)

    d2d = evaluate_disease_to_drugs(
        score_fn, test, kg.drug_ids, filtered_triples=filt, device=device
    )
    drug2dis = evaluate_drug_to_diseases(
        score_fn,
        test,
        kg.disease_ids,
        filtered_triples=filt,
        device=device,
    )
    return {"disease_to_drugs": d2d, "drug_to_diseases": drug2dis, "relation": relation}
