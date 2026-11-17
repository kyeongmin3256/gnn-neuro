"""Train R-GCN + DistMult on drug–disease edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gnn_neuro.data.tensorize import relation_edge_index
from gnn_neuro.eval.metrics import evaluate_disease_to_drugs, evaluate_drug_to_diseases
from gnn_neuro.models.rgcn import RGCNLinkPredictor
from gnn_neuro.train.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from gnn_neuro.train.negatives import corrupt_drug_for_disease


def train_rgcn(
    kg,
    *,
    dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 512,
    num_neg: int = 32,
    epochs: int = 40,
    device: torch.device,
    relation: str = "INDICATION",
    ckpt_dir: str | Path | None = None,
    save_every: int = 5,
    resume: bool = True,
    final_path: str | Path | None = None,
) -> tuple[RGCNLinkPredictor, dict[str, Any]]:
    model = RGCNLinkPredictor(
        kg.num_entities, kg.num_relations, dim=dim, num_layers=num_layers, dropout=dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    triples = kg.triples.to(device)
    edge_index_per_rel = [e.to(device) for e in relation_edge_index(triples, kg.num_relations)]

    train_pos = kg.clinical["train"][relation]
    if train_pos.numel() == 0:
        raise RuntimeError(f"No training edges for relation={relation}")

    drug_ids = torch.tensor(kg.drug_ids, dtype=torch.long, device=device)
    loader = DataLoader(
        TensorDataset(train_pos),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: list[float] = []
    start_epoch = 1
    prefix = f"rgcn_{relation}"
    ckpt_path_dir = Path(ckpt_dir) if ckpt_dir is not None else None

    if resume and ckpt_path_dir is not None:
        latest = latest_checkpoint(ckpt_path_dir, prefix)
        if latest is not None:
            blob = load_checkpoint(latest, model, opt, map_location=device)
            history = list(blob.get("history") or [])
            start_epoch = int(blob.get("epoch", 0)) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for (batch,) in tqdm(loader, desc=f"R-GCN epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            neg = corrupt_drug_for_disease(batch, drug_ids, num_neg)
            loss = model.loss(batch, neg, edge_index_per_rel)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.size(0)
            n += batch.size(0)
        history.append(total / max(n, 1))

        if ckpt_path_dir is not None and (epoch % save_every == 0 or epoch == epochs):
            save_checkpoint(
                ckpt_path_dir / f"{prefix}_epoch{epoch:03d}.pt",
                model=model,
                optimizer=opt,
                epoch=epoch,
                history=history,
                meta={"relation": relation},
            )

    if final_path is not None:
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        print(f"[ckpt] final weights saved {final_path}", flush=True)

    metrics = _eval_rgcn(model, kg, edge_index_per_rel, relation=relation, device=device)
    return model, {"train_loss": history, "metrics": metrics}


@torch.no_grad()
def _eval_rgcn(
    model: RGCNLinkPredictor,
    kg,
    edge_index_per_rel: list[torch.Tensor],
    *,
    relation: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ent = model.encode(edge_index_per_rel)
    test = kg.clinical["test"][relation].to(device)
    train = kg.clinical["train"][relation].to(device)
    val = kg.clinical["val"][relation].to(device)
    parts = [t for t in (train, val, test) if t.numel()]
    filt = torch.cat(parts, dim=0) if parts else test

    def score_fn(h, r, t):
        return model.decoder.score(h, r, t, ent)

    d2d = evaluate_disease_to_drugs(
        score_fn, test, kg.drug_ids, filtered_triples=filt, device=device
    )
    drug2dis = evaluate_drug_to_diseases(
        score_fn, test, kg.disease_ids, filtered_triples=filt, device=device
    )
    return {"disease_to_drugs": d2d, "drug_to_diseases": drug2dis, "relation": relation}
