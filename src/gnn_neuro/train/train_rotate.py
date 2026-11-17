"""Train RotatE on a knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gnn_neuro.eval.metrics import evaluate_disease_to_drugs, evaluate_drug_to_diseases
from gnn_neuro.models.rotate import RotatE
from gnn_neuro.train.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
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
    ckpt_dir: str | Path | None = None,
    save_every: int = 5,
    resume: bool = True,
    final_path: str | Path | None = None,
    disease_candidate_ids: list[int] | None = None,
) -> tuple[RotatE, dict[str, Any]]:
    model = RotatE(kg.num_entities, kg.num_relations, dim=dim, margin=margin).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if train_all_relations:
        train_triples = kg.triples
    else:
        train_triples = kg.clinical["train"][relation]
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
    start_epoch = 1
    prefix = f"rotate_{relation}"
    ckpt_path_dir = Path(ckpt_dir) if ckpt_dir is not None else None

    if resume and ckpt_path_dir is not None:
        latest = latest_checkpoint(ckpt_path_dir, prefix)
        if latest is not None:
            blob = load_checkpoint(latest, model, opt, map_location=device)
            history = list(blob.get("history") or [])
            start_epoch = int(blob.get("epoch", 0)) + 1

    model.train()
    for epoch in range(start_epoch, epochs + 1):
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

        if ckpt_path_dir is not None and (epoch % save_every == 0 or epoch == epochs):
            save_checkpoint(
                ckpt_path_dir / f"{prefix}_epoch{epoch:03d}.pt",
                model=model,
                optimizer=opt,
                epoch=epoch,
                history=history,
                meta={"relation": relation},
            )

    # Persist weights BEFORE evaluation so an eval crash does not lose training.
    if final_path is not None:
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        print(f"[ckpt] final weights saved {final_path}", flush=True)

    metrics = _eval_rotate(
        model,
        kg,
        relation=relation,
        device=device,
        disease_candidate_ids=disease_candidate_ids,
    )
    return model, {"train_loss": history, "metrics": metrics}


@torch.no_grad()
def _eval_rotate(
    model: RotatE,
    kg,
    *,
    relation: str,
    device: torch.device,
    disease_candidate_ids: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    test = kg.clinical["test"][relation].to(device)
    train = kg.clinical["train"][relation].to(device)
    val = kg.clinical["val"][relation].to(device)
    parts = [t for t in (train, val, test) if t.numel()]
    filt = torch.cat(parts, dim=0) if parts else test
    disease_pool = disease_candidate_ids if disease_candidate_ids is not None else kg.disease_ids

    def score_fn(h, r, t):
        return model.score(h, r, t)

    d2d = evaluate_disease_to_drugs(
        score_fn, test, kg.drug_ids, filtered_triples=filt, device=device
    )
    drug2dis = evaluate_drug_to_diseases(
        score_fn,
        test,
        disease_pool,
        filtered_triples=filt,
        device=device,
    )
    return {
        "disease_to_drugs": d2d,
        "drug_to_diseases": drug2dis,
        "relation": relation,
        "n_disease_candidates": len(disease_pool),
    }
