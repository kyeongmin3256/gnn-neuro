"""Disease-centric / drug-centric ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _hits_at_k(ranks: list[float], k: int) -> float:
    if not ranks:
        return float("nan")
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


def rank_metrics(
    ranks: list[float],
    y_true_list: list[np.ndarray],
    y_score_list: list[np.ndarray],
) -> dict[str, float]:
    """Aggregate MRR / Hits and macro AUPRC over queries."""
    if not ranks:
        return {
            "mrr": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "auprc": float("nan"),
            "n_queries": 0,
        }
    mrr = float(np.mean([1.0 / r for r in ranks]))
    auprcs = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        auprcs.append(average_precision_score(y_true, y_score))
    return {
        "mrr": mrr,
        "hits@1": _hits_at_k(ranks, 1),
        "hits@3": _hits_at_k(ranks, 3),
        "hits@10": _hits_at_k(ranks, 10),
        "auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_queries": len(ranks),
    }


def summarize_rank_results(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return results


@torch.no_grad()
def evaluate_disease_to_drugs(
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_triples: torch.Tensor,
    drug_ids: list[int],
    *,
    filtered_triples: torch.Tensor | None = None,
    device: torch.device,
    batch_candidates: int = 2048,
) -> dict[str, float]:
    """For each test (drug, rel, disease), rank all drugs for that disease.

    Uses filtered setting: other known true (drug, rel, disease) pairs are
    removed from the candidate list when ranking a given positive.
    """
    if test_triples.numel() == 0:
        return rank_metrics([], [], [])

    drugs = torch.tensor(drug_ids, dtype=torch.long, device=device)
    drug_pos = {int(d): i for i, d in enumerate(drug_ids)}
    # Build filter set: (rel, disease) -> set(drug)
    filt: dict[tuple[int, int], set[int]] = defaultdict(set)
    if filtered_triples is not None and filtered_triples.numel():
        for h, r, t in filtered_triples.tolist():
            filt[(r, t)].add(h)

    # Group test positives by (relation, disease)
    by_query: dict[tuple[int, int], list[int]] = defaultdict(list)
    for h, r, t in test_triples.tolist():
        by_query[(r, t)].append(h)

    ranks: list[float] = []
    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for (r, disease), pos_drugs in by_query.items():
        pos_set = set(pos_drugs)
        known = set(filt.get((r, disease), set()))
        scores = torch.empty(drugs.size(0), device=device)
        for start in range(0, drugs.size(0), batch_candidates):
            batch = drugs[start : start + batch_candidates]
            rel = torch.full((batch.size(0),), r, dtype=torch.long, device=device)
            dis = torch.full((batch.size(0),), disease, dtype=torch.long, device=device)
            scores[start : start + batch_candidates] = score_fn(batch, rel, dis)

        score_np = scores.detach().cpu().numpy()
        drug_np = drugs.detach().cpu().numpy()
        y_true = np.array([1 if int(d) in pos_set else 0 for d in drug_np], dtype=np.int32)
        y_true_list.append(y_true)
        y_score_list.append(score_np)

        for pos in pos_drugs:
            idx = drug_pos.get(int(pos))
            if idx is None:
                continue
            s_pos = float(score_np[idx])
            better = 0
            for d_i, sc in zip(drug_np, score_np):
                d = int(d_i)
                if d == pos:
                    continue
                if d in known and d not in pos_set:
                    continue
                if sc > s_pos:
                    better += 1
            ranks.append(better + 1)

    return rank_metrics(ranks, y_true_list, y_score_list)


@torch.no_grad()
def evaluate_drug_to_diseases(
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_triples: torch.Tensor,
    disease_ids: list[int],
    *,
    filtered_triples: torch.Tensor | None = None,
    device: torch.device,
    batch_candidates: int = 2048,
) -> dict[str, float]:
    """For each test triple, rank all diseases for that drug (secondary metric)."""
    if test_triples.numel() == 0:
        return rank_metrics([], [], [])

    diseases = torch.tensor(disease_ids, dtype=torch.long, device=device)
    filt: dict[tuple[int, int], set[int]] = defaultdict(set)
    if filtered_triples is not None and filtered_triples.numel():
        for h, r, t in filtered_triples.tolist():
            filt[(h, r)].add(t)

    by_query: dict[tuple[int, int], list[int]] = defaultdict(list)
    for h, r, t in test_triples.tolist():
        by_query[(h, r)].append(t)

    ranks: list[float] = []
    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    disease_id_list = disease_ids
    for (drug, r), pos_dis in by_query.items():
        pos_set = set(pos_dis)
        known = set(filt.get((drug, r), set()))
        scores = torch.empty(diseases.size(0), device=device)
        for start in range(0, diseases.size(0), batch_candidates):
            batch = diseases[start : start + batch_candidates]
            rel = torch.full((batch.size(0),), r, dtype=torch.long, device=device)
            dru = torch.full((batch.size(0),), drug, dtype=torch.long, device=device)
            scores[start : start + batch_candidates] = score_fn(dru, rel, batch)

        score_np = scores.detach().cpu().numpy()
        dis_np = diseases.detach().cpu().numpy()
        y_true = np.array([1 if int(d) in pos_set else 0 for d in dis_np], dtype=np.int32)
        y_true_list.append(y_true)
        y_score_list.append(score_np)

        for pos in pos_dis:
            if pos not in disease_id_list:
                continue
            s_pos = float(scores[disease_id_list.index(pos)])
            better = 0
            for d_i, sc in zip(dis_np, score_np):
                d = int(d_i)
                if d == pos:
                    continue
                if d in known and d not in pos_set:
                    continue
                if sc > s_pos:
                    better += 1
            ranks.append(better + 1)

    return rank_metrics(ranks, y_true_list, y_score_list)
