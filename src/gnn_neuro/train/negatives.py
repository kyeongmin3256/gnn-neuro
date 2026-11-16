"""Negative sampling utilities."""

from __future__ import annotations

import torch


def corrupt_tail(
    pos: torch.Tensor,
    num_entities: int,
    num_neg: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace tail with random entities -> [B * num_neg, 3]."""
    b = pos.size(0)
    neg = pos.repeat_interleave(num_neg, dim=0).clone()
    neg[:, 2] = torch.randint(
        0, num_entities, (b * num_neg,), generator=generator, device=pos.device
    )
    return neg


def corrupt_head(
    pos: torch.Tensor,
    num_entities: int,
    num_neg: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    b = pos.size(0)
    neg = pos.repeat_interleave(num_neg, dim=0).clone()
    neg[:, 0] = torch.randint(
        0, num_entities, (b * num_neg,), generator=generator, device=pos.device
    )
    return neg


def corrupt_drug_for_disease(
    pos: torch.Tensor,
    drug_ids: torch.Tensor,
    num_neg: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Corrupt drug (head) by sampling from the drug candidate set only."""
    b = pos.size(0)
    neg = pos.repeat_interleave(num_neg, dim=0).clone()
    idx = torch.randint(
        0, drug_ids.size(0), (b * num_neg,), generator=generator, device=pos.device
    )
    neg[:, 0] = drug_ids[idx]
    return neg
