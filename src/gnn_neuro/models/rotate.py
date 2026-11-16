"""RotatE knowledge-graph embedding baseline (Sun et al., ICLR 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatE(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 128,
        margin: float = 9.0,
        epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        # Complex embeddings: real/imag each of size dim.
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / dim]),
            requires_grad=False,
        )
        self.entity = nn.Embedding(num_entities, dim * 2)
        self.relation = nn.Embedding(num_relations, dim)
        self._init()

    def _init(self) -> None:
        nn.init.uniform_(
            self.entity.weight,
            -self.embedding_range.item(),
            self.embedding_range.item(),
        )
        nn.init.uniform_(
            self.relation.weight,
            -self.embedding_range.item(),
            self.embedding_range.item(),
        )

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Higher is better (negative distance)."""
        pi = 3.141592653589793
        re_h, im_h = torch.chunk(self.entity(h), 2, dim=-1)
        re_t, im_t = torch.chunk(self.entity(t), 2, dim=-1)
        phase = self.relation(r) / (self.embedding_range.item() / pi)
        re_r, im_r = torch.cos(phase), torch.sin(phase)
        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r
        re_score = re_score - re_t
        im_score = im_score - im_t
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)
        return self.margin - dist

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2])

    def loss(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        adversarial_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss."""
        pos_score = self.forward(pos)
        neg_score = self.forward(neg).view(pos.size(0), -1)
        pos_loss = -F.logsigmoid(pos_score).mean()
        weights = F.softmax(neg_score * adversarial_temperature, dim=-1).detach()
        neg_loss = -(weights * F.logsigmoid(-neg_score)).sum(dim=-1).mean()
        return (pos_loss + neg_loss) / 2
