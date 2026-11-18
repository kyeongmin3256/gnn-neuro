"""R-GCN encoder + DistMult decoder for drug–disease link prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        dropout: float = 0.0,
        edge_chunk_size: int = 500_000,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.edge_chunk_size = edge_chunk_size
        self.weight = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
        )
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_per_rel: list[torch.Tensor]) -> torch.Tensor:
        # Chunked per-relation aggregation so full-graph MP fits in ~24GB.
        out = self.self_loop(x)
        for r, edge_index in enumerate(edge_index_per_rel):
            if edge_index.numel() == 0:
                continue
            # Edges may live on CPU; move only the active chunk to x.device.
            src_all, dst_all = edge_index[0], edge_index[1]
            n_edges = src_all.size(0)
            agg = torch.zeros(x.size(0), out.size(-1), device=x.device, dtype=x.dtype)
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            for start in range(0, n_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, n_edges)
                src = src_all[start:end].to(x.device, non_blocking=True)
                dst = dst_all[start:end].to(x.device, non_blocking=True)
                msg = self.weight[r](x[src])
                agg.index_add_(0, dst, msg)
                deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
                del msg, src, dst
            deg = deg.clamp(min=1).unsqueeze(-1)
            out = out + agg / deg
            del agg, deg
        return self.dropout(F.relu(out))


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        in_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_entities, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [
                RGCNLayer(dims[i], dims[i + 1], num_relations, dropout=dropout)
                for i in range(num_layers)
            ]
        )

    def forward(self, edge_index_per_rel: list[torch.Tensor]) -> torch.Tensor:
        h = self.emb.weight
        for layer in self.layers:
            h = layer(h, edge_index_per_rel)
        return h


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, dim: int) -> None:
        super().__init__()
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
        return (ent[h] * self.rel(r) * ent[t]).sum(dim=-1)


class RGCNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(
            num_entities, num_relations, in_dim=dim, hidden_dim=dim, num_layers=num_layers, dropout=dropout
        )
        self.decoder = DistMultDecoder(num_relations, dim)

    def encode(self, edge_index_per_rel: list[torch.Tensor]) -> torch.Tensor:
        return self.encoder(edge_index_per_rel)

    def score_triples(
        self, triples: torch.Tensor, ent: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder.score(triples[:, 0], triples[:, 1], triples[:, 2], ent)

    def loss(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        edge_index_per_rel: list[torch.Tensor] | None = None,
        *,
        ent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ent is None:
            if edge_index_per_rel is None:
                raise ValueError("Provide edge_index_per_rel or ent")
            ent = self.encode(edge_index_per_rel)
        pos_s = self.score_triples(pos, ent)
        neg_s = self.score_triples(neg, ent)
        scores = torch.cat([pos_s, neg_s], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_s.size(0), device=scores.device),
                torch.zeros(neg_s.size(0), device=scores.device),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(scores, labels)
