"""Checkpoint helpers for long GPU runs."""

from __future__ import annotations

from pathlib import Path

import torch


def latest_checkpoint(ckpt_dir: Path, prefix: str) -> Path | None:
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob(f"{prefix}_epoch*.pt"))
    return cands[-1] if cands else None


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[float],
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "meta": meta or {},
    }
    torch.save(payload, path)
    print(f"[ckpt] saved {path} (epoch={epoch})", flush=True)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    print(f"[ckpt] loaded {path} (epoch={blob.get('epoch')})", flush=True)
    return blob
