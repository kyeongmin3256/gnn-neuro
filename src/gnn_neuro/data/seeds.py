"""Load pruning seed IDs from YAML / CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _norm(x: str) -> str:
    return str(x).strip().replace(":", "_")


def load_seed_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_seed_ids(cfg: dict[str, Any], project_root: Path) -> dict[str, set[str]]:
    """Return seed groups: bpo, gene, disease_roots, diseases (roots+descendants)."""
    bpo = {_norm(x) for x in cfg.get("bpo_ids", [])}

    extra_csv = cfg.get("extra_term_csv")
    if extra_csv:
        p = project_root / extra_csv
        if p.exists():
            terms = pl.read_csv(p)
            id_col = "id" if "id" in terms.columns else terms.columns[0]
            if "label" in terms.columns:
                bpo |= {
                    _norm(r[id_col])
                    for r in terms.filter(pl.col("label") == "BPO").to_dicts()
                }
            else:
                bpo |= {_norm(x) for x in terms[id_col].to_list()}

    genes: set[str] = set()
    gene_csv = cfg.get("gene_seed_csv")
    if gene_csv:
        p = project_root / gene_csv
        if p.exists():
            gdf = pl.read_csv(p)
            id_col = "id" if "id" in gdf.columns else gdf.columns[0]
            if "label" in gdf.columns:
                genes = {
                    _norm(r[id_col])
                    for r in gdf.filter(pl.col("label") == "GEN").to_dicts()
                }
            else:
                genes = {_norm(x) for x in gdf[id_col].to_list()}

    disease_roots = {_norm(x) for x in cfg.get("disease_root_ids", [])}
    diseases = set(disease_roots)

    desc_csv = cfg.get("disease_descendants_csv")
    if desc_csv:
        p = project_root / desc_csv
        if p.exists():
            ddf = pl.read_csv(p)
            id_col = "id" if "id" in ddf.columns else ddf.columns[0]
            diseases |= {_norm(x) for x in ddf[id_col].to_list()}

    return {
        "bpo": bpo,
        "gene": genes,
        "disease_roots": disease_roots,
        "diseases": diseases,
    }
