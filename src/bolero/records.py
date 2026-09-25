"""Helpers for turning batched predictions into analysis tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# BatchPredictor column -> suffix used in analysis tables (e.g. clean_pred, pert_conf).
PRED_COLUMNS = {
    "pred_index": "pred",
    "pred_conf": "conf",
    "true_prob": "true_prob",
    "correct": "correct",
    "classify_ms": "classify_ms",
}


def prefixed(rows: list[dict], prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.rename(columns={k: f"{prefix}_{v}" for k, v in PRED_COLUMNS.items()})


def save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
