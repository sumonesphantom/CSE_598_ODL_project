"""Failure-case gallery helpers: pick cases, isolate the entity, summarize confusions.

Used by scripts/failure_gallery.py to show *why* the frozen classifier fails on a
class (e.g. church): the input, its depth outline, the SAM entity mask, and what
the classifier predicts on the image and on the entity alone (background removed,
the "information removal" spatial instruction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .perturbations import GRAY


def foreground_only(image: np.ndarray, mask: np.ndarray | None, fill: float = GRAY) -> np.ndarray:
    """Keep the entity, replace everything else with a constant fill. No mask -> unchanged."""
    if mask is None or not mask.any():
        return image
    out = np.full_like(image, fill)
    out[mask] = image[mask]
    return out


def select_cases(table: pd.DataFrame, n_fail: int, n_ok: int, seed: int = 0) -> pd.DataFrame:
    """Most confidently wrong failures first, plus a seeded sample of successes for contrast.

    `table` needs columns `correct` (bool) and `confidence` (top-1 probability).
    The returned frame has a `case` column: "fail" or "ok".
    """
    fails = table[~table["correct"]].sort_values("confidence", ascending=False).head(n_fail)
    oks = table[table["correct"]]
    oks = oks.sample(n=min(n_ok, len(oks)), random_state=seed) if len(oks) else oks
    return pd.concat([fails.assign(case="fail"), oks.assign(case="ok")])


def confusion_counts(table: pd.DataFrame) -> pd.DataFrame:
    """What the failures were called instead: predicted label -> count and share of failures."""
    fails = table[~table["correct"]]
    counts = fails["pred_label"].value_counts().rename_axis("pred_label").reset_index(name="count")
    counts["share_of_failures"] = counts["count"] / max(len(fails), 1)
    return counts


def transitions(before: pd.Series, after: pd.Series) -> dict[str, int]:
    """Correct/wrong transitions between two boolean correctness columns."""
    before, after = before.astype(bool), after.astype(bool)
    return {
        "fixed": int((~before & after).sum()),
        "broken": int((before & ~after).sum()),
        "still_wrong": int((~before & ~after).sum()),
        "still_right": int((before & after).sum()),
    }
