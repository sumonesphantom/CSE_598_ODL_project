"""Choosing cue and baseline strengths on development data, and loading the choice.

Strengths are selected on the Imagenette *train* split (scripts/tune_strengths.py)
and frozen before the restoration study runs on the *val* split. Every perturbed
version of an image is generated from that image, so all versions stay on the
same side of the split.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from . import baselines, cues


def method_key(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def select_strengths(summary: pd.DataFrame, max_clean_drop_pp: float = 1.0) -> pd.DataFrame:
    """Pick one strength per method from a dev-set sweep.

    `summary` has one row per (method, strength) with columns `acc_perturbed`
    (mean dev accuracy over all perturbation x severity conditions) and
    `clean_drop_pp` (accuracy lost on clean dev images vs. no processing).

    Among positive strengths whose clean drop is within `max_clean_drop_pp`, the
    one with the highest perturbed accuracy wins (ties go to the weaker strength).
    If no strength meets the constraint, the one with the smallest clean drop is
    kept and flagged, so every method is still evaluated rather than switched off.
    """
    rows = []
    for method, g in summary[summary["strength"] > 0].groupby("method", sort=False):
        ok = g[g["clean_drop_pp"] <= max_clean_drop_pp]
        if len(ok):
            best = ok.sort_values(["acc_perturbed", "strength"], ascending=[False, True]).iloc[0]
        else:
            best = g.sort_values(["clean_drop_pp", "strength"]).iloc[0]
        rows.append({**best.to_dict(), "constraint_met": bool(len(ok))})
    return pd.DataFrame(rows)


def default_strengths() -> dict[str, float]:
    out = {method_key("cue", k): v for k, v in cues.DEFAULT_STRENGTH.items()}
    out.update({method_key("baseline", k): v for k, v in baselines.DEFAULT_STRENGTH.items()})
    return out


def load_strengths(path: Path) -> tuple[dict[str, float], str]:
    """Tuned strengths from `path`, falling back to hand-set defaults for anything missing."""
    strengths = default_strengths()
    if not path.exists():
        warnings.warn(f"{path} not found; using hand-set default strengths. "
                      "Run scripts/tune_strengths.py to choose them on dev data.")
        return strengths, "default"
    tuned = json.loads(path.read_text())["strengths"]
    strengths.update(tuned)
    return strengths, "tuned" if set(tuned) >= set(strengths) else "partially tuned"
