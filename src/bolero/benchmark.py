"""Standard robustness metrics, so our numbers line up with published benchmarks.

Implements the ImageNet-C protocol (Hendrycks & Dietterich, ICLR 2019):

  CE_c          = sum_s E(model, c, s) / sum_s E(ref, c, s)
  relative CE_c = sum_s [E(model, c, s) - E(model, clean)]
                  / sum_s [E(ref, c, s) - E(ref, clean)]
  mCE           = mean over corruptions c of CE_c

E is top-1 error. The reference is usually AlexNet; here any backbone or
method can serve as the reference (e.g. ResNet-50 with no restoration), so a
CE below 1 means "more robust than the reference".
"""

from __future__ import annotations

import pandas as pd


def error_table(summary: pd.DataFrame, acc_col: str = "acc_pert") -> pd.DataFrame:
    """Diagnostic or restoration summary -> one row per (perturbation, level) with top-1 error."""
    out = summary.groupby(["perturbation", "level"], as_index=False)[acc_col].mean()
    return out.assign(error=1.0 - out[acc_col]).drop(columns=acc_col)


def corruption_errors(model: pd.DataFrame, reference: pd.DataFrame,
                      model_clean_acc: float, reference_clean_acc: float) -> pd.DataFrame:
    """Per-perturbation CE and relative CE. Inputs come from `error_table`."""
    merged = model.merge(reference, on=["perturbation", "level"], suffixes=("_model", "_ref"))
    rows = []
    for name, g in merged.groupby("perturbation", sort=False):
        e_m, e_r = g["error_model"].sum(), g["error_ref"].sum()
        gap_m = (g["error_model"] - (1.0 - model_clean_acc)).sum()
        gap_r = (g["error_ref"] - (1.0 - reference_clean_acc)).sum()
        rows.append({
            "perturbation": name,
            "levels": len(g),
            "error_model": g["error_model"].mean(),
            "error_ref": g["error_ref"].mean(),
            "ce": e_m / e_r if e_r > 0 else float("nan"),
            "relative_ce": gap_m / gap_r if gap_r > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def mean_corruption_error(ce: pd.DataFrame, exclude: tuple[str, ...] = ()) -> dict[str, float]:
    kept = ce[~ce["perturbation"].isin(exclude)]
    return {"mce": float(kept["ce"].mean()), "relative_mce": float(kept["relative_ce"].mean()),
            "perturbations": int(len(kept))}
