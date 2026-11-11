"""Split-conformal thresholds for the decision-engine deciders (Jev, RF, ...).

Reads the per-image scores from scripts/run_decider_compare.py and replaces each
decider's own yes/no with a gate on its `p_degraded`: flag an image as degraded only
when p_degraded > q, where q is the conformal quantile of p_degraded over *clean*
images. A new clean image is then flagged with probability <= alpha (exchangeability
assumed), whatever the decider's own calibration.

Evaluation, with the small clean sample in mind:
  - Degraded recall uses the degraded images, which never enter calibration.
  - Clean false alarms are estimated by leave-one-out: each clean image is scored
    against the threshold fit on the other n-1. In-sample false alarm is 0 by
    construction and is not reported as evidence.
  - With n clean images the smallest reachable alpha is 1/(n+1); smaller alphas give
    q = inf (never flag), reported as such.

Outputs (in --out):
  thresholds_alpha<A>.json   {decider: threshold} for run_decider_compare.py --conformal
  summary.csv                per decider x alpha: q, LOO clean false alarm, recall, per-condition recall
  headline.json              the same, compact

Usage:
  python scripts/calibrate_deciders.py --scores results/deciders/main/scores.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from bolero.conformal import conformal_quantile
from bolero.config import RESULTS_DIR

# run_decider_compare.build_row flags p >= threshold; the conformal rule is p > q.
_ABOVE = 1e-9


def loo_false_alarm(clean_scores: np.ndarray, alpha: float) -> float:
    """Leave-one-out clean false-alarm rate of the conformal gate."""
    n = len(clean_scores)
    if n < 2:
        return float("nan")
    flags = [clean_scores[i] > conformal_quantile(np.delete(clean_scores, i), alpha) for i in range(n)]
    return float(np.mean(flags))


def calibrate(scores: pd.DataFrame, alphas: list[float], skip: set[str]) -> pd.DataFrame:
    rows = []
    for decider, g in scores.groupby("decider", sort=False):
        if decider in skip:
            continue
        g = g.dropna(subset=["p_degraded"])
        clean = g.loc[~g["true_degraded"], "p_degraded"].to_numpy(float)
        degraded = g[g["true_degraded"]]
        native_fa = float(g.loc[~g["true_degraded"], "pred_degraded"].astype(bool).mean())
        native_recall = float(degraded["pred_degraded"].astype(bool).mean())
        for alpha in alphas:
            q = conformal_quantile(clean, alpha)
            flagged = degraded["p_degraded"].to_numpy(float) > q
            per_cond = degraded.assign(flag=flagged).groupby("true_condition")["flag"].mean()
            rows.append({
                "decider": decider,
                "alpha": alpha,
                "n_clean": len(clean),
                "n_degraded": len(degraded),
                "q": q,
                "native_clean_false_alarm": native_fa,
                "native_recall": native_recall,
                "loo_clean_false_alarm": loo_false_alarm(clean, alpha),
                "recall": float(flagged.mean()),
                **{f"recall_{c}": float(v) for c, v in per_cond.items()},
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=RESULTS_DIR / "deciders" / "main" / "scores.csv")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--skip", nargs="*", default=["majority"],
                        help="deciders without a meaningful score (constant baselines)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "deciders" / "conformal")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    scores["true_degraded"] = scores["true_degraded"].astype(bool)
    summary = calibrate(scores, args.alphas, set(args.skip))

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    for alpha, g in summary.groupby("alpha"):
        thresholds = {r.decider: (r.q + _ABOVE if np.isfinite(r.q) else 1.0 + _ABOVE) for r in g.itertuples()}
        (args.out / f"thresholds_alpha{alpha:g}.json").write_text(json.dumps(thresholds, indent=2))

    cols = ["decider", "alpha", "q", "native_clean_false_alarm", "native_recall",
            "loo_clean_false_alarm", "recall"]
    headline = {"scores": str(args.scores), "rows": summary[cols].round(4).to_dict(orient="records")}
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2))
    with pd.option_context("display.width", 200):
        print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
