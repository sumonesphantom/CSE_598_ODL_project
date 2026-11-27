"""Tests for scripts/calibrate_deciders.py (conformal thresholds for the deciders)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

cd = importlib.import_module("calibrate_deciders")


def _scores(rng, n_clean=200, n_deg=400):
    rows = []
    for name, shift in [("good", 3.0), ("weak", 0.5), ("majority", None)]:
        for i in range(n_clean + n_deg):
            degraded = i >= n_clean
            if shift is None:
                p = 1.0
            else:
                p = 1 / (1 + np.exp(-(rng.standard_normal() + (shift if degraded else -1.0))))
            rows.append({"decider": name, "image_id": f"im{i}", "true_degraded": degraded,
                         "p_degraded": p, "pred_degraded": p >= 0.5,
                         "true_condition": "noise" if degraded else "clean"})
    return pd.DataFrame(rows)


def test_loo_false_alarm_holds_alpha():
    rng = np.random.default_rng(0)
    clean = rng.uniform(size=200)
    assert cd.loo_false_alarm(clean, 0.1) <= 0.1 + 1e-9


def test_calibrate_ranks_deciders_and_skips_constant():
    out = cd.calibrate(_scores(np.random.default_rng(1)), [0.05, 0.1], skip={"majority"})
    assert set(out["decider"]) == {"good", "weak"}
    assert (out["loo_clean_false_alarm"] <= out["alpha"] + 0.02).all()
    at = out.set_index(["decider", "alpha"])["recall"]
    assert at["good", 0.05] > at["weak", 0.05]
    assert at["good", 0.1] >= at["good", 0.05]  # looser alpha never lowers recall
    assert "recall_noise" in out.columns


def test_threshold_file_is_compatible_with_run_decider_compare(tmp_path, monkeypatch):
    scores = _scores(np.random.default_rng(2), n_clean=20, n_deg=40)
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--scores", str(path), "--alphas", "0.05", "--out", str(tmp_path / "o")])
    cd.main()
    thresholds = json.loads((tmp_path / "o" / "thresholds_alpha0.05.json").read_text())
    clean_max = scores[(scores.decider == "good") & ~scores.true_degraded]["p_degraded"].max()
    # n=20, alpha=0.05 -> q is the clean max; run_decider_compare flags p >= threshold,
    # so the threshold must sit just above q for the max clean image not to be flagged.
    assert thresholds["good"] > clean_max
    assert thresholds["good"] - clean_max < 1e-6
