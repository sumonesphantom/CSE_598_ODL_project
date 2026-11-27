"""Tests for scripts/audit_plans.py decision logic (no model loads)."""

import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
ap = importlib.import_module("audit_plans")


def test_summarize_and_decide():
    rows = []
    for i in range(10):
        # gaussian_blur: plan fixes half the wrong images; gaussian_noise: plan breaks half the right ones
        rows += [
            {"image_id": f"i{i}", "perturbation": "gaussian_blur", "level": 1, "method": "none", "correct": False},
            {"image_id": f"i{i}", "perturbation": "gaussian_blur", "level": 1, "method": "oracle", "correct": i < 5},
            {"image_id": f"i{i}", "perturbation": "gaussian_noise", "level": 1, "method": "none", "correct": True},
            {"image_id": f"i{i}", "perturbation": "gaussian_noise", "level": 1, "method": "oracle", "correct": i < 5},
        ]
    summary = ap.summarize(pd.DataFrame(rows))
    s = summary.set_index("perturbation")
    assert s.loc["gaussian_blur", "delta_pp"] == 50 and s.loc["gaussian_blur", "fix_rate"] == 0.5
    assert s.loc["gaussian_noise", "delta_pp"] == -50 and s.loc["gaussian_noise", "break_rate"] == 0.5
    per = ap.decide(summary, 0.0).set_index("perturbation")
    assert not per.loc["gaussian_blur", "skip"] and per.loc["gaussian_noise", "skip"]
