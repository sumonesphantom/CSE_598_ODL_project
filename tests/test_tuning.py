import json

import pandas as pd

from bolero.tuning import default_strengths, load_strengths, select_strengths


def sweep(rows):
    return pd.DataFrame(rows, columns=["method", "strength", "acc_perturbed", "clean_drop_pp"])


def test_picks_best_strength_within_clean_budget():
    summary = sweep([
        ("cue:a", 0.0, 0.50, 0.0),
        ("cue:a", 0.2, 0.55, 0.5),
        ("cue:a", 0.5, 0.60, 3.0),   # best on perturbed, but too costly on clean
    ])
    chosen = select_strengths(summary, max_clean_drop_pp=1.0).iloc[0]
    assert chosen["strength"] == 0.2 and chosen["constraint_met"]


def test_ties_go_to_the_weaker_strength():
    summary = sweep([("cue:a", 0.3, 0.6, 0.0), ("cue:a", 0.1, 0.6, 0.0)])
    assert select_strengths(summary).iloc[0]["strength"] == 0.1


def test_never_picks_zero_and_flags_when_budget_unmet():
    summary = sweep([("cue:a", 0.0, 0.9, 0.0), ("cue:a", 0.1, 0.4, 2.0), ("cue:a", 0.5, 0.5, 5.0)])
    chosen = select_strengths(summary, max_clean_drop_pp=1.0).iloc[0]
    assert chosen["strength"] == 0.1 and not chosen["constraint_met"]


def test_load_strengths_overrides_defaults(tmp_path):
    path = tmp_path / "strengths.json"
    path.write_text(json.dumps({"strengths": {"cue:thermal_injection": 0.05}}))
    strengths, status = load_strengths(path)
    assert strengths["cue:thermal_injection"] == 0.05
    assert set(strengths) == set(default_strengths()) and status == "partially tuned"
