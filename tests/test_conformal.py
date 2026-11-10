import numpy as np

from bolero.conformal import (
    ConformalClassifier,
    ConformalGate,
    alerts,
    conformal_quantile,
    max_size_for_budget,
    raps_scores,
    raps_sets,
    selective_report,
    set_metrics,
)
from bolero.detection import CLEAN, PerturbationDetector, extract_features


def _dirichlet_probs(rng, n, c, labels, sharpness=3.0):
    logits = rng.standard_normal((n, c))
    logits[np.arange(n), labels] += sharpness
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def test_quantile_is_conservative_and_inf_for_tiny_n():
    scores = np.arange(1, 101) / 100.0
    assert conformal_quantile(scores, 0.1) == 0.91  # ceil(101 * 0.9) = 91st smallest
    assert conformal_quantile(scores[:5], 0.1) == float("inf")


def test_aps_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n, c, alpha = 4000, 20, 0.1
    labels = rng.integers(0, c, n)
    probs = _dirichlet_probs(rng, n, c, labels)
    cal, test = slice(0, 2000), slice(2000, None)
    model = ConformalClassifier.calibrate(probs[cal], labels[cal], alpha)
    m = set_metrics(model.sets(probs[test]), labels[test])
    assert m["coverage"] >= 1 - alpha - 0.02
    assert 1 <= m["mean_set_size"] < c


def test_sets_are_nonempty_prefixes_and_contain_true_label_iff_score_le_qhat():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 10, 200)
    probs = _dirichlet_probs(rng, 200, 10, labels)
    qhat = 0.8
    sets = raps_sets(probs, qhat, lam=0.05, k_reg=2)
    assert sets.sum(axis=1).min() >= 1
    covered = sets[np.arange(200), labels]
    scores = raps_scores(probs, labels, lam=0.05, k_reg=2)
    # Beyond the forced top-1, membership matches the score rule exactly.
    not_top1 = np.argmax(probs, 1) != labels
    assert np.array_equal(covered[not_top1], scores[not_top1] <= qhat)


def test_raps_penalty_makes_set_sizes_more_stable():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 50, 1000)
    probs = _dirichlet_probs(rng, 1000, 50, labels, sharpness=1.0)
    aps = ConformalClassifier.calibrate(probs, labels, 0.1)
    raps = ConformalClassifier.calibrate(probs, labels, 0.1, lam=0.02, k_reg=1)
    assert raps.sets(probs).sum(1).std() < aps.sets(probs).sum(1).std()


def test_gate_false_alarm_rate_is_controlled():
    rng = np.random.default_rng(3)
    clean = rng.normal(0.0, 1.0, (600, 4))
    pert = rng.normal(2.5, 1.0, (600, 4))
    x = np.vstack([clean[:200], pert[:200]])
    y = np.array([CLEAN] * 200 + ["darken"] * 200)
    detector = PerturbationDetector(n_estimators=50).fit(x, y)
    gate = ConformalGate.calibrate(detector, clean[200:400], alpha=0.05)
    assert (gate.route(clean[400:]) != CLEAN).mean() <= 0.05 + 0.04
    assert (gate.route(pert[400:]) == "darken").mean() > 0.8


def test_alerts_and_selective_report():
    sizes = np.array([1, 1, 3, 5, 1])
    correct = np.array([True, False, False, True, True])
    a = alerts(sizes, max_size=2)
    assert a.tolist() == [False, False, True, True, False]
    r = selective_report(correct, a)
    assert r["alert_rate"] == 0.4
    assert r["silent_miss_rate"] == 0.2      # the one wrong answer given without an alert
    assert r["errors_caught"] == 0.5
    assert alerts(sizes, 2, gate_fired=[True, 0, 0, 0, 0])[0]
    assert max_size_for_budget(sizes, 0.2) == 3


def test_gate_decide_matches_detector_interface(image):
    x = np.stack([extract_features(image)] * 4 + [extract_features(1.0 - image)] * 4)
    y = np.array([CLEAN] * 4 + ["inversion"] * 4)
    detector = PerturbationDetector(n_estimators=10).fit(x, y)
    gate = ConformalGate(detector, threshold=0.5, alpha=0.05)
    assert gate.decide(1.0 - image).label == "inversion"
    assert gate.decide(image).label == CLEAN
    assert ConformalGate(detector, threshold=float("inf"), alpha=0.0).decide(1.0 - image).label == CLEAN
