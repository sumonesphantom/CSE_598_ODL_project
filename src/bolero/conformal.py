"""Split-conformal calibration: a guaranteed-false-alarm gate, prediction sets, and abstention.

Three uses, all calibrated on held-out *clean* images and checked on perturbed ones:

1. Gate. The perturbation detector's fixed 0.6 threshold is replaced by a
   threshold on the nonconformity score 1 - p(clean), set so that a new clean
   image is flagged with probability <= alpha (exchangeability assumed).
2. Prediction sets (APS / RAPS) for the classifier. Calibrated on clean images,
   they cover the true class with probability >= 1 - alpha on clean inputs.
   Under perturbation the guarantee no longer holds, so coverage and set size per
   perturbation show how much the classifier's uncertainty grows, and whether
   restoration shrinks it again.
3. Abstain-and-alert. A set larger than `max_size`, or a gate that fires, means
   "this input cannot be trusted": report an alert instead of a confident answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import CLEAN, Decision, PerturbationDetector, extract_features


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The k-th smallest calibration score, k = ceil((n + 1)(1 - alpha)).

    Returns +inf when n is too small for the requested alpha (every set is then the full set).
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    return float("inf") if k > scores.size else float(scores[k - 1])


# ---------------------------------------------------------------------------
# 1. Gate on the perturbation detector
# ---------------------------------------------------------------------------

def gate_scores(detector: PerturbationDetector, features: np.ndarray) -> np.ndarray:
    """Nonconformity of each input with "clean": 1 - p(clean)."""
    proba = detector.predict_proba(features)
    clean_col = int(np.flatnonzero(detector.classes == CLEAN)[0])
    return 1.0 - proba[:, clean_col]


@dataclass
class ConformalGate:
    """Routes an image to restoration only when its score exceeds the clean-calibrated threshold."""

    detector: PerturbationDetector
    threshold: float
    alpha: float

    @classmethod
    def calibrate(cls, detector: PerturbationDetector, clean_features: np.ndarray, alpha: float) -> "ConformalGate":
        """`clean_features` must come from clean images the detector was not trained on."""
        return cls(detector, conformal_quantile(gate_scores(detector, clean_features), alpha), alpha)

    def route(self, features: np.ndarray) -> np.ndarray:
        """Per row: the perturbation to undo (most probable non-clean class), or "clean"."""
        features = np.atleast_2d(features)
        proba = self.detector.predict_proba(features)
        classes = self.detector.classes
        non_clean = np.where(classes == CLEAN, -np.inf, proba)
        best = classes[non_clean.argmax(axis=1)]
        flagged = gate_scores(self.detector, features) > self.threshold
        return np.where(flagged, best, CLEAN)

    def decide(self, image: np.ndarray) -> Decision:
        features = extract_features(image)
        proba = self.detector.predict_proba(features)[0]
        label = str(self.route(features)[0])
        return Decision(label, str(self.detector.classes[proba.argmax()]), float(proba.max()))


# ---------------------------------------------------------------------------
# 2. APS / RAPS prediction sets (Romano et al. 2020; Angelopoulos et al. 2021)
# ---------------------------------------------------------------------------

def restrict_to_classes(probs: np.ndarray, class_index, labels: np.ndarray | None = None):
    """Keep only the columns in `class_index` and renormalize each row to sum to 1.

    With 1000 ImageNet classes and a frozen classifier whose clean error rate exceeds
    alpha, APS must absorb almost the whole softmax tail to reach the true class, so
    qhat ~ 1 and sets are near-trivial. Building sets over the task's own label space
    (the 10 Imagenette classes) avoids that. If `labels` (ImageNet indices) are given,
    also returns them remapped to positions 0..len(class_index)-1.
    """
    class_index = np.asarray(class_index)
    sub = np.asarray(probs, dtype=np.float64)[:, class_index]
    sub = sub / sub.sum(axis=1, keepdims=True)
    if labels is None:
        return sub
    position = {int(c): i for i, c in enumerate(class_index)}
    return sub, np.array([position[int(l)] for l in labels])


def _penalty(n_classes: int, lam: float, k_reg: int) -> np.ndarray:
    """RAPS rank penalty: lam for every rank beyond the first k_reg (cumulated)."""
    return lam * np.clip(np.arange(1, n_classes + 1) - k_reg, 0, None)


def _sorted_scores(probs: np.ndarray, lam: float, k_reg: int, rng: np.random.Generator | None):
    """Per-rank scores: mass ranked at or above each class, plus the rank penalty.

    With `rng`, each row subtracts U * p(class), U ~ Uniform(0, 1) (randomized APS,
    Romano et al. 2020). Without it, a sharp model's confidently *correct* prediction
    already scores ~1.0, so qhat lands on 1.0 and floating-point ties at 1.0 put
    almost every class in the set.
    """
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1) + _penalty(probs.shape[1], lam, k_reg)
    if rng is not None:
        cum = cum - rng.uniform(size=(len(probs), 1)) * sorted_p
    return order, cum


def raps_scores(probs: np.ndarray, labels: np.ndarray, lam: float = 0.0, k_reg: int = 0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Nonconformity of the true label. lam = 0 gives APS; pass `rng` for the randomized version."""
    order, cum = _sorted_scores(probs, lam, k_reg, rng)
    rank = np.argmax(order == np.asarray(labels)[:, None], axis=1)
    return cum[np.arange(len(order)), rank]


def raps_sets(probs: np.ndarray, qhat: float, lam: float = 0.0, k_reg: int = 0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean (N, C) set membership: every class whose score is <= qhat (never empty)."""
    order, cum = _sorted_scores(probs, lam, k_reg, rng)
    keep_sorted = cum <= qhat
    keep_sorted[:, 0] = True
    sets = np.zeros_like(keep_sorted)
    np.put_along_axis(sets, order, keep_sorted, axis=1)
    return sets


@dataclass
class ConformalClassifier:
    """APS (lam = 0) or RAPS prediction sets on top of a frozen classifier's softmax."""

    qhat: float
    alpha: float
    lam: float = 0.0
    k_reg: int = 0
    rng: np.random.Generator | None = None   # set => randomized scores and sets

    @classmethod
    def calibrate(cls, probs: np.ndarray, labels: np.ndarray, alpha: float, lam: float = 0.0,
                  k_reg: int = 0, randomized: bool = True, seed: int = 0) -> "ConformalClassifier":
        rng = np.random.default_rng(seed) if randomized else None
        qhat = conformal_quantile(raps_scores(probs, labels, lam, k_reg, rng), alpha)
        return cls(qhat, alpha, lam, k_reg, rng)

    def sets(self, probs: np.ndarray) -> np.ndarray:
        return raps_sets(probs, self.qhat, self.lam, self.k_reg, self.rng)


def set_metrics(sets: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    sizes = sets.sum(axis=1)
    return {
        "coverage": float(sets[np.arange(len(labels)), labels].mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float((sizes == 1).mean()),
    }


# ---------------------------------------------------------------------------
# 3. Abstain-and-alert
# ---------------------------------------------------------------------------

def alerts(set_sizes: np.ndarray, max_size: int, gate_fired: np.ndarray | None = None) -> np.ndarray:
    """True where the system should alert "input unreliable" instead of answering."""
    alert = np.asarray(set_sizes) > max_size
    return alert if gate_fired is None else alert | np.asarray(gate_fired, bool)


def selective_report(correct: np.ndarray, alert: np.ndarray) -> dict:
    """What an operator sees: how often the system alerts, and how often it is silently wrong.

    `silent_miss_rate` is the security-relevant number: wrong answers delivered without an alert.
    """
    correct, alert = np.asarray(correct, bool), np.asarray(alert, bool)
    answered = ~alert
    return {
        "alert_rate": float(alert.mean()),
        "answered_accuracy": float(correct[answered].mean()) if answered.any() else float("nan"),
        "silent_miss_rate": float((~correct & answered).mean()),
        "errors_caught": float(alert[~correct].mean()) if (~correct).any() else float("nan"),
    }


def max_size_for_budget(set_sizes: np.ndarray, budget: float) -> int:
    """Smallest set-size cutoff whose alert rate on these (typically clean) inputs is <= budget."""
    sizes = np.asarray(set_sizes)
    for k in range(1, int(sizes.max()) + 1):
        if (sizes > k).mean() <= budget:
            return k
    return int(sizes.max())
