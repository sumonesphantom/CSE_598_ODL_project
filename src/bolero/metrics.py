"""Recognition, stability, calibration, and fidelity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, image: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, image, data_range=1.0))


def ssim(reference: np.ndarray, image: np.ndarray) -> float:
    return float(structural_similarity(reference, image, data_range=1.0, channel_axis=-1))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        in_bin = idx == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compare(df: pd.DataFrame, before: str, after: str, by: list[str]) -> pd.DataFrame:
    """Compare two conditions stored as `{before}_*` and `{after}_*` columns.

    Returns per-group accuracy, prediction-change rate, correct->wrong and
    wrong->correct transition counts, confidence shift, and calibration.
    """

    def summarize(g: pd.DataFrame) -> pd.Series:
        b_ok, a_ok = g[f"{before}_correct"].to_numpy(bool), g[f"{after}_correct"].to_numpy(bool)
        return pd.Series({
            "n": len(g),
            f"acc_{before}": b_ok.mean(),
            f"acc_{after}": a_ok.mean(),
            "delta_pp": 100.0 * (a_ok.mean() - b_ok.mean()),
            "changed_rate": (g[f"{before}_pred"] != g[f"{after}_pred"]).mean(),
            "correct_to_wrong": int((b_ok & ~a_ok).sum()),
            "wrong_to_correct": int((~b_ok & a_ok).sum()),
            "conf_shift": (g[f"{after}_conf"] - g[f"{before}_conf"]).mean(),
            "true_prob_shift": (g[f"{after}_true_prob"] - g[f"{before}_true_prob"]).mean(),
            f"ece_{after}": expected_calibration_error(g[f"{after}_conf"].to_numpy(), a_ok),
        })

    return df.groupby(by, sort=False).apply(summarize, include_groups=False).reset_index()


def recovery_rate(acc_clean: float, acc_perturbed: float, acc_restored: float) -> float:
    """Fraction of the perturbation's accuracy loss that restoration wins back."""
    lost = acc_clean - acc_perturbed
    return float("nan") if lost <= 0 else (acc_restored - acc_perturbed) / lost


def transitions(
    before_ok: np.ndarray,
    after_ok: np.ndarray,
    before_pred: np.ndarray,
    after_pred: np.ndarray,
    clean_pred: np.ndarray,
) -> dict:
    """Correctness and agreement transitions from a reference input to a processed one.

    Correctness is measured against the ground-truth label. Agreement is measured
    against the classifier's prediction on the clean image, which is itself wrong
    for some images: `back_to_clean_pred_wrong` counts processed images that
    returned to the clean prediction without becoming correct.
    """
    before_ok, after_ok = np.asarray(before_ok, bool), np.asarray(after_ok, bool)
    before_agree = np.asarray(before_pred) == np.asarray(clean_pred)
    after_agree = np.asarray(after_pred) == np.asarray(clean_pred)
    fixed, broken = ~before_ok & after_ok, before_ok & ~after_ok
    back = ~before_agree & after_agree
    n_wrong, n_correct = int((~before_ok).sum()), int(before_ok.sum())
    return {
        "wrong_to_correct": int(fixed.sum()),
        "correct_to_wrong": int(broken.sum()),
        "net_fixed": int(fixed.sum() - broken.sum()),
        "fix_rate": fixed.sum() / n_wrong if n_wrong else float("nan"),
        "break_rate": broken.sum() / n_correct if n_correct else float("nan"),
        "agree_clean_before": before_agree.mean(),
        "agree_clean_after": after_agree.mean(),
        "back_to_clean_pred": int(back.sum()),
        "back_to_clean_pred_wrong": int((back & ~after_ok).sum()),
        "left_clean_pred": int((before_agree & ~after_agree).sum()),
    }
