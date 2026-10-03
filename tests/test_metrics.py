import numpy as np
import pandas as pd

from bolero.metrics import compare, expected_calibration_error, recovery_rate, transitions


def test_compare_counts_transitions():
    df = pd.DataFrame({
        "group": ["a"] * 4,
        "x_correct": [True, True, False, False], "y_correct": [True, False, True, False],
        "x_pred": [1, 1, 2, 3], "y_pred": [1, 5, 1, 3],
        "x_conf": [0.9] * 4, "y_conf": [0.8] * 4,
        "x_true_prob": [0.9] * 4, "y_true_prob": [0.7] * 4,
    })
    row = compare(df, "x", "y", ["group"]).iloc[0]
    assert row["correct_to_wrong"] == 1 and row["wrong_to_correct"] == 1
    assert row["changed_rate"] == 0.5
    assert np.isclose(row["conf_shift"], -0.1)


def test_recovery_rate():
    assert np.isclose(recovery_rate(0.8, 0.4, 0.6), 0.5)
    assert np.isnan(recovery_rate(0.8, 0.8, 0.9))


def test_ece_perfect_calibration_is_zero():
    conf = np.array([1.0, 1.0, 0.0])
    assert expected_calibration_error(conf, np.array([True, True, False])) == 0.0


def test_transitions_separate_correctness_from_agreement():
    # label is class 1; the clean prediction is 2 (wrong).
    clean_pred = np.array([2, 2, 2, 2])
    before_pred = np.array([1, 3, 3, 2])
    after_pred = np.array([3, 1, 2, 2])
    t = transitions(before_pred == 1, after_pred == 1, before_pred, after_pred, clean_pred)
    assert t["wrong_to_correct"] == 1 and t["correct_to_wrong"] == 1 and t["net_fixed"] == 0
    assert t["fix_rate"] == 1 / 3 and t["break_rate"] == 1.0
    # image 2 returned to the clean prediction, which is still wrong
    assert t["back_to_clean_pred"] == 1 and t["back_to_clean_pred_wrong"] == 1
    assert t["agree_clean_before"] == 0.25 and t["agree_clean_after"] == 0.5
