import numpy as np
import pandas as pd

from bolero.metrics import compare, expected_calibration_error, recovery_rate


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
