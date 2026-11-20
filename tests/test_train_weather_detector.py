"""Day-level VIRADA split and Wilson intervals used by scripts/train_weather_detector.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_weather_detector import virada_split, wilson  # noqa: E402


def test_virada_split_holds_out_whole_days():
    frames = pd.DataFrame({
        "path": [f"f{i}" for i in range(8)],
        "label": ["rain", "no_rain", "rain", "no_rain", "no_rain", "no_rain", "no_rain", "rain"],
        "day": ["d2", "d2", "d4", "d4", "d1", "d3", "d5", "d2"],
    })
    out = virada_split(frames)
    by_day = out.groupby("day")["split"].nunique()
    assert (by_day == 1).all()                                   # no day is in both splits
    assert set(out.loc[out.day == "d4", "split"]) == {"test"}    # the last rain day is held out
    assert set(out.loc[out.day == "d2", "split"]) == {"train"}
    assert set(out.loc[out.label == "rain", "split"]) == {"train", "test"}


def test_wilson():
    low, high = wilson(50, 100)
    assert low < 0.5 < high and high - low == pytest.approx(0.19, abs=0.01)
    assert wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-9) and wilson(10, 10)[1] == pytest.approx(1.0)
