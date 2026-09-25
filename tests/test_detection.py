import numpy as np

from bolero.detection import CLEAN, PerturbationDetector, extract_features
from bolero.perturbations import PERTURBATIONS


def test_feature_vector_is_fixed_length(image, depth):
    n = extract_features(image).shape
    for name, p in PERTURBATIONS.items():
        assert extract_features(p(image, 2, depth)).shape == n


def test_detector_is_conservative_below_threshold(image):
    x = np.stack([extract_features(image)] * 4 + [extract_features(1.0 - image)] * 4)
    y = np.array([CLEAN] * 4 + ["inversion"] * 4)
    detector = PerturbationDetector(threshold=1.01, n_estimators=10).fit(x, y)
    decision = detector.decide(1.0 - image)
    assert decision.predicted == "inversion"
    assert decision.label == CLEAN  # threshold unreachable -> pass through
