import pytest

from bolero.config import CLASSIFIER_MODEL, IMAGENETTE_CLASSES, IMAGENETTE_INDEX


def test_mapping_matches_model_labels():
    transformers = pytest.importorskip("transformers")
    try:
        config = transformers.AutoConfig.from_pretrained(CLASSIFIER_MODEL)
    except OSError:
        pytest.skip("model config not available offline")
    for wnid, index in IMAGENETTE_INDEX.items():
        assert config.id2label[index].startswith(IMAGENETTE_CLASSES[wnid])
