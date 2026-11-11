"""mAP and loader tests (no model loads)."""

import numpy as np

from bolero.detection_eval import Detections, average_precision, evaluate, iou_matrix
from bolero.weather_data import load_yolo_dataset, parse_voc


def test_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_ap_perfect_and_half():
    gt = [np.array([[0, 0, 10, 10]], np.float32), np.array([[0, 0, 10, 10]], np.float32)]
    perfect = [(g, np.array([0.9])) for g in gt]
    assert average_precision(gt, perfect, 0.5) == (1.0, 1.0)
    # one hit (top score) and one miss: recall 0.5, precision 1 up to it -> AP ~ 51/101
    half = [(gt[0], np.array([0.9])), (np.array([[50, 50, 60, 60]], np.float32), np.array([0.8]))]
    ap, rec = average_precision(gt, half, 0.5)
    assert rec == 0.5 and abs(ap - 51 / 101) < 1e-9


def test_ap_ranks_false_positive_first():
    gt = [np.array([[0, 0, 10, 10]], np.float32)]
    dets = [(np.array([[50, 50, 60, 60], [0, 0, 10, 10]], np.float32), np.array([0.9, 0.8]))]
    ap, rec = average_precision(gt, dets, 0.5)
    assert rec == 1.0 and abs(ap - 0.5) < 1e-9   # precision 0.5 at every recall level
    # a duplicate of a matched box is a false positive
    dup = [(np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32), np.array([0.9, 0.8]))]
    assert average_precision(gt, dup, 0.5)[0] == 1.0


def test_evaluate_skips_absent_classes():
    gt_boxes = [np.array([[0, 0, 10, 10], [20, 20, 40, 40]], np.float32)]
    gt_labels = [["car", "person"]]
    det = [Detections(np.array([[0, 0, 10, 10]], np.float32), np.array([0.9], np.float32), ["car"])]
    out = evaluate(gt_boxes, gt_labels, det)
    assert out["per_class"]["car"]["ap50"] == 1.0 and out["per_class"]["person"]["ap50"] == 0.0
    assert np.isnan(out["per_class"]["bus"]["ap50"])
    assert out["map50"] == 0.5


def test_parse_voc_and_yolo_layout(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><size><width>100</width><height>50</height></size>"
                   "<object><name>Car</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>"
                   "<object><name>tree</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>"
                   "</annotation>")
    boxes, labels, w, h = parse_voc(xml)
    assert labels == ["car"] and (w, h) == (100, 50)
    np.testing.assert_array_equal(boxes, [[1, 2, 30, 40]])

    from PIL import Image
    root = tmp_path / "xwod"
    (root / "images/test/rain").mkdir(parents=True)
    (root / "images/test/snow").mkdir(parents=True)
    (root / "labels/test/rain").mkdir(parents=True)
    (root / "data.yaml").write_text("names: {0: car, 1: person, 2: traffic light}\n")
    Image.new("RGB", (200, 100)).save(root / "images/test/rain/r1.jpg")
    Image.new("RGB", (200, 100)).save(root / "images/test/snow/s1.jpg")
    (root / "labels/test/rain/r1.txt").write_text("0 0.5 0.5 0.2 0.4\n2 0.1 0.1 0.1 0.1\n")
    samples = load_yolo_dataset(root, "test", "xwod")
    assert [s.condition for s in samples] == ["rain"]          # snow dropped
    assert samples[0].labels == ["car"]                         # non-target class dropped
    np.testing.assert_allclose(samples[0].boxes, [[80, 30, 120, 70]])
