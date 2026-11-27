"""Conformal gate, prediction sets and abstain-and-alert on the frozen classifier.

Calibration uses clean Imagenette *train* images that the detector was NOT trained
on. Evaluation uses *val* images: clean plus every perturbation x severity, each
classified as-is (`none`), after detector-routed restoration (`detector`) and after
restoration for the true perturbation (`oracle`).

What it answers:
  - Gate: does the conformal threshold hold the clean false-alarm rate at
    --gate-alpha, compared with the fixed 0.6 threshold?
  - Sets: on clean inputs APS/RAPS sets cover the true class at >= 1 - alpha. How far
    does coverage fall and set size grow under each perturbation, and does
    restoration bring them back?
  - Alerts: with a set-size cutoff fixed on clean calibration images (at most
    --alert-budget alerts), how many wrong answers get through without an alert?

Outputs (in --out):
  records.csv.gz   one row per val image x condition x stage x set method
  summary.csv      per perturbation x level x stage x method: accuracy, coverage, set size, alerts
  gate.csv         per perturbation x level: conformal vs fixed-threshold routing
  headline.json    calibration values and the main numbers
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import FrozenClassifier
from bolero.conformal import (
    ConformalClassifier,
    ConformalGate,
    alerts,
    max_size_for_budget,
    restrict_to_classes,
    selective_report,
)
from bolero.config import IMAGENETTE_INDEX, RESULTS_DIR, SEED
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.detection import CLEAN, PerturbationDetector, extract_features
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import save
from bolero.restoration import restore

STAGES = ("none", "detector", "oracle")


def calibration_records(detector_per_class: int, per_class: int, seed: int):
    """Clean train images disjoint from the detector's training sample (scripts/train_detector.py)."""
    used = {r.image_id for r in list_images("train", per_class=detector_per_class, seed=seed)}
    rng = np.random.default_rng(seed + 100)
    available = [r for r in list_images("train") if r.image_id not in used]
    out = []
    for wnid in sorted({r.wnid for r in available}):
        pool = [r for r in available if r.wnid == wnid]
        out += [pool[i] for i in np.sort(rng.choice(len(pool), size=min(per_class, len(pool)), replace=False))]
    return out


def classify(classifier: FrozenClassifier, images: list[np.ndarray], batch_size: int) -> np.ndarray:
    return np.concatenate([classifier.predict_proba(images[i:i + batch_size])
                           for i in range(0, len(images), batch_size)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.1, help="classifier set miscoverage on clean inputs")
    parser.add_argument("--gate-alpha", type=float, default=0.05, help="target clean false-alarm rate")
    parser.add_argument("--alert-budget", type=float, default=0.05, help="max alert rate on clean calibration")
    parser.add_argument("--raps-lambda", type=float, default=0.01)
    parser.add_argument("--raps-k", type=int, default=2)
    parser.add_argument("--cal-per-class", type=int, default=50)
    parser.add_argument("--per-class", type=int, default=20, help="val images per class")
    parser.add_argument("--detector-train-per-class", type=int, default=40,
                        help="must match train_detector.py --train-per-class")
    parser.add_argument("--fixed-threshold", type=float, default=0.6)
    parser.add_argument("--detector", type=Path, default=RESULTS_DIR / "detector" / "detector.joblib")
    parser.add_argument("--label-space", choices=["imagenette", "imagenet"], default="imagenette",
                        help="classes the prediction sets range over (imagenet = all 1000, near-trivial sets)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="deterministic APS/RAPS (ties at 1.0 make APS sets near-trivial)")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "conformal")
    args = parser.parse_args()

    detector = PerturbationDetector.load(args.detector)
    classifier = FrozenClassifier(args.backbone, device=args.device)

    # --- Calibration (clean train images, disjoint from detector training) ---
    cal = calibration_records(args.detector_train_per_class, args.cal_per_class, args.seed)
    cal_images = [load_image(r.path) for r in tqdm(cal, desc="calibration images")]
    cal_labels = np.array([r.label_index for r in cal])
    cal_probs = classify(classifier, cal_images, args.batch_size)
    class_index = (np.array(sorted(IMAGENETTE_INDEX.values())) if args.label_space == "imagenette"
                   else np.arange(cal_probs.shape[1]))
    cal_probs, cal_labels = restrict_to_classes(cal_probs, class_index, cal_labels)
    gate = ConformalGate.calibrate(detector, np.stack([extract_features(im) for im in cal_images]),
                                   args.gate_alpha)
    methods = {
        "aps": ConformalClassifier.calibrate(cal_probs, cal_labels, args.alpha,
                                             randomized=not args.no_randomize, seed=args.seed),
        "raps": ConformalClassifier.calibrate(cal_probs, cal_labels, args.alpha, lam=args.raps_lambda,
                                              k_reg=args.raps_k, randomized=not args.no_randomize,
                                              seed=args.seed + 1),
    }
    max_size = {name: max_size_for_budget(m.sets(cal_probs).sum(1), args.alert_budget)
                for name, m in methods.items()}

    # --- Evaluation (val images: clean + perturbations, three stages) ---
    depth_cache = DepthCache()
    meta, results = [], []
    for record in tqdm(list_images("val", per_class=args.per_class, seed=args.seed), desc="val images"):
        image = load_image(record.path)
        depth = depth_cache.get(record, image)
        conditions = [(CLEAN, 0, image)] + [
            (name, level, p(image, level, depth, rng_for(record.image_id, name, level, args.seed)))
            for name, p in PERTURBATIONS.items() for level in LEVELS
        ]
        images = []
        for name, level, x in conditions:
            features = extract_features(x)
            routed = str(gate.route(features)[0])
            proba = detector.predict_proba(features)[0]
            top = str(detector.classes[proba.argmax()])
            fixed = top if top != CLEAN and proba.max() >= args.fixed_threshold else CLEAN
            row = {"image_id": record.image_id, "class_name": record.class_name,
                   "label_index": record.label_index, "perturbation": name, "level": level,
                   "gate_score": 1.0 - proba[list(detector.classes).index(CLEAN)],
                   "routed_conformal": routed, "routed_fixed": fixed}
            stage_images = {
                "none": x,
                "detector": x if routed == CLEAN else restore(x, routed),
                "oracle": x if name == CLEAN else restore(x, name),
            }
            for stage in STAGES:
                meta.append({**row, "stage": stage})
                images.append(stage_images[stage])
        # Classify one image's conditions at a time; only set sizes and coverage are kept.
        probs = classify(classifier, images, args.batch_size)
        # Top-1 accuracy stays 1000-way (as in every other stage); sets use the chosen label space.
        result = {"correct": probs.argmax(1) == record.label_index}
        probs, label = restrict_to_classes(probs, class_index, np.full(len(probs), record.label_index))
        for name, method in methods.items():
            sets = method.sets(probs)
            result[f"{name}_size"] = sets.sum(1)
            result[f"{name}_covered"] = sets[np.arange(len(probs)), label]
        results.append(result)

    base = pd.DataFrame(meta)
    stacked = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    gate_fired = (base["routed_conformal"] != CLEAN).to_numpy()
    df = pd.concat([
        base.assign(method=name, correct=stacked["correct"], set_size=stacked[f"{name}_size"],
                    covered=stacked[f"{name}_covered"],
                    alert=alerts(stacked[f"{name}_size"], max_size[name], gate_fired))
        for name in methods
    ], ignore_index=True)
    save(df, args.out / "records.csv.gz")

    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "coverage": g["covered"].mean(),
            "mean_set_size": g["set_size"].mean(),
            **selective_report(g["correct"].to_numpy(), g["alert"].to_numpy()),
        })

    summary = (df.groupby(["method", "stage", "perturbation", "level"], sort=False)
               .apply(summarize, include_groups=False).reset_index())
    save(summary, args.out / "summary.csv")

    routing = base[base["stage"] == "none"]
    gate_table = (routing.assign(
        flagged_conformal=routing["routed_conformal"] != CLEAN,
        flagged_fixed=routing["routed_fixed"] != CLEAN,
        routed_correctly_conformal=routing["routed_conformal"] == routing["perturbation"],
        routed_correctly_fixed=routing["routed_fixed"] == routing["perturbation"],
    ).groupby(["perturbation", "level"], sort=False)[[
        "flagged_conformal", "flagged_fixed", "routed_correctly_conformal", "routed_correctly_fixed",
    ]].mean().reset_index())
    save(gate_table, args.out / "gate.csv")

    is_clean = routing["perturbation"] == CLEAN
    none = df[df["stage"] == "none"]
    headline = {
        "calibration_images": len(cal),
        "label_space": args.label_space,
        "randomized_sets": not args.no_randomize,
        "val_images": int(is_clean.sum()),
        "gate": {
            "alpha": args.gate_alpha,
            "threshold_on_1_minus_p_clean": gate.threshold,
            "clean_false_alarm_conformal": float((routing.loc[is_clean, "routed_conformal"] != CLEAN).mean()),
            "clean_false_alarm_fixed": float((routing.loc[is_clean, "routed_fixed"] != CLEAN).mean()),
            "perturbed_flagged_conformal": float((routing.loc[~is_clean, "routed_conformal"] != CLEAN).mean()),
            "perturbed_flagged_fixed": float((routing.loc[~is_clean, "routed_fixed"] != CLEAN).mean()),
        },
        "sets": {
            name: {
                "qhat": methods[name].qhat,
                "alert_max_set_size": max_size[name],
                **{f"{stage}_{subset}": {
                    "coverage": float(g["covered"].mean()),
                    "mean_set_size": float(g["set_size"].mean()),
                    "accuracy": float(g["correct"].mean()),
                } for stage in STAGES for subset, g in [
                    ("clean", df[(df.method == name) & (df.stage == stage) & (df.perturbation == CLEAN)]),
                    ("perturbed", df[(df.method == name) & (df.stage == stage) & (df.perturbation != CLEAN)]),
                ]},
                "alerts_perturbed_none": selective_report(
                    *none[(none.method == name) & (none.perturbation != CLEAN)][["correct", "alert"]]
                    .to_numpy(bool).T),
                "alerts_clean_none": selective_report(
                    *none[(none.method == name) & (none.perturbation == CLEAN)][["correct", "alert"]]
                    .to_numpy(bool).T),
            } for name in methods
        },
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
