"""Stage 3 - restoration study: can preprocessing win back what perturbation took?

For each val image, each perturbation x severity, and the clean image itself,
the perturbed input is passed through each restoration method and re-classified:

  none                   the perturbed image as-is (reference)
  oracle_classical       the classical plan for the *true* perturbation (upper bound)
  detector_classical     the plan for the *detected* perturbation; pass-through if unsure
  cue:<name>             one depth-derived contextual cue, depth estimated on the input
  detector+cue:<name>    detector_classical followed by the cue

Depth for restoration is estimated on the perturbed input (deployment-realistic);
pass --depth-source clean to use the clean depth map as an oracle instead.

Outputs (in --out):
  records.csv.gz   one row per image x perturbation x severity x method
  summary.csv      per perturbation x severity x method: accuracy, recovery rate,
                   transitions vs. the perturbed input, latency
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.cues import CUES, apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator
from bolero.detection import CLEAN, PerturbationDetector
from bolero.metrics import psnr, recovery_rate, ssim
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import prefixed, save
from bolero.restoration import restore


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, 1000.0 * (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--perturbations", nargs="*", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    parser.add_argument("--cues", nargs="*", default=list(CUES), choices=list(CUES))
    parser.add_argument("--combo-cue", default="border_strengthening", choices=[*CUES, "none"])
    parser.add_argument("--detector", type=Path, default=RESULTS_DIR / "detector" / "detector.joblib")
    parser.add_argument("--depth-source", default="perturbed", choices=["perturbed", "clean"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "restoration")
    args = parser.parse_args()

    if not args.detector.exists():
        raise SystemExit(f"{args.detector} not found. Run scripts/train_detector.py first.")
    detector = PerturbationDetector.load(args.detector)
    classifier = FrozenClassifier(device=args.device)
    estimator = DepthEstimator(device=args.device)
    depth_cache = DepthCache(estimator)
    predictor = BatchPredictor(classifier, args.batch_size)

    conditions = [(CLEAN, 0)] + [(name, level) for name in args.perturbations for level in LEVELS]

    for record in tqdm(list_images("val", per_class=args.per_class, seed=args.seed), desc="images"):
        clean = load_image(record.path)
        clean_depth = depth_cache.get(record, clean)
        base = {"image_id": record.image_id, "wnid": record.wnid,
                "class_name": record.class_name, "label_index": record.label_index}

        def emit(meta: dict, image: np.ndarray, restore_ms: float) -> None:
            predictor.add({**base, **meta, "restore_ms": restore_ms,
                           "psnr": psnr(clean, image), "ssim": ssim(clean, image)}, image)

        for name, level in conditions:
            if name == CLEAN:
                x = clean
            else:
                x = PERTURBATIONS[name](clean, level, clean_depth, rng_for(record.image_id, name, level, args.seed))
            cond = {"perturbation": name, "level": level}

            emit({**cond, "method": "none"}, x, 0.0)
            if name != CLEAN:
                oracle, ms = timed(restore, x, name)
                emit({**cond, "method": "oracle_classical"}, oracle, ms)

            decision, detect_ms = timed(detector.decide, x)
            routed, restore_ms = timed(restore, x, decision.label)
            detect_meta = {"detected": decision.label, "detected_raw": decision.predicted,
                           "detected_prob": decision.probability}
            emit({**cond, "method": "detector_classical", **detect_meta}, routed, detect_ms + restore_ms)

            if args.cues or args.combo_cue != "none":
                if args.depth_source == "clean":
                    depth, depth_ms = clean_depth, 0.0
                else:
                    depth, depth_ms = timed(estimator, x)
                for cue in args.cues:
                    out, ms = timed(apply_cue, cue, x, depth)
                    emit({**cond, "method": f"cue:{cue}"}, out, depth_ms + ms)
                if args.combo_cue != "none":
                    out, ms = timed(apply_cue, args.combo_cue, routed, depth)
                    emit({**cond, "method": f"detector+cue:{args.combo_cue}", **detect_meta},
                         out, detect_ms + restore_ms + depth_ms + ms)
    predictor.flush()

    df = prefixed(predictor.rows, "out")
    save(df, args.out / "records.csv.gz")
    save(summarize(df), args.out / "summary.csv")
    print(summarize_overall(df).to_string(index=False))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["image_id", "perturbation", "level"]
    ref = df[df["method"] == "none"].set_index(keys)
    clean_acc = df[(df["perturbation"] == CLEAN) & (df["method"] == "none")]["out_correct"].mean()
    rows = []
    for (name, level, method), g in df.groupby(["perturbation", "level", "method"], sort=False):
        before = ref.loc[pd.MultiIndex.from_frame(g[keys])]
        b_ok, a_ok = before["out_correct"].to_numpy(bool), g["out_correct"].to_numpy(bool)
        rows.append({
            "perturbation": name, "level": level, "method": method, "n": len(g),
            "acc_clean": clean_acc,
            "acc_perturbed": b_ok.mean(),
            "acc_restored": a_ok.mean(),
            "delta_pp": 100.0 * (a_ok.mean() - b_ok.mean()),
            "recovery_rate": recovery_rate(clean_acc, b_ok.mean(), a_ok.mean()),
            "wrong_to_correct": int((~b_ok & a_ok).sum()),
            "correct_to_wrong": int((b_ok & ~a_ok).sum()),
            "changed_rate": (g["out_pred"].to_numpy() != before["out_pred"].to_numpy()).mean(),
            "true_prob_shift": (g["out_true_prob"].to_numpy() - before["out_true_prob"].to_numpy()).mean(),
            "psnr": g["psnr"].replace(np.inf, np.nan).mean(),
            "ssim": g["ssim"].mean(),
            "restore_ms": g["restore_ms"].mean(),
            "classify_ms": g["out_classify_ms"].mean(),
        })
    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    perturbed = df[df["perturbation"] != CLEAN]
    clean = df[df["perturbation"] == CLEAN]
    return pd.DataFrame({
        "acc_on_perturbed": perturbed.groupby("method", sort=False)["out_correct"].mean(),
        "acc_on_clean": clean.groupby("method", sort=False)["out_correct"].mean(),
        "restore_ms": df.groupby("method", sort=False)["restore_ms"].mean(),
    }).reset_index()


if __name__ == "__main__":
    main()
