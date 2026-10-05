"""Stage 3 - restoration study: can preprocessing win back what perturbation took?

For each val image, each perturbation x severity, and the clean image itself,
the perturbed input is passed through each restoration method and re-classified:

  none                   the perturbed image as-is (reference)
  oracle_classical       the classical plan for the *true* perturbation (upper bound)
  detector_classical     the plan for the *detected* perturbation; pass-through if unsure
  baseline:<name>        a depth-free preprocessing baseline, applied blindly
  cue:<name>             one depth-derived contextual cue (see depth_source)
  detector+cue:<name>    detector_classical followed by the cue (perturbed depth)

Each cue runs with up to three depth maps, recorded in the `depth_source` column:

  perturbed    re-estimated on the perturbed input. The deployable setting and
               the main result.
  clean        the clean image's depth. Idealized: unavailable at deployment.
  mismatched   the clean depth of a different image (another class). A control:
               if a cue still helps with an unrelated depth map, the gain comes
               from the image operation, not from depth information.

Strengths come from results/tuning/strengths.json (scripts/tune_strengths.py,
chosen on the train split); hand-set defaults are used if that file is missing.

Outputs (in --out):
  records.csv.gz   one row per image x perturbation x severity x method x depth source
  summary.csv      per perturbation x severity x method x depth source: accuracy against
                   ground truth, agreement with the clean prediction, fixes and breaks
                   vs. the perturbed input, depth agreement, latency
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.baselines import BASELINES, apply_baseline
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.cues import CUES, apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator, rank_agreement, resize_depth
from bolero.detection import CLEAN, PerturbationDetector
from bolero.metrics import psnr, recovery_rate, ssim, transitions
from bolero.perturbations import ALL_PERTURBATIONS, HELD_OUT, LEVELS, rng_for
from bolero.records import prefixed, save
from bolero.restoration import load_plans, restore
from bolero.tuning import load_strengths, method_key

DEPTH_SOURCES = ["perturbed", "clean", "mismatched"]
NO_DEPTH = "unused"


def plans_used(plans: dict) -> dict:
    """Record each perturbation's plan as a list of operation names, for run.json."""
    return {name: [getattr(op, "__name__", repr(op)) for op in ops] for name, ops in plans.items()}


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, 1000.0 * (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--perturbations", nargs="*", default=list(ALL_PERTURBATIONS),
                        choices=list(ALL_PERTURBATIONS))
    parser.add_argument("--cues", nargs="*", default=list(CUES), choices=list(CUES))
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--combo-cues", nargs="*", default=list(CUES), choices=list(CUES),
                        help="Cues to run after detector restoration (pass no names to skip)")
    parser.add_argument("--depth-sources", nargs="*", default=DEPTH_SOURCES, choices=DEPTH_SOURCES)
    parser.add_argument("--strengths", type=Path, default=RESULTS_DIR / "tuning" / "strengths.json")
    parser.add_argument("--plans", type=Path, default=RESULTS_DIR / "tuning" / "plans.json",
                        help="Dev-selected restoration plans (scripts/tune_plans.py)")
    parser.add_argument("--detector", type=Path, default=RESULTS_DIR / "detector" / "detector.joblib")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "restoration")
    args = parser.parse_args()

    if not args.detector.exists():
        raise SystemExit(f"{args.detector} not found. Run scripts/train_detector.py first.")
    if args.combo_cues and "perturbed" not in args.depth_sources:
        raise SystemExit("--combo-cues needs the 'perturbed' depth source.")
    strengths, strength_status = load_strengths(args.strengths)
    plans, plan_status = load_plans(args.plans)
    detector = PerturbationDetector.load(args.detector)
    classifier = FrozenClassifier(device=args.device)
    estimator = DepthEstimator(device=args.device)
    depth_cache = DepthCache(estimator)
    predictor = BatchPredictor(classifier, args.batch_size)

    conditions = [(CLEAN, 0)] + [(name, level) for name in args.perturbations for level in LEVELS]
    needs_depth = bool(args.cues or args.combo_cues)
    records = list_images("val", per_class=args.per_class, seed=args.seed)
    # Records are grouped by class, so an offset of half the list pairs each image
    # with a donor from a different class for the mismatched-depth control.
    donor_of = {r.image_id: records[(i + len(records) // 2) % len(records)] for i, r in enumerate(records)}

    for record in tqdm(records, desc="images"):
        clean = load_image(record.path)
        clean_depth = depth_cache.get(record, clean)
        base = {"image_id": record.image_id, "wnid": record.wnid,
                "class_name": record.class_name, "label_index": record.label_index}
        if "mismatched" in args.depth_sources and args.cues:
            donor = donor_of[record.image_id]
            mismatched_depth = resize_depth(depth_cache.get(donor, load_image(donor.path)), clean.shape[:2])

        def emit(meta: dict, image: np.ndarray, restore_ms: float) -> None:
            predictor.add({**base, "depth_source": NO_DEPTH, "strength": np.nan, **meta,
                           "restore_ms": restore_ms, "psnr": psnr(clean, image), "ssim": ssim(clean, image)},
                          image)

        for name, level in conditions:
            if name == CLEAN:
                x = clean
            else:
                x = ALL_PERTURBATIONS[name](clean, level, clean_depth,
                                            rng_for(record.image_id, name, level, args.seed))
            cond = {"perturbation": name, "level": level}

            depth_ms, perturbed_depth = 0.0, None
            if needs_depth:
                perturbed_depth, depth_ms = timed(estimator, x)
                cond["depth_rank_corr"] = rank_agreement(perturbed_depth, clean_depth)

            emit({**cond, "method": "none"}, x, 0.0)
            if name != CLEAN and name not in HELD_OUT:  # held-out perturbations have no plan
                oracle, ms = timed(restore, x, name, plans)
                emit({**cond, "method": "oracle_classical"}, oracle, ms)

            decision, detect_ms = timed(detector.decide, x)
            routed, restore_ms = timed(restore, x, decision.label, plans)
            detect_meta = {"detected": decision.label, "detected_raw": decision.predicted,
                           "detected_prob": decision.probability}
            emit({**cond, "method": "detector_classical", **detect_meta}, routed, detect_ms + restore_ms)

            for baseline in args.baselines:
                key = method_key("baseline", baseline)
                out, ms = timed(apply_baseline, baseline, x, strengths[key])
                emit({**cond, "method": key, "strength": strengths[key]}, out, ms)

            depths = {"perturbed": (perturbed_depth, depth_ms), "clean": (clean_depth, 0.0)}
            if "mismatched" in args.depth_sources and args.cues:
                depths["mismatched"] = (mismatched_depth, 0.0)
            for source in args.depth_sources if args.cues else []:
                depth, d_ms = depths[source]
                for cue in args.cues:
                    key = method_key("cue", cue)
                    out, ms = timed(apply_cue, cue, x, depth, strengths[key])
                    emit({**cond, "method": key, "depth_source": source, "strength": strengths[key]},
                         out, d_ms + ms)

            for cue in args.combo_cues:
                s = strengths[method_key("cue", cue)]
                out, ms = timed(apply_cue, cue, routed, perturbed_depth, s)
                emit({**cond, "method": f"detector+cue:{cue}", "depth_source": "perturbed", "strength": s,
                      **detect_meta}, out, detect_ms + restore_ms + depth_ms + ms)
    predictor.flush()

    df = prefixed(predictor.rows, "out")
    save(df, args.out / "records.csv.gz")
    summary = summarize(df)
    save(summary, args.out / "summary.csv")
    (args.out / "run.json").write_text(json.dumps({
        "split": "val", "images": len(records), "per_class": args.per_class, "seed": args.seed,
        "strengths_file": str(args.strengths), "strength_status": strength_status,
        "plans_file": str(args.plans), "plan_status": plan_status, "plans": plans_used(plans),
        "strengths": strengths, "depth_sources": args.depth_sources,
    }, indent=2))
    print(f"Strengths: {strength_status}; plans: {plan_status}")
    print(summarize_overall(df).to_string(index=False))


GROUP = ["perturbation", "level", "method", "depth_source"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition x method: compare each processed input with the unprocessed one.

    `acc_*` and fixes/breaks are against the ground-truth label. `agree_clean_*`
    is agreement with the classifier's prediction on the clean image.
    """
    keys = ["image_id", "perturbation", "level"]
    ref = df[df["method"] == "none"].set_index(keys)
    clean_ref = df[(df["perturbation"] == CLEAN) & (df["method"] == "none")].set_index("image_id")
    clean_acc = clean_ref["out_correct"].mean()
    rows = []
    for (name, level, method, source), g in df.groupby(GROUP, sort=False):
        before = ref.loc[pd.MultiIndex.from_frame(g[keys])]
        clean_pred = clean_ref.loc[g["image_id"], "out_pred"].to_numpy()
        b_ok, a_ok = before["out_correct"].to_numpy(bool), g["out_correct"].to_numpy(bool)
        rows.append({
            "perturbation": name, "level": level, "method": method, "depth_source": source,
            "strength": g["strength"].iloc[0], "n": len(g),
            "acc_clean": clean_acc,
            "acc_perturbed": b_ok.mean(),
            "acc_restored": a_ok.mean(),
            "delta_pp": 100.0 * (a_ok.mean() - b_ok.mean()),
            "recovery_rate": recovery_rate(clean_acc, b_ok.mean(), a_ok.mean()),
            **transitions(b_ok, a_ok, before["out_pred"].to_numpy(), g["out_pred"].to_numpy(), clean_pred),
            "changed_rate": (g["out_pred"].to_numpy() != before["out_pred"].to_numpy()).mean(),
            "true_prob_shift": (g["out_true_prob"].to_numpy() - before["out_true_prob"].to_numpy()).mean(),
            "depth_rank_corr": g["depth_rank_corr"].mean() if "depth_rank_corr" in g else np.nan,
            "psnr": g["psnr"].replace(np.inf, np.nan).mean(),
            "ssim": g["ssim"].mean(),
            "restore_ms": g["restore_ms"].mean(),
            "classify_ms": g["out_classify_ms"].mean(),
        })
    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    held_out = df["perturbation"].isin(HELD_OUT)
    perturbed = df[(df["perturbation"] != CLEAN) & ~held_out]
    clean = df[df["perturbation"] == CLEAN]
    by = ["method", "depth_source"]
    return pd.DataFrame({
        "acc_on_perturbed": perturbed.groupby(by, sort=False)["out_correct"].mean(),
        "acc_on_held_out": df[held_out].groupby(by, sort=False)["out_correct"].mean(),
        "acc_on_clean": clean.groupby(by, sort=False)["out_correct"].mean(),
        "restore_ms": df.groupby(by, sort=False)["restore_ms"].mean(),
    }).reset_index()


if __name__ == "__main__":
    main()
