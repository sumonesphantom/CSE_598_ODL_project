"""Which spatial evidence makes the best contextual cue: depth, SAM masks or DETR boxes?

The depth-outline cue helped foreground entities but missed flat, background-like
ones (churches). This study holds the cue fixed and swaps only the structure map
it is built from. All structure is estimated on the perturbed input (the
deployable setting). Results are broken out per class so the church case is visible.

Outputs (in --out):
  records.csv.gz   one row per image x perturbation x severity x cue x structure source
  summary.csv      per perturbation x level x cue x source: accuracy and delta vs. no cue
  per_class.csv    per class x cue x source: accuracy and delta vs. no cue
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.cues import CUES, apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator
from bolero.detection import CLEAN
from bolero.perturbations import ALL_PERTURBATIONS, LEVELS, rng_for
from bolero.records import prefixed, save
from bolero.segmentation import Segmenter, StructureCache, structure_map
from bolero.tuning import load_strengths, method_key

SOURCES = ["depth", "sam", "detr"]
DEFAULT_PERTURBATIONS = ["gaussian_blur", "boundary_blur", "foreground_blur", "darken", "gaussian_noise"]
DEFAULT_CUES = ["depth_boundary_emphasis", "entity_background_separation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--perturbations", nargs="*", default=DEFAULT_PERTURBATIONS,
                        choices=list(ALL_PERTURBATIONS))
    parser.add_argument("--cues", nargs="*", default=DEFAULT_CUES, choices=list(CUES))
    parser.add_argument("--sources", nargs="*", default=SOURCES, choices=SOURCES)
    parser.add_argument("--strengths", type=Path, default=RESULTS_DIR / "tuning" / "strengths.json")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "structure")
    args = parser.parse_args()

    strengths, _ = load_strengths(args.strengths)
    predictor = BatchPredictor(FrozenClassifier(args.backbone, device=args.device), args.batch_size)
    estimator = DepthEstimator(device=args.device)
    depth_cache = DepthCache(estimator)
    segmenters = {s: StructureCache(Segmenter(s, device=args.device)) for s in args.sources if s != "depth"}

    def structure(source, x):
        if source == "depth":
            return estimator(x)
        return structure_map(segmenters[source].entity_mask(x), x.shape[:2])

    conditions = [(CLEAN, 0)] + [(n, lv) for n in args.perturbations for lv in LEVELS]
    for record in tqdm(list_images("val", per_class=args.per_class, seed=args.seed), desc="images"):
        clean = load_image(record.path)
        clean_depth = depth_cache.get(record, clean)
        base = {"image_id": record.image_id, "wnid": record.wnid,
                "class_name": record.class_name, "label_index": record.label_index}
        for name, level in conditions:
            x = clean if name == CLEAN else ALL_PERTURBATIONS[name](
                clean, level, clean_depth, rng_for(record.image_id, name, level, args.seed))
            cond = {**base, "perturbation": name, "level": level}
            predictor.add({**cond, "cue": "none", "source": "none", "structure_ms": 0.0}, x)
            for source in args.sources:
                t0 = time.perf_counter()
                s_map = structure(source, x)
                ms = 1000.0 * (time.perf_counter() - t0)
                for cue in args.cues:
                    out = apply_cue(cue, x, s_map, strengths[method_key("cue", cue)])
                    predictor.add({**cond, "cue": cue, "source": source, "structure_ms": ms}, out)
    predictor.flush()

    df = prefixed(predictor.rows, "out")
    save(df, args.out / "records.csv.gz")
    save(compare_to_none(df, ["perturbation", "level"]), args.out / "summary.csv")
    save(compare_to_none(df, ["class_name"]), args.out / "per_class.csv")
    print(compare_to_none(df, ["class_name"]).to_string(index=False))


def compare_to_none(df, by):
    keys = ["image_id", "perturbation", "level"]
    ref = df[df["cue"] == "none"].set_index(keys)["out_correct"]
    df = df.assign(ref_correct=ref.loc[list(df[keys].itertuples(index=False, name=None))].to_numpy())
    g = df[df["cue"] != "none"].groupby(by + ["cue", "source"], as_index=False)
    out = g.agg(n=("out_correct", "size"), acc=("out_correct", "mean"), acc_none=("ref_correct", "mean"),
                structure_ms=("structure_ms", "mean"))
    return out.assign(delta_pp=100.0 * (out["acc"] - out["acc_none"]))


if __name__ == "__main__":
    main()
