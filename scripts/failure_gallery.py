"""Failure-case gallery for one Imagenette class (default: church).

Classifies every clean val image of the class, then lays out the most confidently
wrong cases (plus a few correct ones for contrast) as rows of:

  input | depth outline | entity mask | entity only (background grayed out)

with the classifier's top-1 on the input and on the entity alone. The last
column tests the "information removal" spatial instruction: does the classifier
recover once the background is removed?

Outputs (in --out):
  predictions.csv   one row per val image of the class: clean top-1/top-3 (+ entity-only with --sam-all)
  confusions.csv    what the failures were called instead
  cases.csv         the gallery rows, with entity-only predictions
  gallery_*.png     the figures (rows_per_page rows each)
  summary.json      accuracy, top confusions, entity-only fixed / broken counts
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

import _bootstrap  # noqa: F401,E402
from bolero.classifier import FrozenClassifier  # noqa: E402
from bolero.config import IMAGENETTE_CLASSES, RESULTS_DIR, SEED  # noqa: E402
from bolero.data import list_images, load_image  # noqa: E402
from bolero.depth import DepthEstimator, boundary_strength  # noqa: E402
from bolero.gallery import confusion_counts, foreground_only, select_cases, transitions  # noqa: E402
from bolero.plotting import apply_style  # noqa: E402
from bolero.segmentation import (Segmenter, StructureCache, TextSegmenter, regions_in_mask,  # noqa: E402
                                 threshold_text_mask)

NAME_TO_WNID = {name: wnid for wnid, name in IMAGENETTE_CLASSES.items()}
MASK_HEADERS = {"sam": "SAM entity mask", "clipseg": "CLIPSeg text mask", "sam_text": "SAM parts in text mask"}


def classify(classifier, images, true_index, batch_size):
    """Top-1 label, confidence, true-class probability and top-3 string for each image."""
    rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        preds = classifier.predict(batch, np.full(len(batch), true_index))
        for i in range(len(batch)):
            cands = classifier.candidates(preds, i)
            rows.append({
                "pred_index": int(preds.index[i]),
                "pred_label": classifier.id2label[int(preds.index[i])],
                "confidence": float(preds.confidence[i]),
                "true_prob": float(preds.true_prob[i]),
                "top3": "; ".join(f"{label} {p:.2f}" for label, p in cands[:3]),
            })
    return rows


def short(label: str, width: int = 22) -> str:
    label = label.split(",")[0]
    return label if len(label) <= width else label[: width - 1] + "…"


def draw_page(cases: list[dict], title: str, path: Path, mask_header: str = "SAM entity mask") -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.1 * n + 0.6), squeeze=False)
    headers = ["Input", "Depth outline", mask_header, "Entity only"]
    for row, case in enumerate(cases):
        ax_in, ax_depth, ax_sam, ax_fg = axes[row]
        mark = "✓" if case["correct"] else "✗"
        ax_in.imshow(case["image"])
        ax_in.set_title(f"{case['image_id']}\n{short(case['pred_label'])} {case['confidence']:.2f} {mark}",
                        fontsize=9)
        ax_depth.imshow(case["image"].mean(axis=-1), cmap="gray", alpha=0.6)
        ax_depth.imshow(case["outline"], cmap="magma", alpha=0.75, vmin=0, vmax=0.5)
        ax_depth.set_title("strong edges = near/far change", fontsize=9)
        ax_sam.imshow(case["image"])
        if case["mask"] is not None:
            overlay = np.zeros((*case["mask"].shape, 4))
            overlay[case["mask"]] = (0.16, 0.47, 0.84, 0.45)
            ax_sam.imshow(overlay)
            ax_sam.set_title(f"entity = {case['mask'].mean():.0%} of frame", fontsize=9)
        else:
            ax_sam.set_title("no entity found", fontsize=9)
        ax_fg.imshow(case["entity_image"])
        fg_mark = "✓" if case["fg_correct"] else "✗"
        ax_fg.set_title(f"{short(case['fg_label'])} {case['fg_confidence']:.2f} {fg_mark}", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, header in zip(axes[0], headers):
        ax.annotate(header, xy=(0.5, 1.32), xycoords="axes fraction", ha="center",
                    fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", default="church", choices=sorted(NAME_TO_WNID))
    parser.add_argument("--n-fail", type=int, default=12, help="most confidently wrong cases to show")
    parser.add_argument("--n-ok", type=int, default=4, help="correct cases to show for contrast")
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--sam-all", action="store_true",
                        help="also run SAM + entity-only classification on every image of the class")
    parser.add_argument("--entity-source", default="sam", choices=["sam", "clipseg", "sam_text"],
                        help="sam: unprompted SAM pick; clipseg: text-prompted mask; sam_text: SAM parts inside the clipseg mask")
    parser.add_argument("--prompt", default=None, help='text prompt for clipseg (default: "a <class>")')
    parser.add_argument("--sam-grid", type=int, default=16, help="SAM prompt points per side")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    start = time.time()
    wnid = NAME_TO_WNID[args.class_name]
    suffix = "" if args.entity_source == "sam" else f"_{args.entity_source}"
    out = args.out or RESULTS_DIR / "failure_gallery" / (args.class_name.replace(" ", "_") + suffix)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    records = [r for r in list_images("val") if r.wnid == wnid]
    true_index = records[0].label_index
    classifier = FrozenClassifier(args.backbone, device=args.device)
    images = [load_image(r.path) for r in records]
    table = pd.DataFrame(classify(classifier, images, true_index, args.batch_size))
    table.insert(0, "image_id", [r.image_id for r in records])
    table["correct"] = table["pred_index"] == true_index
    print(f"{args.class_name}: {len(table)} val images, clean accuracy {table['correct'].mean():.3f}")

    prompt = args.prompt or f"a {args.class_name}"
    segmenter = Segmenter("sam", device=args.device, sam_grid=args.sam_grid) if args.entity_source != "clipseg" else None
    sam = StructureCache(segmenter) if args.entity_source == "sam" else None
    text = TextSegmenter(device=args.device) if args.entity_source != "sam" else None
    index = {r.image_id: i for i, r in enumerate(records)}

    def entity(i):
        if args.entity_source == "sam":
            return sam.entity_mask(images[i], records[i])
        guide = threshold_text_mask(text(images[i], prompt))
        if args.entity_source == "clipseg":
            return guide
        return regions_in_mask(segmenter(images[i]), guide)

    if args.sam_all:
        masks = [entity(i) for i in tqdm(range(len(records)), desc="SAM")]
        fg = classify(classifier, [foreground_only(im, m) for im, m in zip(images, masks)],
                      true_index, args.batch_size)
        table["fg_label"] = [r["pred_label"] for r in fg]
        table["fg_confidence"] = [r["confidence"] for r in fg]
        table["fg_correct"] = [r["pred_index"] == true_index for r in fg]
        table["entity_found"] = [m is not None for m in masks]
        table["entity_area"] = [float(m.mean()) if m is not None else 0.0 for m in masks]

    cases = select_cases(table, args.n_fail, args.n_ok, seed=args.seed)
    estimator = DepthEstimator(device=args.device)
    drawn = []
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="gallery"):
        i = index[row["image_id"]]
        mask = entity(i)
        entity_image = foreground_only(images[i], mask)
        fg = classify(classifier, [entity_image], true_index, 1)[0]
        drawn.append({**row.to_dict(), "image": images[i], "outline": boundary_strength(estimator(images[i])),
                      "mask": mask, "entity_image": entity_image, "fg_label": fg["pred_label"],
                      "fg_confidence": fg["confidence"], "fg_correct": fg["pred_index"] == true_index,
                      "entity_area": float(mask.mean()) if mask is not None else 0.0})

    keep = ["image_id", "case", "pred_label", "confidence", "true_prob", "top3", "correct",
            "fg_label", "fg_confidence", "fg_correct", "entity_area"]
    case_table = pd.DataFrame([{k: d[k] for k in keep} for d in drawn])
    case_table.to_csv(out / "cases.csv", index=False)
    table.to_csv(out / "predictions.csv", index=False)
    confusions = confusion_counts(table)
    confusions.to_csv(out / "confusions.csv", index=False)

    pages = []
    for case_kind, label in [("fail", "misclassified"), ("ok", "correct, for contrast")]:
        subset = [d for d in drawn if d["case"] == case_kind]
        for p in range(0, len(subset), args.rows_per_page):
            path = out / f"gallery_{case_kind}_{p // args.rows_per_page + 1}.png"
            draw_page(subset[p:p + args.rows_per_page],
                      f"{args.class_name.capitalize()}: {label} ({args.backbone}, clean val images)", path,
                      MASK_HEADERS[args.entity_source])
            pages.append(path.name)

    summary = {
        "class": args.class_name,
        "backbone": args.backbone,
        "entity_source": args.entity_source,
        "images": int(len(table)),
        "clean_accuracy": float(table["correct"].mean()),
        "top_confusions": confusions.head(8).to_dict(orient="records"),
        "gallery_cases": transitions(case_table["correct"], case_table["fg_correct"]),
        "pages": pages,
        "wall_minutes": (time.time() - start) / 60,
    }
    if args.sam_all:
        summary["entity_only_accuracy"] = float(table["fg_correct"].mean())
        summary["entity_found_rate"] = float(table["entity_found"].mean())
        summary["all_images"] = transitions(table["correct"], table["fg_correct"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "top_confusions"}, indent=2))
    print(confusions.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
