"""Generative degradation: InstructPix2Pix rain / dust storm versions of clean images.

Severity is the edit strength (see bolero/generative.py). Every edit is kept on disk with
its CLIP similarity to the clean image; whether the content survived is decided at analysis
time with a similarity threshold, never with the classifier or detector under test.

Sources:
  coco        clean COCO val2017 traffic scenes with boxes (weather_data.load_coco), for detection
  imagenette  the Imagenette val sample used by the weather diagnostic, for classification

Outputs (in --out, default data/generative/<source>, git-ignored):
  <condition>/<level>/<image_id>.png   edited image, same size as the clean one
  manifest.csv                         one row per edit: ids, paths, clip_sim, edit seconds

Re-running skips edits that already exist, so an interrupted run resumes.

Usage:
  python scripts/make_generative_weather.py --source coco --n 300
  python scripts/make_generative_weather.py --source imagenette --per-class 20
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, SEED
from bolero.data import list_images
from bolero.generative import IMAGE_GUIDANCE, PROMPTS, ClipEmbedder, WeatherEditor, edit_seed
from bolero.weather_data import load_coco, open_rgb


def sources(args) -> list[dict]:
    if args.source == "coco":
        return [{"image_id": s.image_id, "clean_path": str(s.path)} for s in load_coco(n=args.n, seed=args.seed)]
    return [{"image_id": r.image_id, "clean_path": str(r.path), "wnid": r.wnid, "label_index": r.label_index}
            for r in list_images("val", per_class=args.per_class, seed=args.seed)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["coco", "imagenette"], required=True)
    parser.add_argument("--n", type=int, default=300, help="COCO images")
    parser.add_argument("--per-class", type=int, default=20, help="Imagenette images per class")
    parser.add_argument("--conditions", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    parser.add_argument("--levels", nargs="+", type=int, default=list(IMAGE_GUIDANCE), choices=list(IMAGE_GUIDANCE))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or DATA_DIR / "generative" / args.source

    items = sources(args)
    jobs = [{**it, "condition": c, "level": lv, "path": str(out / c / str(lv) / f"{it['image_id']}.png")}
            for it in items for c in args.conditions for lv in args.levels]
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(items)} images, {len(jobs)} edits, {len(todo)} to do")

    seconds = {}
    if todo:
        editor = WeatherEditor(args.device)
        start = time.time()
        for j in tqdm(todo, desc="edits"):
            clean = open_rgb(j["clean_path"])
            edited = editor(clean, j["condition"], j["level"], edit_seed(j["image_id"], j["condition"], j["level"], args.seed))
            Path(j["path"]).parent.mkdir(parents=True, exist_ok=True)
            edited.save(j["path"])
            seconds[j["path"]] = editor.last_seconds
        print(f"edited {len(todo)} in {(time.time() - start) / 60:.1f} min")
        del editor

    clip = ClipEmbedder(args.device)
    clean_emb = dict(zip([it["image_id"] for it in items],
                         clip([open_rgb(it["clean_path"]) for it in items])))
    rows = []
    for i in tqdm(range(0, len(jobs), 64), desc="clip"):
        batch = jobs[i:i + 64]
        emb = clip([Image.open(j["path"]) for j in batch])
        for j, e in zip(batch, emb):
            rows.append({**j, "clip_sim": float(e @ clean_emb[j["image_id"]]), "edit_seconds": seconds.get(j["path"])})
    manifest = pd.DataFrame(rows)
    old = out / "manifest.csv"
    if old.exists():  # keep edit times measured by earlier, interrupted runs
        prev = pd.read_csv(old).set_index("path")["edit_seconds"]
        manifest["edit_seconds"] = manifest["edit_seconds"].fillna(manifest["path"].map(prev))
    manifest.to_csv(old, index=False)
    print(manifest.groupby(["condition", "level"])["clip_sim"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
