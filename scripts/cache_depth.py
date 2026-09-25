"""Precompute clean-image depth maps so later stages never re-run the depth model on them."""

import argparse

from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--per-class", type=int, default=None, help="images per class (default: all)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = list_images(args.split, per_class=args.per_class)
    cache = DepthCache(DepthEstimator(device=args.device))
    missing = [r for r in records if not cache.path(r).exists()]
    print(f"{len(records):,} images, {len(missing):,} need depth -> {cache.root}")
    for record in tqdm(missing, desc="depth"):
        cache.get(record, load_image(record.path))


if __name__ == "__main__":
    main()
