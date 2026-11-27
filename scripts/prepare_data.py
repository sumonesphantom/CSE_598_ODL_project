"""Locate Imagenette, downloading it into ./data if it is not already available.

Both the 160 px and 320 px releases share the same train/val/wnid/ILSVRC*.JPEG
layout. The decision-engine and VLM experiments use the 320 px release because
TypeSafe's image input rejects images under 280 px on a side.
"""

import argparse
import tarfile
import urllib.request

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, IMAGENETTE_VERSIONS, find_imagenette_root, imagenette_dirname


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="160", choices=sorted(IMAGENETTE_VERSIONS),
                        help="Imagenette release to fetch (default: 160)")
    parser.add_argument("--force", action="store_true", help="download even if a copy is found")
    args = parser.parse_args()

    root = None if args.force else find_imagenette_root(args.version)
    if root is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dirname = imagenette_dirname(args.version)
        archive = DATA_DIR / f"{dirname}.tgz"
        if not archive.exists():
            url = IMAGENETTE_VERSIONS[args.version]
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
        print(f"Extracting {archive}")
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(DATA_DIR, filter="data")
            except TypeError:  # Python < 3.10.12 has no extraction filters
                tar.extractall(DATA_DIR)
        root = DATA_DIR / dirname

    for split in ("train", "val"):
        n = sum(1 for p in (root / split).rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        print(f"{split:5s}: {n:,} images")
    print(f"Imagenette root: {root}")
    print(f"Set IMAGENETTE_VERSION={args.version} to use it by default.")


if __name__ == "__main__":
    main()
