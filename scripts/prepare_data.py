"""Locate Imagenette2-160, downloading it into ./data if it is not already available."""

import argparse
import tarfile
import urllib.request

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, IMAGENETTE_URL, find_imagenette_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download even if a copy is found")
    args = parser.parse_args()

    root = None if args.force else find_imagenette_root()
    if root is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        archive = DATA_DIR / "imagenette2-160.tgz"
        if not archive.exists():
            print(f"Downloading {IMAGENETTE_URL} (~94 MB)")
            urllib.request.urlretrieve(IMAGENETTE_URL, archive)
        print(f"Extracting {archive}")
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(DATA_DIR, filter="data")
            except TypeError:  # Python < 3.10.12 has no extraction filters
                tar.extractall(DATA_DIR)
        root = DATA_DIR / "imagenette2-160"

    for split in ("train", "val"):
        n = sum(1 for p in (root / split).rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        print(f"{split:5s}: {n:,} images")
    print(f"Imagenette root: {root}")
    print("Set IMAGENETTE_ROOT to this path if it is outside ./data.")


if __name__ == "__main__":
    main()
