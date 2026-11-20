"""Sample labelled frames from VIRADA (AAU Visual Rain Dataset, Zenodo 4715681, CC BY 4.0).

The full dataset is 167 GB of video. This fetches only the clips it needs: zip archives keep
their table of contents at the end, so the archive is opened over HTTP range requests and each
chosen clip (about 2.4 MB) is read on its own.

A clip is used only if every minute in it has the same label from both rain sensors (the laser
disdrometer and the tipping-bucket gauge), so no frame sits on a rain / no-rain boundary. All
such rain clips are taken (46 in Crossing2); dry clips are sampled `--dry-per-rain` times as
many, half of them from the days that also have rain, so "rain" cannot be learned from the
day's lighting alone. `--frames` evenly spaced frames are saved per clip.

Outputs (default data/virada/, git-ignored):
  frames/<rain|no_rain>/<clip>_<k>.jpg
  frames.csv      one row per frame: path, label, clip, crop, day

Usage:
  python scripts/prepare_virada.py
"""

import argparse
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, SEED

RECORD = "https://zenodo.org/records/4715681/files/{name}?download=1"
CROPS = ("brick", "asphalt")
USER_AGENT = "bolero-coursework-data-prep"


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(self._request(0, 0)) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def _request(self, start: int, end: int) -> urllib.request.Request:
        return urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size or not len(buffer):
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        with urllib.request.urlopen(self._request(self.pos, end)) as r:
            data = r.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def consistent_clips(labels_zip: zipfile.ZipFile, crop: str) -> pd.DataFrame:
    laser = json.loads(labels_zip.read(f"Labels/Crossing2-{crop}_laser_binary.json"))
    mech = json.loads(labels_zip.read(f"Labels/Crossing2-{crop}_mech_binary.json"))
    rows = []
    for clip, v in laser.items():
        a, b = v["labels"], mech.get(clip, {}).get("labels", [])
        if a and a == b and len(set(a)) == 1:
            month, day = clip.split("-")[:2]
            rows.append({"clip": clip, "crop": crop, "label": "rain" if a[0] else "no_rain",
                         "day": f"2013-{int(month):02d}-{int(day):02d}", "frames_in_clip": v["frameCount"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=3, help="frames saved per clip")
    parser.add_argument("--dry-per-rain", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "virada")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / "Labels.zip"
    if not labels_path.exists():
        urllib.request.urlretrieve(RECORD.format(name="Labels.zip"), labels_path)
    clips = pd.concat([consistent_clips(zipfile.ZipFile(labels_path), c) for c in CROPS], ignore_index=True)

    rng = np.random.default_rng(args.seed)
    rain = clips[clips.label == "rain"]
    dry = clips[clips.label == "no_rain"]
    n_dry = args.dry_per_rain * len(rain)
    same_day = dry[dry.day.isin(rain.day)]
    take_same = min(len(same_day), n_dry // 2)
    picked_same = same_day.iloc[rng.choice(len(same_day), take_same, replace=False)]
    rest = dry.drop(picked_same.index)
    picked_rest = rest.iloc[rng.choice(len(rest), n_dry - take_same, replace=False)]
    chosen = pd.concat([rain, picked_same, picked_rest]).sort_values(["crop", "clip"])
    print(f"{len(rain)} rain clips over {rain.day.nunique()} days, {n_dry} dry clips "
          f"({take_same} from rain days); {args.frames} frames each")

    frames_csv = args.out / "frames.csv"
    done = set(pd.read_csv(frames_csv)["clip"]) if frames_csv.exists() else set()
    rows = pd.read_csv(frames_csv).to_dict("records") if frames_csv.exists() else []
    for crop, group in chosen.groupby("crop"):
        archive = zipfile.ZipFile(io.BufferedReader(RangeFile(RECORD.format(name=f"Crossing2-{crop}.zip")), 1 << 20))
        for clip in tqdm(group.itertuples(), total=len(group), desc=crop):
            if clip.clip in done:
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(archive.read(f"Crossing2-{crop}/{clip.clip}"))
                tmp.flush()
                cap = cv2.VideoCapture(tmp.name)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or clip.frames_in_clip
                for k, index in enumerate(np.linspace(0, n - 1, args.frames + 2)[1:-1].astype(int)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    path = args.out / "frames" / clip.label / f"{Path(clip.clip).stem}_{k}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append({"path": str(path), "label": clip.label, "clip": clip.clip, "crop": crop,
                                 "day": clip.day, "frame": int(index)})
                cap.release()
            pd.DataFrame(rows).to_csv(frames_csv, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["label", "crop"]).agg(frames=("path", "size"), clips=("clip", "nunique"), days=("day", "nunique")))


if __name__ == "__main__":
    main()
