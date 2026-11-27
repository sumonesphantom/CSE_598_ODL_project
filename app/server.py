"""Live demo: one weather-degraded frame through Decide -> Restore -> Detect, in the browser.

Run:
    python app/server.py                # http://127.0.0.1:8000
    python app/server.py --port 8010 --diffusion   # also offer InstructPix2Pix restoration (needs diffusers)

What runs where:
  Decide   the real-trained weather detector (results/weather_detector/detector_real.joblib, from
           scripts/train_weather_detector.py) when that file exists. Without it, a DAWN sample's
           condition comes from its dataset label and an upload needs a manual choice.
  Restore  "auto" follows our measured results: auto-contrast on dust (the only method that helped
           on real dust), nothing on rain (no method helped). Any method can be forced.
  Detect   frozen COCO DETR on the raw and the restored image; boxes shown at score >= 0.5.
  Changes  boxes added / removed / relabeled by restoration; with ground truth (DAWN samples),
           objects lost / gained / invented (bolero/hallucination.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from bolero.config import RESULTS_DIR  # noqa: E402
from bolero.detection_eval import Detections  # noqa: E402
from bolero.hallucination import MATCH_IOU, compare_image, keep_above  # noqa: E402
from bolero.weather_data import load_dawn, open_rgb  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SHOW_SCORE = 0.5
MAX_SIDE = 1280
WEATHER_DETECTOR = RESULTS_DIR / "weather_detector" / "detector_real.joblib"
CONDITIONS = ("rain", "dust", "clean")

# Measured on real data with the frozen detector (TODO.md, detection tables, after the EXIF fix).
EVIDENCE = {
    "rain": {"policy": "none",
             "why": "No method helped on real rain: every baseline's interval includes zero.",
             "known": [{"set": "DAWN rain", "map50": 0.481, "person_recall": 0.89},
                       {"set": "XWOD rain", "map50": 0.252, "person_recall": 0.58}]},
    "dust": {"policy": "baseline:autocontrast",
             "why": "Auto-contrast is the only method that helped on real dust: +6.2 pp mAP50 on XWOD, +1.3 pp on DAWN.",
             "known": [{"set": "DAWN dust", "map50": 0.457, "person_recall": 0.83},
                       {"set": "XWOD dust", "map50": 0.446, "person_recall": 0.85}]},
    "clean": {"policy": "none", "why": "A clear view is passed through untouched.", "known": []},
}
RESTORERS = {
    "auto": "Auto (from our results)",
    "none": "None",
    "baseline:autocontrast": "Auto-contrast",
    "baseline:clahe": "CLAHE",
    "baseline:unsharp": "Unsharp mask",
    "ip2p": "InstructPix2Pix (diffusion)",
}


class State:
    detector = None
    weather = None
    weather_error: str | None = None
    load_error: str | None = None
    remover = None
    diffusion = False
    strengths: dict = {}
    samples: dict = {}
    lock = threading.Lock()


S = State()


def load_models() -> None:
    from bolero.detection_eval import FrozenDetector
    from bolero.tuning import load_strengths

    S.detector = FrozenDetector("detr", min_score=0.05)
    S.strengths, _ = load_strengths(RESULTS_DIR / "tuning" / "strengths.json")
    if WEATHER_DETECTOR.exists():
        from bolero.detection import PerturbationDetector
        S.weather = PerturbationDetector.load(WEATHER_DETECTOR)
    else:
        S.weather_error = f"{WEATHER_DETECTOR.relative_to(ROOT)} not found"
    if S.diffusion:
        from bolero.generative import WeatherRemover
        S.remover = WeatherRemover()


def pick_samples(per_condition: int = 6) -> dict:
    """DAWN frames with a readable number of labeled objects, spread over each condition."""
    out = {}
    try:
        samples = load_dawn()
    except FileNotFoundError:
        return out
    for cond in ("rain", "dust"):
        pool = [s for s in samples if s.condition == cond and 4 <= len(s.labels) <= 14]
        step = max(1, len(pool) // per_condition)
        for s in pool[::step][:per_condition]:
            out[f"dawn-{cond}-{s.image_id}"] = s
    return out


def fit(image: Image.Image) -> tuple[Image.Image, float]:
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image, scale


def data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def decide(image: Image.Image, sample, override: str | None) -> dict:
    probs = None
    if S.weather is not None:
        import cv2
        from bolero.detection import extract_features
        x = np.asarray(image, np.float32) / 255.0
        s = 320 / min(x.shape[:2])
        x = cv2.resize(x, (max(1, round(x.shape[1] * s)), max(1, round(x.shape[0] * s))), interpolation=cv2.INTER_AREA)
        p = S.weather.predict_proba(extract_features(x))[0]
        probs = {str(c): float(v) for c, v in zip(S.weather.classes, p)}
        d = S.weather.decide(x)
        detected = d.label if d.label in CONDITIONS else "clean"
    else:
        detected = None
    if override in CONDITIONS:
        return {"condition": override, "source": "override", "probs": probs, "detected": detected,
                "threshold": getattr(S.weather, "threshold", None)}
    if detected is not None:
        return {"condition": detected, "source": "detector", "probs": probs, "detected": detected,
                "threshold": S.weather.threshold}
    if sample is not None:
        return {"condition": sample.condition, "source": "label", "probs": None, "detected": None, "threshold": None}
    return {"condition": None, "source": "missing", "probs": None, "detected": None, "threshold": None}


def restore(image: Image.Image, condition: str | None, choice: str, seed_key: str) -> tuple[Image.Image, str, float]:
    method = choice
    if choice == "auto":
        method = EVIDENCE.get(condition or "clean", EVIDENCE["clean"])["policy"]
    start = time.perf_counter()
    if method == "none":
        out = image
    elif method.startswith("baseline:"):
        from bolero.baselines import apply_baseline
        from bolero.tuning import method_key
        name = method.split(":", 1)[1]
        y = apply_baseline(name, np.asarray(image, np.float32) / 255.0, S.strengths[method_key("baseline", name)])
        out = Image.fromarray(np.round(np.clip(y, 0, 1) * 255).astype(np.uint8))
    elif method == "ip2p":
        if S.remover is None:
            raise HTTPException(400, "Diffusion restoration is off. Start the server with --diffusion (needs diffusers).")
        if condition not in ("rain", "dust"):
            raise HTTPException(400, "Diffusion restoration needs a rain or dust condition.")
        from bolero.generative import edit_seed
        out = S.remover(image, condition, 1.5, edit_seed(seed_key, condition, 1.5))
    else:
        raise HTTPException(400, f"Unknown restorer {method!r}")
    return out, method, time.perf_counter() - start


def boxes_json(det: Detections) -> list[dict]:
    return [{"box": [round(float(v), 1) for v in b], "label": l, "score": round(float(s), 3)}
            for b, l, s in zip(det.boxes, det.labels, det.scores)]


def box_changes(before: Detections, after: Detections) -> dict:
    """Ground-truth-free changes: which shown boxes restoration added, removed or relabeled."""
    from bolero.detection_eval import iou_matrix
    cross = iou_matrix(after.boxes, before.boxes)
    added, removed, relabeled = [], [], []
    for k in range(len(after.scores)):
        if not cross.shape[1] or cross[k].max() < MATCH_IOU:
            added.append(k)
    for j in range(len(before.scores)):
        if not cross.shape[0] or cross[:, j].max() < MATCH_IOU:
            removed.append(j)
        else:
            k = int(cross[:, j].argmax())
            if after.labels[k] != before.labels[j]:
                relabeled.append({"before": j, "after": k, "from": before.labels[j], "to": after.labels[k]})
    return {"added": added, "removed": removed, "relabeled": relabeled}


def counts(det: Detections) -> dict:
    out: dict[str, int] = {}
    for l in det.labels:
        out[l] = out.get(l, 0) + 1
    return out


class RunRequest(BaseModel):
    sample_id: str | None = None
    image: str | None = None          # data URL or bare base64 for uploads
    condition: str | None = None      # rain | dust | clean | None (= let Decide choose)
    restorer: str = "auto"


app = FastAPI(title="Bolero live demo")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"ready": S.detector is not None, "load_error": S.load_error, "weather_detector": S.weather is not None,
            "weather_detector_note": S.weather_error, "diffusion": S.remover is not None,
            "restorers": [{"id": k, "label": v, "available": k != "ip2p" or S.remover is not None}
                          for k, v in RESTORERS.items()],
            "evidence": EVIDENCE, "show_score": SHOW_SCORE}


@app.get("/api/samples")
def samples():
    return [{"id": k, "condition": s.condition, "dataset": "DAWN", "objects": len(s.labels),
             "name": s.image_id} for k, s in S.samples.items()]


@app.get("/api/thumb/{sample_id}")
def thumb(sample_id: str):
    s = S.samples.get(sample_id)
    if s is None:
        raise HTTPException(404, "No such sample")
    im = open_rgb(s.path)
    im.thumbnail((360, 240))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/run")
def run(req: RunRequest):
    if S.detector is None:
        raise HTTPException(503, "Models are still loading. Try again in a few seconds.")
    sample = None
    if req.sample_id:
        sample = S.samples.get(req.sample_id)
        if sample is None:
            raise HTTPException(404, "No such sample")
        raw = open_rgb(sample.path)
        key = sample.image_id
    elif req.image:
        try:
            payload = req.image.split(",", 1)[1] if req.image.startswith("data:") else req.image
            from PIL import ImageOps
            with Image.open(io.BytesIO(base64.b64decode(payload))) as im:
                raw = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            raise HTTPException(400, "That file isn't an image we can read. Try a JPEG or PNG.")
        key = "upload"
    else:
        raise HTTPException(400, "Pick a sample or upload an image.")
    if req.restorer not in RESTORERS:
        raise HTTPException(400, f"Unknown restorer {req.restorer!r}")

    raw, scale = fit(raw)
    with S.lock:
        dec = decide(raw, sample, req.condition)
        if dec["condition"] is None and req.restorer == "auto":
            return {"needs_condition": True, "decide": dec, "raw": data_url(raw),
                    "size": list(raw.size)}
        restored, method, restore_s = restore(raw, dec["condition"], req.restorer, key)
        det_raw, det_res = S.detector(raw), S.detector(restored)

    shown_raw, shown_res = keep_above(det_raw, SHOW_SCORE), keep_above(det_res, SHOW_SCORE)
    out = {
        "size": list(raw.size), "raw": data_url(raw),
        "restored": data_url(restored) if method != "none" else None,
        "decide": dec,
        "restore": {"choice": req.restorer, "method": method, "label": RESTORERS.get(method, method),
                    "seconds": restore_s,
                    "why": EVIDENCE.get(dec["condition"] or "clean", EVIDENCE["clean"])["why"] if req.restorer == "auto" else None},
        "detect": {"raw": boxes_json(shown_raw), "restored": boxes_json(shown_res),
                   "counts_raw": counts(shown_raw), "counts_restored": counts(shown_res),
                   "ms": [1000 * det_raw.seconds, 1000 * det_res.seconds]},
        "changes": box_changes(shown_raw, shown_res),
        "known": EVIDENCE.get(dec["condition"] or "clean", EVIDENCE["clean"])["known"],
        "gt": None, "hallucination": None,
    }
    if sample is not None:
        gt = sample.boxes * scale
        out["gt"] = [{"box": [round(float(v), 1) for v in b], "label": l} for b, l in zip(gt, sample.labels)]
        out["hallucination"] = compare_image(det_raw, det_res, gt, sample.labels, SHOW_SCORE)
    return out


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--diffusion", action="store_true", help="load InstructPix2Pix (needs diffusers, ~5 GB)")
    args = parser.parse_args()
    S.diffusion = args.diffusion
    S.samples = pick_samples()
    def load():
        try:
            load_models()
        except Exception as e:  # surfaced in /api/status and the terminal
            S.load_error = f"{type(e).__name__}: {e}"
            print(f"model loading failed: {S.load_error}")

    threading.Thread(target=load, daemon=True).start()
    print(f"Bolero demo on http://{args.host}:{args.port}  ({len(S.samples)} DAWN samples)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
