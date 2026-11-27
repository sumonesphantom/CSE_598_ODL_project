"""Probe which decision models accept which image resolutions.

TypeSafe rejects images under 280 px on a side; the other Decisions API models
do not publish a minimum. This script sends one real image at several short-side
resolutions to each model and records accept/reject, so the README can state the
result instead of guessing.

Outputs (in --out):
  resolution_probe.csv   model x resolution: status, accepted, cost, latency
"""

import argparse
import base64
import io
import time
from pathlib import Path

import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, DECISION_MODELS, RESULTS_DIR, load_env
from bolero.data import list_images, load_image
from bolero.openrouter import BudgetExceeded, DecisionsClient, OpenRouterError

QUESTION = {
    "degraded": {
        "type": "noul",
        "instructions": "Is this view degraded by weather or another image corruption?",
        "criteria": {"true": "A visible corruption is present.", "false": "The view looks clean."},
    }
}


def encode_at_short_side(image, short_side: int, quality: int = 95) -> tuple[str, int, int]:
    """Resize so the shorter side is `short_side`, then return a JPEG data URL."""
    pil = Image.fromarray((image * 255).round().clip(0, 255).astype("uint8"))
    w, h = pil.size
    scale = short_side / min(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    pil = pil.resize(size, Image.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", size[0], size[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=list(DECISION_MODELS),
                        help="keys in config.DECISION_MODELS or full model ids")
    parser.add_argument("--sizes", nargs="*", type=int, default=[160, 224, 280, 320],
                        help="short-side resolutions to probe")
    parser.add_argument("--image", type=Path, default=None,
                        help="image to send (default: first Imagenette val image)")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR / "imagenette2-320",
                        help="Imagenette root (default: the 320 px release)")
    parser.add_argument("--max-cost", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "deciders")
    args = parser.parse_args()

    load_env()
    image_path = args.image
    if image_path is None:
        records = list_images("val", root=args.data_root, per_class=1)
        image_path = records[0].path
    image = load_image(image_path)

    client = DecisionsClient(max_cost=args.max_cost)
    rows: list[dict] = []
    for alias in args.models:
        model = DECISION_MODELS.get(alias, alias)
        for size in args.sizes:
            url, width, height = encode_at_short_side(image, size)
            state = ["Resolution probe image.", {"type": "image_url", "image_url": {"url": url}}]
            started = time.perf_counter()
            row = {"model": model, "requested_short_side": size, "width": width, "height": height,
                   "accepted": False, "status": None, "error": "not attempted",
                   "cost": 0.0, "latency_ms": 0.0, "cached": False, "answers": ""}
            try:
                response = client.ask(model, state, QUESTION, use_cache=False)
                row = {"model": model, "requested_short_side": size, "width": width, "height": height,
                       "accepted": True, "status": 200, "error": "",
                       "cost": response.cost, "latency_ms": response.latency_ms, "cached": response.cached,
                       "answers": str(response.answers)}
            except OpenRouterError as exc:
                row = {"model": model, "requested_short_side": size, "width": width, "height": height,
                       "accepted": False, "status": exc.status, "error": exc.message,
                       "cost": 0.0, "latency_ms": (time.perf_counter() - started) * 1000.0,
                       "cached": False, "answers": ""}
            except BudgetExceeded:
                print("Stopping: cost budget reached.")
                break
            rows.append(row)
            print(f"{model:45s} {size:>4}px -> {'ok' if row['accepted'] else 'rejected'} ({row['status']})")

    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "resolution_probe.csv", index=False)
    print()
    if not frame.empty:
        print(frame.pivot_table(index="model", columns="requested_short_side", values="accepted",
                                aggfunc="first").to_string())
    print(f"\nwrote {args.out / 'resolution_probe.csv'}")


if __name__ == "__main__":
    main()
