"""Stage 7 - build a 30-case human review packet for the VLM predictions.

The VLM evaluation in Stage 6 scores answers automatically against the true
class and against geometry derived from the clean depth map. Those automatic
labels can be wrong in interesting ways, so this script samples a small,
stratified packet of cases, reconstructs the exact degraded view the VLM saw,
and lays out the clean / degraded / overlay thumbnails next to the model's
answer and the automatic scores. A human fills the three `manual_*_correct`
columns with 1, 0, or blank; `--score` then compares those judgements with the
automatic labels and reports agreement and Cohen's kappa.

The packet is deterministic: cases are selected by sorting on
(perturbation, class_name, image_id) and taking evenly spaced images from each
of the four occlusion conditions. Images are reconstructed with the same
`SEED` and `rng_for` call as `scripts/run_vlm_eval.py`, so the degraded
thumbnail matches the image the model actually saw.

Outputs (in --out):
  review.csv         one row per image x config, with empty manual columns
  contact_sheet.html self-contained page with the three thumbnails per row
  *_clean.jpg        clean view thumbnail (max width 320)
  *_degraded.jpg     reconstructed degraded view thumbnail
  *_overlay.jpg      degraded view with entity/damaged boxes drawn
  agreement.json     written by --score next to the filled review CSV
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from bolero.config import CACHE_DIR, DATA_DIR, RESULTS_DIR, SEED
from bolero.data import ImageRecord, load_image, to_uint8
from bolero.depth import DepthCache, near_mask
from bolero.perturbations import ALL_PERTURBATIONS, rng_for
from bolero.prompting import draw_regions, region_evidence
from bolero.vlm_scoring import damaged_mask

REVIEW_FIELDS = [
    "case_id",
    "image_id",
    "class_name",
    "perturbation",
    "level",
    "config",
    "clean_image",
    "degraded_image",
    "overlay_image",
    "vlm_entity",
    "vlm_damage",
    "vlm_relation",
    "vlm_confidence",
    "auto_entity_hit",
    "auto_damage_hit",
    "auto_relation_hit",
    "auto_relation_geometric",
    "manual_entity_correct",
    "manual_damage_correct",
    "manual_relation_correct",
    "notes",
]

# field -> (manual column, automatic column)
SCORE_FIELDS = {
    "entity": ("manual_entity_correct", "auto_entity_hit"),
    "damage": ("manual_damage_correct", "auto_damage_hit"),
    "relation": ("manual_relation_correct", "auto_relation_hit"),
}

_TRUE_VALUES = {"1", "true", "yes", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "n", "f"}
_BLANK_VALUES = {"", "nan", "none", "null", "na"}

THUMB_MAX_WIDTH = 320


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    """Lenient boolean for values read back from CSV or pandas."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and value != 0.0)
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def parse_manual(value: Any) -> bool | None:
    """Parse a human mark: 1/0/true/false/yes/no, case-insensitive; blank -> None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return {0: False, 1: True}.get(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value) in (0.0, 1.0):
            return bool(value)
        return None
    text = str(value).strip().lower()
    if text in _BLANK_VALUES:
        return None
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _cell(value: Any) -> Any:
    """CSV-safe value: missing -> empty, booleans -> 1/0."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    return value


def _to_level(value: Any, default: int = 10**6) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return int(number) if np.isfinite(number) else default


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------


def _preference_key(record: dict) -> tuple:
    """Deterministic row preference: parse ok first, then level 2, then lowest level."""
    level = _to_level(record.get("level"))
    return (
        str(record.get("image_id", "")),
        str(record.get("perturbation", "")),
        str(record.get("config", "")),
        0 if _truthy(record.get("parse_ok", True)) else 1,
        0 if level == 2 else 1,
        level,
    )


def select_cases(
    df: pd.DataFrame,
    cases: int = 30,
    configs: Sequence[str] = ("none", "text"),
    seed: int = SEED,
) -> list[dict]:
    """Pick a deterministic, perturbation-stratified set of image/config rows.

    `seed` is accepted for API compatibility; selection itself is deterministic.
    Returns one row dict per selected image and requested config.
    """
    configs = [str(config) for config in configs]
    cases = int(cases)
    if cases <= 0 or not configs or df.empty:
        return []

    frame = df.copy()
    frame = frame[frame["config"].astype(str).isin(configs)]
    if "perturbation" in frame:
        frame = frame[frame["perturbation"].astype(str) != "clean"]
    if frame.empty:
        return []

    if "parse_ok" in frame.columns:
        ok = frame[frame["parse_ok"].map(_truthy)]
        target = min(cases, int(frame["image_id"].nunique()))
        present = set(frame["perturbation"].astype(str))
        covered = set(ok["perturbation"].astype(str))
        if ok["image_id"].nunique() >= target and covered == present:
            frame = ok

    chosen: dict[tuple[str, str], dict[str, dict]] = {}
    for record in sorted(frame.to_dict("records"), key=_preference_key):
        image_id = str(record["image_id"])
        perturbation = str(record["perturbation"])
        config = str(record["config"])
        by_config = chosen.setdefault((image_id, perturbation), {})
        by_config.setdefault(config, record)

    entries: list[dict] = []
    for (image_id, perturbation), by_config in chosen.items():
        if not all(config in by_config for config in configs):
            continue
        first = by_config[configs[0]]
        entries.append(
            {
                "image_id": image_id,
                "perturbation": perturbation,
                "class_name": str(first.get("class_name", "")),
                "rows": by_config,
            }
        )

    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["perturbation"], []).append(entry)
    perturbations = sorted(groups)
    if not perturbations:
        return []
    for perturbation in perturbations:
        groups[perturbation].sort(
            key=lambda entry: (entry["class_name"], entry["image_id"])
        )

    base, extra = divmod(cases, len(perturbations))
    selected: list[dict] = []
    used_images: set[str] = set()
    for index, perturbation in enumerate(perturbations):
        quota = base + (1 if index < extra else 0)
        if quota <= 0:
            continue
        candidates = [
            entry
            for entry in groups[perturbation]
            if entry["image_id"] not in used_images
        ]
        if not candidates:
            continue
        stride = max(1, len(candidates) // quota)
        picked = candidates[::stride][:quota]
        used_images.update(entry["image_id"] for entry in picked)
        selected.extend(picked)

    rows: list[dict] = []
    for entry in selected:
        for config in configs:
            rows.append(entry["rows"][config])
    return rows


# ---------------------------------------------------------------------------
# Packet building
# ---------------------------------------------------------------------------


def _find_image(data_root: Path, wnid: str | None, image_id: str) -> Path:
    """Locate `val/{wnid}/{image_id}.JPEG`, falling back to any matching extension."""
    val = Path(data_root) / "val"
    if wnid:
        folder = val / str(wnid)
        direct = folder / f"{image_id}.JPEG"
        if direct.is_file():
            return direct
        if folder.is_dir():
            matches = sorted(folder.glob(f"{image_id}.*"))
            if matches:
                return matches[0]
    matches = sorted(val.glob(f"*/{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"image {image_id!r} not found under {val}"
        + (f" (wnid {wnid})" if wnid else "")
    )


def _save_thumbnail(image: np.ndarray, path: Path, max_width: int = THUMB_MAX_WIDTH) -> None:
    """Save a JPEG thumbnail, downscaling to at most `max_width` pixels wide."""
    thumb = Image.fromarray(to_uint8(image))
    if thumb.width > max_width:
        height = max(1, int(round(thumb.height * max_width / thumb.width)))
        thumb = thumb.resize((max_width, height), Image.LANCZOS)
    thumb.save(path, format="JPEG", quality=85)


def build_case(
    row: dict,
    record: ImageRecord,
    clean: np.ndarray,
    depth: np.ndarray,
    out: Path,
    seed: int = SEED,
) -> dict:
    """Reconstruct one case, write its three thumbnails, and return the review row."""
    perturbation_name = str(row["perturbation"])
    level = int(row["level"])
    config = str(row["config"])
    perturbation = ALL_PERTURBATIONS[perturbation_name]

    degraded = perturbation(
        clean,
        level,
        depth if perturbation.needs_depth else None,
        rng_for(record.image_id, perturbation_name, level, seed),
    )

    entity = region_evidence(near_mask(depth, 0.4), depth)
    mask = damaged_mask(perturbation_name, degraded, clean, depth)
    damaged = (
        region_evidence(mask, depth)
        if mask is not None and bool(mask.any())
        else None
    )
    overlay = draw_regions(degraded, entity, damaged)

    case_id = f"{record.image_id}_{config}"
    clean_name = f"{case_id}_clean.jpg"
    degraded_name = f"{case_id}_degraded.jpg"
    overlay_name = f"{case_id}_overlay.jpg"
    _save_thumbnail(clean, Path(out) / clean_name)
    _save_thumbnail(degraded, Path(out) / degraded_name)
    _save_thumbnail(overlay, Path(out) / overlay_name)

    return {
        "case_id": case_id,
        "image_id": record.image_id,
        "class_name": row.get("class_name", record.class_name),
        "perturbation": perturbation_name,
        "level": level,
        "config": config,
        "clean_image": clean_name,
        "degraded_image": degraded_name,
        "overlay_image": overlay_name,
        "vlm_entity": _cell(row.get("entity")),
        "vlm_damage": _cell(row.get("damaged_region_content")),
        "vlm_relation": _cell(row.get("relation_answer")),
        "vlm_confidence": _cell(row.get("confidence")),
        "auto_entity_hit": _cell(row.get("entity_hit")),
        "auto_damage_hit": _cell(row.get("damage_hit")),
        "auto_relation_hit": _cell(row.get("relation_hit")),
        "auto_relation_geometric": _cell(row.get("relation_geometric")),
        "manual_entity_correct": "",
        "manual_damage_correct": "",
        "manual_relation_correct": "",
        "notes": "",
    }


def build_review(
    rows: Iterable[dict],
    out: Path,
    data_root: Path = DATA_DIR / "imagenette2-320",
    depth_root: Path = CACHE_DIR / "depth-320",
    depth_for=None,
    seed: int = SEED,
) -> dict:
    """Build review.csv, contact_sheet.html, and the thumbnails for `rows`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    cache = None if depth_for is not None else DepthCache(root=Path(depth_root))
    loaded: dict[str, tuple[ImageRecord, np.ndarray]] = {}
    review_rows: list[dict] = []

    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in loaded:
            path = _find_image(Path(data_root), row.get("wnid"), image_id)
            record = ImageRecord(image_id, path, path.parent.name)
            loaded[image_id] = (record, load_image(path))
        record, clean = loaded[image_id]
        depth = (
            depth_for(record, clean)
            if depth_for is not None
            else cache.get(record, clean)
        )
        review_rows.append(build_case(row, record, clean, depth, out, seed=seed))

    frame = pd.DataFrame(review_rows, columns=REVIEW_FIELDS)
    frame.to_csv(out / "review.csv", index=False)
    (out / "contact_sheet.html").write_text(
        render_contact_sheet(review_rows), encoding="utf-8"
    )
    return {
        "out": out,
        "n_images": len({row["image_id"] for row in review_rows}),
        "n_rows": len(review_rows),
    }


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------


def _display(value: Any) -> str:
    """Human-readable, escaped cell text for the contact sheet."""
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "-"
    text = str(value).strip()
    return html.escape(text) if text else "-"


def _hit_text(value: Any) -> str:
    parsed = parse_manual(value)
    if parsed is None:
        return "-"
    return "1 (correct)" if parsed else "0 (incorrect)"


def _card(row: dict) -> str:
    case_id = _display(row.get("case_id"))
    images = "".join(
        f'<figure><img src="{html.escape(str(row[column]))}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
        for column, label in (
            ("clean_image", "clean"),
            ("degraded_image", "degraded"),
            ("overlay_image", "overlay"),
        )
    )
    return f"""<article class="card">
  <h2>{case_id}</h2>
  <p class="meta"><strong>image</strong> {_display(row.get("image_id"))} &middot;
     <strong>class</strong> {_display(row.get("class_name"))} &middot;
     <strong>condition</strong> {_display(row.get("perturbation"))}
     (level {_display(row.get("level"))}) &middot;
     <strong>config</strong> {_display(row.get("config"))}</p>
  <div class="thumbs">{images}</div>
  <dl>
    <dt>VLM entity</dt><dd>{_display(row.get("vlm_entity"))}</dd>
    <dt>VLM damaged region</dt><dd>{_display(row.get("vlm_damage"))}</dd>
    <dt>VLM relation</dt><dd>{_display(row.get("vlm_relation"))}</dd>
    <dt>VLM confidence</dt><dd>{_display(row.get("vlm_confidence"))}</dd>
    <dt>Auto entity hit</dt><dd>{_hit_text(row.get("auto_entity_hit"))}</dd>
    <dt>Auto damage hit</dt><dd>{_hit_text(row.get("auto_damage_hit"))}</dd>
    <dt>Auto relation hit</dt><dd>{_hit_text(row.get("auto_relation_hit"))}</dd>
    <dt>Geometric relation</dt><dd>{_display(row.get("auto_relation_geometric"))}</dd>
  </dl>
  <p class="manual">Fill <code>manual_entity_correct</code>,
     <code>manual_damage_correct</code> and <code>manual_relation_correct</code>
     in review.csv with 1 (the automatic label is right), 0 (it is wrong), or
     leave blank to skip the field.</p>
</article>"""


def render_contact_sheet(rows: Sequence[dict]) -> str:
    """Self-contained HTML page (relative image paths, all text escaped)."""
    cards = "\n".join(_card(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM review packet</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 24px; max-width: 1180px; }}
  h1 {{ font-size: 22px; }}
  .card {{ border: 1px solid #8884; border-radius: 10px; padding: 14px 18px;
           margin: 18px 0; }}
  .card h2 {{ font-size: 17px; margin: 0 0 6px; font-family: ui-monospace, monospace; }}
  .meta {{ margin: 0 0 10px; opacity: 0.85; }}
  .thumbs {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; flex: 1 1 300px; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
                opacity: 0.7; margin-top: 4px; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px;
        margin: 12px 0 8px; }}
  dt {{ font-weight: 600; }}
  dd {{ margin: 0; white-space: pre-wrap; }}
  .manual {{ font-size: 13px; opacity: 0.8; margin: 0; }}
  code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<h1>VLM review packet</h1>
<p>One card per row of <code>review.csv</code>. Compare the clean view, the
reconstructed degraded view, and the overlay (green = entity region, red =
damaged region) with the VLM's answer and the automatic score.</p>
{cards}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Scoring a filled packet
# ---------------------------------------------------------------------------


def _cohen_kappa(manual: np.ndarray, auto: np.ndarray) -> float | None:
    """Cohen's kappa for two boolean arrays; None when there are no pairs."""
    n = int(manual.size)
    if n == 0:
        return None
    observed = float(np.mean(manual == auto))
    p_manual = float(np.mean(manual))
    p_auto = float(np.mean(auto))
    expected = p_manual * p_auto + (1.0 - p_manual) * (1.0 - p_auto)
    if expected >= 1.0 - 1e-12:
        return 1.0 if observed >= 1.0 - 1e-12 else 0.0
    return float((observed - expected) / (1.0 - expected))


def _agreement_stats(pairs: Sequence[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "agreement": None,
            "kappa": None,
            "manual_correct": 0,
            "auto_correct": 0,
            "both_correct": 0,
            "manual_only": 0,
            "auto_only": 0,
            "neither": 0,
        }
    manual = np.array([pair[0] for pair in pairs], dtype=bool)
    auto = np.array([pair[1] for pair in pairs], dtype=bool)
    return {
        "n": n,
        "agreement": float(np.mean(manual == auto)),
        "kappa": _cohen_kappa(manual, auto),
        "manual_correct": int(manual.sum()),
        "auto_correct": int(auto.sum()),
        "both_correct": int(np.sum(manual & auto)),
        "manual_only": int(np.sum(manual & ~auto)),
        "auto_only": int(np.sum(~manual & auto)),
        "neither": int(np.sum(~manual & ~auto)),
    }


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def score_review(path: Path) -> dict:
    """Compare the filled manual columns with the automatic labels.

    Writes `agreement.json` next to the review CSV and prints a compact table.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    configs = _column(frame, "config")

    field_stats: dict[str, dict] = {}
    pooled: list[tuple[bool, bool]] = []
    per_config_pairs: dict[str, list[tuple[bool, bool]]] = {}

    for field, (manual_column, auto_column) in SCORE_FIELDS.items():
        pairs: list[tuple[bool, bool]] = []
        manual_values = _column(frame, manual_column)
        auto_values = _column(frame, auto_column)
        for manual_raw, auto_raw, config in zip(manual_values, auto_values, configs):
            manual = parse_manual(manual_raw)
            if manual is None:
                continue
            auto = parse_manual(auto_raw)
            if auto is None:
                continue
            pairs.append((manual, auto))
            pooled.append((manual, auto))
            per_config_pairs.setdefault(str(config), []).append((manual, auto))
        field_stats[field] = _agreement_stats(pairs)

    result = {
        "path": str(path),
        "n_rows": int(len(frame)),
        "fields": field_stats,
        "overall": _agreement_stats(pooled),
        "per_config": {
            config: _agreement_stats(pairs)
            for config, pairs in sorted(per_config_pairs.items())
        },
    }
    (path.parent / "agreement.json").write_text(json.dumps(result, indent=2))
    _print_agreement(result)
    return result


def _format_rate(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _print_agreement(result: dict) -> None:
    print(f"review: {result['path']}")
    header = f"{'field':<10} {'n':>4} {'agree':>7} {'kappa':>7} {'manual':>7} {'auto':>6}"
    print(header)
    print("-" * len(header))
    for field, stats in result["fields"].items():
        print(
            f"{field:<10} {int(stats['n']):>4} {_format_rate(stats['agreement']):>7} "
            f"{_format_rate(stats['kappa']):>7} {int(stats['manual_correct']):>7} "
            f"{int(stats['auto_correct']):>6}"
        )
    overall = result["overall"]
    print(
        f"{'overall':<10} {int(overall['n']):>4} {_format_rate(overall['agreement']):>7} "
        f"{_format_rate(overall['kappa']):>7} {int(overall['manual_correct']):>7} "
        f"{int(overall['auto_correct']):>6}"
    )
    for config, stats in result["per_config"].items():
        print(
            f"  {config:<8} {int(stats['n']):>4} {_format_rate(stats['agreement']):>7} "
            f"{_format_rate(stats['kappa']):>7}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS_DIR / "vlm" / "predictions.csv.gz",
    )
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--configs", nargs="*", default=["none", "text"])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vlm" / "review")
    parser.add_argument(
        "--data-root", type=Path, default=DATA_DIR / "imagenette2-320"
    )
    parser.add_argument("--depth-root", type=Path, default=CACHE_DIR / "depth-320")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--score", type=Path, default=None)

    args = parser.parse_args(argv)
    args.configs = [
        piece.strip()
        for value in args.configs
        for piece in str(value).split(",")
        if piece.strip()
    ]
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.score is not None:
        return score_review(args.score)

    if not args.predictions.is_file():
        raise SystemExit(
            f"{args.predictions} not found. Run scripts/run_vlm_eval.py first."
        )
    frame = pd.read_csv(args.predictions)
    rows = select_cases(frame, cases=args.cases, configs=args.configs, seed=args.seed)
    result = build_review(
        rows,
        out=args.out,
        data_root=args.data_root,
        depth_root=args.depth_root,
        seed=args.seed,
    )
    print(
        f"images={result['n_images']} rows={result['n_rows']} out={result['out']}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
