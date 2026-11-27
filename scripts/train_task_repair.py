"""Train the task-driven repair network (bolero.task_repair) through the car's frozen detector.

Data (real photos only; nothing simulated):
  weather   XWOD v1.1 train split, rain and dust (586 photos, box labels for the six classes)
  clean     clean COCO val2017 street photos, excluding the 300 used for every clean test
            (478 left: 95 for model selection, 383 for training)
Model selection: every --eval-every steps the repairer is scored on the XWOD *validation* split
(mAP50 change vs. no repair, rain and dust) and on 95 held-out clean COCO photos; the checkpoint
with the best weather gain whose clean mAP50 drops by no more than --clean-tolerance is kept.
XWOD test, DAWN (another dataset entirely) and the CARLA car frames stay untouched for testing.

Loss = frozen Mask R-CNN training loss (RPN + box head; the mask head is not needed)
     + --clean-weight * mean (repair(x) - x)^2 on clean photos
     + --change-weight * mean (repair(x) - x)^2 on weather photos
(Squared penalties: an absolute-value penalty pins the network to the identity, since its pull
does not weaken near zero change; the first trial run with it changed images by ~0.0002.)

Usage: python scripts/train_task_repair.py
Outputs: data/cache/restoration/task_repair.pt, results/task_repair/{train_log.csv, selection.json}
"""

import argparse
import json
import math
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from bolero.config import RESULTS_DIR, SEED
from bolero.degradation_compare import image_matches, map50
from bolero.detection_eval import Detections
from bolero.task_repair import TASK_REPAIR_PATH, RepairNet
from bolero.weather_data import CLASSES, load_coco, load_xwod, open_rgb

LOAD_SIDE = 1024     # photos are kept in memory at most this large (the detector resizes to 800-1333 anyway)


def load_images(samples):
    import cv2

    out = []
    for s in samples:
        x = np.asarray(open_rgb(s.path), np.uint8)
        h, w = x.shape[:2]
        k = min(1.0, LOAD_SIDE / max(h, w))
        if k < 1.0:
            x = cv2.resize(x, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        sx, sy = x.shape[1] / w, x.shape[0] / h
        boxes = s.boxes.astype(np.float32).reshape(-1, 4) * np.array([sx, sy, sx, sy], np.float32)
        out.append((x, boxes, list(s.labels)))
    return out


def frozen_detector(device):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05).to(device)
    model.roi_heads.mask_roi_pool = model.roi_heads.mask_head = model.roi_heads.mask_predictor = None
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.meta["categories"]


def set_loss_mode(model):
    """Training-mode forward (returns losses) with frozen batch-norm statistics."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def sample_batch(data, n, rng, categories, device, crop=(0.6, 1.0)):
    imgs, targets = [], []
    for i in rng.choice(len(data), n, replace=False):
        x, boxes, labels = data[i]
        h, w = x.shape[:2]
        s = rng.uniform(*crop)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        x = x[y0:y0 + ch, x0:x0 + cw]
        b = boxes - np.array([x0, y0, x0, y0], np.float32)
        b = np.clip(b, 0, [cw, ch, cw, ch])
        keep = ((b[:, 2] - b[:, 0]) >= 4) & ((b[:, 3] - b[:, 1]) >= 4)
        b, lab = b[keep], [l for l, k in zip(labels, keep) if k]
        if rng.random() < 0.5:
            x = x[:, ::-1]
            b = np.stack([cw - b[:, 2], b[:, 1], cw - b[:, 0], b[:, 3]], 1) if len(b) else b
        imgs.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255).to(device))
        targets.append({"boxes": torch.from_numpy(b.reshape(-1, 4).astype(np.float32)).to(device),
                        "labels": torch.tensor([categories.index(l) for l in lab], dtype=torch.int64, device=device)})
    return imgs, targets


@torch.no_grad()
def evaluate(net, model, categories, sets, device):
    """mAP50 per named set, with and without the repairer (no-repair numbers cached by the caller)."""
    model.eval()
    if net is not None:
        net.eval()
    res = {}
    for name, (data, base) in sets.items():
        dets = []
        for x, _, _ in data:
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255).to(device)[None]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                y = net(t) if net is not None else t
                o = model([y[0].float()])[0]
            names = [categories[i] for i in o["labels"].tolist()]
            keep = [i for i, n in enumerate(names) if n in CLASSES]
            dets.append(Detections(o["boxes"][keep].float().cpu().numpy().reshape(-1, 4),
                                   o["scores"][keep].float().cpu().numpy(), [names[i] for i in keep]))
        res[name] = map50(image_matches([d[1] for d in data], [d[2] for d in data], dets))
    if net is not None:
        net.train()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-weather", type=int, default=2)
    p.add_argument("--batch-clean", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--clean-weight", type=float, default=10.0)
    p.add_argument("--change-weight", type=float, default=1.0)
    p.add_argument("--clean-tolerance", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "task_repair")
    p.add_argument("--weights", type=Path, default=TASK_REPAIR_PATH)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load_xwod(split="train")
    valid = load_xwod(split="valid")
    eval_ids = {s.image_id for s in load_coco(n=300, seed=args.seed)}
    pool = [s for s in load_coco(n=10_000, seed=args.seed) if s.image_id not in eval_ids]
    order = rng.permutation(len(pool))
    n_sel = min(100, len(pool) // 5)   # COCO val has 778 eligible street photos; 300 are the clean test set
    clean_train, clean_valid = [pool[i] for i in order[n_sel:]], [pool[i] for i in order[:n_sel]]
    print(f"train: {len(train)} weather + {len(clean_train)} clean; select: {len(valid)} weather + {len(clean_valid)} clean",
          flush=True)
    weather_data, clean_data = load_images(train), load_images(clean_train)
    sel = {f"xwod_valid/{c}": load_images([s for s in valid if s.condition == c]) for c in ("rain", "dust")}
    sel["coco_clean_select"] = load_images(clean_valid)

    model, categories = frozen_detector(device)
    base = evaluate(None, model, categories, {k: (v, None) for k, v in sel.items()}, device)
    print("no repair:", {k: round(v, 4) for k, v in base.items()}, flush=True)

    net = RepairNet(args.width).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    log, best, start = [], None, time.time()
    run = {"det": 0.0, "clean": 0.0, "change": 0.0}
    for step in range(1, args.steps + 1):
        wi, wt = sample_batch(weather_data, args.batch_weather, rng, categories, device)
        ci, ct = sample_batch(clean_data, args.batch_clean, rng, categories, device)
        set_loss_mode(model)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            wr = [net(x[None])[0] for x in wi]
            cr = [net(x[None])[0] for x in ci]
            losses = model([x.float() for x in wr + cr], wt + ct)
            det = sum(losses.values())
        change = torch.stack([(a.float() - b).pow(2).mean() for a, b in zip(wr, wi)]).mean()
        clean = torch.stack([(a.float() - b).pow(2).mean() for a, b in zip(cr, ci)]).mean()
        loss = det + args.clean_weight * clean + args.change_weight * change
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        for k, v in (("det", det), ("clean", clean), ("change", change)):
            run[k] = 0.98 * run[k] + 0.02 * float(v.detach()) if step > 1 else float(v.detach())
        if step % 50 == 0:
            print(f"step {step} det {run['det']:.3f} clean_mse {run['clean']:.5f} weather_mse {run['change']:.5f} "
                  f"({time.time() - start:.0f}s)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(net, model, categories, {k: (v, None) for k, v in sel.items()}, device)
            gain = np.mean([r[k] - base[k] for k in r if k.startswith("xwod")])
            clean_drop = base["coco_clean_select"] - r["coco_clean_select"]
            ok = clean_drop <= args.clean_tolerance
            row = {"step": step, **{f"map50_{k}": v for k, v in r.items()},
                   **{f"change_{k}": r[k] - base[k] for k in r}, "weather_gain": gain, "clean_ok": ok,
                   "det_loss": run["det"], "seconds": time.time() - start}
            log.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if ok and (best is None or gain > best["weather_gain"]):
                best = row
                torch.save({"width": args.width, "state": net.state_dict(), "step": step,
                            "args": {k: str(v) for k, v in vars(args).items()}}, args.weights)
            pd.DataFrame(log).to_csv(args.out / "train_log.csv", index=False)
    (args.out / "selection.json").write_text(json.dumps(
        {"no_repair": base, "chosen": best, "weights": str(args.weights),
         "train": {"xwod_train": len(train), "coco_clean": len(clean_train)}}, indent=2, default=float))
    print("chosen:", best, flush=True)


if __name__ == "__main__":
    main()
