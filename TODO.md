# TODO: Post-presentation revamp (branch `restructure2`)

Direction from the professor: go deeper rather than broader, benchmark against
existing methods, narrow to specific failure conditions, add spatial models
(SAM/DETR), and use generative AI in a way that differs from a drop-in.
Follow-up suggestions: generative AI for both degradation and restoration, a
calibrated decision engine (Jev or an open-source one such as
[Laya](https://github.com/NandhaKishorM/laya)), and conformal prediction.

## Long-term goal: security perception in heavy rain and dust storms
A security / surveillance camera has to keep detecting people and vehicles when
heavy rain or a dust storm degrades the view, and it has to tell an operator when
it can no longer be trusted. The goal has two parts:
1. **Robustness:** detect objects (person, car, truck, bus, motorcycle, bicycle) in real heavy rain and sand/dust.
2. **Security:** know when the system is unreliable (conformal guarantees, abstention) and resist weather-shaped attacks (adversarial raindrops / rain streaks).

This moves the task from Imagenette **classification** to **object detection**
(mAP), with classification kept as the controlled lab setting.

### Verified real-data targets
Checked on the web on 2026-10-19. Confirm licenses and downloads before use.

| Dataset | Real? | Rain | Dust / sand | Labels | Use |
|---|---|---|---|---|---|
| [XWOD](https://arxiv.org/abs/2605.11521) ([Kaggle](https://www.kaggle.com/datasets/kuantinglai/exwod)) | Real only | 665 heavy-rain images | 560 haze/sand/dust images | Boxes, 6 traffic classes, fixed train/val/test split | **Main eval set.** Baseline YOLOv11m mAP50: rain 60.5%, haze/sand/dust 29.9% |
| [DAWN](https://arxiv.org/abs/2008.05402) ([IEEE DataPort](https://ieee-dataport.org/documents/dawn-vehicle-detection-adverse-weather-nature)) | Real (web) | Yes | Sandstorm subset | Boxes, 6 classes, surveillance + driving | Second eval set, ~1,000 images, CC BY 4.0 |
| [RID / RIS](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Single_Image_Deraining_A_Comprehensive_Benchmark_Analysis_CVPR_2019_paper.html) (Li et al., CVPR 2019) | Real | Driving (RID) + **traffic surveillance cams (RIS)** | No | Boxes | Rain detection after deraining. Paper found deraining usually *hurts* detection mAP: this is the result to beat |
| [SIRB](https://arxiv.org/abs/2202.03031) / [SIEB](https://www.sciencedirect.com/science/article/abs/pii/S0030399225003366) | Real | No | 904 sand-dust images (SIEB) | No boxes (restoration) | Train/tune a desanding restorer |
| [SDSI](https://doi.org/10.3390/rs15215227) | Real, surveillance cams | No | 6,578 storm / 6,638 clear | Image-level label | Train/eval the "is there a dust storm?" detector |
| [VIRADA](https://arxiv.org/abs/1908.04034) | Real, surveillance video | 215 h | No | Rain / no-rain | Train/eval the "is it raining?" detector |

### Pipeline roles
Restoration does not need a VLM or Jev. Each stage is its own open-source model and runs locally:

| Role | Question | Options |
|---|---|---|
| **Decide** | Rain, dust or clean? How severe? Restore or pass through? | RF detector (`detection.py`), Jev-style deciders (below) |
| **Restore** | Image in, cleaner image out | Classical plans (`restoration.py`), Restormer, NAFNet, MPRNet, DRSformer (rain); dark channel prior + color correction or a SIRB/SIEB-trained model (dust); DiffBIR, StableSR (diffusion); LaMa (inpainting) |
| **Detect** | Where are the people and vehicles? | DETR (`segmentation.py`), YOLO |

### Open-source Jev alternatives (decide stage only, text in / typed answers out)
All are a few weeks old; benchmark numbers are self-reported. Jev itself is hosted by TypeSafe with unclear access, so it is an optional baseline only.

| Project | Notes |
|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | ModernBERT, Apache-2.0, ~33 ms on a T4. Ships over-confident; recalibrate |
| [open-alternative-jev](https://github.com/femisapien/open-alternative-jev) | Wraps any open-weights LLM (HF / vLLM). README reports ~580 ms/case with a 27B model, so pick a small one |
| poorjev | Local, temperature scaling + conformal abstention built in |
| decider / OpenDecider, kev | Qwen-based reproductions matching Jev's API |
| [von](https://github.com/wfzyx/von) | Archived 2026-10-17; skip |

More: [systemonemodels.org list](https://systemonemodels.org/examples/alternatives/), [VoltAgent ecosystem](https://github.com/VoltAgent/typesafe-jev-ecosystem).

## Venkat: Restructure, detection harness, conformal + security gate
- [x] Make the classifier backbone swappable (`--backbone resnet50|vit_b16|convnext_t`) with top-k candidates
- [x] Add SAM / DETR spatial evidence (`segmentation.py`) usable by every existing cue
- [x] Add `scripts/run_structure_compare.py` (depth vs SAM vs DETR outline, per-class breakdown)
- [x] Add ImageNet-C style CE / mCE (`benchmark.py`, `scripts/benchmark.py`)
- Structure compare (SAM fixes the church case?) moved to Sachin's list (2026-11-11): SAM runs on CPU at ~9 s/image, so the full run is hours long
- [x] Backbone mCE vs ResNet-50, full val (3,925 images; `results/diagnostic_{vit_b16,convnext_t}/`, `results/benchmark_*/`, 2026-11-11):
      | Backbone | Clean acc | Perturbed acc | mCE | Relative mCE |
      |---|---|---|---|---|
      | ResNet-50 | 0.794 | 0.648 | 1.00 | 1.00 |
      | ViT-B/16 | 0.831 | 0.669 | 1.03 | 3.01 |
      | ConvNeXt-T | 0.835 | 0.719 | **0.79** | **0.72** |
      - ConvNeXt-T beats ResNet-50 on 13/14 perturbations (grayscale even); biggest gains on low_contrast / brighten / darken
      - ViT-B/16 is much better on texture damage (noise CE 0.58, blur 0.69, jpeg 0.71) but much worse on color / lighting (inversion CE 2.74, error 0.24 → 0.65; low_contrast 1.27, brighten 1.21, darken 1.18), so overall mCE is slightly worse despite higher clean accuracy
- [x] Wire a structure source into `run_restoration.py`: `--structure-source depth|sam|detr` (default depth; sam/detr need `--depth-sources perturbed`). Picking the default waits on Sachin's structure-compare run
- [x] Detection eval harness (`src/bolero/detection_eval.py`: frozen DETR + COCO-style mAP, 5 tests; `scripts/run_detection.py`). DAWN results, frozen COCO DETR (`results/detection/`):
      | Condition | Method | mAP50 | mAP50-95 | Person recall |
      |---|---|---|---|---|
      | Rain (200 img, 1,588 boxes) | none | 0.481 | 0.314 | 0.89 |
      | | autocontrast / clahe / unsharp | **0.490** / 0.480 / 0.472 | **0.317** / 0.312 / 0.304 | 0.93 / 0.93 / 0.93 |
      | Dust (323 img, 2,235 boxes) | none | 0.457 | 0.287 | 0.83 |
      | | autocontrast / clahe / unsharp | **0.470** / 0.464 / 0.455 | **0.294** / 0.289 / 0.285 | 0.85 / 0.82 / 0.82 |
      - Corrected 2026-11-19 (Aravind's branch): the first version of this table had an EXIF orientation bug. 36 of the 323 DAWN sand images (and 4 rain) are mirrored by their EXIF tag and their VOC boxes match the mirrored view; the `none` method saw it, the baselines did not. The old dust row (0.381 / 0.376 / 0.375) showed preprocessing costing 8 pp; with the fix, auto-contrast and CLAHE help slightly on both conditions and unsharp is neutral. `none` is unchanged (0.481 / 0.457), which confirms the fix touches only the baselines. Output now in `results/detection_dawn/`
      - ~~No preprocessing helps; under dust every baseline costs ~8 pp mAP50 and 6–8 pp person recall~~ (EXIF artifact, see above). Corrected: auto-contrast is the best method on both conditions (+0.9 pp rain, +1.3 pp dust mAP50), the others are within ±1 pp of none, so the Li et al. (CVPR 2019) "preprocessing hurts" result is not reproduced here. Trucks / buses are still the weak classes (AP50 0.26–0.40), bicycles worst under dust (0.20)
- [ ] YOLO baseline for the detection harness (`ultralytics` not installed; add it to requirements and a `YOLODetector` next to `FrozenDetector`)
- [ ] Person miss rate at a fixed alert budget on DAWN (abstain-and-alert for detection, now that the harness exists)
- [x] Add `src/bolero/conformal.py`: split-conformal threshold for the perturbation gate (target clean false-alarm rate, e.g. ≤5%) replacing the fixed 0.6 in `detection.py`. `ConformalGate` is a drop-in for the detector; use it in Stage 3 with `run_restoration.py --conformal-gate results/conformal/headline.json`
- [x] Add APS / RAPS prediction sets for the classifier (`ConformalClassifier`, `FrozenClassifier.predict_proba`)
- [x] Abstain-and-alert logic: `alerts()` (set too large or gate fired), `selective_report()` (alert rate, silent miss rate), `max_size_for_budget()`. Person miss rate waits on the detection harness
- [x] `scripts/run_conformal.py` + `tests/test_conformal.py` (7 tests, full suite passes)
- [x] Run it (~13 min on MPS): `python scripts/run_conformal.py --per-class 20 --cal-per-class 50` → `results/conformal/` (run 2026-11-10, 500 cal / 200 val images)
      - Gate: clean false alarms 5.5% for both conformal and fixed 0.6; conformal flags 80.2% of perturbed images vs 71.4%. Big routing gains on boundary_erase, foreground_occlusion, background_removal, low_contrast L1; worse on jpeg L1, grayscale L1–2, gaussian_noise L1, inversion L3. Neither gate catches foreground_blur / boundary_blur
      - RAPS: clean coverage 0.97, set size 4.5 → perturbed 0.92, 9.8. Alerts catch 88% of perturbed errors (4% silent miss), but clean alert rate is 9.5%
      - Restoration (`detector` / `oracle`) does not recover coverage: accuracy 0.68 → 0.65, coverage 0.92 → 0.90
- [x] **Fix APS** (old run kept in `results/conformal_v1_1000class_deterministic/`): qhat was ≈ 1.0 with sets of 806/1000 classes. Two causes: sets ranged over 1000 ImageNet classes, and deterministic APS scores a confident *correct* prediction at ~1.0, so qhat hit 1.0 and float ties let every class in. Fix: `restrict_to_classes()` (sets over the 10 Imagenette classes, `--label-space`) + randomized APS/RAPS (`--no-randomize` restores old behavior); 2 new tests, 124 pass. Rerun (same 200 val images):
      - APS qhat 0.886: clean coverage 0.995, set size 1.03 → perturbed 0.987, 1.26. RAPS now near-identical (its penalty had masked the APS bug)
      - Alerts: clean alert rate 6.0% (budget 5%, within noise; RAPS 9.5% issue gone), catch 85% of perturbed errors, silent miss 4.9%
      - 10-class coverage stays high under perturbation while 1000-way top-1 drops 0.825 → 0.68; the alert rule (set > 1 or gate fired) carries the safety
      - Restoration still lowers coverage (0.987 → 0.977) and accuracy (0.68 → 0.65)
- [x] Stage 3, conformal gate vs fixed 0.6, `detector_classical`, same 100 val images, no cues (`results/restoration_conformal/`, `results/restoration_fixed_small/`):
      - Mean Δacc over perturbed conditions: conformal −2.9 pp vs fixed −2.6 pp (fix rate 6.4% vs 5.2%, break rate 8.0% vs 6.9%). Clean: 0.80 → 0.80 vs 0.81
      - The gate routes more images, but routing more just applies the classical plans more often, and those plans hurt on average: gaussian_noise L3 −32 pp, background_removal L3 −20, foreground_occlusion L2 −11 (conformal) vs −4 (fixed). Gains only on gaussian_blur (+6–8) and brighten/jpeg L3 (+4)
      - **Takeaway:** the gate is better at detection; the bottleneck is the restoration plans. Fix or drop the noise / background_removal / occlusion plans (or pass through for them) before gating matters
- [x] Restoration plan audit (`scripts/audit_plans.py`, decided on 100 **train** images disjoint from detector training → `results/restoration_audit/skip_plans.json`):
      - Pass through: background_removal (−14.7 pp), gaussian_noise (−17.3), foreground_occlusion (−9.0), region_removal (−6.3), boundary_erase (−2.7), foreground_blur (−1.3), low_contrast (−1.3)
      - Keep: gaussian_blur (+6.0), jpeg_compression (+2.3), boundary_blur (+1.0), brighten (+1.0), darken (+0.3), inversion (0.0)
      - `run_restoration.py --skip-plans results/restoration_audit/skip_plans.json` (detector path only; `oracle_classical` keeps every plan)
      - Stage 3, same 100 val images: fixed −2.56 → **+0.69 pp** (break rate 6.9% → 0.4%); conformal −2.93 → **+0.60 pp** (8.0% → 0.7%). Restoration now beats doing nothing; with the skip list the two gates are equal within noise, so the conformal gate's value is its false-alarm guarantee
      - Overlaps with the unmerged `origin/Trial_2` branch (`scripts/tune_plans.py`), which tries several candidate plans per perturbation and reached the same conclusion. Decide with the team which to keep; Trial_2 is the more complete version
- [ ] Note: the system Python 3.10 (`/Library/Frameworks/Python.framework/Versions/3.10/bin/python3`) has the deps; Anaconda's `python` lacks `transformers` / `cv2`

## Rohith: Benchmarking lead + learned / generative restoration
- [ ] Pick the reference benchmark protocol (ImageNet-C / Imagenette-C severities) and document how ours maps to it
- [ ] Add 2–3 published baselines as methods: a learned restorer (Restormer or NAFNet, plus DRSformer for rain) and test-time adaptation (TENT or MEMO)
- [ ] Add a dust restorer: dark channel prior + color correction, then a model tuned on SIRB/SIEB
- [ ] Add diffusion restorers (DiffBIR or StableSR) for rain/dust and SAM-masked inpainting (LaMa / SD-inpaint) for occlusion
- [ ] Measure hallucination: does restoration invent or change objects? Use agreement with the clean prediction, box changes, and conformal set size (with Venkat)
- [ ] Reproduce the RID/RIS finding (deraining preprocessing vs raw rainy input) as a sanity check
- [ ] Produce one results table: method × perturbation → accuracy, CE, latency
- [ ] Extend the table to detection: method × {XWOD rain, XWOD dust, DAWN rain, DAWN sandstorm, RIS} → mAP50, coverage, latency
- [ ] Write a one-page "how we benchmark" note for the team

## Aravind: Rain + dust data (physics, generative, real)

- [x] Driving study, Phases 1-2 of 6 (2026-11-25), see [DRIVING.md](DRIVING.md): the algorithm on a CARLA car's camera. 15 drives in 5 towns (1,800 frames, 701 in-path objects); physics weather cuts Mask R-CNN mAP50 by 0.20-0.44. A condition detector that also saw clean car frames (`combined_car`) routes 85 % of video frames correctly and sends only 9 % of clean frames to a repair. The full pipeline (route, repair, detect, isolate) runs live at 16 frames/s. Isolation flags in-path objects with about 90 % precision and about 10 % distance error. Repair: dehazing recovers +0.07 to +0.14 in-path recall on simulated dust, but **not on real dust photos** (neutral to -0.05 mAP50); nothing helps much on rain. The next step is a repair that works on real photos
- [x] Driving study, the repair redone and Phases 3, 4 and 6 (2026-11-26), see [DRIVING.md](DRIVING.md):
  - **The repair, redone.** A less invertible simulated weather (v2: uneven dust plumes, sky-dependent haze, camera noise and JPEG) and a repair network trained through the frozen detector on real XWOD photos (`task_repair`). Repairs are now chosen on real photos (the XWOD validation split). No repair reliably helps on the real test photos.
  - **Phase 3.** A simple driving agent (lane following from the map, braking only on perception) and 4 scenarios (pedestrian crossing, stopped car, braking lead car, car in the next lane) at 8 locations in the held-out towns.
  - **Phase 4.** 736 runs. Ground truth never crashes. With no repair the car crashes in 29 % of weather runs (only heavy rain mist and dust storms); our algorithm brings that to 27 % (-2.5 pp, CI [-5.4, +0.4]). It helps in moderate rain mist (58 % → 29 %) and causes 3 crashes in light rain streaks. No false braking, no harm on clean runs.
  - **Phase 6.** Statistics, and a side-by-side demo video.
- [ ] ~~Driving study, Phase 5: real dashcam check (ACDC)~~ skipped by decision (needs a dataset registration; XWOD/DAWN evidence points the same way)
- [ ] Driving study, next: a repair specific to the kind of weather (dehazing helps mist but hurts streaks), e.g. a router that separates mist from streaks
- [x] Extra, outside the agreed list (2026-11-25): CARLA 0.9.16 as a fourth rain / dust source, see [CARLA.md](CARLA.md). 287 paired traffic-camera scenes (clean + rain / dust levels 1-3, exact boxes and depth). CARLA weather barely degrades the detector (rain level 3: -0.07 mAP50), its "dust" is grey fog, and a detector trained on it recognizes almost no real rain (0-6%) or dust (19-30%). Its viewpoint, not its weather, is what matches real data. True vs estimated depth changes the physics filters by at most 2.3 pp mAP50
- [x] DAWN downloaded (done by Venkat, 2026-11-11): Rain (200 images) + Sand (323) from Mendeley 10.17632/766ygrbt8y.3 into `data/dawn/` (git-ignored). License per Mendeley metadata: **CC BY-NC 3.0** (attribute, non-commercial; fine for coursework, cite the DAWN paper)
- [x] XWOD downloaded (2026-11-19): v1.1 from Kaggle `kuantinglai/exwod` into `data/xwod/XWOD_dataset.v1.1/XWOD_dataset.v1.1/` (git-ignored, 1.4 GB). License **CC BY-NC-SA 4.0**, non-commercial academic research only; no redistribution of the full dataset; cite the XWOD paper. v1.1 anonymizes faces, plates and other PII; boxes and splits are unchanged from v1.0
      - Real layout differs from the assumed one: `data.v1.1.yaml`, `<split>.v1.1/{images,labels}/`, splits `train` / `valid` / `test`, 7 conditions in the filename (heavy_rain, dust, fog, snow, flooding, tornado, wildfire), class `bike` for bicycle. Loader fixed and tested for all of it
      - Two loader bugs fixed: `"rain" in name` matched the "rain" in "train", tagging every training image (all 5,945) as rain; and `bike` was silently dropped. Boxes are now clipped to the image (91 ran past the edge)
      - Loaded counts match the paper: rain 301 / 106 / 258 and dust 285 / 130 / 145 images (train / valid / test), 665 rain and 560 dust in total. Fog (306) is dropped, as it is not in the weather taxonomy
      - Frozen COCO DETR on the XWOD test split (`python scripts/run_detection.py --datasets xwod --out results/detection_xwod`):
        | Condition | Method | mAP50 | mAP50-95 | Person recall |
        |---|---|---|---|---|
        | Rain (258 img, 851 boxes) | none | 0.252 | 0.141 | 0.58 |
        | | autocontrast / clahe / unsharp | 0.259 / 0.250 / 0.244 | 0.145 / 0.141 / 0.138 | 0.50 / 0.58 / 0.50 |
        | Dust (145 img, 963 boxes) | none | 0.446 | 0.273 | 0.85 |
        | | autocontrast / clahe / unsharp | 0.508 / 0.443 / 0.473 | 0.286 / 0.273 / 0.274 | 0.86 / 0.86 / 0.85 |
        - XWOD rain is much harder for DETR than DAWN rain (mAP50 0.25 vs 0.48). Rain has only 12 test persons, so person recall moves in steps of 1/12
        - Auto-contrast is the only baseline that helps on both conditions (dust mAP50 +6.2 pp, rain +0.7 pp). No confidence intervals here; see the degradation comparison below
      - EXIF orientation bug fixed in `run_detection.py` (new `weather_data.open_rgb`): 35 XWOD images carry an EXIF orientation tag (33 train, 2 dust test) and their boxes match the oriented view. transformers' pipeline applies the tag to a PIL image from disk but not to an array, so the `none` method saw the oriented image and every baseline saw the raw one. On XWOD dust this understated each baseline by about 2 pp mAP50 (autocontrast 0.489 → 0.508) and made baselines look like they lower recall (0.67 → 0.72 after the fix)
      - The same bug affected the DAWN table in Venkat's section (36 mirrored sand images); it is corrected there
- [x] Loaders (`src/bolero/weather_data.py`, tested): `load_dawn()` (VOC XML; Sand → `dust`), `load_xwod()` / `load_yolo_dataset()` (XWOD v1.1 layout, see above; plain Ultralytics layout also accepted), `load_condition_folders()` for SDSI / VIRADA (one folder per label). SDSI / VIRADA themselves are not downloaded yet
- [x] Physics-based weather in `perturbations.py` (`WEATHER`, opt-in; not in `ALL_PERTURBATIONS`, so earlier results and the trained detector are unchanged): `rain_streaks`, `rain_mist` (depth haze I·t + A·(1−t), t = e^(−β·d), plus streaks), `rain_lens` (refracting, defocused drops), `dust_haze` (yellow-brown airlight, β per channel R 0.8 / G 1.0 / B 1.35 so blue fades first, plus contrast loss), `dust_storm` (dense haze + wind-blown sand). Names map onto the `weather` taxonomy; usable via `--perturbations` in `run_diagnostic.py` / `run_decider_compare.py` and `conditions.iter_samples`. 4 tests
- [x] Classifier drop on the weather set (2026-11-19): `python scripts/run_diagnostic.py --perturbations rain_streaks rain_mist rain_lens dust_haze dust_storm --per-class 20 --out results/diagnostic_weather`. Frozen ResNet-50, 200 val images (20 per class), clean accuracy 82.5%. Accuracy change vs clean in pp, 95% bootstrap CI over images:
      | Effect | Level 1 | Level 2 | Level 3 | All levels |
      |---|---|---|---|---|
      | rain_streaks | -10.5 | -13.0 | -21.5 | -15.0 [-19.3, -11.2] |
      | rain_mist | -14.0 | -35.5 | -54.5 | -34.7 [-39.8, -30.0] |
      | rain_lens | -7.5 | -25.0 | -52.5 | -28.3 [-32.8, -24.2] |
      | dust_haze | -2.5 | -5.5 | -9.5 | -5.8 [-10.2, -1.7] |
      | dust_storm | -18.5 | -48.0 | -76.5 | -47.7 [-52.8, -42.3] |
      - Weather is about twice as damaging as the original 14 perturbations: -26.3 pp [-30.0, -22.6] overall (rain -26.0, dust -26.8) vs -14.6 pp for the original set on the full val split. Only the dust_haze level 1 interval includes zero
      - Severities are uneven across effects: dust_storm level 3 leaves 6% accuracy (below the 10% chance level), while dust_haze level 3 costs only 9.5 pp. If levels should mean the same thing across effects, recalibrate them (e.g. against XWOD mAP) before the physics/generative/real comparison
- [x] Generative degradation (2026-11-19): `python scripts/make_generative_weather.py --source coco --n 300` and `--source imagenette --per-class 20`, then `python scripts/run_generative_diagnostic.py` → `data/generative/` (git-ignored), `results/generative_imagenette/`
      - InstructPix2Pix (`timbrooks/instruct-pix2pix`, MIT, fp16, 20 steps), prompts "make it raining heavily" / "make it a sandstorm". Severity = edit strength: image guidance 2.0 / 1.75 / 1.5 for levels 1 / 2 / 3 (lower = stronger), text guidance 7.5. Chosen on 3 pilots: below 1.5 the model replaces the scene (new skyline, new people, bus gone) instead of adding weather. Edits are made at a 512 short side and resized back, so boxes stay valid. ~3 s per edit on the RTX 5080 (1,800 COCO edits in 108 min, 1,200 Imagenette in 60 min). Code in `src/bolero/generative.py`
      - "Label survives" is judged by CLIP (ViT-B/32) similarity to the clean image, never by the classifier or detector under test (filtering on "the detector still finds the boxes" would drop the hard cases and inflate scores). Cutoff = 5th percentile of the physics effects' similarity on the same images (0.604 COCO, 0.593 Imagenette). It removes invented scenes but cannot check single objects: some kept edits lost a car or a dog. Results are reported for all edits and for kept edits
      - What it produces: dust is a convincing orange haze that gets heavy fast; rain is weak, mostly wet roads and a cooler tint, with few visible streaks. Kept rate COCO dust 94 / 88 / 72%, rain 100 / 99 / 94%; Imagenette dust 78 / 58 / 47%, rain 99 / 95 / 80%
      - Classifier (frozen ResNet-50, same 200 Imagenette images as the physics run above, clean 82.5%), accuracy change in pp, 95% bootstrap CI:
        | | Level 1 | Level 2 | Level 3 | All levels |
        |---|---|---|---|---|
        | generative rain, all edits | -9.5 | -15.0 | -42.0 | -22.2 [-27.0, -17.0] |
        | generative rain, kept | -8.1 | -11.1 | -29.6 | -15.7 [-20.3, -11.0] |
        | generative dust, all edits | -36.0 | -54.5 | -65.5 | -52.0 [-58.2, -46.3] |
        | generative dust, kept | -21.8 | -37.4 | -50.5 | -34.0 [-40.9, -27.2] |
        | physics rain / dust (above) | | | | -26.0 / -26.8 |
      - Generative dust is about twice as damaging as physics dust, and most of the gap is content loss: even kept edits cost 22 pp at level 1. Imagenette images are 160-pixel close-ups upscaled to 512, where a sandstorm easily swallows the object. On COCO street scenes (detection, below) generative dust is milder than physics dust_storm
- [x] Compare physics-based, generative and real degradation on the same detector (2026-11-19): `python scripts/run_degradation_compare.py` → `results/degradation_compare/` (`summary.csv`, `predict.csv`, `kid.csv`). Frozen COCO DETR, every set with none + the 3 baselines; mAP50 CIs are 95% image bootstrap (500)
      - Sets: 300 clean COCO val2017 traffic scenes (every image has a vehicle, no crowd boxes); the same 300 through the 5 physics effects × 3 levels and through InstructPix2Pix rain / dust × 3 levels; real XWOD test and DAWN rain / dust. Clean COCO mAP50 0.708 [0.672, 0.749]
      - mAP50 drop vs clean COCO (level 1 / 2 / 3): rain_streaks -0.10 / -0.15 / -0.21; rain_mist -0.31 / -0.52 / -0.64; rain_lens -0.04 / -0.08 / -0.17; dust_haze -0.09 / -0.18 / -0.31; dust_storm -0.42 / -0.63 / -0.66; generative rain -0.07 / -0.08 / -0.17; generative dust -0.15 / -0.23 / -0.35. Real: XWOD rain 0.252, XWOD dust 0.446, DAWN rain 0.481, DAWN dust 0.457
      - Reference: the other real dataset of the same condition, scored the same way. A synthetic set should not be expected to beat it
        | Predicts → | XWOD rain | DAWN rain | XWOD dust | DAWN dust |
        |---|---|---|---|---|
        | Class profile (Spearman of per-class AP50, mean over levels): other real set | **0.80** | **0.80** | **0.89** | **0.89** |
        | physics (best effect) | -0.27 (streaks) | 0.47 (mist, streaks) | 0.31 (storm) | -0.03 (storm) |
        | generative | -0.40 | -0.30 | 0.05 | -0.33 |
        | Same best method as real (share of levels): other real set | 1 | 1 | 1 | 1 |
        | physics haze-type (rain_mist / dust_haze, dust_storm) | 1 | 1 | 1 | 1 |
        | generative | 0.33 | 0.33 | **0** | **0** |
        | Looks like real (CLIP KID ×1000, lower = closer; clean COCO 2.14 / 1.41 / 1.56 / 1.75): other real set | 1.01 | 1.01 | 0.24 | 0.24 |
        | best physics level | 1.35 (mist) | 0.93 (mist) | 1.47 (haze) | 1.53 (haze) |
        | best generative level | 1.62 | 0.99 | 1.22 | 1.14 |
      - **No synthetic degradation predicts real performance nearly as well as a second real dataset does.** The two real sets agree on which classes suffer (0.80-0.89); no synthetic source gets above 0.47, most are near zero or negative
      - Physics haze-type effects (rain_mist, dust_haze / dust_storm) are the most useful proxy: they always pick the method that is best on real data (auto-contrast), and rain_mist / rain_streaks give the best class profile for DAWN rain. Lens drops and generative edits are the weakest
      - Generative dust **looks** the most like real dust (lowest KID of any synthetic set) but **behaves** the least like it: it never picks auto-contrast and ranks classes against the real order. Looking realistic is not the same as degrading a detector realistically
      - Matching difficulty is not informative: most sources come within 0.06 of a real mAP50 at their best-matching level (dust_storm 0.16-0.17; on XWOD rain only rain_mist gets within 0.07), because the level can be chosen after the fact. XWOD rain (0.25) and DAWN rain (0.48) themselves differ by 0.23
      - On real data only dust benefits from preprocessing: auto-contrast +6.2 pp [2.3, 14.9] on XWOD dust, +1.3 pp [0.3, 2.3] on DAWN dust; on rain no baseline's interval excludes zero
      - Generative edits that change the scene (CLIP similarity to the clean image below 0.604, the 5th percentile of the physics effects) are reported separately as `kept`. The filter removes invented scenes (similarity 0.38-0.45) but is coarse near the cutoff: some kept edits lost a car or a dog, and some rejected ones are fine. With a stricter cutoff (0.75, `results/degradation_compare_clip075/`) generative dust level 3 loses only 0.17 mAP50 instead of 0.35, so much of its "weather damage" is content destruction. Conclusions above hold at both cutoffs
      - Caveats: the class-profile score mostly reflects the camera viewpoint, not the weather: clean CARLA traffic-camera frames already match the real sets at 0.54-0.90 (see CARLA.md), while clean COCO close-ups do not; COCO scenes are the detector's training domain, so synthetic sets start easier than real ones; class profiles use only 5-6 classes; auto-contrast is best on every real set, so "same best method" mostly asks whether a source agrees that auto-contrast helps
- [x] Retrain the RF condition detector on real rain and dust (2026-11-19, VIRADA done; **SDSI still open**): `python scripts/prepare_virada.py && python scripts/train_weather_detector.py` → `results/weather_detector/`
      - VIRADA (Zenodo 4715681, CC BY 4.0) is 167 GB of video, so `prepare_virada.py` reads the Crossing2 zips over HTTP range requests and fetches only the chosen clips (8.6 MB of frames kept). A clip is used only if every minute has the same label from both rain sensors: all 46 such rain clips (3 days) plus 92 dry clips, half from the rain days; 3 frames per clip. VIRADA rain is wet, reflective asphalt, not visible streaks
      - SDSI has no public download (paper: Remote Sensing 15(21):5227, 2023); the MDPI page blocks automated reading. Needs a request to the authors. The script takes `--sdsi data/sdsi` (folders storm/ clear/) once it arrives
      - Same features, same random forest (threshold 0.6), same 510 clean images (300 COCO traffic scenes not in the comparison sample + VIRADA dry frames from training days); only the source of rain / dust examples differs. Synthetic = the 5 physics effects at levels 1-3 on the clean pool. Real = XWOD train + valid (479 rain incl. VIRADA training days, 415 dust). VIRADA is split by day (test: the last rain day + alternate dry-only days)
      - Share of held-out images routed correctly (decided label = truth; for clean sets, the share not falsely alarmed), 95% Wilson CIs in `summary.csv`:
        | Test set | Synthetic-trained | Real-trained | Combined |
        |---|---|---|---|
        | DAWN rain (dataset never trained on) | 0.64 | **0.90** | 0.95 |
        | DAWN dust (dataset never trained on) | 0.44 | **0.96** | 0.92 |
        | XWOD rain (test split) | 0.57 | 0.30 | 0.68 |
        | XWOD dust (test split) | 0.15 | 0.40 | 0.36 |
        | VIRADA rain (held-out day) | **0.00** | 0.20 | 0.14 |
        | VIRADA clean (held-out days) | 0.88 | 0.76 | 0.70 |
        | COCO clean | 0.63 | **0.96** | 0.56 |
        | Imagenette clean (source never trained on) | 0.53 | **0.96** | 0.51 |
      - The synthetic-trained detector says "rain" to almost anything: 60% of clean COCO, 73% of real XWOD dust, 78% of clean Imagenette (top-1). It never recognizes VIRADA's wet-road rain. Adding synthetic data to real data (combined) brings those false alarms back (only 51-56% of clean COCO and Imagenette images pass)
      - The real-trained detector generalizes to DAWN, a dataset it never saw, and keeps 96% of clean images from an unseen source clean, so it is recognizing weather rather than "not one of my clean datasets". Its weak spots: XWOD dust is often called rain (45%), XWOD rain is mostly found (top-1 0.78) but below the 0.6 confidence bar, and VIRADA rain is hard for every detector
      - Caveats: VIRADA rain comes from one crossing (2 crops of one camera) on 3 days; real dust training data is XWOD only until SDSI arrives; the synthetic training set is heavily imbalanced (4,590 rain / 3,060 dust vs 510 clean; the forest's class weighting only partly offsets it)
      - Recommendation: route on the real-trained detector, not one trained on physics perturbations. Physics weather is useful to stress the classifier / detector, not to teach a detector what real weather looks like

## Sachin: Decision engine (Jev alternatives) + VLM
- [x] Read the in-context learning paper from the professor (internal, do not share). The instruction taxonomy is mapped in the README
- [x] Serialize `SceneEvidence` (`prompting.py`) for the deciders: `noul` "is the view degraded?", `choice` condition, `score` severity 1–3 (`conditions.py`, `build_jev_questions`)
- [x] Compare deciders on the OpenRouter Decisions API: RF vs Jev 1.14 vs Clef Flash vs Perplexity Decider Flash vs Liquid d1 vs Joey 1, plus a local Qwen prompt-and-parse adapter. Laya and poorjev were dropped as install-risk alternatives and documented. Metrics: routing accuracy, clean false alarms, severity MAE, latency, cost (`scripts/run_decider_compare.py`)
- [x] Probe image resolution limits: Jev needs >=280 px; Imagenette2-320 is the common input (`scripts/probe_resolutions.py`)
- [x] Recalibrate each decider with Venkat's conformal gate (done by Venkat, 2026-11-11): `python scripts/calibrate_deciders.py` → `results/deciders/conformal/` (`thresholds_alpha{0.05,0.1,0.2}.json` work with `run_decider_compare.py --conformal`). Calibrated on the 20 clean images; recall on the 540 degraded ones; leave-one-out clean false alarm = alpha
      | Decider | Native FA / recall | Recall at α = 0.05 | Recall at α = 0.10 |
      |---|---|---|---|
      | RF | 0.10 / 0.70 | **0.735** | **0.837** |
      | Jev | 0.20 / 0.92 | 0.533 | 0.667 |
      | joey-1 | 0.00 / 0.46 | 0.533 | 0.667 |
      | pplx-decider-flash | 0.05 / 0.52 | 0.469 | 0.644 |
      | d1 | 0.10 / 0.61 | 0.446 | 0.583 |
      | clef-flash | 0.05 / 0.60 | 0.288 | 0.612 |
      - At a matched clean false-alarm rate the local RF beats every hosted decider; Jev's 0.92 recall came from a loose threshold. Only 20 clean images, so α < 1/21 is unreachable; use a larger clean sample for the final numbers
- [x] Choose a VLM (Qwen2.5-VL-3B) and compare image only vs image + evidence, with text, overlay, depth-image and field ablations (`scripts/run_vlm_eval.py`)
- [x] Define the VLM scoring: entity match, damage match (only when it overlaps the entity), geometric relation with a VLM fallback, and label utility (`vlm_scoring.py`)
- [x] Build the 30-case human review packet and agreement/kappa scorer (`scripts/make_vlm_review.py`); run it on the final VLM predictions
- [ ] Rerun the decider comparison on the `weather` taxonomy once Aravind's physics rain/dust perturbations land; retrain the RF on those classes
- [x] Structure source swap, short version (2026-11-11): no SAM/DETR win. 100 val images, `gaussian_blur` + `foreground_blur`, 3 sources, 2 cues, paired deltas with 95% bootstrap CIs:
      | Source | Time | Delta (boundary) | Delta (separation) | Fix / break |
      |---|---|---|---|---|
      | depth | 49 ms | -0.71 pp [-2.71, +1.29] | -0.43 pp [-1.71, +0.71] | 17 / 22 |
      | SAM | 4.7 s | -0.43 pp [-2.43, +1.71] | -1.29 pp [-2.86, +0.14] | 17 / 20 |
      | DETR | 190 ms | -3.14 pp [-5.29, -1.29] | -1.14 pp [-2.71, +0.14] | 7 / 29 |
      - DETR is the only source whose CI excludes zero, and it is harmful (breaks 29, fixes 7). SAM is neutral and ~95x slower than depth
      - Church: SAM +1.4 pp (1/70, CI [-7.1, +10.0]), depth -1.4, DETR 0.0. Church clean accuracy is 30.8% over the full val set, so this is a classifier limitation, not a missing structure source
      - Decision: keep depth as the default; do not rerun Stage 3 with SAM/DETR. Run: `python scripts/run_structure_compare.py --per-class 10 --perturbations gaussian_blur foreground_blur --out results/structure`

## Kushi: Literature, failure analysis + attacks
- [ ] Summarize the professor's papers (incl. Aditya's DARPA/ECML paper): method, dataset, metric, what we can reuse
- [ ] Summarize XWOD, DAWN, RID/RIS, AdvRain, Laya and one conformal-detection paper in the same format
- [ ] Reproduce weather-shaped attacks against the detector: [AdvRain](https://arxiv.org/abs/2303.01338) (adversarial raindrop sticker) and [adversarial rain streaks](https://arxiv.org/abs/2009.09205)
- [ ] Check whether the condition detector + conformal gate catch these attacks, or whether they pass as natural rain (with Venkat)
- [ ] Defense: adversarial rain augmentation for the restorer, as in arXiv 2009.09205 (with Rohith)
- [ ] Build a failure-case gallery for church / altar / background entities: input, depth outline, SAM mask, predictions
- [ ] Build a second gallery for real rain/dust misses (XWOD/DAWN): missed persons and vehicles, and what the restorer did to them
- [ ] Track per-class results from `results/structure/per_class.csv` as the runs come in

## Shared
- [x] Agree on the target condition: heavy rain and dust storm (replaces fog vs night)
- [x] Assign owners for the generative AI and security tracks
- [ ] Review the owner split at the next sync and rebalance if anyone is overloaded
- [ ] Weekly sync: review benchmark table changes
