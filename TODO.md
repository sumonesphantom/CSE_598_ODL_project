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
      | Rain (200 img, 1,588 boxes) | none | **0.481** | **0.314** | 0.89 |
      | | autocontrast / clahe / unsharp | 0.480 / 0.471 / 0.462 | 0.297 / 0.294 / 0.284 | 0.93 / 0.89 / 0.93 |
      | Dust (323 img, 2,235 boxes) | none | **0.457** | **0.287** | 0.84 |
      | | autocontrast / clahe / unsharp | 0.381 / 0.376 / 0.375 | 0.235 / 0.229 / 0.230 | 0.78 / 0.76 / 0.77 |
      - No preprocessing helps; under dust every baseline costs ~8 pp mAP50 and 6–8 pp person recall (same direction as Li et al. CVPR 2019). Trucks / buses are the weak classes (AP50 0.26–0.40 under dust)
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
- [x] DAWN downloaded (done by Venkat, 2026-11-11): Rain (200 images) + Sand (323) from Mendeley 10.17632/766ygrbt8y.3 into `data/dawn/` (git-ignored). License per Mendeley metadata: **CC BY-NC 3.0** (attribute, non-commercial; fine for coursework, cite the DAWN paper)
- [ ] XWOD: needs a Kaggle API key (`~/.kaggle/kaggle.json`), then `kaggle datasets download kuantinglai/exwod` into `data/xwod/`. Confirm its research-use license, and check the loader against the real layout
- [x] Loaders (`src/bolero/weather_data.py`, tested): `load_dawn()` (VOC XML; Sand → `dust`), `load_xwod()` / `load_yolo_dataset()` (Ultralytics YOLO layout, assumed until XWOD is downloaded), `load_condition_folders()` for SDSI / VIRADA (one folder per label). SDSI / VIRADA themselves are not downloaded yet
- [x] Physics-based weather in `perturbations.py` (`WEATHER`, opt-in; not in `ALL_PERTURBATIONS`, so earlier results and the trained detector are unchanged): `rain_streaks`, `rain_mist` (depth haze I·t + A·(1−t), t = e^(−β·d), plus streaks), `rain_lens` (refracting, defocused drops), `dust_haze` (yellow-brown airlight, β per channel R 0.8 / G 1.0 / B 1.35 so blue fades first, plus contrast loss), `dust_storm` (dense haze + wind-blown sand). Names map onto the `weather` taxonomy; usable via `--perturbations` in `run_diagnostic.py` / `run_decider_compare.py` and `conditions.iter_samples`. 4 tests
- [ ] Measure the classifier drop on the weather set: `python scripts/run_diagnostic.py --perturbations rain_streaks rain_mist rain_lens dust_haze dust_storm --per-class 20 --out results/diagnostic_weather`
- [ ] Generative degradation: heavy rain / dust storm versions of clean images (InstructPix2Pix or SDXL img2img). Control severity with edit strength. Keep an image only if the label survives (clean-image prediction / CLIP similarity / DETR boxes still match)
- [ ] Compare physics-based, generative and real (XWOD/DAWN) degradation on the same detector. Which synthetic one predicts real-data performance best?
- [ ] Retrain the RF condition detector on SDSI / VIRADA so it recognizes real rain and dust

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

## Khushi: Literature, failure analysis + attacks
- [ ] Summarize the professor's papers (incl. Aditya's DARPA/ECML paper): method, dataset, metric, what we can reuse
- [x] Summarize XWOD, DAWN, RID/RIS, AdvRain, Laya and one conformal-detection paper in the same format → [`docs/literature_summaries.md`](docs/literature_summaries.md)
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
