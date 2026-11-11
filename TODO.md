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
- [ ] Run `run_structure_compare.py --per-class 20` and check whether SAM fixes the church failure case
- [ ] Run `run_diagnostic.py` for `vit_b16` and `convnext_t`, then report mCE vs ResNet-50 (~35–45 min + ~25–30 min on MPS):
      `python scripts/run_diagnostic.py --backbone vit_b16 --out results/diagnostic_vit_b16`, then
      `python scripts/benchmark.py --model results/diagnostic_vit_b16 --out results/benchmark_vit_b16` (same for `convnext_t`)
- [ ] Wire the best structure source into `run_restoration.py` as an option
- [ ] Add a detection eval harness on Aravind's loaders: frozen DETR + a YOLO baseline, mAP50 / mAP50-95 per weather condition
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
- [ ] Restoration plan audit: for each perturbation where `oracle_classical` Δacc < 0, pass through instead of restoring, then rerun the comparison
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
- [ ] Download XWOD + DAWN and confirm license terms before any results go in the report
- [ ] Write loaders for XWOD and DAWN (rain + sand/dust subsets only) and for SDSI / VIRADA (condition labels)
- [ ] Add physics-based synthetic rain to `perturbations.py`: streaks, rain + mist using depth (I·t + A·(1−t), t = e^(−β·d)), and lens raindrops
- [ ] Add physics-based dust: the same scattering model with a yellow-brown airlight A and wavelength-dependent β (blue fades first), plus contrast loss
- [ ] Generative degradation: heavy rain / dust storm versions of clean images (InstructPix2Pix or SDXL img2img). Control severity with edit strength. Keep an image only if the label survives (clean-image prediction / CLIP similarity / DETR boxes still match)
- [ ] Compare physics-based, generative and real (XWOD/DAWN) degradation on the same detector. Which synthetic one predicts real-data performance best?
- [ ] Retrain the RF condition detector on SDSI / VIRADA so it recognizes real rain and dust

## Sachin: Decision engine (Jev alternatives) + VLM
- [x] Read the in-context learning paper from the professor (internal, do not share). The instruction taxonomy is mapped in the README
- [x] Serialize `SceneEvidence` (`prompting.py`) for the deciders: `noul` "is the view degraded?", `choice` condition, `score` severity 1–3 (`conditions.py`, `build_jev_questions`)
- [x] Compare deciders on the OpenRouter Decisions API: RF vs Jev 1.14 vs Clef Flash vs Perplexity Decider Flash vs Liquid d1 vs Joey 1, plus a local Qwen prompt-and-parse adapter. Laya and poorjev were dropped as install-risk alternatives and documented. Metrics: routing accuracy, clean false alarms, severity MAE, latency, cost (`scripts/run_decider_compare.py`)
- [x] Probe image resolution limits: Jev needs >=280 px; Imagenette2-320 is the common input (`scripts/probe_resolutions.py`)
- [ ] Recalibrate each decider with Venkat's conformal gate before gating on it. `results/deciders/main/scores.csv` is ready; a split-half threshold sweep is in the README, but the real split-conformal threshold waits on `conformal.py`
- [x] Choose a VLM (Qwen2.5-VL-3B) and compare image only vs image + evidence, with text, overlay, depth-image and field ablations (`scripts/run_vlm_eval.py`)
- [x] Define the VLM scoring: entity match, damage match (only when it overlaps the entity), geometric relation with a VLM fallback, and label utility (`vlm_scoring.py`)
- [x] Build the 30-case human review packet and agreement/kappa scorer (`scripts/make_vlm_review.py`); run it on the final VLM predictions
- [ ] Rerun the decider comparison on the `weather` taxonomy once Aravind's physics rain/dust perturbations land; retrain the RF on those classes

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
