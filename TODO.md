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
- [ ] Run `run_diagnostic.py` for `vit_b16` and `convnext_t`, then report mCE vs ResNet-50
- [ ] Wire the best structure source into `run_restoration.py` as an option
- [ ] Add a detection eval harness on Aravind's loaders: frozen DETR + a YOLO baseline, mAP50 / mAP50-95 per weather condition
- [ ] Add `src/bolero/conformal.py`: split-conformal threshold for the perturbation gate (target clean false-alarm rate, e.g. ≤5%) replacing the fixed 0.6 in `detection.py`
- [ ] Add APS / RAPS prediction sets for the classifier; report coverage and set size per perturbation, before and after restoration
- [ ] Abstain-and-alert: when the conformal set is too large or the condition detector fires, flag "camera unreliable" instead of reporting a confident miss. Report the person miss rate at a fixed alert budget

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
- [ ] Read the in-context learning paper from the professor (internal, do not share)
- [ ] Serialize `SceneEvidence` (`prompting.py`) for the deciders: `noul` "is the view degraded?", `choice` "rain / dust / clean", `score` severity 1–3
- [ ] Compare deciders: RF detector vs Laya vs open-alternative-jev (small Qwen) vs poorjev; optional Jev baseline if we get access. Metrics: routing accuracy, clean false alarms, latency
- [ ] Recalibrate each decider with Venkat's conformal gate before gating on it
- [ ] Choose a VLM (local Qwen2.5-VL / LLaVA) and feed it `prompting.build_prompt` output plus the image; compare image only vs image + evidence on occlusion images, ablating each evidence field
- [ ] Define the VLM scoring: does it name the hidden entity / its relation correctly?

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
