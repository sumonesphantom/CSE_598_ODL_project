# Literature summaries: datasets, attacks, decider, conformal detection

**TODO item (Khushi):** Summarize XWOD, DAWN, RID/RIS, AdvRain, Laya and one conformal-detection paper.
**Format:** what it is → method/dataset → metric → key results → what we can reuse.
**Sources:** abstracts / project pages, read 2026-11-11. Numbers marked *(team)* come from our own `TODO.md` / runs, not the paper.

---

## Quick comparison

| Paper | Type | Real data? | Rain | Dust | Boxes | Main use for us |
|---|---|---|---|---|---|---|
| XWOD | Detection dataset | Yes | Yes | Yes (haze/sand/dust) | Yes, 6 classes | **Main eval set** |
| DAWN | Detection dataset | Yes | Yes | Yes (sandstorm) | Yes | **Second eval set** (already downloaded) |
| RID / RIS (MPID) | Deraining benchmark | Yes (RID/RIS) | Yes | No | Yes, 5 classes | **The result to beat**: deraining hurts detection |
| AdvRain | Physical attack | – | Fake raindrops | – | – | **Attack to reproduce** |
| Adversarial rain streaks (2009.09205) | Digital attack + defense | – | Rain streaks | – | – | **Attack + defense to reproduce** |
| Laya | Decision engine | – | – | – | – | Optional open-source decider |
| Two-step conformal detection | Uncertainty method | – | – | – | Yes | Guarantees for boxes (security part) |

---

## 1. XWOD: real-world object detection under extreme weather

- **Citation:** Chen, Liu, Fadillah, Lai, Liu. *XWOD: A Real-World Benchmark for Object Detection under Extreme Weather Conditions.* arXiv 2605.11521, May 2026 (preprint, not yet peer-reviewed). [arXiv](https://arxiv.org/abs/2605.11521) · [Kaggle](https://www.kaggle.com/datasets/kuantinglai/exwod)
- **What it is:** a large real-world traffic detection benchmark.
- **Dataset:** 10,010 images, 42,924 boxes. **7 conditions:** rain, snow, fog, haze/sand/dust, flooding, tornado, wildfire. **6 classes:** car, person, truck, motorcycle, bicycle, bus. Fixed train/val/test splits released.
  - *(team)* Subsets we care about: 665 heavy-rain images, 560 haze/sand/dust images.
- **Metric:** mAP@50.
- **Key results:** YOLO detectors trained on XWOD transfer zero-shot to other weather benchmarks much better than published baselines: RTTS 63.0% vs 40.4%, **DAWN 59.9% vs 32.8%**, WEDGE 61.1% vs 45.4%.
  - *(team)* Baseline YOLOv11m mAP50: rain 60.5%, haze/sand/dust 29.9%. Dust is much harder than rain.
- **License:** "research-use license" (exact terms to confirm in the release; the CC BY-NC-SA 4.0 on arXiv covers the *paper*, not the data).
- **Reuse:** main evaluation set for rain + dust; its baselines give us numbers to compare against. Needs a Kaggle API key to download.

## 2. DAWN: vehicle detection in adverse weather

- **Citation:** Kenk, Hassaballah. *DAWN: Vehicle Detection in Adverse Weather Nature Dataset.* arXiv 2008.05402, 2020. [arXiv](https://arxiv.org/abs/2008.05402)
- **What it is:** a small real-world dataset for detection in bad weather, for driving and surveillance.
- **Dataset:** ~1,000 real traffic images (urban, highway, freeway), **4 conditions:** fog, snow, rain, sandstorm. Bounding boxes on vehicles (and people).
  - *(team)* We use Rain (200 images, 1,588 boxes) + Sand (323 images, 2,235 boxes). License per Mendeley: **CC BY-NC 3.0** (fine for coursework; cite the paper).
- **Metric:** paper mainly releases data; detection mAP is what users report.
- **Key results** *(team, frozen COCO DETR)*: rain mAP50 0.481, dust 0.457. **No preprocessing helped**; under dust, autocontrast/CLAHE/unsharp cost ~8 pp mAP50 and 6–8 pp person recall.
- **Reuse:** second evaluation set (already loaded by `load_dawn()`); source images for the **rain/dust miss gallery** (TODO item 7).

## 3. RID / RIS (MPID benchmark): does deraining help detection?

- **Citation:** Li et al. *Single Image Deraining: A Comprehensive Benchmark Analysis.* CVPR 2019 (arXiv 1903.08558); journal extension in IJCV 2020. [CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Single_Image_Deraining_A_Comprehensive_Benchmark_Analysis_CVPR_2019_paper.html) · [ar5iv](https://ar5iv.labs.arxiv.org/html/1903.08558)
- **What it is:** a benchmark (MPID) that evaluates deraining methods four ways: full-reference (PSNR/SSIM), no-reference, human judgment, and **task-driven** (does detection improve?).
- **Dataset:** synthetic + real rain (streaks, raindrops, rain + mist). Two real, box-annotated sets for the detection test:
  - **RID (Rain in Driving):** 2,496 frames from car-mounted cameras, 10,316 boxes; car, person, bus, bicycle, motorcycle.
  - **RIS (Rain in Surveillance):** 2,048 frames from 154 traffic surveillance cameras, 15,538 boxes; car, person, bus, truck, motorcycle.
- **Metric:** mAP, with COCO-pretrained Faster R-CNN, YOLOv3, SSD-512, RetinaNet (no rain adaptation).
- **Key result:** **deraining usually *hurts* detection.** Raw rainy input beats every deraining method for almost all detector/dataset pairs, e.g. YOLOv3 RID 27.8 (rainy) vs 26.7 (best derained); RetinaNet RIS 12.8 vs 10.9. Only Faster R-CNN on RID improved (16.5 → 23.4), and it is the weakest detector overall.
- **Reuse:** this is exactly our own finding (repair looks better, recognition gets worse). **RIS matches our surveillance goal.** Reproducing it is a sanity check (Rohith's TODO), and beating it is a strong claim for the report.

## 4. AdvRain: adversarial raindrops (physical attack)

- **Citation:** Guesmi, Hanif, Shafique. *AdvRain: Adversarial Raindrops to Attack Camera-based Smart Vision Systems.* arXiv 2303.01338, 2023. [arXiv](https://arxiv.org/abs/2303.01338)
- **What it is:** a physical attack: a **translucent sticker with printed raindrops on the camera lens**. No access to the system's software or hardware is needed.
- **Method:** each drop blurs the region it covers. An **iterative random search** finds drop positions that make the classifier wrong. Designed to look natural and to fool all images of a target class.
- **Targets / data:** classifiers only: VGG19 on ImageNet, ResNet34 on Caltech-101.
- **Metric:** drop in accuracy.
- **Key result:** with just **20 raindrops**, accuracy drops by **>45% (VGG19/ImageNet)** and **>40% (ResNet34/Caltech-101)**.
- **Reuse:** attack to reproduce on our classifier, then on DETR (paper didn't test detectors, so that's new). Key question for us: **does our condition detector + conformal gate flag it, or does it pass as natural rain?** Our existing `rain_lens` perturbation is a natural starting point.

## 5. Adversarial rain streaks + defensive deraining

- **Citation:** Zhai, Juefei-Xu, Guo, Xie, Ma, Feng, Qin, Liu. *Adversarial Rain Attack and Defensive Deraining for DNN Perception.* arXiv 2009.09205, 2020 (rev. 2022). [arXiv](https://arxiv.org/abs/2009.09205)
- **What it is:** a digital attack that makes **realistic rain streaks** whose parameters are optimized to fool the model, plus a defense.
- **Method:** "factor-aware rain generation" models rain streaks from the camera exposure process with **learnable rain factors** (e.g., angle, length, density), which the attack optimizes.
- **Targets:** image classification and object detection DNNs (datasets not named in the abstract).
- **Defense:** **adversarial rain augmentation**: train the deraining model on mixed adversarial rain layers so it protects downstream perception.
- **Key result:** synthesized rain is strongly adversarial, and the augmented derainer improves downstream performance (numbers are in the full paper; abstract gives none).
- **Reuse:** attack for TODO item 3; the defense is TODO item 5 (with Rohith). Our `rain_streaks` perturbation can be extended with learnable parameters.

## 6. Laya: open-source decision engine (Jev alternative)

- **Source:** [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0; numbers are author-reported).
- **What it is:** a fast, non-generative "System 1" model that answers **typed questions** over text: `choice` (pick a label), `score` (ordinal), `noul` (yes/no probability). Same API idea as Jev.
- **Model:** ModernBERT-family encoders (~150M–421M parameters); English, multilingual and typed-decision checkpoints, with a language router. Optional "vega" mode with a 0.8B reader that accepts images.
- **Speed:** ~33 ms per question on a T4 GPU.
- **Calibration:** temperature scaling + **conformal prediction sets** (α = 0.1) with `abstain` / `route` outputs. The README itself warns confidence can be high on wrong answers, so **gate on `route`, not raw confidence**. *(team)* Ships over-confident; recalibrate.
- **Reuse:** a free, local decider to compare with our RF. *(team)* Sachin dropped it as an install risk, and Venkat's results show the local RF already beats hosted deciders at a matched false-alarm rate, so this is low priority.

## 7. Conformal detection: Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction

- **Citation:** Timans, Straehle, Sakmann, Nalisnick. ECCV 2024 (arXiv 2403.07263). [arXiv](https://arxiv.org/abs/2403.07263)
- **What it is:** a way to put a **guaranteed safety margin** around each detected box.
- **Method:** two steps. (1) Conformal prediction over the **class label** (a set of possible classes). (2) Use that to build **box intervals** that cover the true box with a chosen probability, **even when the object is misclassified**. Ensemble and quantile-regression versions make the margin **adapt to object size**.
- **Metric:** coverage (does the interval contain the true box at the target rate?) and interval tightness.
- **Key result:** target coverage is met with "practically tight" intervals on real 2D detection datasets; size-adaptive versions balance coverage better.
- **Reuse:** extends Venkat's classifier-level conformal work (`conformal.py`) to **detection**, for the "know when the camera can't be trusted" part of our goal. Example: "with 90% guarantee, a person is inside this enlarged box." Pairs with the person-miss-rate-at-alert-budget task.
- **Alternatives:** Ries et al., *Probabilistic Object Detection with Conformal Prediction* (arXiv 2605.07549, 2026): corner-wise conformal with Bonferroni correction. A runway-detection application (arXiv 2505.16740) proposes a C-mAP metric.

---

## Takeaways for the team

1. **Evaluation:** XWOD (main) + DAWN (second) for real rain/dust; RIS for surveillance-camera rain. All have the same kind of traffic classes as our detection harness.
2. **The bar to clear:** Li et al. (RID/RIS) and our own DAWN run both show **simple restoration hurts detection**. A method that restores *and* keeps or improves mAP is a real contribution.
3. **Attacks:** AdvRain (physical, lens drops) and adversarial rain streaks (digital) are the two weather-shaped attacks; both were tested on classifiers, and only the second on detectors. Testing whether our gate catches them is new.
4. **Trust:** two-step conformal detection is the natural next step after Venkat's classifier-level conformal sets.
