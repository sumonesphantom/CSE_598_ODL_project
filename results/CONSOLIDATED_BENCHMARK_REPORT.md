# Bolero: Consolidated Benchmark Report

**Generated from existing experiment artifacts.**

This report consolidates the available audit, diagnostic, detector, tuning, restoration, cue-sweep, and distortion results. These artifacts come from distinct experiment stages and are not one single end-to-end run.

> Negative and neutral outcomes are retained. This report does not assume that every restoration method improves recognition.

## 1. Research objective

The project investigates whether depth-aware, context-preserving image processing can help a classifier retain recognition performance under visual perturbations, and whether a detector can route selected inputs through a restoration stage.

## 2. Experimental organization

| Stage | Data / configuration | Primary artifact |
| --- | --- | --- |
| Initial audit | 3,925 images | `results/audit/` |
| Perturbation diagnostic | 3,925 images; 14 perturbations; 3 levels | `results/diagnostic/` |
| Detector evaluation | 18,000 train; 9,000 validation samples | `results/detector/` and `results/detector_retrained/` |
| Strength tuning | 100 training images; 10 per class | `results/tuning/strengths.json` |
| Plan tuning | 200 training images; 20 per class | `results/tuning/plans.json` |
| Original restoration | 500 validation images; 50 per class | `results/restoration/` |
| Tuned-plan validation | 500 validation images; 50 per class | `results/restoration/plan_validation_50_retry/` |

**Counting note:** In restoration tables, evaluated samples are the sum of the row-level `n` values across perturbation/severity conditions. They are not unique images. The same validation image is evaluated in multiple conditions and, in the original benchmark, under multiple depth sources.

## 3. Initial Bolero audit

| Metric | Result |
| --- | --- |
| images | 3,925 |
| accuracy, original | 79.36% |
| accuracy, bolero | 74.14% |
| accuracy change | -5.22 pp |
| relative error increase | +25.3% |
| predictions changed | 15.64% |
| mean confidence, original | 0.848 |
| mean confidence, bolero | 0.811 |
| mean confidence change | -0.037 |

The audit indicates that the evaluated Bolero transformation did not preserve the original classifier's accuracy in this run. It motivates more selective processing rather than establishing an overall gain.

Source: `results/audit/headline_metrics.csv`.

## 4. Perturbation diagnostic

| Metric | Result |
| --- | --- |
| Images | 3925 |
| Records | 164850 |
| Perturbations | 14 |
| Severity levels | 3 |
| Clean accuracy | 79.36% |
| Perturbed accuracy | 64.77% |
| Prediction changed rate | 26.23% |
| Correct to wrong | 27445 |
| Wrong to correct | 3386 |
| Held-out rand_augment accuracy | 72.99% |
| Wall time (minutes) | 24.68 |

Correct-to-wrong and wrong-to-correct counts show that perturbations can both break previously correct predictions and occasionally correct previously incorrect ones.

Source: `results/diagnostic/headline.json` and `summary.csv`.

## 5. Detector evaluation

| Detector | Train samples | Validation samples | Top-1 accuracy | Routed correctly | Clean false-alarm rate | Perturbed detection rate |
| --- | --- | --- | --- | --- | --- | --- |
| Original detector | 18000 | 9000 | 83.16% | 71.74% | 5.50% | 72.02% |
| Retrained detector | 18000 | 9000 | 83.12% | 71.74% | 6.00% | 72.20% |

Top-1 accuracy, routing correctness, clean false-alarm rate, and perturbed detection rate are distinct measurements. The retrained detector is the one named in the newer plan-validation metadata.

Sources: `results/detector/` and `results/detector_retrained/`.

## 6. Restoration parameter tuning

### 6.1 Strength tuning

Tuning split: **train**; 100 images; 10 per class; seed 42. Maximum allowed clean accuracy drop: 1.0 pp.

| Method | Selected strength | Constraint met |
| --- | --- | --- |
| baseline:autocontrast | 0.5 | Yes |
| baseline:clahe | 0.1 | Yes |
| baseline:unsharp | 0.1 | Yes |
| cue:border_strengthening | 0.1 | Yes |
| cue:contrast_luminance | 0.2 | Yes |
| cue:depth_boundary_emphasis | 0.3 | Yes |
| cue:entity_background_separation | 0.1 | Yes |
| cue:shadow_reinforcement | 0.2 | Yes |
| cue:thermal_injection | 0.1 | Yes |

Strength selection used training data, a candidate grid, and a maximum permitted clean-accuracy drop. These are tuning outputs, not independent validation results.

Source: `results/tuning/strengths.json` and `summary.csv`.

### 6.2 Restoration plan tuning

Tuning split: **train**; 200 images; 20 per class; seed 42. Minimum gain criterion: 0.5 pp.

| Perturbation | Selected plan |
| --- | --- |
| background_removal | pass_through |
| boundary_erase | pass_through |
| foreground_occlusion | pass_through |
| gaussian_noise | bilateral |
| region_removal | pass_through |

Plan selection is separate from strength selection. Pass-through selections mean that no restoration operation was retained for those perturbations; they are not successful image reconstructions.

Source: `results/tuning/plans.json` and `plan_summary.csv`.

## 7. Original broad restoration benchmark

| Metadata | Value |
| --- | --- |
| Split | val |
| Images | 500 |
| Images per class | 50 |
| Seed | 42 |
| Strength status | tuned |
| Depth sources | perturbed, clean, mismatched |

### Results by method and depth source

| Method | Depth source | Evaluated samples | Conditions | Perturbed accuracy | Restored accuracy | Change | Pooled fix rate | Pooled break rate | Restore ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline:autocontrast | unused | 22,500 | 45 | 65.76% | 66.17% | 0.41 pp | 3.34% | 1.12% | 0.71 |
| baseline:clahe | unused | 22,500 | 45 | 65.76% | 65.77% | 0.01 pp | 2.32% | 1.19% | 0.68 |
| baseline:unsharp | unused | 22,500 | 45 | 65.76% | 65.50% | -0.26 pp | 3.61% | 2.28% | 0.37 |
| cue:border_strengthening | clean | 22,500 | 45 | 65.76% | 65.74% | -0.02 pp | 2.87% | 1.53% | 0.19 |
| cue:border_strengthening | mismatched | 22,500 | 45 | 65.76% | 65.24% | -0.52 pp | 3.36% | 2.55% | 0.19 |
| cue:border_strengthening | perturbed | 22,500 | 45 | 65.76% | 65.64% | -0.12 pp | 2.52% | 1.50% | 44.20 |
| cue:contrast_luminance | clean | 22,500 | 45 | 65.76% | 65.77% | 0.01 pp | 3.04% | 1.57% | 0.88 |
| cue:contrast_luminance | mismatched | 22,500 | 45 | 65.76% | 65.77% | 0.01 pp | 3.14% | 1.62% | 0.87 |
| cue:contrast_luminance | perturbed | 22,500 | 45 | 65.76% | 65.73% | -0.03 pp | 3.02% | 1.62% | 44.87 |
| cue:depth_boundary_emphasis | clean | 22,500 | 45 | 65.76% | 63.98% | -1.78 pp | 6.76% | 6.22% | 0.39 |
| cue:depth_boundary_emphasis | mismatched | 22,500 | 45 | 65.76% | 61.47% | -4.29 pp | 6.62% | 9.97% | 0.38 |
| cue:depth_boundary_emphasis | perturbed | 22,500 | 45 | 65.76% | 63.21% | -2.55 pp | 5.49% | 6.74% | 44.36 |
| cue:entity_background_separation | clean | 22,500 | 45 | 65.76% | 65.81% | 0.05 pp | 3.75% | 1.88% | 0.35 |
| cue:entity_background_separation | mismatched | 22,500 | 45 | 65.76% | 65.05% | -0.71 pp | 3.60% | 2.95% | 0.35 |
| cue:entity_background_separation | perturbed | 22,500 | 45 | 65.76% | 65.78% | 0.02 pp | 3.26% | 1.66% | 44.32 |
| cue:shadow_reinforcement | clean | 22,500 | 45 | 65.76% | 65.65% | -0.11 pp | 5.33% | 2.94% | 0.10 |
| cue:shadow_reinforcement | mismatched | 22,500 | 45 | 65.76% | 63.92% | -1.84 pp | 4.71% | 5.25% | 0.10 |
| cue:shadow_reinforcement | perturbed | 22,500 | 45 | 65.76% | 65.50% | -0.26 pp | 4.52% | 2.75% | 44.07 |
| cue:thermal_injection | clean | 22,500 | 45 | 65.76% | 63.40% | -2.36 pp | 7.94% | 7.73% | 0.48 |
| cue:thermal_injection | mismatched | 22,500 | 45 | 65.76% | 61.79% | -3.97 pp | 7.06% | 9.72% | 0.47 |
| cue:thermal_injection | perturbed | 22,500 | 45 | 65.76% | 63.07% | -2.69 pp | 6.98% | 7.73% | 44.46 |
| detector+cue:border_strengthening | perturbed | 22,500 | 45 | 65.76% | 63.88% | -1.88 pp | 8.41% | 7.24% | 70.59 |
| detector+cue:contrast_luminance | perturbed | 22,500 | 45 | 65.76% | 63.60% | -2.16 pp | 8.18% | 7.54% | 71.23 |
| detector+cue:depth_boundary_emphasis | perturbed | 22,500 | 45 | 65.76% | 62.52% | -3.24 pp | 9.46% | 9.86% | 70.77 |
| detector+cue:entity_background_separation | perturbed | 22,500 | 45 | 65.76% | 63.86% | -1.90 pp | 8.40% | 7.26% | 70.74 |
| detector+cue:shadow_reinforcement | perturbed | 22,500 | 45 | 65.76% | 63.67% | -2.09 pp | 8.84% | 7.79% | 70.49 |
| detector+cue:thermal_injection | perturbed | 22,500 | 45 | 65.76% | 62.30% | -3.46 pp | 9.62% | 10.27% | 70.85 |
| detector_classical | unused | 22,500 | 45 | 65.76% | 63.48% | -2.28 pp | 6.84% | 7.02% | 26.42 |
| none | unused | 22,500 | 45 | 65.76% | 65.76% | 0.00 pp | 0.00% | 0.00% | 0.00 |
| oracle_classical | unused | 21,000 | 42 | 65.23% | 62.37% | -2.87 pp | 9.92% | 9.68% | 3.74 |

Each cue-based method is reported separately for `perturbed`, `clean`, and `mismatched` depth. Baseline methods and detector/oracle controls use `unused` because they do not consume a depth source. Clean perturbation rows are excluded from these restoration-effect aggregates.

**Aggregation:** Accuracy and timing are weighted by each row's sample count. Fix rate is pooled as total wrong-to-correct transitions divided by the total number initially incorrect. Break rate is pooled as total correct-to-wrong transitions divided by the total number initially correct. These denominators differ, so averaging row-level rates by `n` would not produce pooled transition rates.

The sample count is repeated condition-level evaluation volume, not a count of distinct validation images. The original broad benchmark is kept separate from the tuned-plan validation because their method sets and configurations differ.

Source: `results/restoration/run.json` and `results/restoration/summary.csv`.

## 8. Tuned-plan validation benchmark

| Metadata | Value |
| --- | --- |
| Split | val |
| Images | 500 |
| Images per class | 50 |
| Seed | 42 |
| Backbone | resnet50 |
| Strength status | tuned |
| Plan status | tuned |
| Detector | results\detector_retrained\detector.joblib |
| Depth sources in run configuration | perturbed, clean, mismatched |

### Plan variants used

| Perturbation | Variant |
| --- | --- |
| background_removal | pass_through |
| boundary_blur | hand_set |
| boundary_erase | pass_through |
| brighten | hand_set |
| clean | hand_set |
| darken | hand_set |
| foreground_blur | hand_set |
| foreground_occlusion | pass_through |
| gaussian_blur | hand_set |
| gaussian_noise | bilateral |
| grayscale | hand_set |
| inversion | hand_set |
| jpeg_compression | hand_set |
| low_contrast | hand_set |
| region_removal | pass_through |

### Results by method

| Method | Depth source | Evaluated samples | Conditions | Perturbed accuracy | Restored accuracy | Change | Pooled fix rate | Pooled break rate | Restore ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| detector_classical | unused | 22,500 | 45 | 65.76% | 66.78% | 1.02 pp | 5.10% | 1.10% | 1204.20 |
| none | unused | 22,500 | 45 | 65.76% | 65.76% | 0.00 pp | 0.00% | 0.00% | 0.00 |
| oracle_classical | unused | 21,000 | 42 | 65.23% | 66.34% | 1.11 pp | 7.56% | 2.33% | 0.77 |

### Actual evaluation coverage

| Method | Depth source | Rows | Evaluated samples | Perturbations represented |
| --- | --- | --- | --- | --- |
| detector_classical | unused | 46 | 23,000 | background_removal, boundary_blur, boundary_erase, brighten, clean, darken, foreground_blur, foreground_occlusion, gaussian_blur, gaussian_noise, grayscale, inversion, jpeg_compression, low_contrast, rand_augment, region_removal |
| none | unused | 46 | 23,000 | background_removal, boundary_blur, boundary_erase, brighten, clean, darken, foreground_blur, foreground_occlusion, gaussian_blur, gaussian_noise, grayscale, inversion, jpeg_compression, low_contrast, rand_augment, region_removal |
| oracle_classical | unused | 42 | 21,000 | background_removal, boundary_blur, boundary_erase, brighten, darken, foreground_blur, foreground_occlusion, gaussian_blur, gaussian_noise, grayscale, inversion, jpeg_compression, low_contrast, region_removal |

Although the run metadata lists `perturbed`, `clean`, and `mismatched` as configured depth sources, the saved validation summary has `depth_source=unused` for all methods. Therefore this validation output does not provide a comparison across those three depth sources.

The `none` and `detector_classical` methods each cover clean plus 15 perturbations. `oracle_classical` has fewer rows and omits clean and `rand_augment`, so its aggregate has a different condition set and should not be treated as a strictly matched overall comparison.

This validation run uses the selected strength and plan configurations and the retrained detector. It remains independent of the original broad restoration benchmark.

Source: `results/restoration/plan_validation_50_retry/run.json` and `summary.csv`.

## 9. Cue and distortion experiments

| Experiment | Rows available | Artifact |
| --- | --- | --- |
| Cue sweep | 66 | `results/cue_sweep/summary.csv` |
| Distortion evaluation | 78 | `results/distortions/summary.csv` |

These experiments provide additional evidence about individual cues and distortion families. Their own sample sizes and evaluation protocols apply; they are not automatically comparable to the main restoration benchmarks.

## 10. Interpretation and limitations

The available results support the following cautious conclusions:

1. **Perturbations affect recognition.** The diagnostic shows a substantial difference between clean and perturbed accuracy.
2. **Detector routing is imperfect.** Clean false alarms and missed perturbations remain relevant when deciding whether to invoke restoration.
3. **Restoration is not universally beneficial.** Some transformations can damage useful classifier cues, producing neutral or negative outcomes.
4. **Selective processing is a meaningful design direction.** Pass-through plan selections indicate that leaving an input unchanged can be preferable to applying a harmful operation.
5. **The two restoration runs answer different questions.** They must remain separate in headline claims and comparisons.

### Limitations

- The results use finite validation samples and a defined set of controlled perturbations.
- Different runs vary in method coverage, detector version, tuning procedure, and condition coverage.
- Pass-through is not evidence of successful reconstruction.
- Classifier accuracy alone does not establish object-level robustness or performance under real weather and surveillance conditions.
- Saved outputs alone do not establish full pipeline reproducibility; source code, model versions, dependencies, and data provenance must also be retained.

## 11. Excluded and pending work

The following smoke-test directories are excluded from headline benchmark aggregation:

- `results/restoration/detector_smoke/`
- `results/restoration/plan_smoke/`

NAFNet adapter smoke tests, if present, demonstrate execution rather than comparative restoration effectiveness. Weather-specific diagnostics, real-world weather evaluation, generative degradation comparisons, object-detection baselines, and security-oriented weather attacks remain pending until their evaluation artifacts are available.

## 12. Artifact inventory

| Artifact | Status |
| --- | --- |
| `results/audit/headline_metrics.csv` | Available |
| `results/diagnostic/headline.json` | Available |
| `results/diagnostic/summary.csv` | Available |
| `results/detector/headline.json` | Available |
| `results/detector/eval_summary.csv` | Available |
| `results/detector_retrained/headline.json` | Available |
| `results/detector_retrained/eval_summary.csv` | Available |
| `results/tuning/strengths.json` | Available |
| `results/tuning/summary.csv` | Available |
| `results/tuning/plans.json` | Available |
| `results/tuning/plan_summary.csv` | Available |
| `results/restoration/run.json` | Available |
| `results/restoration/summary.csv` | Available |
| `results/restoration/plan_validation_50_retry/run.json` | Available |
| `results/restoration/plan_validation_50_retry/summary.csv` | Available |
| `results/cue_sweep/summary.csv` | Available |
| `results/distortions/summary.csv` | Available |

## 13. Reproducibility notes

- All artifact paths in this report are repository-relative.
- The generator reads existing files only; it does not load model weights or execute experiment scripts.
- `results/SUMMARY.md` is not overwritten.
- Smoke-test outputs are excluded from the main restoration tables.
- Missing artifacts are marked unavailable rather than replaced with inferred values.

---

_Generated from the current contents of `results/` by `scripts/generate_report.py`._
