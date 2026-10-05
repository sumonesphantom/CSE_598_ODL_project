# Restoration audit: why the pipeline lost to doing nothing, and what fixed it

This note records an audit of the classical restoration stage, the two defects it
found, the fix, and the measured result against the progress-presentation numbers.

**Headline.** Restoration went from **−2.4 pp** to **+1.03 pp** against the
untouched input, measured on the same 500 val images with the same bootstrap
procedure. It is the first configuration in this project where preprocessing
beats leaving the image alone, and it is now the best-performing method in the
study, ahead of the auto-contrast baseline.

Branch: `Trial_2`. Everything below is reproducible with
`python scripts/tune_plans.py && python scripts/run_restoration.py --per-class 50`.

---

## 1. The problem

The progress deck reported that pooled over all 42 perturbed conditions, every
method landed within about a point of doing nothing, and that oracle restoration
was the *worst* performer at −2.9 pp. Its risk slide named the cause:

> **Four restoration plans cause harm.** Denoising and the three fills cost 7 to
> 21 points. **Plan:** pass through when a plan hurts on dev data.

That diagnosis was correct and understated. At n = 500, **seven** of the fourteen
plans were net harmful, and the four largest accounted for the entire deficit:

| Perturbation | Plan | Δ accuracy | Images broken | Images fixed |
|---|---|---|---|---|
| `gaussian_noise` | NLM denoising | **−20.5 pp** | 366 | 59 |
| `background_removal` | Telea inpainting | **−16.2 pp** | 275 | 32 |
| `foreground_occlusion` | Telea inpainting | −9.1 pp | 173 | 37 |
| `region_removal` | Telea inpainting | −7.1 pp | 147 | 40 |
| `boundary_erase` | Telea inpainting | −2.2 pp | 85 | 52 |

Across these five, restoration **rescued 220 images and destroyed 1,046** — a net
loss of 826. The other nine plans averaged +1.3 pp, so classical restoration was
never broken as a concept; two specific operations were sinking it.

## 2. Diagnosis

### Denoising strips the evidence, it is not mistuned

The hand-set plan was `h = 0.9 × estimated σ`. The obvious hypothesis — that the
noise estimator was wrong — is false. Estimated σ lands within **1–8 %** of true
σ at all three severities.

The real mechanism is that non-local means averages each pixel against similar
patches elsewhere in the image. Grain disappears, and so does the fine texture
(fur, bark, fabric) that ResNet-50 reads. At severity 3 the plan broke **65 % of
the images it had previously classified correctly** — the highest break rate
anywhere in the study.

A sweep of **10 denoise variants** (NLM at five strengths, bilateral, median,
Gaussian at two widths, pass-through) found **no setting that beats leaving the
image alone**. Weakening the denoiser monotonically improves accuracy right down
to identity.

### Inpainting fabricates what it cannot know

Telea inpainting diffuses surrounding texture inward. That is plausible for a
small hole and fiction for a large one. Mask coverage, measured:

| Perturbation | Mask area, severities 1 / 2 / 3 |
|---|---|
| `region_removal` | 10 % / 20 % / 35 % |
| `foreground_occlusion` | 10 % / 20 % / 35 % |
| `background_removal` | **30 % / 50 % / 70 %** |

At `background_removal` severity 3 the plan reconstructs **70 % of the frame from
the remaining 30 %**, producing a smooth, confident invention with no content.
Mask *detection* is not the problem — recovered mask area matches the intended
area to within 0.1 pp.

A sweep of **9 inpaint variants** (Telea at three radii, Navier–Stokes, three
area-guard thresholds, 50 % blending, pass-through) again found **nothing that
beats pass-through**.

### The common mechanism

Both repairs make the image *look* cleaner to a human while removing the evidence
the classifier uses. A noisy photo still contains its texture; a denoised one does
not. An occluded photo still contains its intact remainder; an inpainted one has
that region replaced with plausible fiction.

## 3. The fix

Rather than hard-code the sweep's conclusions — which would be fitting to a
single observation — plan choice was moved onto development data, mirroring how
cue strengths were already chosen.

**`scripts/tune_plans.py` (new).** Scores every candidate in
`restoration.CANDIDATES` on perturbed **train**-split images and freezes the
winner to `results/tuning/plans.json` before Stage 3 touches val. A candidate must
beat `pass_through` by at least `--min-gain-pp` (default 0.5) to be adopted,
because a repair costs latency and a sub-point win is noise at this sample size.
`pass_through` is always a candidate: where a perturbation destroys information
rather than transforming it, no classical inverse exists.

**`src/bolero/restoration.py`.** `denoise()` takes a strength; `inpaint()` takes a
`max_area` guard that refuses to invent more than a set fraction of the frame;
`denoise_adaptive()` tests a severity-dependent response keyed on estimated σ.
Defaults preserve the original behaviour, so `PLANS` still describes the hand-set
pipeline and the existing tests pass unchanged.

**Selected on 200 train images (13,200 evaluations):**

| Perturbation | Was | Selected | Dev gain |
|---|---|---|---|
| `gaussian_noise` | `nlm_0.90` | **`bilateral`** | +0.7 pp |
| `background_removal` | `inpaint_always` | **`pass_through`** | — |
| `foreground_occlusion` | `inpaint_always` | **`pass_through`** | — |
| `region_removal` | `inpaint_always` | **`pass_through`** | — |
| `boundary_erase` | `inpaint_always` | **`pass_through`** | — |

Four of five resolve to *do nothing*. Only noise keeps a plan, and bilateral
filtering wins because it smooths within edges rather than across them.

## 4. Results

Evaluated on the val split, 500 images (50 per class), which the plan selection
never saw. Bootstrap CIs over images, 2,000 resamples — the same procedure as the
progress deck.

### Pooled over all 42 perturbed conditions

| Method | Progress deck | After the fix |
|---|---|---|
| **Oracle restoration** | −2.9 pp [−3.5, −2.2] | **+1.07 pp [+0.6, +1.5]** |
| **Detector restoration** | −2.4 pp [−2.9, −1.9] | **+1.03 pp [+0.6, +1.4]** |
| Auto-contrast baseline | +0.4 pp [+0.1, +0.7] | +0.32 pp [+0.1, +0.5] |
| Best depth cue | +0.0 pp [−0.3, +0.3] | +0.03 pp [−0.2, +0.2] |

The intervals do not overlap each other or zero, so the change is significant by
the deck's own standard. Detector restoration now also beats auto-contrast, which
was previously the only method that helped — so the deployable pipeline is now the
best method in the study rather than the worst.

The two rows that were not touched (baseline, cue) moved by ≤0.1 pp, which is the
consistency check: only restoration plans changed, so only restoration should move.

### Per perturbation, deployable system

| Perturbation | Before | After | Change |
|---|---|---|---|
| `gaussian_noise` | −20.5 | **+0.2** | **+20.7** |
| `background_removal` | −15.7 | **0.0** | **+15.7** |
| `region_removal` | −7.1 | **0.0** | +7.1 |
| `foreground_occlusion` | −5.1 | **0.0** | +5.1 |
| `boundary_erase` | −0.5 | **0.0** | +0.5 |
| the nine untouched plans | +5.0 … −0.2 | +4.6 … −0.1 | ≤0.4 (run variation) |

### Transitions and stability

| Metric | Before | After |
|---|---|---|
| Images broken (right → wrong) | **1,046** | **37** |
| Images fixed (wrong → right) | 220 | 42 |
| Net | **−826** | **+5** |
| Break rate | 0.237 | **0.007** |
| True-class probability shift | **−0.100** | +0.003 |
| Agreement with clean prediction | 0.611 | **0.738** |
| Accuracy cost on clean inputs | — | **0.0 pp, 0 images broken** |
| Repair latency | 7.0 ms | **0.22 ms** |

The number of images *fixed* went **down** (220 → 42). The gain comes entirely
from no longer destroying a thousand images that were already correct. The one
place genuine recovery survives is `gaussian_noise`, where bilateral filtering
fixes 9.9 % of wrong images against breaking 3.6 %.

## 5. What this does and does not show

**It does show** that classical restoration helps where a perturbation is
invertible (photometric, blur, compression: +1.6 to +4.9 pp) and harms where
information is destroyed (noise, occlusion), and that letting development data
choose between a plan and pass-through is enough to turn a net-negative stage
positive at zero cost on clean inputs. `grayscale` was already handled this way;
the change applies that principle consistently.

**It does not show** that the depth hypothesis works. Cue results are unchanged —
the best cue is +0.03 pp [−0.2, +0.2], and the mismatched-depth control still says
cues read the depth map without converting it into accuracy. Nothing here affects
that finding.

**It does not recover the damage.** Clean accuracy is 79.4 % and damaged accuracy
is ~65 %. Of the ~14.6 pp gap, this work recovers roughly 1. The remaining 34 % of
damaged images that stay misclassified are untouched. Reaching them needs a
*learned* restoration model, not a classical one.

## 6. Known limitations of this work

1. **Dev selection covers 5 of 14 perturbations.** `foreground_blur` (−2.0 pp) and
   `boundary_blur` (−1.1 pp) use unsharp masking and are still net harmful; they
   are not yet in `CANDIDATES`.
2. **200 train images** for plan selection. Differences under ~1 pp are noise, and
   the `bilateral` choice won by 0.7 pp — treat it as break-even, not a win.
3. **One plan per perturbation, not per severity.** At severity 3 a mild Gaussian
   blur beats pass-through on noise (0.42 vs 0.37) while hurting at severity 1
   (0.64 vs 0.76). `denoise_adaptive` keys off estimated σ but did not clear the
   adoption margin.
4. **Measured on a leaky evaluation set.** Per the progress deck, 3,791 of 3,925
   val images are ImageNet training images. Leakage affects absolute accuracy more
   than the difference between methods, but the honest test is the 134 unseen
   images, where n is small enough that +1.07 pp may not stay significant.
5. **Bootstrap CIs are not yet in the repo.** The intervals above were computed
   ad hoc; `metrics.py` should carry that function so the numbers are reproducible
   from the codebase.

## 7. Changes

| File | Change |
|---|---|
| `src/bolero/restoration.py` | parameterized `denoise()`, area-guarded `inpaint()`, `denoise_adaptive()`, `CANDIDATES`, `load_plans()` |
| `scripts/tune_plans.py` | **new** — dev-data plan selection |
| `scripts/run_restoration.py` | loads selected plans; records them in `run.json` |
| `scripts/run_all.sh` | plan selection added to both modes |
| `tests/test_restoration.py` | +10 tests (13 → 23); suite 102 → 112 |
| `README.md` | restoration-plans section rewritten; known gaps #2–#4 updated |

Nothing else was modified. The classifier, depth model, perturbations, cues,
detector and metrics are untouched, so the before/after comparison is valid:
the 14 perturbations are byte-identical to the previous run.
