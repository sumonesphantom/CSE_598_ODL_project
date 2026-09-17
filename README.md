# Bolero: auditing depth-based layered reconstruction as a classifier front-end

**Question.** Depth-layer reconstruction is visually faithful. Is it *semantically*
faithful? We measure what it costs a deployed ImageNet classifier in accuracy,
prediction stability and calibration — and whether the damage is spread evenly.

**Main result.** On 3,925 Imagenette validation images, depth-layer reconstruction
drops ResNet-50 top-1 accuracy from **79.4% to 74.1%** (−5.2 pp, a 25% relative
increase in errors) and changes **15.6%** of predictions. The loss is highly
non-uniform: English springer loses **23.3 pp** while tench loses 1.6. Calibration
degrades with it (ECE 0.057 → 0.073). One third of all breakages are English
springer, and 86% of those land on another spaniel/setter/pointer — silhouette
survives, coat coloration does not.

## Reproducing the main result

```bash
git clone <this-repo> && cd CSE_598_ODL_project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Point at the Imagenette validation images (see "Data" below).
export IMAGENETTE_ROOT=/path/to/imagenette2-160

jupyter nbconvert --to notebook --execute --inplace 02_bolero_audit.ipynb
```

Or open `02_bolero_audit.ipynb` and run all cells. Runtime is under a minute on CPU
for the audit; §6 additionally downloads Depth-Anything-V2-Small (~100 MB) and runs
it on three images.

Outputs land in `results/`:

| File | Contents |
|---|---|
| `correctness_audit.csv` | Shipped vs. recomputed accuracy (the label-mapping defect) |
| `headline_metrics.csv` | Accuracy, flip rate, confidence |
| `per_class_accuracy.csv` | Per-class accuracy and delta |
| `outcome_transitions.csv` | correct/wrong transition counts |
| `calibration_bins.csv` | Reliability bins for both conditions |
| `top_broken_transitions.csv` | Where broken predictions land |
| `figures/*.png` | The three figures |

## Data

`bolero_evidence.csv` (in this repo, 3,925 rows) holds the recorded predictions and
is all the audit in §1–§5 needs. §6 additionally needs the Imagenette images:

```bash
curl -L -O https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar xzf imagenette2-160.tgz
export IMAGENETTE_ROOT=$PWD/imagenette2-160
```

If `IMAGENETTE_ROOT` is unset, the notebook looks for
`../clip_corruption_calibration/data/imagenette2-160` and then `./data/imagenette2-160`,
and skips §6 cleanly if it finds neither.

## Repository contents

| Path | Status |
|---|---|
| `02_bolero_audit.ipynb` | **The audit.** Runs end-to-end, zero errors. Start here. |
| `bolero_evidence.csv` | Recorded predictions for 3,925 images |
| `01_bolero.ipynb` | Original exploratory notebook. **Does not run** — kept as a record. |
| `boleropaper1.pdf` | Fan et al., *Revisiting Deep Intrinsic Image Decompositions* |
| `boleropaper2.pdf` | Birkl et al., *MiDaS v3.1* |

## Known gaps

1. **The reconstruction step is not in this repo.** `bolero_evidence.csv` records
   `bolero_prediction` for every image, but the code that produced it — layer
   decomposition, displacement, recomposition — exists only on the machine that
   generated the CSV. The audit is therefore reproducible *from the recorded
   predictions*, not *end-to-end from images*. Highest-priority fix.
2. **Two columns in `bolero_evidence.csv` are wrong.** `original_correct` and
   `bolero_correct` compare a wnid against an ImageNet index, so only tench can ever
   score; they report −0.15 pp against the true −5.22 pp. §3 of the notebook
   recomputes correctness properly and quantifies the defect. **Do not use those two
   columns.**
3. `01_bolero.ipynb` has hard-coded `F:\` paths, two missing imports
   (`AutoImageProcessor`, `AutoModelForDepthEstimation`), and a final cell
   referencing six undefined variables. It is superseded by `02_bolero_audit.ipynb`.
4. Single model, single dataset, single reconstruction setting. The texture/color
   mechanism in §6 is a hypothesis supported by the error structure, not a tested
   causal claim.

## Generative AI disclosure

Per §9 of the assignment brief: AI coding assistance was used for the analysis code
and figures. Disclose this in the final report.
