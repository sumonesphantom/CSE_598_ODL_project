# Generative restoration: pilot results

Results of the run steps for the generative-restoration task (InstructPix2Pix asked to remove rain
or dust, then frozen DETR on every version). Code: branch `feat/diffusion-restore` (commit
bc584d9). Run by Aravind on 2026-11-26.

**Only the pilot was run** (step 4: 100 images per dataset × condition, 400 in all), not the
full set. The run steps say to stop and send examples if the edits look broken, and at image
guidance 1.5 they often do: whole scenes are replaced, objects erased, and colour casts or dust
added instead of removed (see `examples/`).

## How it was run

```bash
python scripts/run_generative_restore.py --limit 100
```

Defaults: image guidance 1.5 and 2.0, text guidance 7.5, 20 steps, 512 px, DETR
(`facebook/detr-resnet-50`), operating score 0.5, 200 bootstrap reps. Datasets: DAWN rain and
sand, and the XWOD test split (rain and dust). All 6 tests in `tests/test_hallucination.py`
pass. RTX 5080 laptop GPU: 3.4-3.8 s per edit.

## Main result (`summary.csv`)

DETR mAP50 change vs. the raw image, 95 % bootstrap interval in brackets. Bold: interval
excludes zero.

| | DAWN rain | DAWN dust | XWOD rain | XWOD dust |
|---|---|---|---|---|
| raw image (mAP50) | 0.527 | 0.483 | 0.336 | 0.447 |
| InstructPix2Pix, image guidance 1.5 | **-0.148** [-0.206, -0.089] | +0.006 [-0.141, +0.044] | **-0.225** [-0.314, -0.127] | **-0.111** [-0.186, -0.047] |
| InstructPix2Pix, image guidance 2.0 | **-0.142** [-0.217, -0.059] | **-0.150** [-0.267, -0.045] | **-0.062** [-0.106, -0.024] | -0.041 [-0.142, +0.106] |
| auto-contrast (baseline) | +0.001 [-0.009, +0.019] | **+0.019** [+0.002, +0.031] | **+0.059** [+0.019, +0.158] | **+0.085** [+0.022, +0.181] |

Hallucination at a 0.5 detection score:

| | true objects lost | invented per 100 images | CLIP similarity to raw |
|---|---|---|---|
| InstructPix2Pix, 1.5 | 25-70 % | 88-242 | 0.73-0.87 |
| InstructPix2Pix, 2.0 | 15-26 % | 118-224 | 0.91-0.94 |
| auto-contrast | 1-3 % | 32-52 | 0.99 |

**Generative restoration hurts detection on real rain and dust.** It lowers mAP50 in 7 of 8
cells (significantly in 6), erases many true objects and invents many false ones. Plain
auto-contrast does better everywhere. Image guidance 2.0 keeps the scene far better than 1.5,
but still costs mAP50 on DAWN.

## Files

| file | contents |
|---|---|
| `summary.csv` | dataset × condition × method: mAP50, mAP50-95, change vs. raw with CI, hallucination rates, CLIP similarity, times |
| `per_class.csv` | AP50 and recall50 per class |
| `per_image.csv` | per image × method: objects found before / after, lost, gained, invented, cleared, relabeled, CLIP similarity |
| `headline.json` | run configuration and the main numbers |
| `detections.pkl.gz` | raw detections, for re-scoring at another threshold without a GPU (`--rescore --threshold 0.3`) |
| `examples/` | raw (left) beside restored (right), with that image's counts in the title |

## Examples

Good (a true object found that was missed before, nothing lost or invented, scene kept):

- `good_sand_storm-012_ig2.jpg`: found 5 → 6
- `good_rain_storm-381_ig2.jpg`: found 0 → 1
- `good_sand_storm-094_ig2.jpg`: found 1 → 2
- `good_rain_storm-066_ig1.5.jpg`: found 3 → 4

Invented or erased:

- `bad_rain_storm-245_ig1.5.jpg`: the motorway is replaced by a green field; 21 of 21 cars lost
- `bad_rain_storm-090_ig1.5.jpg`: the cars dissolve into smears; 14 of 15 lost
- `bad_rain_storm-026_ig1.5.jpg`: 35 objects invented, the one true object lost
- `bad_sand_storm-002_ig1.5.jpg`: the truck and people are erased and dust is added; 4 of 4 lost

The examples are all DAWN images (CC BY-NC 3.0). XWOD's licence does not allow redistributing
its images, so its worst cases are listed by ID only; they are in `per_image.csv` and can be
viewed from a local XWOD copy: `dust_test_00003` and `dust_test_00002` (image guidance 1.5: 9
and 8 true objects lost), `heavy_rain_test_00035` (1.5: 7 lost). Its best case is
`dust_test_00087` (1.5: 3 more found, nothing lost or invented).
