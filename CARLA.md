# CARLA weather testbed (extra, outside the agreed TODO)

Aravind, 2026-11-25. Adds the CARLA simulator as a fourth source of rain / dust, next to the
physics filters, the InstructPix2Pix edits and the real XWOD / DAWN photos, and asks the same
questions of it as of the others (TODO.md, Aravind's section).

**Short version.** CARLA gives perfectly paired frames with exact boxes and depth, and its
traffic-camera viewpoint matches real surveillance data far better than COCO does. Its weather,
though, is a poor stand-in for real weather: rain and dust barely hurt the detector (real rain
and dust hurt it a lot), its "dust" is grey fog, and a detector trained on CARLA weather does
not recognize real rain or dust at all. Use CARLA for viewpoint, geometry and true depth, not for
weather appearance.

## What was recorded

`scripts/record_carla_weather.py` (CARLA 0.9.16, headless, synchronous mode at 20 Hz):

- 287 scenes, 5 towns (Town01, 02, 03, 05, 10HD), 12 camera positions per town, 5 instants per
  camera, 1280×720, 90° FOV. Cameras sit 5-9 m up beside the road, looking along it (pitch -12°
  to -22°), like pole-mounted traffic cameras. Sun altitude 15-70°, random per scene
- 30 vehicles at the spawn points nearest the camera (Traffic Manager autopilot) and 25
  pedestrians on walkable ground within 45 m
- Every scene is rendered 7 times with all actors frozen: clean, rain 1-3, dust 1-3. The frames
  of a scene differ only in the weather
- Boxes from CARLA's instance-segmentation camera (visible extent; riders count as person, as in
  COCO; boxes under 12 px or 150 visible pixels dropped): 3,667 boxes, 1,533 person, 1,416 car,
  471 truck, 154 motorcycle, 58 bus, 35 bicycle. Spot-checked visually; boxes are tight
- Ground-truth depth (metres) for every scene
- Rain levels = CARLA's SoftRainNoon / MidRainyNoon / HardRainNoon presets (with the scene's
  sun): wet, reflective roads and splashes, little visible streaking
- **CARLA cannot render yellow-brown dust.** Its `dust_storm` parameter mostly greys the sky,
  the `DustStorm` preset looks like an overcast day, and fog colour is not controllable from
  Python. "Dust" here is `dust_storm` 100 plus grey fog of density 20 / 40 / 65
- 647 MB in `data/carla/` (git-ignored). About 20 minutes of simulation on the RTX 5080

## 1. Does CARLA weather degrade the detector like real weather?

Frozen COCO DETR, mAP50 with 95% image-bootstrap CIs (`results/degradation_compare/`). Clean
CARLA: **0.511** [0.473, 0.556], already much harder than clean COCO (0.708), because the
elevated cameras see many small, distant objects, as real surveillance cameras do.

| mAP50 change vs. clean CARLA | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| CARLA rain | -0.001 | -0.004 | -0.069 |
| CARLA dust | **+0.048** | -0.004 | -0.056 |
| physics rain_mist on CARLA frames | -0.344 | -0.438 | -0.484 |
| physics dust_haze on CARLA frames | -0.059 | -0.129 | -0.254 |

Real data for scale: XWOD rain 0.252, XWOD dust 0.446, DAWN rain 0.481, DAWN dust 0.457.
CARLA's own weather barely touches the detector; its light dust even helps slightly (the grey
veil hides background clutter). The physics filters, applied to the same frames, are far harsher.

Preprocessing agrees with real data in direction: on CARLA dust, auto-contrast helps at levels
2 and 3 (+0.019 [0.004, 0.033], +0.049 [0.027, 0.068]); on CARLA rain no baseline helps by more
than 0.014. On real data auto-contrast helps on dust and nothing helps on rain.

## 2. Which synthetic source predicts real data? (CARLA added to the TODO comparison)

Same scores as in TODO.md (class profile = Spearman of per-class AP50 with the real set; same
best = share of levels that pick the real set's best method; KID = CLIP kernel distance ×1000,
lower = looks more like real):

| vs. → | | XWOD rain | DAWN rain | XWOD dust | DAWN dust |
|---|---|---|---|---|---|
| class profile | other real set | 0.80 | 0.80 | 0.89 | 0.89 |
| | **clean CARLA (no weather)** | 0.60 | 0.90 | 0.66 | 0.54 |
| | CARLA weather | 0.63 | 0.90 | 0.18 | -0.07 |
| | physics on CARLA (rain_mist / dust_haze) | 0.67 | 0.73 | 0.58 | 0.51 |
| | clean COCO (no weather) | -0.10 | 0.10 | 0.54 | 0.14 |
| same best method | CARLA weather | 0.33 | 0.33 | 0.67 | 0.67 |
| | physics (rain_mist / dust_haze) on CARLA / on COCO | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| KID to real | clean CARLA / clean COCO | 3.35 / 2.14 | 2.74 / 1.41 | 2.94 / 1.56 | 3.08 / 1.75 |
| | other real set | 1.01 | 1.01 | 0.24 | 0.24 |

- **The class-profile score measures the camera viewpoint, not the weather.** Clean CARLA
  frames, with no weather at all, already match the real sets' class pattern (0.54-0.90), and
  adding CARLA weather does not improve it. Traffic-camera geometry (small, distant people and
  bikes; trucks and buses seen at an angle) is what makes CARLA match. This also qualifies the
  TODO comparison: COCO-based synthetic sets score low on class profile partly because COCO's
  close-up photos rank classes differently even when clean
- **CARLA looks the least like real data** (highest KID of any source, clean or weathered), so
  photographic realism is not what drives its better class profile
- **For choosing preprocessing, physics filters stay the most reliable**, on either base images;
  CARLA's native weather picks the real-best method only a third to two thirds of the time

## 3. Condition detector: does simulator weather teach real weather?

`scripts/train_weather_detector.py` with CARLA frames present (`results/weather_detector/`).
A fourth detector, `carla`, trains on the same clean pool plus CARLA clean / rain / dust frames
from Town01, 03 and 05 (688 clean, 534 rain, 534 dust); Town10HD and Town02 are held out.
Share of images routed correctly (threshold 0.6):

| Test set | CARLA-trained | Real-trained | Physics-trained | Combined |
|---|---|---|---|---|
| XWOD rain | **0.00** | 0.30 | 0.57 | 0.68 |
| DAWN rain | 0.06 | 0.90 | 0.64 | 0.95 |
| VIRADA rain | 0.00 | 0.20 | 0.00 | 0.14 |
| XWOD dust | 0.30 | 0.40 | 0.15 | 0.36 |
| DAWN dust | 0.19 | 0.96 | 0.44 | 0.92 |
| CARLA rain (held-out towns) | 0.27 | 0.10 | 0.55 | 0.75 |
| CARLA dust (held-out towns) | 0.36 | 0.00 | 0.00 | 0.00 |
| COCO / Imagenette / VIRADA / CARLA clean | 0.99 / 1.00 / 1.00 / 0.95 | 0.96 / 0.96 / 0.76 / 0.95 | 0.63 / 0.53 / 0.88 / 0.72 | 0.56 / 0.51 / 0.70 / 0.69 |

- A detector trained on CARLA weather **almost never recognizes real rain** (0-6%) and finds
  only 19-30% of real dust. It is cautious (it keeps 95-100% of clean images clean), but what it
  learned is CARLA's look, not weather
- The real-trained detector sees CARLA rain as clean (71% of the time) and CARLA's grey dust as
  rain (81%): to a model that knows real weather, CARLA weather does not look like either
- Even on its own held-out towns the CARLA detector catches only a quarter to a third of CARLA
  rain and dust: CARLA's level-1 rain is nearly invisible
- Recommendation unchanged: route on the real-trained detector

## 4. Does depth accuracy matter for the physics filters? (true depth)

CARLA's depth is exact, so the depth-based physics effects were also built from true depth
(converted to the same normalized inverse-depth convention as Depth-Anything's output). True
depth minus estimated depth, mAP50, 95% CI:

| | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| rain_mist | +0.011 [0.004, 0.024] | +0.012 [0.004, 0.026] | +0.005 [0.003, 0.007] |
| dust_haze | +0.011 [0.000, 0.025] | +0.008 [0.002, 0.015] | +0.023 [0.009, 0.037] |
| dust_storm | +0.001 [-0.003, 0.007] | +0.007 [0.002, 0.011] | +0.002 [-0.001, 0.007] |

True depth makes the haze a little milder (often significant, never more than 2.3 pp). Depth
estimation errors are not a meaningful weakness of the physics filters.

## Caveats

- The 5 instants per camera are a few seconds apart and strongly correlated; image-level
  bootstrap intervals for CARLA sets are therefore somewhat too narrow (camera-level resampling
  would be stricter)
- 287 scenes instead of 300: the first long run stopped when CARLA failed a tick on an actor it
  had removed itself. The recorder now abandons that camera and continues
- CARLA dust is grey fog, not dust (see above); a mod with coloured fog, or CARLA 0.10 (Unreal
  Engine 5), may do better
- Ubuntu 24.04 is not officially supported by CARLA 0.9.16; it ran headless without problems here
- Found later (DRIVING.md): the recorder reused one Traffic Manager across towns, and a Traffic
  Manager keeps the road map of the first town it served. In the towns recorded after Town10HD,
  some vehicles therefore drove onto sidewalks or into each other before being frozen. Boxes,
  depth and camera geometry are unaffected, but those scenes show less natural traffic. The
  car recorder (record_ego_drive.py) uses a fresh Traffic Manager per town

## Reproduce

```bash
# CARLA 0.9.16 release, unpacked; client wheel from its PythonAPI/carla/dist (cp311 for .venv)
uv pip install --python .venv ~/carla/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl
~/carla/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 &
python scripts/record_carla_weather.py --pilot --out /tmp/carla_pilot   # 1 scene, setup check
python scripts/record_carla_weather.py --scenes 300                     # ~20 min, data/carla/
python scripts/run_degradation_compare.py      # picks up data/carla automatically
python scripts/train_weather_detector.py       # adds the CARLA-trained detector and CARLA tests
```
