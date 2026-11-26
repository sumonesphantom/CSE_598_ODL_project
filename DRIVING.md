# Driving study: the algorithm on a car's camera

Aravind, 2026-11-25 to 2026-11-26. Goal of the study: our algorithm runs on a car's camera. Rain
or dust corrupts what the camera sees. The algorithm recognizes the weather, repairs the frame and
hands the car the objects that matter (what is in its path, how far away), so the car drives more
safely. This document covers:

- Phase 1: the car and its weathered camera;
- Phase 2: the algorithm running on it;
- the repair stage, redone;
- Phase 3: a driving agent and four hazard scenarios;
- Phase 4: the collision experiment;
- Phase 6: the statistics and a demo video.

Phase 5 (a real dashcam check) was skipped (see the conclusions).

**Short version.**
- Weather on the car's camera hurts detection a lot (mAP50 -0.20 to -0.44), so the setup can
  show whether the algorithm helps.
- Routing works. A condition detector that has also seen clean frames from the car's own camera
  routes 85 % of video frames correctly and reacts within 1 s. It sends only 9 % of clean frames
  to a repair, against 20-50 % for the detectors we had before.
- The full pipeline (route, repair, detect, isolate) runs at **16 frames/s** live in CARLA,
  with 99.9 % of frames under 100 ms.
- Isolation works. About 9 in 10 objects it flags as "in the car's path" really are, and the
  distances it estimates from the camera alone are off by about 10 % (median).
- **The repair step is the weak link.**
  - The first simulated dust was exactly the haze model that dehazing inverts, so simulation
    flattered that repair. It was replaced by a camera model that is harder to undo (weather v2).
  - A repair network was trained through the frozen detector on real weather photos.
  - Even so, **no repair reliably improves detection on real rain or dust photos**. Every
    test-set interval includes zero except one tiny gain.
- **Driving (736 simulated runs).**
  - With perfect perception the car never crashes.
  - With the car's detector and no repair it crashes in 29 % of the weather runs, all of them in
    heavy rain mist or a dust storm.
  - The algorithm lowers that to 27 % (-2.5 percentage points, 95 % CI [-5.4, +0.4]: not
    significant).
  - It clearly helps in moderate rain mist (crashes 58 % → 29 %), causes 3 crashes in light rain
    streaks, never brakes falsely and never harms clean driving.

## Phase 1: the car and its weather

**Recording** (`scripts/record_ego_drive.py`, `data/carla_ego/`, 2.1 GB, git-ignored)

- A Lincoln MKZ drives on autopilot among 50 cars and 120 pedestrians (30 % of them cross roads)
  in 5 towns (Town01, 02, 03, 05, 10HD), 3 drives of 60 s per town
- A front camera at bumper height (1.4 m, 1280x720, 90 deg), with CARLA's depth and
  instance-segmentation cameras at the same pose; one frame every 0.5 s
- 1,800 frames, 7,260 objects, 701 of them in the car's path (ahead, less than 2 m to either
  side, within 40 m)
- Ground truth for every object: its visible box, its class, its distance and lateral offset
  (from exact depth), and whether it is in the path. The car's own body is excluded
- A drive in which the car moved in fewer than 30-50 % of frames (stuck at lights or in a jam)
  is recorded again from a new start

**Weather**: the project's own physics effects (rain streaks, rain mist, raindrops on the lens,
dust haze, dust storm; levels 1-3), applied from CARLA's exact depth (`bolero.ego.apply_weather`).
CARLA's built-in weather barely affects a detector (CARLA.md), so it is not used.

**1.3: Does the weather hurt?** Yes, and enough to matter. These are frozen detectors on the
held-out towns (Town10HD, Town02; 144 frames):

| mAP50 | clean | rain L1 / L2 / L3 (change) | dust L1 / L2 / L3 (change) |
|---|---|---|---|
| Mask R-CNN (the car's detector) | 0.731 | -0.20 / -0.32 / -0.37 | -0.27 / -0.32 / -0.44 |
| DETR (the project's detector) | 0.506 | -0.16 / -0.24 / -0.28 | -0.24 / -0.32 / -0.40 |

Without repair, Mask R-CNN finds 84 % of in-path objects on clean frames. Under heavy rain mist
it finds 29 %, and under a dust storm 33 %.

## Phase 2: the algorithm on the car

All choices were made on the three training towns. All numbers below are from the two held-out
towns. Intervals are 95 % block-bootstrap intervals over 10-second stretches of each drive
(neighbouring frames are correlated).

### 2.1 Routing: which weather is it? (`scripts/run_car_routing.py`)

Each drive becomes a weather video: clean, then rain or dust rising from level 1 to 3, clean,
the other weather rising, then clean. That is 6-second segments and 4 changes of condition per
minute.

| Condition detector | frames routed correctly | clean frames sent to a repair | route changes /min | delay after weather starts |
|---|---|---|---|---|
| real-trained (smoothed) | 72 % | 21 % | 4.3 | 2.0 s |
| physics-trained (smoothed) | 83 % | 14 % | 4.5 | 1.3 s |
| combined (no smoothing / smoothed) | 74 % / 82 % | 49 % / 17 % | 10.8 / 8.7 | 0.0 / 1.3 s |
| CARLA-weather-trained (smoothed) | 44 % | 26 % | 5.7 | 4.5 s |
| **combined + clean car frames** | **85 %** | **9 %** | **4.2** | **1.0 s** |

- **New detector, `combined_car`** (`train_weather_detector.py`). It is the combined detector
  plus 360 clean frames from the car's own camera, from the training towns only. A car has plenty
  of clean footage of its own, and without it every detector mistook clean car frames for
  weather. It keeps the combined detector's skill on real photos: DAWN rain 0.96, DAWN dust 0.91,
  XWOD rain 0.57 (0.68 before), XWOD dust 0.37. The other detectors' results are unchanged
- **Smoothing over time** (`car_perception.ConditionRouter`: moving average plus hysteresis,
  tuned on the training towns) halves the flicker of the older detectors. `combined_car` does not
  need it: its route already changes as often as the weather does. It keeps its own per-frame
  rule, because tuning on drives whose clean frames it was trained on would be biased
- **What it misses**: raindrops on the lens (recognized 3 % of the time; they are also the least
  harmful rain effect) and light rain (57 % at level 1). Dust is recognized 100 % of the time,
  and rain is never confused with dust

### 2.2 Repair: which fix wins the image back? (`scripts/run_car_perception.py`)

Candidates (`bolero.learned_restoration`; official weights only, SHA-256 pinned, model code
vendored under MIT in `src/bolero/third_party/`):

- Restormer deraining (CVPR 2022)
- InstructIR (ECCV 2024), told either "remove the rain" or "remove the fog"
- Dark-channel-prior dehazing with a per-channel (yellow-brown) airlight
- The project's auto-contrast baseline

Chosen per weather on the training towns by mAP50. For Mask R-CNN: dust is repaired by
**dark-channel dehazing** and rain by **auto-contrast**. For DETR: dehazing and Restormer.

The routed pipeline compared with no repair, Mask R-CNN, held-out towns (in-path recall = share
of objects in the car's path that are found):

| change vs. no repair | level 1 | level 2 | level 3 |
|---|---|---|---|
| dust: in-path recall | +0.074 [0.025, 0.122] | +0.080 [0.027, 0.135] | +0.136 [0.078, 0.195] |
| dust: mAP50 | +0.039 [0.009, 0.064] | +0.060 [0.030, 0.088] | +0.108 [0.072, 0.142] |
| rain: in-path recall | +0.019 [0.000, 0.040] | +0.027 [0.000, 0.049] | +0.023 [0.000, 0.048] |
| rain: mAP50 | +0.015 [0.003, 0.025] | +0.021 [0.007, 0.033] | +0.005 [-0.004, 0.010] |

- **Routing loses almost nothing.** Routing on `combined_car` gives the same result as being told
  the true weather (oracle) to within 0.004. For DETR the routed gains are larger on rain
  (+0.07 to +0.11 in-path recall with Restormer)
- **The rain choice is fragile.** On the training towns, auto-contrast and Restormer were tied
  (mAP50 0.476 vs. 0.472). On the held-out towns Restormer would have found +0.09 more in-path
  objects than no repair, against +0.02 for auto-contrast. The pre-registered choice is
  reported above
- **InstructIR's dehazing does nothing for dust.** The learned dehazer was trained on grey fog
- **In simulation, routing is not even needed**: dehazing every frame (no router) finds
  0.602 of in-path objects under weather vs. 0.588 routed, and on clean CARLA frames it lifts
  in-path recall from 0.84 to 0.90 at the same mAP50

**Real-photo check** (`scripts/run_car_real_check.py`). Mask R-CNN on real photos, mAP50 change
vs. no repair, 95 % image-bootstrap CI:

| | COCO clean | XWOD rain | DAWN rain | XWOD dust | DAWN dust |
|---|---|---|---|---|---|
| no repair (mAP50) | 0.790 | 0.390 | 0.613 | 0.624 | 0.650 |
| dehazing, every image | +0.003 [-0.003, 0.008] | +0.037 [-0.020, 0.094] | -0.025 [-0.110, 0.009] | -0.046 [-0.094, 0.002] | -0.013 [-0.032, 0.002] |
| Restormer, every image | +0.002 | **+0.007 [0.002, 0.013]** | -0.002 | +0.003 | -0.008 |
| auto-contrast, every image | +0.000 | +0.026 [-0.004, 0.042] | -0.011 | -0.005 | -0.004 |
| routed (`combined_car`) | +0.001 | +0.005 [-0.002, 0.013] | -0.012 | **-0.026 [-0.059, -0.001]** | -0.010 |

- No repair hurts real clean photos, so routing is safe, but it is not needed for safety either
- **The simulator's dust gain does not carry over to real dust.** On real dust, dehazing is
  neutral to slightly harmful, and the routed pipeline costs 0.026 mAP50 on XWOD dust. The
  physics dust follows exactly the haze model that dark-channel dehazing inverts, so in simulation
  the repair partly undoes our own forward model. Real dust is not that uniform haze
- On real rain, only Restormer gives a significant gain, and it is tiny (+0.007)

### 2.3 Isolation: what matters to the car? (`bolero.car_perception.isolate`)

Each confident detection becomes an object with:

- a distance, from where its box meets the road, using only the camera's height and tilt (a
  flat-road assumption, nothing a real car would not have)
- a lateral offset and an in-path flag
- for the nearest in-path object, a "critical" mark

Mask R-CNN also returns each object's mask. Results with Mask R-CNN on the held-out towns:

| | clean | rain L1-3 (routed) | dust L1-3 (routed) |
|---|---|---|---|
| flagged in-path objects that really are in the path | 0.92 | 0.86-0.90 | 0.91-0.94 |
| true in-path objects flagged | 0.76 | 0.43-0.58 | 0.52-0.59 |
| frames whose most critical object is identified correctly | 0.74 | 0.42-0.56 | 0.49-0.58 |
| median distance error (objects within 40 m) | 9.8 % | 9-10 % | 8-9 % |

The isolation stage itself is accurate: what it flags is right, and the distances are good. What
limits it under weather is the detector missing objects, which is the repair stage's job.

### 2.4 Speed (`scripts/run_car_live.py`)

The stages alone, on the RTX 5080 laptop GPU, 1280x720:

| stage | ms |
|---|---|
| routing (features + random forest, single job) | 51; runs in a background thread, off the critical path |
| auto-contrast / dehazing at 640 px / dehazing full size | 21 / 25 / 75 |
| InstructIR at 640 px / full | 41 / 260 |
| Restormer at 480 / 640 px / full | 106 / 217 / 983 (too slow for real time) |
| Mask R-CNN fp32 / fp16 | 61 / 39 |
| DETR | 37-69 |
| isolation | < 0.2 |

The speed settings (dehazing at 640 px, Mask R-CNN in fp16) change detection by -0.002 in-path
recall and +0.003 mAP50 on average over all 15 weather settings (`results/car_live/fidelity.csv`).

**Live in CARLA**: the car drove 90 s through Town10HD while the weather changed every 8-10 s.
900 camera frames went through the full pipeline at 10 frames/s, with CARLA sharing the GPU:
**median 62 ms per frame (16 frames/s), 95th percentile 74 ms, 1 of 900 frames over 100 ms**.
The route matched the true weather on 94 % of frames. `results/car_live/live.mp4` shows the
weathered camera view beside the repaired view with the isolated objects (red = in the path,
thick = most critical).

## The repair, redone

Phase 2 ended with two problems. First, the simulated dust (v1) is exactly the haze model that
dark-channel dehazing inverts, so simulation flattered that repair. Second, no repair helped on
real dust photos. Both are addressed here, before any driving.

### Weather v2: a camera that is harder to undo (`bolero.ego.weather_v2`)

- **Dust** is no longer one uniform haze:
  - Visibility is set in metres (200 / 90 / 45 m for levels 1-3; 0.6 times that in a dust
    storm), and the extinction comes from CARLA's metric depth.
  - Density is uneven. A slowly drifting plume field scales it by 0.4 to 1.6 across the frame,
    and in a live drive it moves with time.
  - The haze colour depends on the sky (brighter and yellower towards the horizon and the sun),
    so there is no single airlight to estimate.
  - Light scattered forwards blurs distant objects.
  - A dust storm adds blowing grains.
- **Rain** keeps the v1 effects.
- **Every frame, clean ones included,** then goes through a camera stage: shot and read noise,
  then JPEG at quality 75, as a real camera would.

Under v2, dehazing no longer gets a free win on dust. On the held-out car frames, averaged over
dust levels 1-3, it adds +0.097 in-path recall under v1 but +0.008 under v2.

### A repair learned for the detector (`bolero.task_repair`, `scripts/train_task_repair.py`)

- **The network.** It is a small residual U-Net (267k parameters) with a global per-channel gain
  and offset. Its output layers start at zero, so before training it is exactly the identity.
- **Training.** It sits in front of the car's frozen Mask R-CNN and is trained to lower that
  detector's own loss. The detector never changes: all its weights are frozen and its batch-norm
  statistics are fixed. Only the preprocessing step is learned, so the project's rule of not
  retraining the recognizer holds.
- **Data.** Real XWOD training photos of rain and dust (586), mixed in every batch with 383 clean
  COCO street photos. A squared penalty on changing the clean photos (weight 10) and a smaller one
  on changing the weather photos (weight 1) keep it from harming the detector.
- **Model selection.** The checkpoint is chosen on the XWOD validation split. The best is at step
  2,500: +0.011 mAP50 on rain and +0.012 on dust, and +0.003 on held-out clean COCO.
- **Speed.** 14 ms per 720p frame in fp16, so it fits the real-time budget.

### Choosing the repair on real photos (`scripts/run_car_repair.py`)

The repair for each condition is now chosen on the **XWOD validation split (real photos)** and
then tested, untouched, on the XWOD test split, on DAWN (a different dataset) and on 300 clean
COCO photos. The table gives the change in Mask R-CNN mAP50 against no repair, with a 95 % image
bootstrap interval:

| | XWOD valid rain | XWOD valid dust | XWOD test rain | XWOD test dust | DAWN rain | DAWN dust | COCO clean |
|---|---|---|---|---|---|---|---|
| no repair (mAP50) | 0.690 | 0.410 | 0.390 | 0.624 | 0.613 | 0.650 | 0.790 |
| auto-contrast | +0.011 | -0.006 | +0.026 [-0.004, 0.042] | -0.005 | -0.011 | -0.004 | +0.000 |
| dehazing | **+0.049** [0.007, 0.109] | -0.022 | +0.037 [-0.020, 0.094] | -0.046 [-0.094, 0.002] | -0.025 | -0.013 | +0.003 |
| Restormer | +0.013 | +0.004 | +0.007 [0.002, 0.013] | +0.003 | -0.002 | -0.008 | +0.002 |
| learned repair | +0.006 | **+0.011** [-0.007, 0.031] | +0.005 [-0.004, 0.018] | -0.007 [-0.029, 0.009] | -0.003 | -0.004 | +0.004 |
| routed (chosen repairs) | | | -0.004 [-0.050, 0.038] | -0.018 [-0.068, 0.013] | -0.024 | -0.005 | +0.003 |

- **Chosen on the validation split:** dehazing for rain and the learned repair for dust. That
  choice is fixed for Phase 4, whatever the test photos say.
- **The test photos do not confirm either choice.** Every interval includes zero except
  Restormer's tiny gain on XWOD rain. The learned repair is the most even of the candidates
  (within ±0.007 on every set and never harmful on clean photos), but it does not reliably help
  either. In short, **no repair we have reliably improves detection on real rain or dust photos.**
- **The router does well on DAWN but not on XWOD test.** It recognizes DAWN's rain and dust 96 %
  and 91 % of the time, but XWOD test rain only 57 % of the time, and it sends 43 % of clean COCO
  photos to the rain repair. That is harmless here (+0.003), because dehazing barely changes clean
  photos.

On the car frames (held-out towns, mean over weather levels 1-3), in-path recall:

| | v1 dust | v2 dust | v1 rain | v2 rain | v2 clean |
|---|---|---|---|---|---|
| no repair | 0.538 | 0.555 | 0.534 | 0.510 | 0.830 |
| dehazing | 0.634 | 0.562 | 0.581 | 0.535 | 0.875 |
| learned repair | 0.562 | **0.570** | 0.543 | 0.524 | 0.841 |
| Restormer | | 0.551 | | **0.609** | 0.830 |

Restormer is the best repair for v2 rain, but at 0.1-1 s per frame it is too slow for the car
(2.4), so it was not a candidate for driving.

## Phase 3: a car that drives on what it sees

### The driving agent (`bolero.driving`)

The agent is deliberately simple and identical for every perception pipeline, so any difference
in safety comes from perception alone.

- **Steering** is pure pursuit along a route fixed at the start: points every 2 m along the
  straightest way ahead, crossing junctions straight over. Perception plays no part in steering.
- **Speed and braking.**
  - The car cruises at 10 m/s (36 km/h).
  - It brakes for the nearest object perception puts in its path, treating it as standing still.
    The deceleration needed to stop 5 m short is a = v² / (2 (d - 5)).
  - It brakes in proportion once that need passes 1 m/s², up to full braking (8 m/s²).
- **Against flicker.** An object must appear in 2 perception frames in a row before the car
  reacts. Once it has, the car remembers it for 0.5 s if it drops out of view.
- **Timing.** Perception runs at 10 Hz, and its decision takes effect 0.1 s later, for every
  pipeline including ground truth.

### The scenarios (`scripts/run_car_scenarios.py`)

The scenarios are staged in the two held-out towns (Town10HD, Town02), at 4 locations per town:
straight road with at least 110 m clear ahead, with the hazard point outside any junction and
next to a second lane. Nothing else moves.

| scenario | what happens | what a safe car does |
|---|---|---|
| pedestrian_cross | a pedestrian 2 m beyond the right edge of the lane walks in (1.6 m/s) when the car is 35 m away, and stops in the middle of the lane | stops |
| stopped_car | a car stands in the lane 70 m ahead | stops |
| lead_brake | a car 30 m ahead at the same speed brakes at 6 m/s² to a stop, 3 s in | stops |
| adjacent_car | a car stands in the next lane 70 m ahead | does **not** brake (false-braking check) |

**Check with ground-truth perception** (`--stage check`): the car stops safely in all 24 hazard
runs, 5.1 m short in almost all of them (it is designed to stop 5 m short), and drives past the
adjacent car without braking in 7 of 8 locations. Its largest distance from the planned route
was 3 cm.

**A bug found on the way.** The first version steered towards the nearest map waypoint, found
afresh every tick. Inside a junction that can be a turning lane, so at 2 of the 8 locations the
car turned off its route and hit roadside objects, even with perfect perception. Those crashes
would have been counted as perception failures. The route is now fixed at the start: at each
branch, each option is followed 12 m ahead and the one that stays on the start's heading line is
taken. Every run logs its largest distance from the route, and the analysis drops any location
where a run strayed more than 1.5 m. The whole experiment was rerun after the fix.

## Phase 4: does the car drive more safely?

Every scenario at every location (2 towns × 4 locations × 4 scenarios) was driven under 11 weather
variants: clean, plus each of the 5 weather effects at levels 2 and 3 (weather v2). Each variant
was driven with each perception pipeline:

- **ground truth**: CARLA's own object labels, blind to the weather. This is the upper bound.
- **no repair**: the car's detector (Mask R-CNN) on the weathered frame.
- **our algorithm**: the condition router, then the repair chosen for that condition (dehazing
  for rain, the learned repair for dust, nothing for clean), then the detector.

That is 736 runs (`results/car_scenarios/`). In "no repair" vs. "our algorithm" pairs, the
scenario, location and weather are the same, so the comparison is paired. Intervals are 95 %
bootstrap intervals over those pairs. No location had to be dropped: the car never strayed more
than 6 cm from its route.

### Collisions

| crash rate | no repair | our algorithm | change [95 % CI] | crashes prevented / caused |
|---|---|---|---|---|
| clean (24 hazard runs) | 0 % | 0 % | 0 | 0 / 0 |
| rain (144) | 26.4 % | 23.6 % | -2.8 pp [-7.6, +2.1] | 8 / 4 |
| dust (96) | 33.3 % | 31.3 % | -2.1 pp [-5.2, 0.0] | 2 / 0 |
| all weather (240) | 29.2 % | 26.7 % | -2.5 pp [-5.4, +0.4] | 10 / 4 |

With ground-truth perception the car never crashes, under any weather (24 of 24 hazard runs
safe, always about 5 m short). Every crash therefore comes from perception.

By weather effect (hazard runs, levels 2 and 3 together, 48 pairs each):

| effect | no repair | our algorithm | change [95 % CI] |
|---|---|---|---|
| rain streaks | 0 % | 6.3 % | **+6.3 pp [0, +14.6]**: 3 crashes caused |
| rain mist | 79.2 % | 64.6 % | **-14.6 pp [-27.1, -4.1]**: 8 prevented, 1 caused |
| raindrops on the lens | 0 % | 0 % | 0 |
| dust haze | 0 % | 0 % | 0 |
| dust storm | 66.7 % | 62.5 % | -4.2 pp [-10.4, 0] |

- **Only heavy weather causes crashes.** Rain mist and dust storms cause them; light streaks,
  lens drops and dust haze do not, even without repair. In those effects the detector still sees
  the hazard in time, so the repair has nothing to win back.
- **The one clear gain is in moderate rain mist.** At level 2, crashes fall from 58 % to 29 % (14
  to 7 of 24). Rain mist is a haze-like veil, which dehazing (the repair chosen for rain) removes.
  At level 3 both pipelines crash every time: the repair cannot recover enough.
- **The algorithm causes crashes in light rain streaks** (3 of 48, all in Town10HD location 1).
  The router correctly calls them rain and sends them to dehazing, which suits mist but not
  streaks, and in those runs the detector lost the car ahead.
- **Dust storms barely change** (2 prevented, none caused). The learned repair is mild by design.
- **Where it crashes, it hits about as hard.** Mean impact speed among crashes is 27-28 km/h for
  both pipelines, against a cruising speed of 36 km/h.
- **By scenario:** in dust, almost all crashes are with the stopped car (47-50 %) and the braking
  lead car (47 %); the pedestrian is hit in only 0-3 % of dust runs. In rain, the three hazards
  are hit about equally often (19-27 %). The algorithm's gain is largest for the pedestrian
  (rain: 27 % → 19 % crashes).

### False braking

In the adjacent-car scenario (a car in the next lane, which needs no braking), no pipeline
stopped or crashed. Brief braking taps (about 0.1-0.35 s on average) happen in 4-12 % of runs for
every pipeline, ground truth included, so the algorithm adds no false braking.

### What Phase 4 shows

**In simulation, the algorithm makes the car slightly safer overall, but not significantly**
(-2.5 percentage points of crashes, interval [-5.4, +0.4]). It clearly helps in one condition
(moderate rain mist), slightly harms in another (light rain streaks), and never harms clean
driving. The pattern follows from the repair stage: routing and isolation work, but the repairs
help only where the weather looks like what they undo. Combined with the real-photo results,
where no repair reliably helps, the evidence does not support a claim that the algorithm makes a
car safer in real rain or dust. What it does show is a working end-to-end pipeline, a measurable
safety gain where the repair matches the weather, and that a mismatched repair can cost safety.

### Demo video (`results/car_scenarios/demo/demo.mp4`)

Five cases, each with the unrepaired car (left) beside the full algorithm (right), same scenario
and weather. Each side shows the camera view with the objects found (red = in the car's path),
and a title gives the run's outcome:

1. Pedestrian crossing in rain mist (level 2), Town10HD: without repair the car hits the
   pedestrian; with the algorithm it stops.
2. Stopped car in a dust storm (level 2), Town02: the algorithm stops in time; no repair does not.
3. Braking lead car in rain streaks (level 3), Town10HD: the algorithm causes the crash.
4. Stopped car in heavy rain mist (level 3), Town10HD: both crash.
5. Pedestrian crossing in clear weather, Town10HD: both stop.

Re-render with `python scripts/run_car_scenarios.py --stage demo --demo-runs <run ids>`; add
`--show` to watch the drive live, with CARLA's camera following the car.

## Conclusions and what is not covered

- **What works:** the car's camera, the weather, routing (85 % of frames correct, 9 % of clean
  frames sent to a repair), isolation (about 90 % precise, about 10 % distance error) and
  real-time speed (16 frames/s). In simulation the whole loop runs: detect, route, repair, isolate,
  brake.
- **What does not yet:** the repair. No repair we tried, including one trained through the
  detector on real weather photos, reliably improves detection on real rain or dust photos. In
  the driving experiment the algorithm prevents crashes only where the repair matches the
  weather (moderate rain mist), and the overall gain (-2.5 percentage points) is not
  significant.
- **Phase 5 (real dashcam photos, ACDC) was skipped** at the project's decision: it needs a
  dataset registration, and the real-photo evidence already on hand (XWOD, DAWN) points the same
  way. The real-world evidence is therefore from traffic-camera and web photos, not a car's
  camera, and it is about detection only, not driving.
- **The most promising next step** is a repair that is specific to the kind of weather, not only
  its family. Dehazing helps mist but hurts streaks, so a router that separates mist from streaks
  (or a repair that does both) would remove the crashes the algorithm causes today.

## Caveats

- Five towns, 15 drives. The held-out towns give 144 evaluation frames (6 drives), so intervals
  are wide for some cells
- The physics weather is uniform over the frame, with no wet roads, spray or lights
- The distance estimate assumes a flat road and a fixed camera tilt, so it will be worse on hills
  and when the car brakes hard
- The real-photo check uses traffic-camera and street photos (XWOD, DAWN), not dashcam footage
- Ground-truth masks were not saved, so mask quality is not scored (boxes are)
- The driving experiment covers 8 locations in 2 towns, one car speed (36 km/h), straight roads
  and one hazard at a time, with nothing else moving. The driving agent is deliberately simple
  (no tracking, so a moving lead car is treated as standing still). Intervals are over 24-48
  paired runs per effect, so effects smaller than about 5 percentage points cannot be resolved
- The demo video re-drives the chosen runs rather than replaying them. All 10 re-drives ended as
  in the experiment (same crashes and safe stops)
- Found on the way: one Traffic Manager reused across towns keeps the first town's road map and
  steers cars into buildings. The car recorder uses a fresh one per town. The traffic-camera set
  in CARLA.md was affected (see its caveats)

## Reproduce

```bash
~/carla/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 &
python scripts/record_ego_drive.py                 # ~45 min with re-drives -> data/carla_ego/
python scripts/train_weather_detector.py           # adds combined_car and the car tests
python scripts/run_car_routing.py                  # 2.1, a few minutes
python scripts/run_car_perception.py               # 1.3, 2.2, 2.3; ~2 h of inference, resumable
python scripts/run_car_real_check.py               # 2.2 on real photos, ~15 min
python scripts/run_car_live.py --benchmark         # 2.4 stage timings and speed-setting check
python scripts/run_car_live.py --seconds 90        # 2.4 live drive + video (needs CARLA)
python scripts/train_task_repair.py                # learned repair, ~20 min on the GPU
python scripts/run_car_repair.py                   # choose repairs on real photos, check on car frames, ~1.5 h
python scripts/run_car_scenarios.py --stage check  # Phase 3: ground-truth driving at every location, ~4 min
python scripts/run_car_scenarios.py                # Phase 4: 736 runs, ~5 h, resumable; then the analysis
python scripts/run_car_scenarios.py --stage demo --show --demo-runs <run ids>   # demo video, watched live
```

Repair weights download on first use from the authors' releases (Restormer GitHub release,
InstructIR Hugging Face) and are hash-checked. InstructIR's text encoder is
TaylorAI/bge-micro-v2 (Hugging Face).
