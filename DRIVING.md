# Driving study: the algorithm on a car's camera (Phases 1-2 of 6)

Aravind, 2026-11-25. Goal of the whole study: our algorithm runs on a car's camera. Rain or dust
corrupts what the camera sees. The algorithm recognizes the weather, repairs the frame and hands
the car the objects that matter (what is in its path, how far away), so the car drives more
safely. This document covers Phases 1 and 2: the car, its weathered camera and the algorithm
running on it. Phases 3-6 (a driving agent that brakes on what we detect, scenarios, the
collision experiment, a real dashcam check) are not done yet.

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
- **The repair step is the weak link.** In simulation, dehazing recovers a lot under dust
  (+0.07 to +0.14 of in-path objects found). On real dust photos it does not help, and slightly
  hurts. Nothing helps much on rain in either. The simulator's dust is the same scattering model
  the dehazer inverts, so the simulated gain is optimistic. Before Phase 4 can show that the car
  drives more safely in real weather, the repair needs a method that helps on real photos.

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

## What this means for Phases 3-6

- The car, the weather, routing, isolation and real-time speed are ready for the driving
  experiment
- With today's repairs, a CARLA driving test would likely show the algorithm helping in dust.
  That result would not carry over to real dust (see the real-photo check). Two options before
  Phase 4:
  1. Find or train a repair that helps on real dust and rain photos (e.g. fine-tune InstructIR
     or a dehazer on XWOD / DAWN-style real data), and keep the physics weather for simulation
  2. Make the simulated weather less "invertible" (dust that is not a uniform scattering
     haze: uneven density, particles, colour variation) so simulation is not tuned to the repair
- Phase 5 (real dashcam weather, e.g. ACDC, which needs a free registration) becomes more
  important: it is the check that decides whether a repair is credited

## Caveats

- Five towns, 15 drives. The held-out towns give 144 evaluation frames (6 drives), so intervals
  are wide for some cells
- The physics weather is uniform over the frame, with no wet roads, spray or lights
- The distance estimate assumes a flat road and a fixed camera tilt, so it will be worse on hills
  and when the car brakes hard
- The real-photo check uses traffic-camera and street photos (XWOD, DAWN), not dashcam footage
- Ground-truth masks were not saved, so mask quality is not scored (boxes are)
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
```

Repair weights download on first use from the authors' releases (Restormer GitHub release,
InstructIR Hugging Face) and are hash-checked. InstructIR's text encoder is
TaylorAI/bge-micro-v2 (Hugging Face).
