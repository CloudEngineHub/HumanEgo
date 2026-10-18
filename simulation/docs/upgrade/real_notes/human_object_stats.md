# HumanEgo serve_bread: bread / plate geometry and layout in the human data vs. the sim sampler

Date: 2026-10-17. Source: `/home/leo/leo/HumanEgo/data/serve_bread/aria` (61 recordings, read-only;
nothing written under `data/`, `data/water_flowers` never touched). Per-episode numbers:
`human_object_stats.json` (same directory). Scripts: `../scripts/measure_episode.py` (per-episode
measurement), `../scripts/analyze.py` (summaries + sim comparison), `../scripts/contact_sheets.py`
(RGB thumbnails, `../img/sheet_*.jpg`).

## 1. What the released data holds (per episode)

`mps_serve_bread_NNN_vrs/`
* `sample.vrs` (raw Aria, ~0.5 GB), `slam/` (MPS: `closed_loop_trajectory.csv` with
  `gravity_world = (0, 0, -9.81)` in every episode, i.e. the MPS world frame is metric and
  gravity aligned with +z up; `semidense_points.csv.gz` ~50-110 k points;
  `semidense_observations.csv.gz` ~10 M rows of (point uid, timestamp, camera, u, v)),
  `hand_tracking/` (MPS wrist/palm), `else/` (VRS health checks).
* `preprocess/` (pipeline output, `preprocess/Preprocess.py`):
  * `aria_cam_rgb_config.json`: rectified pinhole, 640x480, f = 320 px (90 deg HFOV), no distortion.
  * `aria_phases_results.json`: walk / transition / manipulation windows.
  * `kptsselector_results.json`: 20 2D contour keypoints per object (`obj1` = "piece of bread .",
    `obj2` = "a plate .") placed on the DINO+SAM2 mask after closing, erosion and a 5-px
    inner-edge erosion (about 4 px inside the true silhouette on each side).
  * `cotracker_results.json`: their 2D tracks over the object-centric window.
  * `camtriangulator_results.json`: `cam0_c2w` and, per object, the 20 keypoints triangulated
    (multi-view DLT + Huber BA, every 3rd frame) in MPS world and in cam0, plus a pca1/vlm pose.
    The object-centric window is the up-to-60 approach frames before the first manipulation
    frame (`CoTracker.object_centric_max_frames: 60`); cam0 is its first frame, so the objects
    are static and the person is still walking in.
  * `all_data/<idx>/`: `rgb.png`, `aria_cam_rgb.json` (K, `c2w = T_world_device @ T_device_cam`),
    `aria_slam.json`, `aria_hands.json`, `aria_phases.json`; from cam0 on also `mask_obj1/2.png`,
    `mask_arm*.png` and the arm-inpainted / keypoint-overlay RGBs; from the first manipulation
    frame on `training_data.json` (c2w, K, right-hand `T_hand_to_world` + grasp flag,
    `T_obj_to_world` for obj1/obj2, anchor = obj1).
* There is no depth for the human data: all metric 3D comes from SLAM poses (triangulated
  keypoints, semidense points).
* Training (`cfg/training/serve_bread/HumanEgo.yaml`): `data_sources: {aria: 60}`, eval = episode
  000, train = 001-060; `frame_mode: camera_frame` (positions are expressed in the current camera
  frame), image `rgb_WoArm_WArmObjKpts.png` resized to 240x320.

Scenes (from RGB contact sheets `../img/sheet_t0_*.jpg`, `../img/sheet_bread_*.jpg`):

| group | episodes | setting | person |
|---|---|---|---|
| A | 000-038 | bread + plate on a cardboard box (Trossen robot box) standing on a lab desk; 038 on a black box top | standing |
| B | 039-044 | directly on the light butcher-block desk | seated (knees visible) |
| C | 045-060 | black kitchen-style countertop | seated |

Bread type: the same straight rolled croissant (orange egg wash, 6-7 chevron bands) in all 61
episodes; there is no other bread in the human data. Plate: the same light-blue plate in all.
Layout is always plate on the left, croissant on the right, croissant long axis roughly
left-right in the image.

Data issues found:
* ep 060: `all_data/*/aria_cam_rgb.json`, `aria_cam_rgb_config.json` and `rgb.png` were
  re-extracted at 1920x1080 (f = 960) after preprocessing, while tracks, masks and
  `training_data.json` are 640x480 with f = 320 (the triangulated keypoints reproject onto the
  2D seeds with 1.1 px error using f = 320, 947 px using f = 960). The 3D data are fine; any tool
  that reads K from `aria_cam_rgb.json` for this episode gets the wrong intrinsics. Included.
* ep 033: the bread's triangulated contour is implausible (19.6 cm horizontal extent; it sits
  7 cm above the support surface found below it). Excluded from all statistics (n = 60).

## 2. Simulator assumptions (simulation/)

* World frame (`configs/serve_bread.yaml`, `humanego_sim/scene.py`): origin on the table top
  directly below the camera, +x forward (away from the robot), +y left, +z up. Camera from the
  real hand-eye calibration `T_base_in_cam`, placed at `cam_height` = 0.436 m: optical axis
  pitched 37.0 deg below horizontal, zero yaw (looks along +x) and roll. Camera RealSense D405,
  640x480 after centre crop, fx = 440 px (72 deg HFOV; the Aria training images are 90 deg).
  Arm bases at (0.12, +-0.324, 0.215).
* Sampler `humanego_sim/sampling.py::sample_scene(rng, mode, difficulty=1.0)` (used by
  `evaluate.py`, `build_web.py`, autotune). All poses are [x, y, yaw_deg] in that world frame,
  i.e. relative to the camera foot and camera heading:
  * `single` (paper protocol, right arm): plate x = 0.47 + U(-0.07, 0.07), plate y =
    0.11 + U(-0.06, 0.06), plate yaw U(0, 360) (irrelevant, round); bread x = 0.47 +
    U(-0.08, 0.08) (independent of the plate), bread y = plate_y - gap with gap = 0.26 +
    U(-0.10, 0.12) (0.16-0.38 m, centre-to-centre along y), bread yaw = 90 + U(-35, 35) deg.
  * `bimanual` (config default `mode: bimanual`, the teaser): plate x = 0.47 + U(-0.06, 0.06),
    y = U(-0.05, 0.05); right bread x = 0.46 + U(-0.08, 0.08), y = plate_y - (0.27 +
    U(-0.07, 0.08)), yaw 90 +- 35; left bread mirrored.
  * Fixed config layouts: bimanual plate (0.47, 0) / breads (0.46, -+0.27, 95/85 deg); single
    plate (0.47, 0.11), bread (0.47, -0.15, 95 deg). Unused constant `X_RANGE = (0.36, 0.56)`.
  * Bread yaw convention: yaw 90 = long axis along world y = left-right in the image.
* Sizes: croissant bounding box 13.8 x 6.2 x 5.0 cm (`scripts/bread_gen.py`, from the
  triangulated contours: "max extent 12.8 cm inner-eroded, ~13.8 outer"), 50 g; danish "bun"
  14.0 x 7.4 x 6.0 cm (robot rollouts only, not in the human data), 75 g; plate R = 12.25 cm
  (24.5 cm), rim height 2.2 cm, 450 g (`scripts/build_assets.py`, README "Object sizes and
  masses"). Breads rest on the table (centre at z = H/2).

## 3. Method (per episode)

Frames and reference times
* t0 = first `training_data.json` frame (start of the manipulation phase = the first observation
  the policy is trained on; the person has stopped walking, the right hand is visible but
  18-47 cm from the bread, median 24 cm, i.e. before contact). Grasp onset = first frame with the
  right-hand grasp flag > 0.5 (t0 to grasp: median 6.8 s).
* cam0 = first object-centric frame (start of the approach; reported for reference only).
* Frame **H** (comparable to the sim world): origin = camera optical centre dropped vertically
  onto the support plane, +x = camera optical axis projected on the horizontal plane (head
  heading), +y left, +z up. Built at t0 from the MPS c2w (MPS world is gravity aligned, so H is
  table aligned up to the support tilt, mostly < 2 deg, see below). Also reported: the
  OpenCV camera frame at t0 (what the policy sees, `frame_mode: camera_frame`) and the angles /
  pixel position of the objects in the 640x480, f = 320 Aria image.

Measurements
* Support plane (the surface the bread rests on: box top, desk or counter): mode of semidense
  point heights in a 25 cm ring around the bread (outside the bread hull dilated 2 cm and outside
  the plate disc), restricted to a 3 cm window just below the 10th percentile of the bread
  contour heights. A free search lands on the wrong surface in a third of the episodes (desk
  under the box, a lower box, the floor), hence the window; it was calibrated on the 43 episodes
  where the free search was self-consistent (support = contour p10 - 1.2 cm, p5-p95 0.7-1.7 cm).
  Found in all 61 episodes from 12-1341 ring points (ep 046 only 12). Fitted tilt 0.0-1.9 deg
  except 048 (3.4), 056 (2.9) and 046 (33.7, meaningless with 12 points).
* Bread centroid and yaw: horizontal PCA of the 20 triangulated contour keypoints (yaw = long-axis
  angle from H +x towards +y, folded to [0, 180); 90 = left-right as in the sim). Cross-check:
  PCA of the mask silhouette back-projected onto a horizontal plane (below).
* Bread length / width: (a) mask silhouette of 5 frames spread over the first half of t0..grasp,
  back-projected onto the horizontal plane at mid bread height, robust (0.5-99.5 %) extents along
  its PCA axes, median over the 5 frames; the width is corrected for the silhouette inflation by
  the bread's side seen at elevation theta (footprint = sqrt(W^2 + (H cot theta)^2) along the view
  direction); (b) contour keypoint extents + 2 x 4 px erosion margin at the cam0 range. Raw
  contour extents are lower bounds.
* Bread height: 95th percentile of the heights above the support of semidense points inside the
  bread contour hull **that were last observed before grasp onset** (filtering on
  `semidense_observations.csv.gz`; without it the hand and the lifted bread dominate). 41-245
  points per episode.
* Plate diameter and centre: plate mask at t0 back-projected onto the plane 2.0 cm above the
  support (rim height; sim 2.2 cm); robust and full extents. Centre = silhouette centroid. The
  contour circle fit is reported but is biased low (inner-eroded contour, the far rim
  triangulates poorly: 20.6 +- 1.4 cm).
* Camera: optical-centre height above the support, pitch (optical axis below horizontal), roll
  at t0; plus medians over the pre-grasp window.
* Bread-to-plate vector: plate centre minus bread centroid in H (dx forward, dy left), distance,
  direction (atan2(dy, dx); 90 = plate straight to the left).
* Sim side: 20 000 draws of `sample_scene(..., difficulty=1)` per mode, mapped through the sim
  camera pose (bread centre at z = 2.5 cm, plate centre at z = 1.1 cm, same as the human side).

## 4. Results (n = 60; ep 033 excluded). Units cm / deg.

Full tables (all quantities, per scene group, sim draws): appendix A. Per-episode values:
`human_object_stats.json` -> `episodes[]`.

### 4.1 Object sizes

| quantity | human mean +- std | p5 - p95 | [min, max] | sim |
|---|---|---|---|---|
| croissant length, mask silhouette | 14.6 +- 0.5 | 14.0 - 15.4 | [13.9, 16.1] | 13.8 |
| croissant length, contour + 4 px erosion | 14.5 +- 0.4 | 14.0 - 15.3 | [13.8, 15.6] | |
| croissant length, raw contour (lower bound) | 12.8 +- 0.6 | 11.9 - 13.7 | | |
| croissant width, mask, height-corrected | 7.1 +- 0.8 (median 7.3) | 5.4 - 8.1 | [5.0, 8.5] | 6.2 |
| croissant width, raw contour (lower bound) | 4.8 +- 0.9 | 3.8 - 6.6 | | |
| croissant height, pre-grasp semidense p95 | 5.7 +- 0.8 (median 5.4) | 4.9 - 7.3 | [4.8, 7.8] | 5.0 |
| plate diameter, rim silhouette (robust 0.5-99.5 %) | 24.8 +- 0.4 | 24.3 - 25.4 | [24.2, 25.6] | 24.5 |
| plate diameter, rim silhouette (full extent) | 25.9 +- 0.4 | 25.4 - 26.5 | | |
| plate diameter, contour circle fit (biased low) | 20.6 +- 1.4 | | | |

Group means of croissant length (mask): A 14.3, B 14.7, C 15.3 cm (the dark counter in group C
probably merges shadow into the mask; the contour-based length differs less, 14.4 / 14.5 /
14.8). Best single estimate for the real croissant: **~14.5 x ~7 x ~5.4 cm** (L x W x H), i.e.
about 5 % longer, 15 % wider and 8 % taller than the sim's 13.8 x 6.2 x 5.0. Width is the least
certain number (+-0.8 cm; it depends on the height correction). Plate: **~24.8 cm** (25.9 cm
including the outer lip), sim 24.5 cm.

### 4.2 Head camera at t0 vs. the fixed sim camera

| quantity | human all | A (standing, box) | B (seated, desk) | C (seated, counter) | sim |
|---|---|---|---|---|---|
| height above support | 43.7 +- 3.7 [36.9, 51.9] | 46.2 +- 1.4 | 42.1 +- 0.7 | 38.3 +- 0.9 | 43.6 |
| pitch below horizontal | 34.3 +- 3.5 [27.5, 40.4] | 34.2 +- 3.3 | 37.2 +- 1.7 | 33.4 +- 3.9 | 37.0 |
| roll | 0.7 +- 1.4 [-1.9, 4.9] | 0.1 | 3.3 | 1.1 | 0 |
| horizontal FOV | 90 (Aria rectified, f = 320 @ 640x480) | | | | 72 (D405, f = 440) |

The head is nearly static between t0 and grasp: pre-grasp-window medians equal the t0 values
(height 43.7, pitch 34.1, bread range 57.4). At cam0 (start of the approach walk) the bread is
53 +- 14 cm ahead instead of 36 cm.

### 4.3 Layout in frame H at t0 (x = forward along the camera heading from the camera foot, y = left)

| quantity | human mean +- std | p5 - p95 | [min, max] | sim single: range (mean +- std) | sim bimanual, right bread |
|---|---|---|---|---|---|
| bread x | 36.2 +- 4.2 | 29.9 - 44.1 | [27.8, 49.5] | [39, 55] (47.0 +- 4.6) | [38, 54] |
| bread y | -15.2 +- 7.0 | -29.2 - -8.1 | [-34.1, -0.8] | [-32.9, 0.9] (-15.9 +- 7.2) | [-39.8, -15.1] (-27.6) |
| plate x | 35.4 +- 6.3 | 26.4 - 45.2 | [23.3, 54.3] | [40, 54] (46.9 +- 4.0) | [41, 53] |
| plate y | 11.5 +- 6.4 | -1.1 - 19.8 | [-8.3, 23.6] | [5, 17] (11.0 +- 3.5) | [-5, 5] |
| bread yaw (contour PCA) | 85.0 +- 10.3 | 67.2 - 100.8 | [61.7, 107.2] | U[55, 125] (90 +- 20.1) | U[55, 125] |
| bread yaw (mask PCA, cross-check) | 91.3 +- 11.0 | 72.5 - 106.2 | [64.4, 116.0] | | |
| plate - bread, dx (forward) | -0.8 +- 5.9 | -10.2 - 9.5 | [-13.2, 12.6] | [-14.9, 14.7] (0.0 +- 6.1) | [-12.8, 14.8] |
| plate - bread, dy (left) | 26.7 +- 3.0 | 21.6 - 30.1 | [20.8, 34.5] | [16, 38] (27.0 +- 6.3) | [20, 35] |
| bread-plate centre distance | 27.3 +- 3.1 | 22.5 - 31.3 | [21.5, 34.5] | [16.0, 40.5] (27.7 +- 6.2) | [20.0, 37.5] |
| direction bread -> plate (90 = left) | 91.8 +- 12.4 | 72.7 - 112.8 | [64.3, 116.6] | [47.6, 131.4] (90 +- 13.6) | [55, 122] |

By group, bread y: A -10.9 +- 2.7, B -17.9 +- 6.0, C -24.3 +- 4.9 (in C the wearer looks more at
the plate). Bread yaw from the two methods differs by 6.3 +- 4.9 deg (mask PCA is biased by the
silhouette of the bread's side); treat the yaw mean as 85-91 deg and its spread as ~10 deg.

### 4.4 What the camera sees at t0 (OpenCV camera frame; angles = atan(x/z), atan(y/z), + = right / down)

| quantity | human mean +- std | p5 - p95 | sim single mean (range) | human inside sim range |
|---|---|---|---|---|
| bread depth z_c | 52.8 +- 3.8 | 47.7 - 59.8 | 62.2 (55.9 - 68.7) | 20 % (80 % closer) |
| bread range | 57.2 +- 3.4 | 52.7 - 63.7 | 64.9 (56.7 - 75.6) | 53 % (47 % closer) |
| bread y_c (down) | 13.7 +- 3.7 | 7.8 - 19.2 | 4.6 (-0.3 - 9.4) | 13 % (87 % lower) |
| bread x_c (right) | 15.1 +- 6.9 | 7.9 - 29.0 | 15.9 (-0.9 - 32.9) | 98 % |
| bread elevation below optical axis | 14.5 +- 3.6 | 9.1 - 20.4 | 4.3 (-0.2 - 9.5) | 12 % |
| bread azimuth | 15.9 +- 7.4 | 8.2 - 30.5 | 14.2 (-0.8 - 30.2) | 93 % |
| bread pixel (Aria 640x480, f = 320) u, v | 413, 322 (median 394, 322) | v 291 - 359 | 403, 264 (v 238 - 293) | v: 12 % |
| plate depth z_c | 53.2 +- 4.7 | 46.6 - 61.4 | 63.1 (57.5 - 68.7) | 13 % |
| plate elevation | 16.0 +- 6.3 | 5.2 - 24.8 | 5.3 (1.2 - 9.7) | 20 % |
| plate azimuth | -12.4 +- 7.0 | -22.2 - 1.0 | -9.9 (-16.4 - -4.2) | 60 % |

The same sim bread projected with the D405 intrinsics lands at u 433 +- 52, v 274 +- 22 px.

### 4.5 Other t0 quantities

Right hand at t0 (MPS hand tracking, frame H): x 34 +- 7, y -34.5 +- 7, z 18 +- 8 cm, 26 +- 6 cm
from the bread centre (min 18 cm, no contact). t0 to grasp onset: 6.6 +- 2.0 s.
Joint test: only 2 of 60 human t0 layouts (3 %) fall inside the single-mode sampler's box on all
of (bread x, plate x, plate y, gap, yaw); bread x alone is inside for 22 %, plate x for 22 %,
plate y for 70 %, gap and yaw for 100 %.

## 5. Sim sampler vs. human data: verdict per quantity

| quantity | verdict |
|---|---|
| bread / plate forward distance (x) | **offset**: sim 10-11 cm further (bread 47 vs 36 cm median; 78 % of human episodes are closer than the sim minimum 39 cm). Same in camera frame: depth 62 vs 53 cm. |
| image elevation of the objects | **offset** (consequence of the above + 3 deg steeper sim pitch): bread 4 deg below the optical axis in sim vs 14.5 deg in the human data; v = 264 vs 322 px; 88 % of human episodes are lower in the image than anything the sim samples. |
| bread lateral y (single mode) | **match** (-15.9 +- 7.2 vs -15.2 +- 7.0; 98 % inside). |
| plate lateral y (single mode) | inside but **narrower** (sim 5-17, human -1-20 p5-p95; 30 % of human outside, both sides). |
| bread-plate gap / distance | **sim ~2x wider**: dy 16-38 (std 6.3) vs human 21-34 (std 3.0). Gaps below ~20 cm (18 % of sim draws) never occur in the human data; at 16 cm the croissant touches the plate rim (12.4 + 3.5 cm). dx matches (0 +- 6 vs -1 +- 6). |
| bread yaw | **sim ~2x wider**: U(55, 125), std 20, vs human 85-91 +- 10 (p5-p95 67-106). Covers all human cases; about half of the sim draws are outside the human p5-p95. |
| camera height | **match on average** (43.6 vs 43.7) but human is three clusters: 46 (standing at box), 42, 38 (seated). |
| camera pitch | **inside**, sim at the steep end: 37.0 vs 34.3 +- 3.5 (27.5-40.4). |
| camera FOV | **different**: 72 deg (D405) vs 90 deg (Aria) - not a sampler issue, but pixel positions and apparent sizes differ by 1.375x. |
| croissant size | **sim smaller**: 13.8 x 6.2 x 5.0 vs ~14.5 x 7.1 x 5.4 (L +5 %, W +15 %, H +8 %). |
| plate size | **match**: 24.5 vs 24.8 (rim silhouette). |
| bread type | human: one croissant only; sim bimanual adds the danish loaf (no human counterpart). |
| bimanual mode (config default) | plate at y = 0 +- 5 and right bread at y = -27.6 +- 5 (azimuth 24 deg): the whole layout sits ~12 cm to the right of the human single-bread layout (85 % of human plates outside the bimanual plate-y range). |

### Most important mismatches (ranked)

1. **Objects too far forward / too high in the image.** The human t0 snapshot has the croissant
   at x = 36 +- 4 cm ahead of the camera foot (camera depth 53 cm, 14.5 deg below the optical
   axis); the sim samples x = 39-55 cm (depth 62 cm, 4 deg below the axis). Only 22 % of human
   episodes are inside the sim's x range, 12-13 % inside its image-elevation / y_c range. Since
   the policy works in the camera frame (`frame_mode: camera_frame`), this is a direct
   input-distribution shift: the human croissant at t0 maps to sim world (x, y) ~ (0.34, -0.15).
   Matching it needs bread x ~ U(0.29, 0.45) and plate x ~ U(0.26, 0.46) (human p5-p95), i.e.
   below the sampler's current 0.39 / 0.40 minimum and below the unused `X_RANGE` 0.36; check
   reach of the WidowX AI from the bases at x = 0.12 before adopting (in view it is: at
   x = 0.30 the croissant is ~17 deg below the D405 axis, half-FOV 28.6 deg).
2. **Bread-plate gap range too wide** (16-38 vs 21-34 cm, std 6.3 vs 3.0) and **yaw range too
   wide** (+-35 vs ~+-17 deg at p5-p95, std 20 vs 10). The sim evaluates many layouts the policy
   never saw; if the goal is "in-distribution" validation, use gap ~ U(0.21, 0.31) (or
   N(0.267, 0.03)) and yaw ~ N(87, 10) clipped to [65, 110]; keep the wide ranges as a separate
   out-of-distribution test.
3. **Croissant ~5-15 % too small**: use ~14.5 x 7.0 x 5.4 cm. (The sim value 13.8 cm came from
   the inner-eroded contour + ~1 cm; the full silhouette is 14.5-14.6 cm; the 6.2 cm width
   ignored that the 3D silhouette contour sees the cross-section obliquely.)
4. **Camera height / pitch are a single point** (43.6 cm, 37 deg) while the human data spans
   37-52 cm and 27-40 deg in three clusters; the pitch is at the human p75. Randomising the
   rig is not possible on the real robot, but the sim could sample camera height 0.37-0.48 and
   pitch 29-40 deg to test sensitivity.
5. **Default mode is bimanual**, whose layout (plate centred, bread 27 cm to the right,
   azimuth 24 deg) is not what the human single-bread demonstrations show (plate 13 deg left,
   bread 13-16 deg right of the gaze). Compare against the human data in `single` mode.

## 6. What is and is not comparable

* **The human camera is head-mounted and moves.** The person walks in (cam0 is up to 0.9 m away),
  stops, and then holds the head almost still until the grasp (pitch range over t0..grasp:
  median 1.8 deg, max 4.3 deg). The policy's training frames start at t0, so t0 is the human analogue
  of the robot's fixed view; during manipulation the head moves again, so the training
  distribution of camera-frame positions is wider than the t0 snapshot reported here.
* **There is no table frame in the human data.** The three scenes have different surfaces and
  no table-edge reference, and the head yaw at t0 is free, so "position on the table" can only be
  expressed relative to the head camera (frame H, built from the gravity-aligned MPS world) or
  relative between objects. In the sim, world x is both the table/robot axis and the camera
  heading (camera yaw 0), so sim positions are table-relative and camera-relative at once.
  Comparable without caveats: sizes, bread-plate distance, camera height and pitch (rotation
  invariant), and everything in the camera frame. Comparable "as seen from the camera": x/y in
  H, dx/dy, bread yaw (all depend on the head yaw at t0, which is where the person chose to
  look, typically between the two objects).
* **Not comparable**: the robot base vs. the human body. The human's right hand at t0 hovers
  ~35 cm to the right, 34 cm ahead and 18 cm above the table; the robot's right base is at
  (0.12, -0.32, 0.21). Shoulder / reach geometry of the human was not measured.
* **Camera height and pitch depend on the scene** (standing at a box on a desk: 46 cm; seated
  at desk: 42 cm; seated at counter: 38 cm). The sim's single rig (43.6 cm, 37 deg) sits between
  the clusters, so per-group comparisons (appendix A) are more meaningful than pooled ones for
  anything involving camera height.
* **Image comparisons need the same intrinsics.** Training images are the 90-deg rectified Aria
  pinhole (f = 320 at 640x480, resized to 240x320); the sim/robot D405 is 72 deg (f = 440). The
  pixel numbers in section 4.4 project the sim with the Aria intrinsics to isolate geometry.
* The sim README's "objects ~0.60 m from the camera, inside the depth range of the released
  human training data (0.65 +- 0.12 m)" does not match the t0 snapshot: human bread depth z_c is
  0.53 +- 0.04 m and range 0.57 +- 0.03 m, while the sim's sampled depth is 0.62 +- 0.04 m.

## 7. Accuracy notes

* MPS SLAM is metric (IMU + stereo); the triangulated keypoints reproject onto their 2D seeds
  with 0.2-1.4 px median error. Support-plane height: +-0.3-0.5 cm (it is fitted within a 3 cm
  window calibrated on the self-consistent episodes). Camera height inherits that.
* Bread height is the 95th percentile of the top-surface points: it slightly underestimates the
  crest if the crest has few points and overestimates if stray points remain (range 4.8-7.8 cm;
  median 5.4 is the better summary than the mean 5.7).
* Bread width is the least certain (+-0.8 cm across episodes; model-based correction for the
  visible side). Length and plate diameter agree between independent methods within 0.1-0.5 cm.
* Plate centre: silhouette centroid; it differs from the contour circle fit by 1.4 cm median
  (max 4.8 cm). Bread centre: contour centroid; agrees with the silhouette centroid to 0.4 cm
  median.

## Appendix A. Full tables (generated by scripts/analyze.py; human n = 60, ep 033 excluded)

Field names: `_H_t0` = frame H at t0; `_cam_` = OpenCV camera frame at t0; `az/el` = angle right / below the optical axis; `u/v_aria` = pixel in the 640x480 f = 320 Aria image; `_cam0` = first approach frame. Sim rows use the same names (sim world = H).

### Human, group ALL

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_L_mask [cm] | 60 | 14.6 | 0.5 | 13.9 | 14.0 | 14.3 | 14.5 | 14.9 | 15.4 | 16.1 |
| bread_L_contour+erosion [cm] | 60 | 14.5 | 0.4 | 13.8 | 14.0 | 14.2 | 14.5 | 14.9 | 15.3 | 15.6 |
| bread_L_contour_raw [cm] | 60 | 12.8 | 0.6 | 11.5 | 11.9 | 12.4 | 12.7 | 13.2 | 13.7 | 14.3 |
| bread_W_mask_corr [cm] | 60 | 7.1 | 0.8 | 5.0 | 5.4 | 7.0 | 7.3 | 7.6 | 8.1 | 8.5 |
| bread_W_contour_raw [cm] | 60 | 4.8 | 0.9 | 2.9 | 3.8 | 4.3 | 4.7 | 5.3 | 6.6 | 6.9 |
| bread_H_semidense_p95 [cm] | 60 | 5.7 | 0.8 | 4.8 | 4.9 | 5.2 | 5.4 | 5.9 | 7.3 | 7.8 |
| plate_D_mask_robust [cm] | 60 | 24.8 | 0.4 | 24.2 | 24.3 | 24.5 | 24.7 | 25.1 | 25.4 | 25.6 |
| plate_D_mask_full [cm] | 60 | 25.9 | 0.4 | 25.3 | 25.4 | 25.6 | 25.8 | 26.2 | 26.5 | 26.9 |
| plate_D_contour_circle [cm] | 60 | 20.6 | 1.4 | 17.7 | 19.0 | 19.6 | 20.4 | 21.1 | 23.3 | 26.5 |
| cam_height_t0 [cm] | 60 | 43.7 | 3.7 | 36.9 | 37.5 | 39.5 | 45.3 | 46.2 | 47.6 | 51.9 |
| cam_pitch_t0 [deg] | 60 | 34.3 | 3.5 | 27.5 | 28.7 | 31.8 | 34.3 | 37.0 | 39.7 | 40.4 |
| cam_roll_t0 [deg] | 60 | 0.7 | 1.4 | -1.9 | -1.3 | -0.1 | 0.5 | 1.2 | 3.0 | 4.9 |
| cam_pitch_pregrasp_med [deg] | 59 | 34.1 | 3.3 | 27.6 | 29.4 | 31.6 | 34.3 | 36.6 | 39.0 | 40.8 |
| cam_height_pregrasp_med [cm] | 59 | 43.7 | 3.7 | 36.6 | 37.4 | 39.6 | 45.3 | 46.3 | 47.5 | 52.3 |
| bread_range_pregrasp_med [cm] | 59 | 57.4 | 3.5 | 47.5 | 52.2 | 55.3 | 57.5 | 59.4 | 63.0 | 68.3 |
| bread_x_H_t0 [cm] | 60 | 36.2 | 4.2 | 27.8 | 29.9 | 33.7 | 35.8 | 38.7 | 44.1 | 49.5 |
| bread_y_H_t0 [cm] | 60 | -15.2 | 7.0 | -34.1 | -29.2 | -19.9 | -12.7 | -10.4 | -8.1 | -0.8 |
| plate_x_H_t0 [cm] | 60 | 35.4 | 6.3 | 23.3 | 26.4 | 30.5 | 35.4 | 39.8 | 45.2 | 54.3 |
| plate_y_H_t0 [cm] | 60 | 11.5 | 6.4 | -8.3 | -1.1 | 9.3 | 13.0 | 15.1 | 19.8 | 23.6 |
| bread_yaw_H_t0 [deg] | 60 | 85.0 | 10.3 | 61.7 | 67.2 | 78.0 | 86.1 | 92.4 | 100.8 | 107.2 |
| bread_yaw_H_t0_mask [deg] | 60 | 91.3 | 11.0 | 64.4 | 72.5 | 85.9 | 92.0 | 98.0 | 106.2 | 116.0 |
| plate_minus_bread_dx [cm] | 60 | -0.8 | 5.9 | -13.2 | -10.2 | -4.9 | -0.6 | 3.0 | 9.5 | 12.6 |
| plate_minus_bread_dy [cm] | 60 | 26.7 | 3.0 | 20.8 | 21.6 | 24.3 | 26.9 | 29.3 | 30.1 | 34.5 |
| bread_plate_dist [cm] | 60 | 27.3 | 3.1 | 21.5 | 22.5 | 25.0 | 27.5 | 29.8 | 31.3 | 34.5 |
| bread_plate_dir [deg] | 60 | 91.8 | 12.4 | 64.3 | 72.7 | 83.7 | 91.3 | 100.6 | 112.8 | 116.6 |
| bread_range_t0 [cm] | 60 | 57.2 | 3.4 | 47.9 | 52.7 | 54.7 | 57.1 | 59.2 | 63.7 | 68.5 |
| plate_range_t0 [cm] | 60 | 57.2 | 4.2 | 47.7 | 52.8 | 54.1 | 57.3 | 59.1 | 63.6 | 73.6 |
| bread_az_t0 [deg] | 60 | 15.9 | 7.4 | 1.0 | 8.2 | 11.0 | 13.0 | 20.6 | 30.5 | 35.5 |
| bread_el_t0 [deg] | 60 | 14.5 | 3.6 | 4.2 | 9.1 | 12.4 | 14.4 | 16.9 | 20.4 | 21.4 |
| plate_az_t0 [deg] | 60 | -12.4 | 7.0 | -25.1 | -22.2 | -16.6 | -13.3 | -10.4 | 1.0 | 8.7 |
| plate_el_t0 [deg] | 60 | 16.0 | 6.3 | 3.9 | 5.2 | 12.1 | 16.2 | 21.1 | 24.8 | 30.4 |
| bread_cam_x_t0 [cm] | 60 | 15.1 | 6.9 | 0.9 | 7.9 | 10.2 | 12.7 | 19.5 | 29.0 | 33.6 |
| bread_cam_y_t0 [cm] | 60 | 13.7 | 3.7 | 3.8 | 7.8 | 11.4 | 13.9 | 16.2 | 19.2 | 20.9 |
| bread_cam_z_t0 [cm] | 60 | 52.8 | 3.8 | 43.4 | 47.7 | 50.5 | 52.4 | 54.8 | 59.8 | 65.2 |
| plate_cam_x_t0 [cm] | 60 | -11.6 | 6.4 | -23.5 | -19.5 | -15.4 | -12.8 | -9.6 | 1.1 | 8.2 |
| plate_cam_y_t0 [cm] | 60 | 15.2 | 5.7 | 3.7 | 4.9 | 11.7 | 16.0 | 19.4 | 23.8 | 27.4 |
| plate_cam_z_t0 [cm] | 60 | 53.2 | 4.7 | 45.6 | 46.6 | 49.9 | 52.7 | 55.4 | 61.4 | 70.2 |
| bread_u_aria_t0 [px] | 60 | 413.4 | 46.9 | 325.9 | 366.7 | 382.7 | 394.4 | 441.0 | 509.0 | 548.5 |
| bread_v_aria_t0 [px] | 60 | 322.5 | 21.4 | 263.1 | 290.6 | 309.9 | 321.8 | 336.8 | 358.6 | 364.6 |
| hand_x_H_t0 [cm] | 60 | 34.0 | 6.8 | 1.1 | 25.5 | 30.5 | 34.3 | 37.6 | 43.7 | 47.2 |
| hand_y_H_t0 [cm] | 60 | -34.5 | 6.9 | -51.7 | -46.0 | -40.0 | -31.4 | -29.5 | -27.2 | -24.8 |
| hand_z_H_t0 [cm] | 60 | 18.3 | 8.1 | -23.6 | 12.2 | 14.7 | 16.5 | 21.8 | 30.4 | 35.1 |
| hand_to_bread_t0 [cm] | 60 | 26.3 | 5.6 | 18.0 | 19.0 | 22.4 | 24.4 | 29.7 | 36.1 | 46.8 |
| t0_to_grasp_s [s] | 60 | 6.6 | 2.0 | 0.0 | 3.6 | 5.0 | 6.8 | 7.9 | 9.2 | 12.1 |
| bread_x_H_cam0 [cm] | 60 | 53.3 | 13.7 | 29.1 | 33.7 | 42.0 | 52.7 | 61.5 | 74.0 | 91.5 |
| cam_height_cam0 [cm] | 60 | 43.8 | 4.6 | 23.5 | 37.5 | 40.2 | 45.8 | 46.6 | 47.7 | 52.7 |

### Human, group A_box_on_desk

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_L_mask [cm] | 38 | 14.3 | 0.2 | 13.9 | 14.0 | 14.2 | 14.4 | 14.5 | 14.6 | 14.7 |
| bread_L_contour+erosion [cm] | 38 | 14.4 | 0.4 | 13.8 | 13.9 | 14.1 | 14.4 | 14.7 | 15.3 | 15.6 |
| bread_L_contour_raw [cm] | 38 | 12.6 | 0.6 | 11.5 | 11.8 | 12.2 | 12.5 | 12.9 | 13.9 | 14.3 |
| bread_W_mask_corr [cm] | 38 | 6.9 | 0.7 | 5.0 | 5.3 | 6.7 | 7.1 | 7.4 | 7.6 | 7.7 |
| bread_W_contour_raw [cm] | 38 | 4.9 | 1.0 | 2.9 | 3.4 | 4.2 | 4.8 | 5.5 | 6.6 | 6.9 |
| bread_H_semidense_p95 [cm] | 38 | 5.9 | 0.8 | 4.9 | 5.0 | 5.3 | 5.5 | 6.3 | 7.4 | 7.8 |
| plate_D_mask_robust [cm] | 38 | 24.6 | 0.2 | 24.2 | 24.3 | 24.4 | 24.6 | 24.7 | 24.9 | 25.2 |
| plate_D_mask_full [cm] | 38 | 25.7 | 0.2 | 25.3 | 25.4 | 25.5 | 25.7 | 25.8 | 26.0 | 26.5 |
| plate_D_contour_circle [cm] | 38 | 20.5 | 1.4 | 17.7 | 19.0 | 19.5 | 20.3 | 21.3 | 23.3 | 23.5 |
| cam_height_t0 [cm] | 38 | 46.2 | 1.4 | 43.9 | 44.6 | 45.4 | 46.0 | 46.6 | 48.2 | 51.9 |
| cam_pitch_t0 [deg] | 38 | 34.2 | 3.3 | 27.5 | 29.3 | 32.2 | 34.1 | 36.5 | 39.1 | 40.4 |
| cam_roll_t0 [deg] | 38 | 0.1 | 1.0 | -1.9 | -1.3 | -0.6 | 0.2 | 0.7 | 1.4 | 2.9 |
| cam_pitch_pregrasp_med [deg] | 38 | 34.0 | 3.2 | 27.6 | 29.3 | 31.4 | 34.2 | 35.8 | 39.0 | 40.4 |
| cam_height_pregrasp_med [cm] | 38 | 46.2 | 1.5 | 43.5 | 44.6 | 45.4 | 46.0 | 46.6 | 48.6 | 52.3 |
| bread_range_pregrasp_med [cm] | 38 | 57.7 | 3.4 | 51.9 | 53.1 | 55.5 | 57.8 | 59.5 | 63.2 | 68.3 |
| bread_x_H_t0 [cm] | 38 | 36.1 | 4.3 | 28.3 | 29.8 | 33.0 | 35.8 | 38.8 | 43.1 | 49.5 |
| bread_y_H_t0 [cm] | 38 | -10.9 | 2.7 | -15.8 | -14.6 | -12.4 | -11.0 | -9.6 | -7.8 | -0.8 |
| plate_x_H_t0 [cm] | 38 | 33.5 | 6.5 | 23.3 | 25.8 | 28.3 | 33.4 | 36.6 | 43.9 | 54.3 |
| plate_y_H_t0 [cm] | 38 | 14.9 | 3.1 | 9.4 | 11.1 | 13.0 | 14.1 | 16.1 | 20.0 | 23.6 |
| bread_yaw_H_t0 [deg] | 38 | 86.8 | 8.9 | 63.1 | 72.3 | 82.4 | 87.9 | 92.4 | 100.8 | 103.6 |
| bread_yaw_H_t0_mask [deg] | 38 | 91.0 | 9.7 | 64.4 | 72.0 | 86.0 | 92.3 | 97.7 | 104.0 | 105.0 |
| plate_minus_bread_dx [cm] | 38 | -2.6 | 4.9 | -13.2 | -10.5 | -6.8 | -2.6 | 1.9 | 4.3 | 4.9 |
| plate_minus_bread_dy [cm] | 38 | 25.9 | 2.9 | 20.8 | 21.5 | 23.2 | 26.2 | 27.8 | 30.1 | 31.0 |
| bread_plate_dist [cm] | 38 | 26.4 | 2.9 | 21.7 | 22.4 | 23.8 | 26.7 | 29.2 | 30.4 | 31.2 |
| bread_plate_dir [deg] | 38 | 95.7 | 10.7 | 80.7 | 80.9 | 85.1 | 95.5 | 104.8 | 113.0 | 116.6 |
| bread_range_t0 [cm] | 38 | 57.5 | 3.4 | 51.8 | 53.2 | 55.2 | 57.2 | 59.3 | 63.8 | 68.5 |
| plate_range_t0 [cm] | 38 | 58.4 | 4.3 | 51.4 | 53.5 | 55.4 | 58.2 | 59.7 | 65.9 | 73.6 |
| bread_az_t0 [deg] | 38 | 11.4 | 2.9 | 1.0 | 8.2 | 9.7 | 11.8 | 12.6 | 15.0 | 17.5 |
| bread_el_t0 [deg] | 38 | 16.1 | 2.6 | 9.4 | 12.7 | 14.2 | 16.1 | 17.6 | 20.4 | 21.4 |
| plate_az_t0 [deg] | 38 | -15.9 | 3.8 | -25.1 | -22.6 | -18.5 | -15.2 | -13.1 | -11.6 | -8.5 |
| plate_el_t0 [deg] | 38 | 19.5 | 4.6 | 11.3 | 12.5 | 16.4 | 19.2 | 22.8 | 26.0 | 30.4 |
| bread_cam_x_t0 [cm] | 38 | 10.9 | 2.7 | 0.9 | 7.9 | 9.5 | 10.8 | 12.5 | 14.5 | 16.2 |
| bread_cam_y_t0 [cm] | 38 | 15.6 | 2.5 | 8.9 | 12.1 | 13.7 | 15.7 | 17.0 | 19.6 | 20.9 |
| bread_cam_z_t0 [cm] | 38 | 54.1 | 3.4 | 48.4 | 49.6 | 51.8 | 54.0 | 56.0 | 60.5 | 65.2 |
| plate_cam_x_t0 [cm] | 38 | -15.0 | 3.0 | -23.5 | -19.6 | -16.5 | -14.1 | -12.7 | -11.4 | -9.6 |
| plate_cam_y_t0 [cm] | 38 | 18.5 | 3.6 | 11.9 | 13.7 | 16.1 | 18.6 | 20.7 | 24.2 | 27.4 |
| plate_cam_z_t0 [cm] | 38 | 53.0 | 5.4 | 45.6 | 46.6 | 48.9 | 51.7 | 55.3 | 62.1 | 70.2 |
| bread_u_aria_t0 [px] | 38 | 385.2 | 16.6 | 325.9 | 366.4 | 375.4 | 387.4 | 392.0 | 406.0 | 421.1 |
| bread_v_aria_t0 [px] | 38 | 332.0 | 15.6 | 292.3 | 311.3 | 320.2 | 331.6 | 340.7 | 358.7 | 364.6 |
| hand_x_H_t0 [cm] | 38 | 35.7 | 4.9 | 25.5 | 29.0 | 32.4 | 35.6 | 39.0 | 43.8 | 46.9 |
| hand_y_H_t0 [cm] | 38 | -30.2 | 2.8 | -37.3 | -35.0 | -31.3 | -30.0 | -28.5 | -26.5 | -24.8 |
| hand_z_H_t0 [cm] | 38 | 15.3 | 2.2 | 10.0 | 12.1 | 13.9 | 15.3 | 16.5 | 19.0 | 21.4 |
| hand_to_bread_t0 [cm] | 38 | 23.2 | 2.9 | 18.0 | 18.8 | 21.8 | 22.7 | 24.0 | 27.9 | 30.4 |
| t0_to_grasp_s [s] | 38 | 7.8 | 1.2 | 6.0 | 6.2 | 6.9 | 7.5 | 8.4 | 9.8 | 12.1 |
| bread_x_H_cam0 [cm] | 38 | 55.9 | 14.9 | 29.1 | 35.2 | 45.2 | 56.0 | 65.1 | 76.9 | 91.5 |
| cam_height_cam0 [cm] | 38 | 46.6 | 1.3 | 44.4 | 45.3 | 45.9 | 46.4 | 47.0 | 48.3 | 52.7 |

### Human, group B_desk

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_L_mask [cm] | 6 | 14.7 | 0.4 | 14.3 | 14.3 | 14.5 | 14.7 | 15.0 | 15.3 | 15.4 |
| bread_L_contour+erosion [cm] | 6 | 14.5 | 0.3 | 14.1 | 14.2 | 14.4 | 14.5 | 14.7 | 14.9 | 14.9 |
| bread_L_contour_raw [cm] | 6 | 13.0 | 0.3 | 12.7 | 12.7 | 12.8 | 12.9 | 13.2 | 13.4 | 13.4 |
| bread_W_mask_corr [cm] | 6 | 6.8 | 1.0 | 5.4 | 5.5 | 6.2 | 7.1 | 7.6 | 7.7 | 7.7 |
| bread_W_contour_raw [cm] | 6 | 5.1 | 0.7 | 4.1 | 4.3 | 4.8 | 5.0 | 5.3 | 5.9 | 6.1 |
| bread_H_semidense_p95 [cm] | 6 | 5.9 | 1.1 | 4.8 | 4.9 | 5.2 | 5.4 | 6.6 | 7.3 | 7.5 |
| plate_D_mask_robust [cm] | 6 | 24.7 | 0.4 | 24.2 | 24.3 | 24.5 | 24.5 | 24.8 | 25.2 | 25.3 |
| plate_D_mask_full [cm] | 6 | 25.7 | 0.4 | 25.3 | 25.3 | 25.5 | 25.7 | 25.8 | 26.2 | 26.4 |
| plate_D_contour_circle [cm] | 6 | 21.6 | 2.6 | 19.1 | 19.4 | 20.4 | 21.1 | 21.6 | 25.3 | 26.5 |
| cam_height_t0 [cm] | 6 | 42.1 | 0.7 | 41.1 | 41.2 | 41.7 | 42.2 | 42.7 | 42.9 | 42.9 |
| cam_pitch_t0 [deg] | 6 | 37.2 | 1.7 | 35.2 | 35.4 | 36.1 | 37.1 | 37.9 | 39.4 | 39.9 |
| cam_roll_t0 [deg] | 6 | 3.3 | 1.7 | 0.7 | 1.1 | 2.2 | 3.6 | 4.7 | 4.8 | 4.9 |
| cam_pitch_pregrasp_med [deg] | 5 | 36.6 | 0.9 | 35.8 | 35.8 | 35.8 | 36.5 | 37.1 | 37.7 | 37.8 |
| cam_height_pregrasp_med [cm] | 5 | 42.1 | 0.7 | 41.3 | 41.4 | 41.5 | 42.0 | 42.7 | 42.9 | 43.0 |
| bread_range_pregrasp_med [cm] | 5 | 56.3 | 2.1 | 53.8 | 53.9 | 54.2 | 57.2 | 57.9 | 58.1 | 58.2 |
| bread_x_H_t0 [cm] | 6 | 35.9 | 2.2 | 33.2 | 33.5 | 34.5 | 35.7 | 37.1 | 38.8 | 39.3 |
| bread_y_H_t0 [cm] | 6 | -17.9 | 6.0 | -27.2 | -25.4 | -20.1 | -18.1 | -15.1 | -10.6 | -9.2 |
| plate_x_H_t0 [cm] | 6 | 36.4 | 5.4 | 32.1 | 32.2 | 32.8 | 34.0 | 38.7 | 44.4 | 45.8 |
| plate_y_H_t0 [cm] | 6 | 10.1 | 6.1 | -1.0 | 1.6 | 9.5 | 10.5 | 14.1 | 15.8 | 16.0 |
| bread_yaw_H_t0 [deg] | 6 | 87.6 | 9.4 | 71.8 | 74.6 | 83.7 | 89.7 | 94.4 | 96.3 | 96.7 |
| bread_yaw_H_t0_mask [deg] | 6 | 96.2 | 9.6 | 81.2 | 83.2 | 91.3 | 97.6 | 103.4 | 105.9 | 106.1 |
| plate_minus_bread_dx [cm] | 6 | 0.5 | 6.5 | -4.9 | -4.7 | -3.9 | -1.6 | 2.0 | 10.2 | 12.6 |
| plate_minus_bread_dy [cm] | 6 | 28.0 | 1.9 | 25.3 | 25.5 | 26.6 | 28.6 | 29.6 | 29.7 | 29.7 |
| bread_plate_dist [cm] | 6 | 28.6 | 1.6 | 25.6 | 26.3 | 28.5 | 29.1 | 29.6 | 29.8 | 29.8 |
| bread_plate_dir [deg] | 6 | 89.1 | 13.4 | 64.3 | 69.3 | 86.1 | 93.0 | 98.5 | 99.9 | 99.9 |
| bread_range_t0 [cm] | 6 | 56.4 | 2.0 | 53.7 | 53.9 | 54.8 | 57.0 | 57.9 | 58.2 | 58.4 |
| plate_range_t0 [cm] | 6 | 56.2 | 3.2 | 53.7 | 53.8 | 54.1 | 54.7 | 57.8 | 60.8 | 61.5 |
| bread_az_t0 [deg] | 6 | 18.1 | 6.4 | 9.1 | 10.5 | 15.1 | 18.1 | 20.2 | 26.2 | 28.1 |
| bread_el_t0 [deg] | 6 | 11.5 | 2.1 | 9.2 | 9.2 | 9.8 | 11.6 | 12.6 | 14.2 | 14.6 |
| plate_az_t0 [deg] | 6 | -11.8 | 7.2 | -19.0 | -18.7 | -16.8 | -12.2 | -10.3 | -1.8 | 0.9 |
| plate_el_t0 [deg] | 6 | 10.5 | 3.6 | 4.3 | 5.6 | 9.7 | 10.5 | 12.2 | 14.5 | 15.2 |
| bread_cam_x_t0 [cm] | 6 | 17.2 | 6.3 | 8.4 | 9.6 | 14.0 | 17.4 | 19.4 | 25.2 | 27.0 |
| bread_cam_y_t0 [cm] | 6 | 10.6 | 1.8 | 8.6 | 8.7 | 9.2 | 10.8 | 11.4 | 12.8 | 13.3 |
| bread_cam_z_t0 [cm] | 6 | 52.3 | 1.6 | 50.5 | 50.6 | 51.2 | 52.0 | 53.0 | 54.5 | 54.9 |
| plate_cam_x_t0 [cm] | 6 | -10.9 | 6.5 | -17.2 | -17.0 | -15.4 | -11.2 | -10.1 | -1.8 | 1.0 |
| plate_cam_y_t0 [cm] | 6 | 9.8 | 3.0 | 4.6 | 5.7 | 9.1 | 9.9 | 11.1 | 13.2 | 13.7 |
| plate_cam_z_t0 [cm] | 6 | 53.8 | 4.5 | 49.9 | 50.1 | 50.9 | 52.0 | 55.8 | 60.3 | 61.3 |
| bread_u_aria_t0 [px] | 6 | 426.2 | 40.3 | 371.9 | 379.9 | 406.6 | 425.5 | 438.1 | 478.4 | 491.7 |
| bread_v_aria_t0 [px] | 6 | 304.8 | 12.1 | 291.1 | 291.4 | 295.1 | 305.2 | 310.9 | 320.3 | 322.9 |
| hand_x_H_t0 [cm] | 6 | 28.2 | 13.7 | 1.1 | 7.8 | 29.3 | 33.3 | 35.8 | 37.0 | 37.0 |
| hand_y_H_t0 [cm] | 6 | -39.0 | 5.1 | -45.7 | -44.4 | -40.6 | -40.1 | -38.1 | -32.0 | -30.1 |
| hand_z_H_t0 [cm] | 6 | 14.3 | 18.6 | -23.6 | -12.8 | 19.7 | 21.2 | 23.2 | 23.8 | 23.9 |
| hand_to_bread_t0 [cm] | 6 | 31.7 | 7.6 | 26.6 | 26.7 | 27.3 | 29.6 | 30.9 | 42.8 | 46.8 |
| t0_to_grasp_s [s] | 6 | 3.3 | 1.8 | 0.0 | 0.7 | 3.1 | 3.5 | 4.2 | 5.1 | 5.4 |
| bread_x_H_cam0 [cm] | 6 | 40.4 | 7.8 | 32.1 | 32.5 | 34.7 | 38.5 | 46.6 | 50.3 | 50.8 |
| cam_height_cam0 [cm] | 6 | 42.9 | 0.9 | 41.7 | 41.7 | 42.3 | 43.2 | 43.5 | 43.8 | 43.9 |

### Human, group C_counter

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_L_mask [cm] | 16 | 15.3 | 0.4 | 14.5 | 14.6 | 15.2 | 15.3 | 15.4 | 15.8 | 16.1 |
| bread_L_contour+erosion [cm] | 16 | 14.8 | 0.4 | 14.2 | 14.2 | 14.5 | 14.9 | 15.0 | 15.4 | 15.6 |
| bread_L_contour_raw [cm] | 16 | 13.1 | 0.3 | 12.6 | 12.7 | 12.9 | 13.1 | 13.4 | 13.5 | 13.7 |
| bread_W_mask_corr [cm] | 16 | 7.9 | 0.4 | 7.2 | 7.3 | 7.7 | 7.8 | 8.1 | 8.4 | 8.5 |
| bread_W_contour_raw [cm] | 16 | 4.6 | 0.6 | 3.8 | 3.8 | 4.2 | 4.6 | 5.0 | 5.6 | 5.7 |
| bread_H_semidense_p95 [cm] | 16 | 5.3 | 0.3 | 4.8 | 4.9 | 5.1 | 5.2 | 5.4 | 5.7 | 5.8 |
| plate_D_mask_robust [cm] | 16 | 25.3 | 0.2 | 25.0 | 25.0 | 25.1 | 25.3 | 25.4 | 25.5 | 25.6 |
| plate_D_mask_full [cm] | 16 | 26.4 | 0.2 | 26.0 | 26.1 | 26.2 | 26.4 | 26.5 | 26.7 | 26.9 |
| plate_D_contour_circle [cm] | 16 | 20.3 | 0.6 | 19.0 | 19.3 | 20.0 | 20.3 | 20.5 | 21.3 | 21.3 |
| cam_height_t0 [cm] | 16 | 38.3 | 0.9 | 36.9 | 36.9 | 37.6 | 38.4 | 39.0 | 39.4 | 39.5 |
| cam_pitch_t0 [deg] | 16 | 33.4 | 3.9 | 27.6 | 27.8 | 30.8 | 32.8 | 37.0 | 38.6 | 39.7 |
| cam_roll_t0 [deg] | 16 | 1.1 | 0.9 | -0.7 | -0.2 | 0.5 | 1.0 | 1.6 | 2.4 | 2.7 |
| cam_pitch_pregrasp_med [deg] | 16 | 33.8 | 3.6 | 28.5 | 29.2 | 31.5 | 32.2 | 36.5 | 39.4 | 40.8 |
| cam_height_pregrasp_med [cm] | 16 | 38.4 | 0.9 | 36.6 | 37.1 | 37.5 | 38.7 | 38.9 | 39.5 | 39.7 |
| bread_range_pregrasp_med [cm] | 16 | 56.9 | 4.0 | 47.5 | 50.9 | 55.0 | 56.8 | 59.6 | 62.7 | 64.2 |
| bread_x_H_t0 [cm] | 16 | 36.4 | 4.7 | 27.8 | 31.7 | 33.9 | 35.1 | 38.1 | 45.4 | 47.9 |
| bread_y_H_t0 [cm] | 16 | -24.3 | 4.9 | -34.1 | -31.7 | -28.1 | -23.7 | -20.7 | -18.2 | -17.4 |
| plate_x_H_t0 [cm] | 16 | 39.5 | 3.8 | 30.6 | 34.1 | 38.1 | 39.5 | 41.0 | 44.8 | 45.2 |
| plate_y_H_t0 [cm] | 16 | 3.7 | 5.7 | -8.3 | -6.0 | 0.5 | 5.2 | 8.2 | 9.8 | 10.1 |
| bread_yaw_H_t0 [deg] | 16 | 79.5 | 12.1 | 61.7 | 65.9 | 71.0 | 77.8 | 84.6 | 100.1 | 107.2 |
| bread_yaw_H_t0_mask [deg] | 16 | 90.1 | 14.4 | 64.4 | 71.3 | 80.7 | 87.7 | 97.5 | 113.5 | 116.0 |
| plate_minus_bread_dx [cm] | 16 | 3.0 | 6.4 | -12.6 | -10.1 | 1.2 | 4.4 | 6.4 | 10.1 | 11.3 |
| plate_minus_bread_dy [cm] | 16 | 28.1 | 3.1 | 20.8 | 23.8 | 26.5 | 28.2 | 29.9 | 32.2 | 34.5 |
| bread_plate_dist [cm] | 16 | 28.9 | 3.2 | 21.5 | 24.4 | 27.0 | 28.9 | 30.5 | 33.3 | 34.5 |
| bread_plate_dir [deg] | 16 | 83.6 | 12.4 | 68.7 | 69.9 | 76.1 | 81.2 | 87.7 | 109.9 | 112.8 |
| bread_range_t0 [cm] | 16 | 56.8 | 4.0 | 47.9 | 50.9 | 54.5 | 56.5 | 59.5 | 62.4 | 64.2 |
| plate_range_t0 [cm] | 16 | 54.7 | 2.8 | 47.7 | 51.4 | 53.4 | 54.4 | 56.0 | 59.0 | 59.5 |
| bread_az_t0 [deg] | 16 | 25.7 | 4.9 | 16.9 | 18.8 | 22.3 | 26.0 | 28.6 | 32.9 | 35.5 |
| bread_el_t0 [deg] | 16 | 11.8 | 3.9 | 4.2 | 6.1 | 9.8 | 12.0 | 13.7 | 16.9 | 20.9 |
| plate_az_t0 [deg] | 16 | -4.4 | 6.2 | -11.3 | -10.9 | -9.8 | -5.8 | -0.7 | 5.8 | 8.7 |
| plate_el_t0 [deg] | 16 | 9.9 | 4.4 | 3.9 | 4.7 | 6.5 | 9.2 | 13.2 | 16.3 | 18.2 |
| bread_cam_x_t0 [cm] | 16 | 24.1 | 4.9 | 17.3 | 17.9 | 20.4 | 23.4 | 28.0 | 31.5 | 33.6 |
| bread_cam_y_t0 [cm] | 16 | 10.4 | 3.4 | 3.8 | 5.3 | 8.6 | 10.2 | 12.3 | 15.3 | 18.0 |
| bread_cam_z_t0 [cm] | 16 | 50.0 | 3.9 | 43.4 | 46.2 | 48.1 | 49.1 | 51.2 | 57.6 | 59.7 |
| plate_cam_x_t0 [cm] | 16 | -4.0 | 5.7 | -10.1 | -10.0 | -8.6 | -5.5 | -0.7 | 5.7 | 8.2 |
| plate_cam_y_t0 [cm] | 16 | 9.3 | 4.0 | 3.7 | 4.4 | 6.1 | 8.5 | 13.0 | 15.3 | 16.3 |
| plate_cam_z_t0 [cm] | 16 | 53.3 | 3.1 | 46.4 | 48.7 | 52.3 | 53.4 | 54.6 | 57.9 | 58.2 |
| bread_u_aria_t0 [px] | 16 | 475.8 | 34.5 | 417.5 | 429.5 | 451.7 | 476.6 | 495.2 | 528.0 | 548.5 |
| bread_v_aria_t0 [px] | 16 | 306.5 | 23.1 | 263.1 | 274.0 | 294.6 | 307.4 | 317.3 | 337.0 | 361.6 |
| hand_x_H_t0 [cm] | 16 | 32.3 | 6.1 | 22.4 | 24.1 | 28.9 | 31.8 | 36.2 | 40.8 | 47.2 |
| hand_y_H_t0 [cm] | 16 | -43.1 | 4.9 | -51.7 | -50.7 | -45.9 | -43.8 | -40.1 | -35.9 | -33.7 |
| hand_z_H_t0 [cm] | 16 | 26.9 | 4.6 | 20.4 | 20.8 | 22.6 | 26.8 | 28.9 | 34.6 | 35.1 |
| hand_to_bread_t0 [cm] | 16 | 31.6 | 4.2 | 23.7 | 24.5 | 29.2 | 32.4 | 34.7 | 36.9 | 37.1 |
| t0_to_grasp_s [s] | 16 | 5.1 | 0.7 | 4.0 | 4.3 | 4.5 | 4.9 | 5.4 | 6.4 | 6.7 |
| bread_x_H_cam0 [cm] | 16 | 51.8 | 9.4 | 34.7 | 37.9 | 48.2 | 51.2 | 57.4 | 65.2 | 71.9 |
| cam_height_cam0 [cm] | 16 | 37.5 | 3.9 | 23.5 | 32.8 | 37.5 | 38.7 | 39.2 | 39.7 | 40.5 |

### Sim sampler, mode single (20k draws, difficulty 1.0)

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_x_H_t0 | 20000 | 47.0 | 4.6 | 39.0 | 39.8 | 42.9 | 46.9 | 51.0 | 54.2 | 55.0 |
| bread_y_H_t0 | 20000 | -15.9 | 7.2 | -32.9 | -27.8 | -21.4 | -15.9 | -10.4 | -4.1 | 0.9 |
| plate_x_H_t0 | 20000 | 46.9 | 4.0 | 40.0 | 40.7 | 43.4 | 46.9 | 50.4 | 53.3 | 54.0 |
| plate_y_H_t0 | 20000 | 11.0 | 3.5 | 5.0 | 5.6 | 8.1 | 11.0 | 14.1 | 16.4 | 17.0 |
| bread_yaw_H_t0 | 20000 | 89.8 | 20.1 | 55.0 | 58.5 | 72.5 | 89.8 | 107.2 | 121.3 | 125.0 |
| plate_minus_bread_dx | 20000 | -0.0 | 6.1 | -14.9 | -10.3 | -4.5 | -0.0 | 4.4 | 10.2 | 14.7 |
| plate_minus_bread_dy | 20000 | 27.0 | 6.3 | 16.0 | 17.1 | 21.5 | 26.9 | 32.4 | 36.9 | 38.0 |
| bread_plate_dist | 20000 | 27.7 | 6.2 | 16.0 | 17.9 | 22.4 | 27.7 | 33.0 | 37.4 | 40.5 |
| bread_plate_dir | 20000 | 90.0 | 13.6 | 47.6 | 67.6 | 80.6 | 90.1 | 99.4 | 112.3 | 131.4 |
| bread_range_t0 | 20000 | 64.9 | 3.8 | 56.7 | 58.7 | 61.9 | 64.9 | 67.8 | 71.0 | 75.6 |
| plate_range_t0 | 20000 | 64.4 | 3.0 | 58.6 | 59.8 | 61.8 | 64.4 | 66.9 | 69.2 | 70.7 |
| bread_az_t0 | 20000 | 14.2 | 6.3 | -0.8 | 3.7 | 9.5 | 14.3 | 19.0 | 24.2 | 30.2 |
| bread_el_t0 | 20000 | 4.3 | 2.8 | -0.2 | 0.2 | 1.9 | 4.2 | 6.7 | 8.9 | 9.5 |
| plate_az_t0 | 20000 | -9.9 | 3.1 | -16.4 | -14.8 | -12.5 | -9.9 | -7.3 | -5.1 | -4.2 |
| plate_el_t0 | 20000 | 5.3 | 2.5 | 1.2 | 1.6 | 3.1 | 5.2 | 7.4 | 9.2 | 9.7 |
| bread_cam_x_t0 | 20000 | 15.9 | 7.2 | -0.9 | 4.1 | 10.4 | 15.9 | 21.4 | 27.8 | 32.9 |
| bread_cam_y_t0 | 20000 | 4.6 | 2.8 | -0.3 | 0.2 | 2.1 | 4.6 | 7.0 | 8.9 | 9.4 |
| bread_cam_z_t0 | 20000 | 62.2 | 3.7 | 55.9 | 56.5 | 59.0 | 62.2 | 65.4 | 68.0 | 68.7 |
| plate_cam_x_t0 | 20000 | -11.0 | 3.5 | -17.0 | -16.4 | -14.1 | -11.0 | -8.1 | -5.6 | -5.0 |
| plate_cam_y_t0 | 20000 | 5.7 | 2.4 | 1.4 | 1.9 | 3.6 | 5.7 | 7.8 | 9.5 | 9.9 |
| plate_cam_z_t0 | 20000 | 63.1 | 3.2 | 57.5 | 58.1 | 60.3 | 63.0 | 65.9 | 68.1 | 68.7 |
| bread_u_aria_t0 | 20000 | 402.6 | 37.6 | 316.2 | 341.4 | 374.0 | 402.2 | 430.8 | 464.1 | 506.8 |
| bread_v_aria_t0 | 20000 | 263.9 | 15.8 | 238.2 | 240.5 | 250.0 | 263.0 | 277.3 | 289.8 | 293.1 |
| bread_u_d405 | 20000 | 432.9 | 51.7 | 314.1 | 348.8 | 393.6 | 432.3 | 471.7 | 517.5 | 576.2 |
| bread_v_d405 | 20000 | 273.5 | 21.8 | 238.2 | 241.3 | 254.4 | 272.3 | 292.0 | 309.1 | 313.6 |

### Sim sampler, mode bimanual_right (20k draws, difficulty 1.0)

| quantity | n | mean | std | min | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| bread_x_H_t0 | 20000 | 46.0 | 4.6 | 38.0 | 38.8 | 42.0 | 45.9 | 50.0 | 53.2 | 54.0 |
| bread_y_H_t0 | 20000 | -27.6 | 5.2 | -39.8 | -36.1 | -31.4 | -27.6 | -23.8 | -18.9 | -15.1 |
| plate_x_H_t0 | 20000 | 47.0 | 3.5 | 41.0 | 41.6 | 44.0 | 46.9 | 50.0 | 52.4 | 53.0 |
| plate_y_H_t0 | 20000 | -0.0 | 2.9 | -5.0 | -4.5 | -2.5 | -0.1 | 2.4 | 4.5 | 5.0 |
| bread_yaw_H_t0 | 20000 | 90.3 | 20.2 | 55.0 | 58.7 | 72.8 | 90.3 | 107.9 | 121.4 | 125.0 |
| plate_minus_bread_dx | 20000 | 1.0 | 5.8 | -12.8 | -8.6 | -3.2 | 1.0 | 5.2 | 10.5 | 14.8 |
| plate_minus_bread_dy | 20000 | 27.5 | 4.3 | 20.0 | 20.8 | 23.8 | 27.6 | 31.3 | 34.3 | 35.0 |
| bread_plate_dist | 20000 | 28.2 | 4.3 | 20.0 | 21.3 | 24.5 | 28.2 | 31.8 | 34.8 | 37.5 |
| bread_plate_dir | 20000 | 88.0 | 12.0 | 55.3 | 68.4 | 79.3 | 88.0 | 96.7 | 107.8 | 121.7 |
| bread_range_t0 | 20000 | 67.8 | 3.8 | 58.2 | 61.7 | 64.9 | 67.8 | 70.6 | 74.0 | 78.3 |
| plate_range_t0 | 20000 | 63.5 | 2.6 | 59.1 | 59.6 | 61.2 | 63.4 | 65.7 | 67.5 | 68.1 |
| bread_az_t0 | 20000 | 24.1 | 4.2 | 12.9 | 17.0 | 21.1 | 24.2 | 27.2 | 31.0 | 35.7 |
| bread_el_t0 | 20000 | 4.9 | 2.9 | 0.3 | 0.7 | 2.4 | 4.8 | 7.3 | 9.7 | 10.2 |
| plate_az_t0 | 20000 | 0.0 | 2.6 | -4.9 | -4.1 | -2.2 | 0.1 | 2.3 | 4.1 | 4.9 |
| plate_el_t0 | 20000 | 5.2 | 2.1 | 1.7 | 2.0 | 3.4 | 5.2 | 7.0 | 8.6 | 9.0 |
| bread_cam_x_t0 | 20000 | 27.6 | 5.2 | 15.1 | 18.9 | 23.8 | 27.6 | 31.4 | 36.1 | 39.8 |
| bread_cam_y_t0 | 20000 | 5.1 | 2.8 | 0.3 | 0.8 | 2.7 | 5.2 | 7.5 | 9.5 | 10.0 |
| bread_cam_z_t0 | 20000 | 61.5 | 3.7 | 55.1 | 55.7 | 58.3 | 61.4 | 64.7 | 67.2 | 67.9 |
| plate_cam_x_t0 | 20000 | 0.0 | 2.9 | -5.0 | -4.5 | -2.4 | 0.1 | 2.5 | 4.5 | 5.0 |
| plate_cam_y_t0 | 20000 | 5.7 | 2.1 | 2.0 | 2.4 | 3.9 | 5.7 | 7.5 | 8.9 | 9.3 |
| plate_cam_z_t0 | 20000 | 63.1 | 2.8 | 58.3 | 58.8 | 60.7 | 63.1 | 65.5 | 67.4 | 67.9 |
| bread_u_aria_t0 | 20000 | 464.6 | 28.5 | 393.9 | 418.2 | 443.8 | 464.1 | 484.9 | 512.4 | 550.1 |
| bread_v_aria_t0 | 20000 | 267.3 | 16.2 | 241.0 | 243.4 | 253.1 | 266.4 | 280.8 | 294.0 | 297.3 |
| bread_u_d405 | 20000 | 518.1 | 39.2 | 420.9 | 454.3 | 489.5 | 517.5 | 546.0 | 583.9 | 635.7 |
| bread_v_d405 | 20000 | 278.2 | 22.2 | 242.1 | 245.4 | 258.7 | 277.0 | 296.7 | 314.9 | 319.5 |

### Coverage (fraction of human episodes inside the sim's min..max)

| quantity | sim single [min,max] | human in / below / above | sim bimanual-right [min,max] | human in / below / above |
|---|---|---|---|---|
| bread_x_H_t0 | [39.0, 55.0] | 0.22 / 0.78 / 0.00 | [38.0, 54.0] | 0.27 / 0.73 / 0.00 |
| bread_y_H_t0 | [-32.9, 0.9] | 0.98 / 0.02 / 0.00 | [-39.8, -15.1] | 0.35 / 0.00 / 0.65 |
| plate_x_H_t0 | [40.0, 54.0] | 0.22 / 0.77 / 0.02 | [41.0, 53.0] | 0.13 / 0.85 / 0.02 |
| plate_y_H_t0 | [5.0, 17.0] | 0.70 / 0.15 / 0.15 | [-5.0, 5.0] | 0.12 / 0.03 / 0.85 |
| bread_yaw_H_t0 | [55.0, 125.0] | 1.00 / 0.00 / 0.00 | [55.0, 125.0] | 1.00 / 0.00 / 0.00 |
| plate_minus_bread_dx | [-14.9, 14.7] | 1.00 / 0.00 / 0.00 | [-12.8, 14.8] | 0.98 / 0.02 / 0.00 |
| plate_minus_bread_dy | [16.0, 38.0] | 1.00 / 0.00 / 0.00 | [20.0, 35.0] | 1.00 / 0.00 / 0.00 |
| bread_plate_dist | [16.0, 40.5] | 1.00 / 0.00 / 0.00 | [20.0, 37.5] | 1.00 / 0.00 / 0.00 |
| bread_plate_dir | [47.6, 131.4] | 1.00 / 0.00 / 0.00 | [55.3, 121.7] | 1.00 / 0.00 / 0.00 |
| bread_range_t0 | [56.7, 75.6] | 0.53 / 0.47 / 0.00 | [58.2, 78.3] | 0.33 / 0.67 / 0.00 |
| plate_range_t0 | [58.6, 70.7] | 0.32 / 0.67 / 0.02 | [59.1, 68.1] | 0.25 / 0.73 / 0.02 |
| bread_az_t0 | [-0.8, 30.2] | 0.93 / 0.00 / 0.07 | [12.9, 35.7] | 0.50 / 0.50 / 0.00 |
| bread_el_t0 | [-0.2, 9.5] | 0.12 / 0.00 / 0.88 | [0.3, 10.2] | 0.13 / 0.00 / 0.87 |
| plate_az_t0 | [-16.4, -4.2] | 0.60 / 0.27 / 0.13 | [-4.9, 4.9] | 0.13 / 0.85 / 0.02 |
| plate_el_t0 | [1.2, 9.7] | 0.20 / 0.00 / 0.80 | [1.7, 9.0] | 0.15 / 0.00 / 0.85 |
| bread_cam_x_t0 | [-0.9, 32.9] | 0.98 / 0.00 / 0.02 | [15.1, 39.8] | 0.35 / 0.65 / 0.00 |
| bread_cam_y_t0 | [-0.3, 9.4] | 0.13 / 0.00 / 0.87 | [0.3, 10.0] | 0.17 / 0.00 / 0.83 |
| bread_cam_z_t0 | [55.9, 68.7] | 0.20 / 0.80 / 0.00 | [55.1, 67.9] | 0.22 / 0.78 / 0.00 |
| plate_cam_x_t0 | [-17.0, -5.0] | 0.68 / 0.17 / 0.15 | [-5.0, 5.0] | 0.13 / 0.85 / 0.02 |
| plate_cam_y_t0 | [1.4, 9.9] | 0.22 / 0.00 / 0.78 | [2.0, 9.3] | 0.18 / 0.00 / 0.82 |
| plate_cam_z_t0 | [57.5, 68.7] | 0.13 / 0.85 / 0.02 | [58.3, 67.9] | 0.08 / 0.90 / 0.02 |
| bread_u_aria_t0 | [316.2, 506.8] | 0.93 / 0.00 / 0.07 | [393.9, 550.1] | 0.50 / 0.50 / 0.00 |
| bread_v_aria_t0 | [238.2, 293.1] | 0.12 / 0.00 / 0.88 | [241.0, 297.3] | 0.13 / 0.00 / 0.87 |

joint inside single-mode box: {"bread_x": 0.21666666666666667, "plate_x": 0.21666666666666667, "plate_y": 0.7, "gap": 1.0, "yaw": 1.0, "all": 0.03333333333333333}

sim camera: {"height": 0.436, "pitch_down_deg": 36.99999994093033, "fx_d405_640x480": 440.0, "hfov_d405_deg": 72.05474677020722, "hfov_aria_deg": 90.0}