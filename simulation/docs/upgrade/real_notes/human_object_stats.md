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
  exactly table aligned up to the table tilt, which measured 0.1-1.9 deg). Also reported: the
  OpenCV camera frame at t0 (what the policy sees, `frame_mode: camera_frame`) and the angles /
  pixel position of the objects in the 640x480, f = 320 Aria image.

Measurements
* Support plane (the surface the bread rests on: box top, desk or counter): mode of semidense
  point heights in a 25 cm ring around the bread (outside the bread hull dilated 2 cm and outside
  the plate disc), restricted to a 3 cm window just below the 10th percentile of the bread
  contour heights. A free search lands on the wrong surface in a third of the episodes (desk
  under the box, a lower box, the floor), hence the window; it was calibrated on the 43 episodes
  where the free search was self-consistent (support = contour p10 - 1.2 cm, p5-p95 0.7-1.7 cm).
  Found with 75-1340 points in all 61 episodes.
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
