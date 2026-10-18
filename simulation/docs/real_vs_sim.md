# Simulation vs the real robot: serve_bread sim-to-real validation

How faithful is the MuJoCo replica, judged on all the real evidence that exists, and what is set up
to measure it properly. Experiment log: `docs/upgrade/real.md`. Raw notes with file:line
citations: `docs/upgrade/real_notes/` (paper numbers, video inventory, human-data object
statistics). Tools: `scripts/real/` (below).

## Summary

1. **The sim's main number does not match the real one.**
   * The paper's real protocol is single arm, croissant, released checkpoint, production controller: **95 %** (38/40).
   * The sim on the same protocol and controller: **12.5 %** (5/40, seed 7777). The tuned sim controller with the grasp skill reaches 82.5 %.
   * Nearly all sim production failures are grasps that slip at lift.
2. **Most likely cause: the table height relative to the robot.**
   * The sim places the arm bases 0.215 m above the table. That figure was inferred from a `safe_z_min` comment.
   * Two real front-view frames give **≈0.15 m** (camera ≈0.37 m instead of 0.436).
   * At the measured height, sim production rises from 12.5 % to **67.5 %** single-arm, and from 41 % to 56 % per arm bimanual (right 72 %, left 41 %). With the table 10 cm higher: 75 %.
   * The tuned controller is insensitive to it single-arm (82.5 → 85 %), but bimanual drops 92 → 80 % per arm: it was fitted at the assumed height.
   * The remaining gap to the real 95 % is grasp misses (9/40) and breads on the rim (3/40).
3. **The video agrees with the sim on:**
   * transport duration (scale-free, relative to the wrap-up);
   * lift height (production);
   * left/right sequencing.
4. **The video disagrees with the sim on:**
   * the release: real sets the bread down, sim drops it from 5–8 cm;
   * the grasp point: real grasps nearer the bread's end;
   * reach speed: sim production is ~1.5–2x slow.
5. **The tuned controller's motion is not real-like.** It lifts 40 cm (real ~20–25) and its transport has 16 speed peaks (production 1–2, real smooth).
6. **Correlation.** Only the released checkpoint exists, so there is no checkpoint-level sim–real correlation. The checkpoints needed are listed in §4.
   * The paper's cross-condition study (same checkpoint) differs by ≤5 points between conditions, within its ±3.4-point standard error. The sim cannot be validated against it beyond "no large drop".
7. **Logging and replay are ready.**
   * AriaMimic branch `sim-logging` records everything needed for a one-to-one comparison.
   * `replay_real.py` rebuilds the scene and compares (tested end to end on simulated logs).
   * §6 lists 2.5 h of real experiments, with the sim's predictions.

## 1. Inventory of real evidence

### 1.1 Paper (`AriaMimic/paper/`)

`AriaMimic/paper.txt`, `paper.pdf` and `arxiv_abs.html` are **another paper** (AlphaNIST, NVIDIA,
arXiv:2610.13354). The HumanEgo numbers come from `paper/example.tex` (CoRL + appendix),
`paper/arxiv_v3/main.tex`, `paper/icra_2026/main.tex`, the rebuttal and the plotting scripts in
`paper/figs/*.py`. The values below appear only in those scripts.

| real result (Serve Bread) | value | trials |
|---|---|---|
| HumanEgo, 30 min human video (released checkpoint) | **95 %** | 38/40 |
| HumanEgo, 15 min | 82.5 % | 33/40 |
| ACT, 30 min robot teleop | 52.5 % | 21/40 |
| SPOT / ZeroMimic / Track2Act / PointPolicy / EgoZero | 62.5 / 47.5 / 35 / 12.5 / 5 % | 40 each |
| data efficiency, no aux, 1 / 2 / 4 / 8 / 12 / 18 / 24 / 30 min | 0 / 2.5 / 12.5 / 37.5 / 67.5 / 85 / 92.5 / 95 % | 40 each |
| hand tracker Aria / WiLoR / HaMeR / MediaPipe | 95 / 45 / 32.5 / 0 % | 40 each |
| human:robot co-training 0 / 25 / 50 / 75 / 100 % | 65 / 72.5 / 77.5 / 90 / 95 % | 40 each |
| cross-condition, **mean of Serve Bread + Downstack Cups**: Trossen / Franka / UR10 / ZED camera / novel height | 91.25 / 88.75 / 87.5 / 90 / 86.25 % | 80 each |

**Protocol** (CoRL App. "Task Details"):
* Single arm (right).
* One croissant and a dinner plate, plate left, bread right.
* Bread–plate horizontal offset U[0, 50] cm; depths independent.
* "Arbitrary, non-aligned" start pose; the real `Control.yaml` has `go_initial_pose: True`, which is a fixed pose.
* Success: the croissant rests on the plate, any orientation.
* Failure modes are defined (off the plate; slip in transport) but **never counted**.
* The rebuttal gives detection / segmentation failures below 5 %, across all tasks.
* No rollout durations or time limits are given.

The bimanual croissant + danish setup appears only in the videos. Other version conflicts are in
`real_notes/paper_real_numbers.md`:
* the demo counts (45 vs 60);
* the data-efficiency x-axis, relabelled from an older 40-minute scale;
* the trial count in the aux-loss script comment.

### 1.2 Real videos

All real-robot footage comes from two edits:
* **A** (labelled x8): one bimanual rollout, `website/static/videos/serve_bread.mp4`.
* **B–D** (labelled x16): a montage, `website/static/videos/cross_condition_3.mp4`.
  * B1–B4 are single-arm runs.
  * B5 is the canonical bimanual run.
  * B6 has swapped breads and a raised plate.

`full_demo.mp4`, the ICRA video, the twitter clip and the gifs are copies or sub-clips of these. The footage is third-person only: no robot-camera view, no failures, no timer.

**The speed labels are wrong.** The real wrap-up after a release is `open_gripper` (blocking, 1.0 s) followed by `go_home` (3.0 s), unchanged in AriaMimic since March 2026:
* A shows it in 9.5 frames (1.3 s at x8);
* B5 / B6 in 4–4.5 frames (1.1 s at x16);
* B1 in 2 frames.

Each rollout is therefore calibrated with motions of code-fixed duration: the wrap-up, `go_initial` (2.5 s) and the left arm's 6 s activation delay. A comes out at ≈x22 (x18–25) and B5 / B6 at ≈x47–54. B1's rulers are too short to use (x75–100 ±50 %). Frame annotations: `scripts/real/real_video_annotations.json`. Calibration and timing: `scripts/real/video_timing.py`.

### 1.3 Real inference code and config (`AriaMimic/inference`, `cfg/inference/serve_bread`)

The sim ports `run_bimanual`, `AsyncRobotController` and the driver behaviour faithfully. Differences found:
* **Wrap-up.** Real: halt, open 1.0 s (blocking), `go_home` 3.0 s, then the motors go idle and the arm hangs limp. Sim: open with a 0.8 s wait, then home.
* **Policy loop period.** Sim assumes 0.3 s. The real loop runs DINO-SAM + LaMa + two inferences + visualisation and is not measured anywhere.
* **D405 intrinsics.** Sim uses a nominal `fx_raw` = 660; the real K is read from the camera at run time and never saved.
* **Single-arm wrap-up.** The real single-arm `run()` path differs from bimanual: non-blocking open, 0.5 s sleep, non-blocking home.
* **Release trigger.** Real releases by idle-done (hand within 0.13 m of the plate for 1.5 s while the ICT grasp flag is on), by policy done (> 0.99), or by operator keys `[` / `]`. The grasp flag is the measured opening < 50 %. The sim adds "gripper stalled closed" as a grasp (`grasp_state_source`); on the real robot that rule does not exist.
* **Saved outputs.** Each real run saves `raw.mp4`, `inpaint_kpts.mp4` and `inpaint_kpts_traj.mp4` plus the init diagnostics in `results/policy/<stamp>/` on the robot PC. **None of these are on this machine.**

### 1.4 Human data (`data/serve_bread/aria`, 61 episodes)

* There is one croissant type and one light-blue plate in all 61 episodes; no danish.
* The layout is always plate left, croissant right.
* Ep 033 has an implausible bread triangulation and is excluded. Ep 060's per-frame intrinsics files were re-extracted at 1920 × 1080 after preprocessing (wrong K for any tool reading them).
* Measured with SLAM-triangulated contour keypoints, masks and semidense points: §3.

### 1.5 What does not exist

* Real per-trial logs and failure counts.
* Real-time video and robot-camera footage.
* Any checkpoint other than the released one.

## 2. Quantitative comparison: real video vs sim (Trossen, bimanual, dev seed 7777, 32 scenes)

Sim runs are logged in `runs/real_dev7777/{production,tuned}`; metrics come from `scripts/real/sim_phases.py`. Real values are frame annotations; seconds use the per-rollout calibration. "Per wrap-up" is phase duration ÷ the same rollout's release→home time, which is scale-free.

* Sim production: 40.6 % per arm.
* Sim tuned (`configs/tuned/trossen_wxai.json` + grasp skill): 92.2 % per arm.

| quantity | real | sim production | sim tuned | verdict |
|---|---|---|---|---|
| reach: first motion → grasp (s) | 1.0–3.7 (A-left, B5, B6) | R 4.8, L 4.2 | 2.4 / 2.8 | production ~1.5–2x slow |
| reach per wrap-up | 0.43–0.90 | 1.2–1.3 | 0.67–0.79 | production slow, tuned matches |
| grasp → lift-off (s) | 0.7–4.2 | 1.5–1.8 | 0.7 | both in range |
| transport: lift → release (s) | A 15–22; B5 / B6 8–12 | R 11.8, L 16.1 | 8.0 / 8.3 | production inside the real range, tuned at its edge |
| transport per wrap-up | 2.6–5.4 | 3.3–4.5 | 2.2–2.3 | production matches, tuned slightly fast |
| left − right timing | left starts ≥6 s later; right releases first (Δ grasp 4–6 s at the calibrated speed) | Δ close 6.0 s, Δ open 8.4 s | 6.7 / 7.2 s | consistent |
| grasp point along the bread (0 = middle, 1 = tip) | ≈0.7: fingers over the outer ~30 % (A, B5, croissant and danish) | 0.39 R / 0.47 L | 0.56 / 0.50 | real grasps nearer the end |
| grasp height | fingers reach down to the table, plates span the bread | tool point +1.5 / +1.3 cm above centre | −0.3 / +0.1 cm | unknown on real |
| release | **bread set down** (touches plate / croissant), then opens | bread bottom 7.9 cm (R) / 4.9 cm (L) above the plate floor; 8 of 26 served below 2 cm | 4.4 / 3.2 cm | sim drops, real places |
| release trigger | unknown | idle-done in 64 of 64 arms | idle-done 59, timeout 2 | — |
| peak lift above table | R ≈25 cm, L ≈20 cm (plate = 24.5 cm scale, A) | R 26, L 15 | **40–41** | production matches; tuned lifts 2x too high |
| smoothness: speed peaks in transport | continuous, no visible stops | 1–2 | **16–17** (0.33 m/s peaks, re-plan rate) | tuned is jerky |
| mean transport speed | ~5 cm/s (A, calibrated) | 3–5 cm/s | 11–12 cm/s | production matches |
| failure modes | none shown; paper: slip / off-plate, not counted | 29 of 64 arms slip at lift, 5 dropped in transit, 3 closed on air | 8 of 64 arms fail | — |

**Discrepancies, ranked by likely impact on success and on validity:**

1. **Success level.** Real 95 % vs production sim 12.5 % on the paper protocol (§4); failures are slips at lift.
   * Prime suspect: table height (§3, §4).
   * Then the grasp point and how a real croissant squashes in the fingers.
2. **Release: placing vs dropping.**
   * Every sim release is idle-done, about 13 cm from the plate pose. The real robot lowers the bread onto the plate first.
   * The danish bouncing off the croissant, a failure the physics session tuned crumb damping for, may therefore be a sim artefact.
   * Hypothesis: with a thick bread the real 50 % opening flag never fires. Then neither idle-done nor latching runs, and the release comes from the policy or the operator, after the policy's own place motion.
   * The sim's "stalled closed" grasp rule would then hide this. The real log's `done_reason` and `g_pos` settle it.
3. **Grasp point.** End grasps on the real robot vs off-centre grasps in the sim; this interacts with 1 and 2 (a thinner, softer end squashes past 50 %).
4. **Tuned-controller realism.** Its motion is faster, higher and jerkier than the real robot's, so the 92 % per arm says little about the real robot. The autotune score should penalise jerk and lift height (`sim_phases.py` has speed peaks, LDLJ and lift).
5. **Reach speed** (production too slow). Candidates: the policy-loop period (sim 0.3 s; real unknown, now logged as `loop_dt`) and the table height.
6. **Minor.** D405 intrinsics (nominal in sim); wrap-up timing; the start pose (paper "arbitrary" vs fixed); the paper's wider offset range than `sampling.py` single.

## 3. Initial conditions and objects

The human data has no depth; positions come from MPS SLAM, which is metric and gravity-aligned. Frame H:
camera foot on the support, x = heading, y = left. Full statistics:
`real_notes/human_object_stats_summary.json`.

| | human data, median (p5–p95) | simulator |
|---|---|---|
| croissant length | 14.5 cm (14.0–15.4) | 13.8 |
| croissant width | 7.3 cm (5.4–8.1), mask corrected for side view | 6.2 |
| croissant height | 5.4 cm (4.9–7.3) | 5.0 |
| plate diameter | 24.7 cm (24.3–25.4) | 24.5 ✓ |
| bread–plate distance | 27.9 cm (22.2–31.4) | single 27.7 (17.9–37.4) ✓ wider |
| bread yaw | 86° (67–101) | 90 ± 35 ✓ wider |
| bread forward from the camera foot | 36 cm (30–44) | single 47 (40–54); 22 % of human inside |
| bread depth along the optical axis | 52 cm (48–60) | 62 (57–68) |
| bread below the optical axis | 14° (9–20) | 4° (0–9) |
| head-camera height / pitch | 45 cm (38–48) / 34° (29–40) | 43.6 cm / 37° |
| hand–bread distance at the first training frame | 24 cm (19–36) | arm start pose: fixed |

**Real rig** (video, §1.2 frames with the arms folded): camera-above-base / base-above-table = 1.47
in two frames. With the calibrated 0.221 m camera-above-base, the real bases are **≈0.15 m above
the table** (sim 0.215 m) and the camera ≈0.37 m (sim 0.436 m).

Proposed values, for the robustness and physics sessions: **`docs/proposed_scene.json`**.
* Table height (`rig.cam_height` 0.366, test 0.346–0.396).
* Croissant 14.5 × 7.3 × 5.4 cm, or `scene.bread.right.scale` 1.06.
* A single-arm offset range widened towards the paper's.
* An in-distribution diagnostic scene set (human-matched camera-frame depth / elevation).
* `docs/proposed_portable_controller.json`: the tuned Trossen values restricted to knobs that exist in the real `Control.yaml` (alpha 0.175, `duration_smooth_param` 2.0, grasp threshold 0.65, idle distance 0.083 m, idle time 2.65 s), for real experiment E4.

## 4. Sim–real correlation

**Checkpoint level: not possible.** Only the released checkpoint exists, locally and on HF. Needed, from `runs/serve_bread/` on the robot or training PC:

| checkpoint | real Serve Bread | sim needs |
|---|---|---|
| `ScalingLaw/01_HumanEgo_HumanData/N{1,3,6,12,24,36,48}` (no aux; N60 = released) | 0 / 2.5 / 12.5 / 37.5 / 67.5 / 85 / 92.5 % | nothing (camera frame, absolute) |
| HumanEgo-15 (the main-table 15-min model) | 82.5 % | nothing |
| `HandTracking/0{1,2,3}_{MediaPipe,WiLoR,HaMeR}` | 0 / 45 / 32.5 % | nothing |
| `DataSource/*` co-training mixes (25 / 50 / 75 % configs do not exist) | 72.5 / 77.5 / 90 % | nothing |
| `XFrame/01_AnchorFrame` | 87.5 % | anchor-frame support in `runner.py` (currently NotImplemented) |
| ACT, SPOT, … baselines | 52.5 … 5 % | their inference code: not portable to this sim |

The data-efficiency series would give 8 points on one curve; it is the best target. The 60 demos are local (`data/serve_bread/aria`), so the N-series can be retrained (400 epochs each). Retrained models are not the paper's weights, but the trend should match.

**Condition level, with the released checkpoint** (the paper's cross-condition study). Sim: single arm, 32 scenes, seed 7777, production / tuned.

| condition | real (mean of 2 tasks, n = 80) | sim production | sim tuned |
|---|---|---|---|
| Trossen + D405 (reference) | 91.25 (Serve Bread alone: 95) | 12.5 (n = 40) | 82.5 (n = 40) |
| Franka | 88.75 | 21.9 | 75.0 |
| UR10 (sim: UR5e + 2F-85) | 87.5 | 12.5 | 75.0 |
| novel camera ZED | 90.0 | 12.5 | 81.2 |
| novel height (sim: table 10 cm higher) | 86.25 | **75.0** | 93.8 |

* All real differences are within ±1.5 standard errors (±3.4 points); the real study cannot rank conditions.
* Over the 4 like-for-like conditions (Trossen, Franka, UR, ZED), tuned sim vs real: Spearman 0.95 (p = 0.05), Pearson 0.93. Including "height": −0.05.
* The "height" mapping is a guess: the paper's novel height probably raised the objects, not the robot.
* Conclusions:
  * like the real robot, the tuned sim loses ≤10 points across embodiments and cameras;
  * production sim fails everywhere at the nominal geometry;
  * the one large sim effect is table height.

**Table-height sweep** (seed 7777; single arm 40 scenes, bimanual 32). The table is the only
thing that moves: the camera keeps its calibration to the arm bases.
* Production rises monotonically as the table comes up (12.5 → 40 → 67.5 → 75 %), plateauing at ~75 %.
* The tuned controller is flat on the single arm (82.5 → 85 %). It loses 12 points per arm bimanual (92 → 80 %): it was tuned at the assumed height.
* **Re-tune at the measured height** before trusting tuned numbers.

| `rig.cam_height` (bases above table) | single production | single tuned | bimanual production (per arm) | bimanual tuned |
|---|---|---|---|---|
| 0.436 (0.215 m, current) | 12.5 % | 82.5 % | 40.6 % | 92.2 % |
| 0.396 (0.175 m) | 40.0 % | | | |
| 0.366 (0.145 m, video estimate) | **67.5 %** | 85.0 % | 56.2 % (R 72, L 41) | 79.7 % (R 81, L 78) |
| 0.346 (0.125 m) | 75.0 % | | | |
| 0.336 (0.115 m) | 75.0 % | 93.8 % | | |

## 5. Real-robot logging and one-to-one replay

**AriaMimic branch `sim-logging`** (pushed to that branch only; `master_v2` untouched).
* Files: `inference/SimLogger.py` plus hooks in `Inference.py` / `InferenceController.py`; usage in `inference/SIM_LOGGING.md`. Logging is on in `cfg/inference/serve_bread/Control.yaml` (`sim_log`); `HUMANEGO_SIM_LOG=0` turns it off.
* Per run (`results/policy/<stamp>/sim_log/`, format `humanego-real-log/1`):
  * calibration, configs, checkpoint;
  * the t = 0 perception: RGB, depth frames, masks, per-side keypoints and poses;
  * per policy step: RGB-D, the policy-input image, plan, grasp and done outputs, latch state, object poses, arm state;
  * every controller command: target, caged, EMA output, goal time, gripper;
  * events: grasp on/off, **done reason**, wrap-up phases, keys, manual override;
  * a 50 Hz joint / gripper / EE sampler;
  * the operator outcome.
* The control loop is unchanged; writers run in threads.
* `python -m inference.SimLogger selftest <dir>` checks the format without hardware.

**`scripts/real/replay_real.py <log>`** has four modes.
* `scene`:
  * fits the table plane in the logged depth (camera height and pitch vs the sim rig);
  * rebuilds plate and breads from the logged perception, then refines them until the simulator's own perception of the rebuilt scene reproduces the logged one;
  * compares sizes, intrinsics and images.
* `kinematic`: drives the sim arms through the recorded joints and gripper commands. Gives the FK residual (model check) and whether the bread ends where it did on the real robot (contact / bread physics check).
* `closed`: reruns the released policy from the rebuilt scene with any `--params` / `--set`. Compares phases, grasp point relative to the perceived bread, release height and EE-path DTW.
* `policy`: re-infers on the logged observations. Checks the plans against the logged ones and a second-noise floor. Before the first grasp it also infers on the sim rendering of the same state, measuring the visual gap in plan space.

**Tested** on `scripts/real/sim_as_real.py`, which writes a simulated episode in the real format with known truth. Production, both arms failing:

| check | result |
|---|---|
| table plane | camera 0.4356 m / 37.07° (truth 0.436 / 37.0) |
| scene from perception | first guess 1.0–2.1 cm / 1–12° off (keypoint bias towards the camera); **refined 0.02 cm / 0.2°** |
| kinematic replay | outcome agrees 2/2; FK residual median 0.3–0.45 cm |
| closed-loop rerun | phases within 0.1–1.6 s; EE-path DTW 3.1–4.1 cm (stochastic physics / GPU) |
| policy replay | plan error 0.2 cm = noise floor 0.2 cm; grasp / done identical; sim-image plan difference 0.0 cm (sim log) |

`scripts/real/sim_phases.py` gives the same metrics for any sim run. `scripts/real/video_timing.py` does the video calibration.

## 6. Real experiments to run (≈2.5 h of robot time)

Check out AriaMimic `sim-logging`. Film every trial at real time (60 fps phone, front view) and show the run folder name in the clip. Label each trial with:
`python -m inference.SimLogger label <dir> right=… left=… --bread_right … --condition …`.

| id | what | trials | checkpoint / settings | measures | sim prediction |
|---|---|---|---|---|---|
| E0 | selftest + 1 logged rollout | 1 | released, production | `loop_dt` (sim assumes 0.3 s), files complete | — |
| E1 | 5 static scenes at tape-measured positions (no policy), plus a 5-pose joint sweep | 5 + 1 | — | **table plane / camera height** (sim 0.436 m vs video ≈0.37 m), reconstruction error, FK residual | — |
| E2 | paper protocol: single right arm, croissant, offsets 15–50 cm, yaw 60–120° | 20 | released, production `Control.yaml` | success, `done_reason`, `g_pos` while holding (does the 50 % flag fire?), release height, phases | 12.5 % at the sim's table height; at the measured height: §4 |
| E3 | bimanual, croissant right, danish left | 20 | released, production | per-arm success, release behaviour, L/R timing | 41 % per arm (sim table height) |
| E4 | E3 with `docs/proposed_portable_controller.json` in `Control.yaml` | 20 | released | does the sim-predicted gain transfer? | 62.5 % per arm (single arm 45 %) vs E3 |
| E5 | if found or retrained: N12 and N24 data-efficiency checkpoints on the E2 protocol | 10 each | — | checkpoint-level correlation | run in sim when available |

Then, for every log:

```bash
python simulation/scripts/real/replay_real.py <sim_log dir>
```

This gives per-trial sim-vs-real outcome, phases, grasp and release. To find the sim settings that best match E2 / E3, rerun with `--set rig.cam_height=…`.

## 7. Recommendations for the other sessions

* **Robustness / eval** (`sampling.py`, `evaluate.py`):
  * report the paper protocol (single arm) next to bimanual;
  * add the table-height sweep and the in-distribution scene set;
  * widen the single-arm offsets;
  * sample the croissant size (`docs/proposed_scene.json`).
* **Physics** (`softbody.py`, `scene.py`):
  * set `rig.cam_height` from E1 (or 0.366 until then) before any further physics tuning;
  * re-check the crumb damping and drop fixes once releases are placements (real) rather than drops;
  * after E2, calibrate squash so the real 50 % opening flag fires as it does on the robot, then drop the sim-only "stalled closed" grasp rule.
* **Controller** (`controller.py`, `runner.py`):
  * add smoothness (speed peaks or LDLJ) and peak-lift terms to the autotune objective;
  * keep a "real-portable" tuned set (`Control.yaml` knobs only) for real experiments;
  * time the policy loop from E0's `loop_dt`.
* **Engineering**:
  * run `sim_as_real.py` + `replay_real.py --modes scene kinematic` in CI as a format / regression test (≈2 min, one GPU episode).
