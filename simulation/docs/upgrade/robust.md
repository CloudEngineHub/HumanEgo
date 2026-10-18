# Robustness and evaluation protocol: experiment log

Owner: the robustness session (`upgrade/robust`). Every step lists its hypothesis, the change, the
measurement (n, seeds, CI), and whether the change was kept or reverted. Per-arm rates are given
with Wilson 95 % intervals. Paired tests run on the same scenes: exact McNemar on discordant
arms, plus a paired scene bootstrap of the difference. Seeds 7777 and 8888 are development
sets; 9001–9099 are reserved for the final report.

## 0. Setup check

* Worktree `HumanEgo-wt/robust` on `origin/main` (a9773c9). One Trossen episode
  (`run_episode.py`, seed 0, grasp skill) served both breads in 46 s wall.

## 1. Evaluation protocol

**Change.** New module `humanego_sim/stats.py`:

* Wilson score intervals.
* A scene bootstrap for per-arm rates (the two arms of a scene are correlated).
* `paired()`: exact McNemar and a paired scene bootstrap.
* The seed protocol: `check_seed`, so reserved seeds need `--final`.

`evaluate()` summaries now carry `ci`. `compare_params.py` prints CIs and tests every set
against the first one on the same scenes. It accepts several `--seed` values (pooled for the
paired test), `--variation`, `--final`, and `name=params.json@'<JSON>'` sets. Its `--out` file
keeps per-scene outcomes.

**Check.** A 2-scene smoke run prints the CIs and the paired line, e.g. "pert vs clean: per-arm
25 % vs 50 %, diff −25.0 pts [−50, +0], discordant arms 0/1, McNemar p=1 → not significant".
Scene sampling without variation is identical to the old `[sample_scene(rng) ...]` list
(checked). **Kept.**

## 2. Perception realism and scene variation (all off by default)

**Change.** `humanego_sim/robustness.py`, plus hooks in `perception.py`, `env.render_*`,
`sampling.py` and the eval worker:

| option | model |
|---|---|
| `perception.mask_noise` | per-object erosion/dilation bias (persistent per episode), smooth boundary jitter, missed detections and spurious blobs per detector frame |
| `perception.distractor_confusion` | a bread-like distractor segmented as bread (per-type rate measured with the real DINO-SAM) |
| `perception.mask_source: dinosam` | the released DINO-SAM (Grounding-DINO tiny + SAM2 tiny) run on the rendered RGB, through `scripts/dinosam_server.py` in an env that has transformers + sam2 |
| `perception.depth_artifacts` | flying pixels and holes in a band at depth edges, random hole blobs (mostly persistent), per-episode depth scale error |
| `perception.calib_error` | the rendering camera is moved off the hand-eye calibration; the robot keeps using the calibration |
| `perception.latency` | the policy image is 66 + U(0, 33) ms old, 5 % of frames are not refreshed |
| scene `_var` (`sampling.sample_scenes(..., variation=)`) | bread scale ±12 %, extra bread yaw ±15°, extra plate offset ±3 cm, table height ±1 cm (the robot assumes the nominal height), light intensity / tint / key-light direction, 1–3 visual-only distractors |

Each perturbation draws from its own random stream (derived from the episode seed without
consuming it). Scene variation draws from per-(scene, kind) streams, so the base layout of every
scene is unchanged and conditions stay paired.

**Check: the clean path is unchanged.** With every option off, the GT masks, all 10 depth
frames, the settled state and every perceived object pose / keypoint set (3 scenes) hash
identically to `main`. The clean RGB render hash differs between any two runs, `main` against
itself included: the EGL renderer is nondeterministic, and nothing in this change causes it.
**Kept.**

**DINO-SAM on the renders** (`scripts/robust_mask_calib.py`, 32 scenes, seed 7777):

| condition | bread IoU (median / p10) | bread boundary bias / jitter | plate IoU | plate jitter | missed | false "bread" blobs |
|---|---|---|---|---|---|---|
| clean | 0.966 / 0.952 | 0 px / 1.0 px | 0.935 | 3.6 px (shadow) | 0 / 96 | 0 / 32 frames |
| lighting | 0.965 / 0.946 | 0 / 1.0 | 0.939 | 3.7 | 0 | 0 |
| distractors | 0.968 / 0.954 | 0 / 0.9 | 0.936 | 3.5 | 0 | 4 / 32 |
| all scene variation | 0.970 / 0.953 | 0 / 0.9 | 0.949 | 2.9 | 0 | 4 / 32 |

On clean renders the real segmenter is close to the ground truth. Its one real failure is
clutter: it took the banana for bread in 3 of 10 scenes and the yellow box in 1 of 13, and no
other distractor. The synthetic `mask_noise` level is therefore set above what DINO-SAM does on
renders (±2 px bias, 1.5 px jitter, 5 % misses, 10 % blobs per frame). Real images are harder
than renders, and this level stands in for that. The real segmenter is reported as its own
condition (`dinosam`).

## 3. Development-seed screen (Trossen, seed 7777, 32 scenes, commit 40bb979)

Tuned settings + grasp skill (`robust_report.py --name trossen_dev7777 --log`). The paired
difference vs clean is a scene bootstrap; p is McNemar.

| condition | per arm [95 % CI] | right | left | both | vs clean | p |
|---|---|---|---|---|---|---|
| clean | **92 %** [83, 97] | 94 % | 91 % | 84 % | — | — |
| masks | **80 %** [68, 88] | 81 % | 78 % | 66 % | −12 [−25, −2] | 0.039 |
| depth | 86 % [75, 92] | 94 % | 78 % | 72 % | −6 [−14, +0] | 0.29 |
| calib | 88 % [77, 94] | 84 % | 91 % | 81 % | −5 [−14, +3] | 0.38 |
| latency | 92 % [83, 97] | 94 % | 91 % | 84 % | +0 | 1 |
| bread_size | 89 % [79, 95] | 94 % | 84 % | 81 % | −3 [−14, +6] | 0.75 |
| layout | 86 % [75, 92] | 88 % | 84 % | 78 % | −6 [−16, +2] | 0.22 |
| table_height | 91 % [81, 96] | 94 % | 88 % | 81 % | −2 [−6, +3] | 1 |
| lighting | 91 % [81, 96] | 94 % | 88 % | 84 % | −2 [−8, +3] | 1 |
| distractors | 89 % [79, 95] | 91 % | 88 % | 81 % | −3 [−9, +0] | 0.5 |
| **all** | **70 %** [58, 80] | 69 % | 72 % | 59 % | −22 [−39, −6] | 0.004 |

Only mask noise and the combination are significant on 32 scenes.

**Audit** (`robust_audit.py trossen_dev7777 clean masks all`): what lost the arms that clean served.

* masks: 10 arms lost. **8 of them had a missed detection**: the bread (ep 5, 13, 21, 26), or the
  plate (ep 16 and 17, both arms each; the plate mask is shared by both arms, so its miss costs
  two). With an empty mask the object's frame is the identity and the policy closes on air. The
  boundary errors (±2 px bias, 1.5 px jitter) move the perceived bread by only 1.6 mm (open
  loop) and lose almost nothing.
* all: 18 arms lost. 8 are the same missed detections. 2 are the box taken for bread (ep 2: the
  right arm's bread perceived 16 cm off; ep 24: 33 cm off). The rest are spread over the
  calibration / layout / size episodes, with no single cause.
* A first reading blamed a "flipped plate frame" (38–108° rotations vs clean). Every one of
  those rotations is a plate or bread with no keypoints (identity frame), i.e. a missed
  detection, not frame instability.

## 4. Perception-side mitigations (development seeds)

**4a. Multi-frame mask fusion** (`perception.mask_frames: 5`). Run the detector on 5 frames, drop
empty frames and frames whose area is off the median by > 35 %, then take a pixel majority
vote. Hypothesis: misses are transient, so fusion removes them; a persistent confusion (the
distractor) stays.

* Open loop, seed 7777, 64 breads: failures 4 → 1 (masks) and 6 → 3 (all). The clean perception
  is identical: the GT frames are identical, so the fused mask is the same mask.
* Closed loop, seed 7777, masks: **80 % → 89 %** per arm (clean 92 %).
* Pending in `scripts/robust_batch.sh`: all on 7777, and clean / masks / all on 8888.
* Clean case: not re-run closed loop, because the perception input is bit-identical (fusing
  identical GT masks).

**4b. Outlier rejection in the keypoint lift** (`perception.lift_reject`). Open loop, seed 7777:

* `erode_px: 3` (lift only from pixels ≥ 3 px inside the mask): no gain under depth (shift 1.8
  vs 2.1 mm) or masks (failures unchanged, 4). It moves the clean perception (centre error vs
  GT 1.97 vs 1.77 cm, axis p90 15.8° vs 14.6°).
* `table_margin: 0.004` (drop keypoints lifted onto the calibrated table plane): never fires, so
  the output is identical.

The lift is not where arms are lost: boundary and depth noise move the bread by about 2 mm.
**Neither variant is adopted.** The code stays as options, off by default.

## Open issues / hand-over

* `scripts/robust_batch.sh` (resumable) runs everything that remains, in this order:
  1. Fusion on the dev seeds.
  2. Reserved set 9001 (n = 100) clean / masks / dinosam / all for all three robots.
  3. The fused rows on 9001.
  4. The other single perturbations on 9001.

  Output goes to `runs/robust_report/<name>/` and the log to `runs/robust/batch.log`. When it
  is done: `robust_report.py --table <names>`, the README table, `robust_audit.py` on the worst
  reserved conditions.
* Distractor confusion: a mask component that is not the bread wins when it is larger. A
  workspace gate (drop detections whose lifted centroid lies outside the reachable table area)
  would remove it. Not tried: the distractors are placed outside the workspace by
  construction, so the gate would be circular here.
* Needs a runner hook (controller session's `runner.py`): the latency option delays only the camera
  image. Proprioception delay and inference-latency jitter (pending delivery at P ± jitter) would
  need runner changes.
* Needs a JS port only if a mitigation becomes default-on (`web/js/live/perception*.js`); none has so far.
* The physics session will change bread-physics defaults on main. Re-run the final report after
  that lands (the scripts resume / re-run per condition).
