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
