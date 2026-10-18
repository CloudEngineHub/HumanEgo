# Engineering quality and website tooling: experiment log

Owner: the engineering session (`upgrade/eng`). Scope: config profiles, the pytest suite and CI,
the one-command site rebuild, the website's weight and reach, and the README's top-level
structure. Every step lists the change, how it was checked, and the result.

## 0. Setup check

* Worktree `HumanEgo-wt/eng` on `origin/main` (a9773c9). Runtime assets are symlinked; the web
  build outputs (`web/data`, `web/assets/*`) are copied, because the packer rewrites them.
* One production Trossen episode (`run_episode.py`, seed 0) served both breads (47 s wall).
* The site as deployed (humanego-ai.github.io 64fab83) was snapshotted as the "before" build.

## 1. Config profiles (`--profile production|tuned`)

**Change.** New `humanego_sim/profiles.py`. `env.load_config` applies a named profile between the
per-robot calibration and explicit overrides:

    serve_bread.yaml  <  per_robot.<robot>  <  profile  <  overrides (--set, autotune candidates)

* `production` (the default) leaves `serve_bread.yaml` unchanged.
* `tuned` applies `configs/tuned/<robot>.json` `params` plus its `requires` (the grasp skill on).

The profile travels inside the overrides dict (`{"profile": "tuned"}`), so it also reaches the
evaluation workers. `--profile` was added to `run_episode`, `evaluate`, `compare_params`,
`autotune`, `autotune_validate`, `build_web` and `rebuild_site`. In `evaluate.py` and
`compare_params.py` only the argument parsing changed. `autotune_validate` now also records its
fixed overrides in `final.json` (`"fixed"`), so later notes can say what the baseline ran with.

**Check 1: identical configs.** The resolved config of `--profile tuned` equals the old route
(`params_to_overrides(tuned params)` + `--set control.grasp_assist={"enabled": true}`) as a dict,
for 3 robots × 2 cameras × 2 arm modes (12/12). This is a unit test
(`test_tuned_profile_equals_the_old_set_route`).

**Check 2: same scenes, compare_params.py.** 4 scenes per robot, seeds 7777 and 8888, 2 workers,
GPU, old route vs `--profile tuned`:

| robot | seed | old route per-arm | profile per-arm |
|---|---|---|---|
| Trossen | 7777 | 0.750 | 0.750 |
| Franka | 7777 | 0.500 | 0.500 |
| UR5e | 7777 | 0.875 | 0.875 |
| Trossen | 8888 | 1.000 | 1.000 |
| Franka | 8888 | 0.500 | 0.375 |
| UR5e | 8888 | 0.750 | 0.750 |

Per-arm rates match in 5 of 6 cells (Franka/8888: 4/8 vs 3/8 arms). Continuous metrics
(distance to the plate) differ episode by episode.

**Check 3: is that the profile or the simulator?** The two routes were rerun on CPU (2 torch
threads, same scenes, fresh env per route). The trajectories still differ: the first difference
is a last-digit flip in the logged qpos (1e-5), sometimes before any policy output differs.
It then grows chaotically. The configs are identical, so this is run-to-run numerical
nondeterminism, not the profile. A control run (same route twice, single-threaded, deterministic
torch) is in §1a.

### 1a. Determinism control

One 12 s Trossen episode (seed 7777 scene) was run three times on CPU, single-threaded, with
`torch.use_deterministic_algorithms(True)` and a fresh env per run: old route, old route again,
and `--profile tuned`.

| pair | identical | first differing frame |
|---|---|---|
| old vs old (repeat) | no | 64 (2.1 s) |
| old vs profile | no | 134 |
| old (repeat) vs profile | no | 64 |

**Result.** The same route repeated is no closer to itself than to the profile. The simulator is
not bit-reproducible run to run, even on CPU with one thread: logged qpos differ by 1e-5 at
first, then the episode diverges chaotically. So "same results" can only mean the same
computation. The resolved configs are identical (check 1), and the outcome counts agree within
run-to-run noise (check 2).

The source of the nondeterminism is not identified. The policy noise is seeded per episode,
`env.rng` is seeded per env, and torch is deterministic here. Candidates are EGL rendering
feeding perception, and the soft-body contact solve. That is for the robustness/physics owners
(see follow-ups).

## 2. Tests and CI

`simulation/tests` (pytest, `simulation/pytest.ini`):

* `test_config_profiles.py`: defaults are the real rig; deep_update; per_robot merge; overrides
  beat per_robot; production is the default; tuned applies every tuned param and the skill; it
  equals the old `--set` route (robot × mode); overrides apply on top of a profile; errors; list
  keys; tuned-file schema; every CLI takes `--profile`.
* `test_grasp_skill.py`: the grasp-skill state machine on a fake arm with a known bread pose:
  refined pinch point (along/lateral/height caps), close on arrival, hold, holding, approach
  timeout, empty-grip re-grasp (lower and more central, cooldown), the attempt cap, aborting when
  the policy lets go, and the production path with the skill off.
* `test_driver_ik.py`: SE(3) round trip, cam↔base conversions, IK round trip (8 random targets per
  arm, ≥ 7/8 within 1 mm / 0.01 rad), Cartesian command → physics → measured pose. Finding: the
  Trossen start pose sits 2.6° from wrist joint 4's lower limit, so a pure 4 cm lift at a fixed
  orientation is not reachable (5 mm / 3.4° short). The test moves down instead.
* `test_perception_geometry.py`: the released perception on a synthetic top-down scene (two
  elliptical breads, a plate): mask split, MAD/DBSCAN outlier removal, keypoints on the
  contour, the PCA frame along the long axis, left-arm mirroring, and `runner._perceived_geometry`
  (centre at half the top height, axis, extent, place height).
* `test_smoke_episode.py` (marker `sim`): 5 s closed loop per robot × profile. It skips without
  the checkpoint or an OpenGL context. Locally 6/6 pass in 54 s.
* `test_web_build.py`: bit-exact packed rollouts (synthetic, and the built ones), the JSON
  fallback for unknown sections, GLB quantization (positions within 2e-5 m, area kept, flat
  meshes lose normals), WebP bread GLBs keep float32 positions, manifest numbers come from data,
  validation catches errors, the tuning note, and the deploy plan.

CI: `.github/workflows/sim-tests.yml` runs `pytest simulation/tests -m "not sim"` on ubuntu with
CPU torch and `simulation/requirements-ci.txt`, with `MUJOCO_GL=disable` and no checkpoint
download. It was checked in a clean Python 3.11 env built from those requirements (§2a).

### 2a. CI check

A clean conda env (Python 3.11, CPU torch from the PyTorch CPU index, then
`requirements-ci.txt`) with `MUJOCO_GL=disable` and no checkpoint: `pytest simulation/tests -m
"not sim"` gives 54 passed, 6 deselected in 15 s. After rebasing onto main (robustness, physics,
parity, protocol commits), the synthetic perception env needed `cfg`, `rng` and a tiny
`policy_cam` model for the robustness layer. The full local suite then passed: 56 + 6 smoke
episodes. The README results table is now generated, and a test fails when it drifts from
`configs/tuned`.

## 3. One-command site rebuild

`python simulation/scripts/rebuild_site.py [--workers 2]` (never deploys):

1. It builds every site config with its own profile (`SITE_CONFIGS`, `robot:camera:mode:n:profile`)
   through `build_web.build_config`.
2. It writes the manifest with `humanego_sim/web_manifest.py`:
   * the results table comes from the build's own rollouts;
   * held-out numbers and the tuning note come from `configs/tuned/<robot>.json`;
   * the tuning plot comes from `autotune_runs/<run>/iterations`.
   No number is hand-typed. `--tuning_note` was removed. The old note quoted a "production"
   baseline from `soft_v6/final.json`, but that trial ran with the grasp skill forced on, which
   final.json did not record, so that baseline is no longer quoted.
3. It packs the build (§4).
4. It records `media/teaser.mp4` + `teaser.jpg` and `media/<robot>.mp4` with `web/tools/record.py`
   from a local gzip server.
5. It validates the manifest against the files and writes `runs/site/report.json`.

Then run `python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim --teaser`.
`--teaser` also refreshes the main page's `static/videos/sim_teaser.mp4` and its poster.

## 4. Website weight and reach

**Measurement.** `web/tools/serve.py --gzip` serves like GitHub Pages: it gzips JSON, JS, GLB,
ONNX and .bin, but not media; this was checked against the live site's headers.
`web/tools/measure_load.py` runs headless Chrome with a cold cache and records CDP wire bytes per
resource class. It can add Chrome's Fast 4G / Slow 4G throttling, a phone profile (Pixel 8
viewport, touch, 4× CPU slowdown), WebGPU removed, and Live mode.

**Before** (deployed build, desktop): 5.59 MB over 48 requests. Breakdown:

| item | size |
|---|---|
| bread GLBs | 3.73 MB |
| Trossen GLB | 0.77 MB |
| three.js from jsdelivr (24 modules) | 0.49 MB |
| textures | 0.34 MB |
| rollout + policy video | 0.17 MB |

Time to ready:

| setting | time to ready |
|---|---|
| unthrottled | 1.31 s |
| Fast 4G | 5.62 s |
| phone on Slow 4G | 28.9 s |
| the real site from this box | 2.1 s |

Live mode before:

| setting | ready after | download | 3 re-plans |
|---|---|---|---|
| WebGPU | 6.0 s | 64 MB | 2.8 s |
| WASM fallback | 3.6 s | 36 MB | 4.1 s |
| phone, WASM | 7.8 s | | 13.8 s |

The whole site is 169 MB on disk.

**Changes** (`humanego_sim/web_pack.py`, applied in place after a build, idempotent):

* **Rollouts → `.bin`.** Every JSON number is stored as an integer at exactly the precision the
  exporter rounded it to. Columns are delta-coded over time in int8/16/32, with a JSON header.
  Decoding divides by the same power of ten, which returns the doubles `JSON.parse` gives.
  * Verified in Python on all 77 rollouts.
  * Verified in Chrome by `web/tools/check_rollouts.py`, comparing each with its JSON.
  * Size: 37.1 → 11.6 MB raw (gzip 5.5 → 2.3 MB).
* **Policy-input videos.** Re-encoded from the logged policy views (or the MP4) at libx264 crf 30
  with a 32-frame GOP: 7.5 → 3.7 MB. On one rollout, crf 28/GOP 8 was 98 KB at SSIM 0.995;
  crf 30/GOP 32 is 49 KB at SSIM 0.994.
* **Robot GLBs.** KHR_mesh_quantization: int16 positions with a uniform scale + offset on the node,
  int8 normals, uint16 UVs and indices. Meshes whose normals are just face normals (≥ 95 % of
  corners within 2°) drop them and are welded by position; the viewer rebuilds flat normals once
  per geometry. The GTAO pass needs a normal attribute, so this is not done with material
  flatShading.

  | robot | raw | gzip |
  |---|---|---|
  | Franka | 11.0 → 1.6 MB (390k → 85k vertices) | 3.55 → 1.06 MB |
  | UR5e | 11.5 → 5.2 MB | 3.69 → 2.12 MB |
  | Trossen | 1.6 → 0.8 MB | 0.75 → 0.49 MB |

* **Bread GLBs.**
  * Textures: 1536×2048 JPEG → 1024 px WebP (EXT_texture_webp). Identical images are shared.
  * Normal maps are WebP q92. Their error against the 1024 px reference is 13.6° / 6.2°, about
    what the original JPEG 4:2:0 already had (14.5° / 7.4°).
  * Positions stay float32, because `softbread.js` deforms them on the CPU. `softbread.js` also
    converts any quantized or interleaved attribute to float32 before skinning.
  * Size: 2.1 / 2.5 → 1.0 / 1.1 MB.
* **Textures.** PNG/JPEG → WebP: trossen_BaseColor 262 → 25 KB, maple 81 → 20 KB. The model JSON
  references are rewritten.
* **Viewer loading.**
  * The rollout is fetched in parallel with the model and its meshes.
  * `modulepreload` covers the whole three.js module graph (24 modules) plus a jsdelivr preconnect.
  * Manifest scenes are rounded to 0.1 mm.
* **Deploy.** `deploy_web.py` ships only what the packed site loads: `.bin` rollouts and WebP
  textures, not their JSON/PNG sources. It validates the manifest first. Deploy plan: 169 →
  127 MB. Of that, 98 MB is the Live-mode policy ONNX (62 MB fp32 + 31 MB fp16) and MJCF, which
  is in `web/js/live` territory (see follow-ups).

**Phone layout** (393 px, before):

* The control bar is a horizontal scroller with no hint.
* The HUD title runs under the policy-input picture.
* "CAM" overflows its button.
* The viewer starts below a full-screen hero.

Fixes (CSS only):

* a fade on the bar's right edge;
* an ellipsised title capped at 58 % width;
* wider camera buttons;
* a tighter hero;
* a horizontally scrollable results table.

Live mode itself worked on the phone with WASM before and after.

**Screenshot diff** (`web/tools/compare_shots.py`, 10 fixed cases across robots, views, times and
the policy camera):

* Before vs after: PSNR 45.9–48.7 dB, 0.12–0.46 % of pixels differing by more than 8/255.
* Noise floor (the same build twice): PSNR 50.5–57.5 dB, 0.04–0.17 %.
* The differences are only texture re-encodes (bread, table, arm) and the re-encoded policy
  video. There are no silhouette or timing differences.

### 4a. After

Headless Chrome with a cold cache, served like GitHub Pages. "Before" is the deployed build; the
box was busier during the "after" runs (load ~30 vs ~6), which only penalises "after".

| first load | before | after |
|---|---|---|
| wire bytes (default config) | 5.59 MB, 48 req | **2.93 MB**, 49 req |
| ready, desktop, unthrottled | 1.31 s | **0.73 s** |
| ready, desktop, Fast 4G | 5.62 s | **3.38 s** |
| ready, phone (4× CPU), Slow 4G | 28.9 s | **16.0 s** |
| Franka config, Fast 4G | 8.25 MB, 8.54 s | **3.51 MB, 3.90 s** |

What is left in the first load (2.93 MB):

| item | size |
|---|---|
| bread GLBs (WebP, not gzip-able) | 1.71 MB |
| Trossen GLB | 0.51 MB |
| three.js from jsdelivr | 0.49 MB |
| rollout `.bin` + policy video | 80 KB |
| textures | 46 KB |
| manifest | 4 KB |

Live mode is unchanged in bytes (the ONNX policy) and works in every setting:

| setting | ready | download | 3 re-plans |
|---|---|---|---|
| desktop, WebGPU | 5.2 s | 64 MB | 2.6 s |
| desktop, WASM fallback | 3.5 s | 36 MB | 4.0 s |
| phone, WASM | 9.4 s | 36 MB | 16 s |

The phone numbers are noisier under the load; there were no errors. `check_rollouts.py`
decoded all packed rollouts in Chrome identically to their JSON.

Screenshot diff, the 3D scene only (HUD, policy-input picture and timeline masked):

| comparison | PSNR | pixels > 8/255 |
|---|---|---|
| before vs after | 45.5–49.3 dB | 0.05–0.45 % |
| noise floor (same build twice) | 49.6–56.6 dB | 0.05–0.21 % |

The remaining differences are texture re-encodes and flat-normal Franka parts. Full frames,
including the HUD: 40.8–48.7 dB; the 40.8 dB case is the HUD title shifted by about a pixel.

Phone layout after: the viewer and its timeline are on the first screen, the HUD title is
ellipsised, the camera buttons fit, and the bar fades at its edge.

## Follow-ups (not done here)

* Live-mode policy download: 62 MB fp32 for WebGPU. Storing the weights as fp16 with Cast nodes
  would halve it at fp32 compute; this needs `export_onnx.py` and `web/js/live/policy.js`.
* Determinism (§1a). A repeated episode diverges after about 2 s even on CPU with one thread.
  Finding the source (rendering, contact solve, perception) would make A/B tests exact. Until
  then, use the paired tests in `stats.py`.
* Trossen start pose: it sits 2.6° from wrist joint 4's lower limit, so lifting at a fixed
  orientation saturates the IK (5 mm / 3.4° short for a 4 cm lift). This is a controller /
  driver question.
