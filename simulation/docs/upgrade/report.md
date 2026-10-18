# Simulator upgrade, 2026-10-17 / 18: report

Goal: fix every weakness found in the review of the simulator, verify each change, and redeploy.
Five sessions worked in parallel, each in its own git worktree and owning its own files:
robustness and evaluation, controller, physics, sim vs real, and engineering and website. Their logs
are in this folder (`robust.md`, `controller.md`, `physics.md`, `real.md`, `eng.md`). The
coordinator then integrated: it adopted the measured scene, re-tuned the controllers, ran the
reserved test and deployed.

## Headline: reserved test set

Seed 9001, 100 scenes per robot (200 arms). These scenes were never used to pick anything.
Everything runs on the final defaults: sourced physics, the table height measured from the real
video, the 14.5 cm croissant and fused masks. Wilson 95 % intervals.

| Robot | Tuned controller + grasp skill, per arm | Right / left | Both breads served | Production controller (real `Control.yaml`), per arm | Both breads served |
|---|---|---|---|---|---|
| Trossen WidowX AI (real rig) | **80.5 %** [74.5, 85.4] | 88 / 73 % | 64 % | 60.5 % [53.6, 67.0] | 27 % |
| Franka Panda | **75.5 %** [69.1, 80.9] | 88 / 63 % | 57 % | 46.0 % [39.2, 52.9] | 17 % |
| UR5e + Robotiq 2F-85 | **76.0 %** [69.6, 81.4] | 99 / 53 % | 53 % | 57.5 % [50.6, 64.1] | 22 % |

Robustness of the tuned controllers on the same 100 scenes (per arm). The perturbation models are
described in `robust.md`:

| Robot | Clean | Segmentation errors | Real DINO-SAM segmenter | All perturbations together |
|---|---|---|---|---|
| Trossen | 80.5 % | 84.5 % | 80.5 % | 78.0 % [71.8, 83.2] |
| Franka | 75.5 % | 71.5 % | — | 64.5 % [57.7, 70.8] |
| UR5e | 76.0 % | 75.0 % | — | 76.5 % [70.2, 81.8] |

All perturbations together are: segmentation errors, depth artefacts, calibration error, camera
latency, bread size, layout, table height, lighting and distractors.

### Before vs after

**Fidelity.** The production controller on the Trossen, the closest thing to the real robot, went
from **41 % per arm (bimanual) and 12.5 % single-arm to 60.5 % and 67.5 %**. The real robot scores
95 % single-arm in the paper. The change comes almost entirely from the table height.

**Tuned controllers.** The headline numbers are lower than the 91–94 % quoted before. That is
expected and honest:
- The old numbers were measured on 24–32 scenes, on the seeds the parameters had been selected on.
- They used physics partly tuned to suit the policy.
- They used a table that was probably 6.5 cm too low.

## What changed, by area

### 1. Sim vs real (`docs/real_vs_sim.md`, `real.md`)

**Table height.**
- The sim put the arm bases 0.215 m above the table, a guess from a `Control.yaml` comment.
- Two real front-view frames give **≈ 0.15 m**.
- Adopted as the new default, `rig.cam_height` 0.366 (c172786).
- Confidence is medium: **please tape-measure it (experiment E1)**.

**Objects.** The croissant is now 14.5 cm long, as measured in the 60 human episodes (scale 1.06).
Further proposed values are in `docs/proposed_scene.json`.

**Where the video agrees with the sim:**
- transport duration;
- lift height;
- left/right sequencing.

**Where it disagrees:**
- The release: the real robot sets the bread down; the sim drops it from 5–8 cm.
- The grasp point: the real robot grasps nearer the bread's end.
- The tuned controller moves faster and more jerkily than the real robot.

**Sim–real correlation across checkpoints is not possible yet.**
- Only the released checkpoint exists locally and on Hugging Face.
- The paper's scaling-law checkpoints N1–N48 (real 0–92.5 %) are needed.

**Real-robot logging.**
- AriaMimic branch `sim-logging` (SimLogger, per-step logs; not merged into main).
- `scripts/real/replay_real.py` rebuilds a logged scene in the sim and compares trajectories.

### 2. Physics (`docs/physics_params.md`, `physics.md`)

- **Sourced parameters:** every parameter now has a source and a range, and the policy-free tests
  (compress, pinch-and-lift, drop, gripper) validate them.
- **Bread contact:**
  - Before: so soft that a 20 kPa crumb behaved like 6.8 kPa, with fingers sinking 5 mm into the contact.
  - Now: stiff, ≤ 1 mm, and still dead on landing.
- **Friction:** was 1.0 everywhere. Now per surface: finger pads 0.8, crust 0.4, plate 0.4, table 0.45.
- **Grippers set to their datasheets:**
  - 2F-85: 295 → 21 N per pad.
  - Franka: 1 → 20 N per finger.
  - Trossen: keeps its system-identified servo.
- **Plate rim:** 2.2 → 3.0 cm (real video 2.9 ± 0.4 cm).
- **Autotune no longer searches physics** (55047dc).
- **Plastic dents:** implemented, but they change nothing significant, so they are off.
- **Per-robot skins still needed:** the UR5e still needs the volumetric flex and the Franka the fine skin.

### 3. Controller (`controller.md`)

- **Python ≡ browser.** `scripts/parity_test.py` runs the same scene through the Python stack and
  the browser Live stack.
  - The two are identical: EE targets differ by 0.000 mm.
  - The policy matches to 1e-6.
  - A few Live-mode bugs were fixed along the way.
- **New options are kept off.** Each is implemented in Python and JS and measured on the new physics:
  - re-perception before a regrasp;
  - an arrival-based timeout;
  - grip checks from effort or touch;
  - wrist yaw across the bread;
  - a slow final approach;
  - release timing.

  None helped, and the slow approach cost the UR5e 20 points. They stay off by default
  (fd4e0b0), available for real-robot work.
- **Re-tuned on the final defaults.** Only controller parameters were searched; the autotune runs
  are `trossen_v7`, `franka_panda_v4` and `ur5e_v4`.
  - **Trossen:** 83 % / 89 % per arm on the dev seeds vs 72 % / 66 % for the previous settings on
    the same scenes. On seed 8888: +23 pts, McNemar p = 0.0015. Motion is also gentler (goal
    time ≈ 0.5 s instead of 0.2 s).
  - **Franka:** 73 % / 73 % vs 60 % / 60 %. The gain repeats on both seeds but is not significant
    on either alone (n = 24).
  - **UR5e:** nothing better found (83 % = 83 %), so the settings are kept.

### 4. Robustness and evaluation (`robust.md`)

- **Evaluation protocol:**
  - Wilson CIs and paired McNemar / scene-bootstrap tests in `evaluate.py` / `compare_params.py`.
  - Seeds 9001–9099 are reserved for final reports, enforced with `--final`.
- **Perturbation models:**
  - realistic segmentation errors;
  - the real DINO-SAM on rendered frames;
  - RealSense depth artefacts;
  - hand-eye calibration error;
  - camera latency;
  - bread size, layout, table height, lighting and distractors.
- **Dev-seed screen:** only segmentation errors (−12 pts) and everything combined (−22 pts)
  mattered. Most losses were missed detections.
- **Mask fusion:** 5-frame fusion is now the default, +9 pts under segmentation errors (p = 0.04)
  and bit-identical on clean masks.

### 5. Engineering and website (`eng.md`)

- **Profiles:** `--profile production|tuned` on every script.
- **Tests:** pytest, 61 tests, plus a CPU GitHub Actions workflow.
- **Site rebuild:** `scripts/rebuild_site.py` rebuilds everything, and every number on the site is
  generated from `configs/tuned/*.json`, never hand-typed.
- **Lighter site:** first load 5.59 → 2.93 MB; ready in 0.73 s instead of 1.31 s on desktop, and
  16 s instead of 29 s on a phone over Slow 4G.
- **WebGPU fallback:** without WebGPU the WASM fallback works.
- **README:** restructured.

## Open issues and caveats

- **The tuned controller exists only in sim.** The real robot runs the production controller.
  `docs/proposed_portable_controller.json` restricts the tuning to knobs the real `Control.yaml`
  has; that is experiment E4.
- **The table height is a video estimate (medium confidence).** If the tape measure disagrees,
  change `rig.cam_height` and re-run the final test.
- **The left arm is still the weak side** on the UR5e (53 %) and the Franka (63 %).
- **The sim is not run-to-run deterministic** (GPU and contact solve). Every comparison is paired
  on the same scenes for this reason.
- **Gripper command values:** the sim's Trossen gripper closes to −2 mm, the real driver to 0. This
  was left unchanged.
- **Release behaviour:** the real robot sets the bread down before releasing; the sim releases
  from 5–8 cm (`docs/real_vs_sim.md` §2).

## Real experiments that would settle the open questions

From `docs/real_vs_sim.md` §6. Run them with the AriaMimic `sim-logging` branch:

| | what | trials |
|---|---|---|
| E0 | self-test + 1 logged rollout (`loop_dt`, file check) | 1 |
| E1 | **tape-measure the rig: arm-base height above the table**; 5 static scenes; joint sweep | 5 + 1 |
| E2 | paper protocol, single arm, production `Control.yaml` | 20 |
| E3 | bimanual, croissant + danish, production | 20 |
| E4 | E3 with the portable tuned controller | 20 |
| E5 | data-efficiency checkpoints (if they can be found or retrained), for the sim–real correlation | 10 each |
