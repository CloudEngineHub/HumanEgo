# Upgrade log: low-level controller and Python ↔ browser parity

Owner: the controller session (`humanego_sim/controller.py`, `runner.py`, `driver.py`, `robots.py`,
`web/js/live/*`, `scripts/parity_test.py`, `control.*` in `configs/tuned/*.json`).

Protocol for every step: hypothesis → change → before / after on the same scenes, dev seeds 7777 and
8888 (Trossen 32 scenes, Franka / UR5e 24), tuned settings + `control.grasp_assist.enabled`. Same-seed
reruns differ by 3–4 points (GPU nondeterminism in the policy), so differences under ~5 points on 32
scenes are noise. Seeds 9001–9099 are reserved for the final report and never used here.

Numbers are per arm / right / left / both served.

## 1. Parity test (`scripts/parity_test.py`)

**Hypothesis.** The browser Live stack (`web/js/live/*`, MuJoCo WASM + ONNX Runtime Web) is a
line-by-line port of the Python stack, but nothing checked it end to end since the grasp skill, the
centre-height estimate and the UR5e place-down release were added.

**Change.** A trace in `runner.py` (off unless `runner.trace` is a dict) records the physics state at
t = 0 and before every control tick, the perception, every policy call (ICT, anchor uv, the network's
output, the first input images) and every command. The browser harness (`web/js/live/parity.js`, run
in headless Chrome by `parity_test.py`) starts from the same physics state with the same flow-ODE
initial noise (`policy.noise`) and logs the same quantities. Modes:

* `sync`: Python's perception and policy outputs; before every control tick the physics state is
  re-synced to Python's (replaying the last physics step, so MuJoCo's lagging kinematics match too).
  Any command difference is a logic difference between the ports; the state just before each re-sync
  measures one 0.1 s tick of WASM-vs-native physics drift.
* `free`: Python's perception and policy outputs replayed by (side, time), free-running WASM physics.
* `full`: the browser on its own: perception from three.js renders of the same t = 0 state, the ONNX
  policy on the same noise (and, first, on Python's exact inputs), free-running physics.

(results below)

**Result (landed, 418f103).** Seed 7777 scene 0, tuned settings + grasp skill. `sync`: EE targets, driver
goals, gripper / skill / release / latching identical on Trossen, UR5e (place-down release) and Franka
(ICT 1e-6 = float32); one-tick WASM-vs-native drift 0.005–0.07 mm. `free`: commands identical, breads
diverge only once the second bread is dropped (t = 20–22 s, 5–12 mm; UR5e none), same outcome. `full`
(Trossen): perception within 0.3 mm / 0.2°, ONNX = PyTorch to 6e-7 on identical inputs, but the
policy-input images differ (PSNR 21 dB: viewer backdrop and table-texture mapping vs MuJoCo's render), so
re-plans diverge from 0.9 s (same outcome). Fixed: the policy camera rendered the scene as of the last
animation frame (now synced to the physics state before every render); `perception.depth_noise` ignored.
Open (viewer, web session): render parity of the policy-camera image.

## 2. Re-perception before a re-grasp (in progress, not landed)

**Audit** (16 Trossen scenes, seed 7777): every failed arm (5/32) had used both re-grasps; 1 of 6 re-grasps
succeeded. Retries close far from the pinch point (Trossen 9 of 11, Franka 9 of 46 retries > 2 cm; first
tries 0/64): e.g. scene 1, the croissant slipped out at 28 cm, landed 9 cm away, both retries went back to
the t = 0 position; the approach from ~40 cm up also hits the 1.5 s timeout 5 cm above the target.

**Change** (`grasp_assist.reperceive`: off | skill | policy; `timeout_near`): after a re-grasp decision
the runner re-runs the released perception once the bread is visible (≥ 60 % of its t = 0 mask) and still
(mask IoU ≥ 0.9 between ticks); updates the skill geometry, and with "policy" the policy's bread pose (plate
frame re-oriented to it); a bread found on the plate ends the arm (served). `timeout_near`: the approach
timeout counts only within that distance of the pinch point (hard cap `approach_max`). Ported to JS
(observe / apply split so `parity_test.py` replays Python's camera observations).

**Parity:** `parity_test.py --scene 1 --set reperceive=policy, timeout_near, grip_check=touch, one_pad_empty,
slip_drop` (Trossen, a scene with a slip and two re-grasps): sync mode identical (6 / 6 re-perception
calls at the same times, EE targets 0.000 mm); free mode diverges from the drop at 5.8 s, same outcome.

**Trossen** (`runs/ctrl_cmp/trossen_wxai_rp.json`, 64 scenes, seeds 7777 + 8888, paired vs base):

| set | per arm | right | left | both | vs base |
|---|---|---|---|---|---|
| base (tuned + skill) | 93 % | 94 | 92 | 88 | |
| timeout_near 0.03 | 92 % | 92 | 92 | 86 | −0.8 pts [−4.7, +3.1], 3 / 4 discordant, p = 1 |
| reperceive skill | 94 % | 97 | 91 | 89 | +0.8 [−2.3, +3.9], 4 / 3, p = 1 |
| reperceive policy | 93 % | 94 | 92 | 89 | +0.0 [−3.9, +3.9], 3 / 3, p = 1 |
| reperceive policy + near | 92 % | 92 | 92 | 89 | −0.8 [−5.5, +3.9], 4 / 5, p = 1 |

Neutral: the Trossen fails 9 of 128 arms, too few re-grasps for these to show. Measured where re-grasps
are frequent (Franka: ~3 re-grasps per 2 arms; UR5e) next. Known corner case: a bread dropped out of reach
(IK fails for the commanded orientation).

## 3. Grip sensing (in progress, not landed)

Real hardware (sourced): Trossen reports gripper position + motor effort (N); the 2F-85 object detection
gOBJ (current limit before the requested position; 3 after 2 = lost), position, current; the Franka Hand
width + `is_grasped` (ε-window), no force. None has fingertip force. Sim signals in `driver.py`:
`grip_effort` (servo force = what the motors report), `pad_forces` (contact normal force per finger on the
breads = a tactile pad the real grippers lack). Logged vs ground truth (both pads on the bread):
Trossen: q and effort equivalent (P-servo), 98 % of no-contact ticks flagged, 0 false alarms; pad touch
adds only one-pad contacts (0.6 %). Franka: same for q, but 6 % of holding ticks have one finger only on
the bread (q reads "held"). `grasp_assist.grip_check` = q | touch | effort, `one_pad_empty`, `slip_drop`
implemented (Python + JS), not yet measured.

## Franka screening, seed 7777 (24 scenes, paired vs base; `runs/ctrl_cmp/q_franka_panda_*_7777.json`)

| set | per arm | right | left | vs base (paired) |
|---|---|---|---|---|
| base (tuned `franka_v3` + skill) | 62 % | 83 | 42 | |
| reperceive policy + timeout_near 0.03 | 69 % | 88 | 50 | +6.2 pts [−8.3, +22.9], 10 / 7, p = 0.63 |
| grip_check touch (+ one_pad_empty, slip_drop 0.5) | 67 % | 92 | 42 | +4.2 [−12.5, +20.8], 9 / 7, p = 0.80 |
| yaw align + max_rot_step 0.06 | 46 % | 67 | 25 | −16.7 [−31.2, −2.1], 4 / 12, p = 0.08 |
| lateral_gain 1.0 (full centring) | 52 % | 75 | 29 | −10.4 [−27.1, +6.2], 7 / 12, p = 0.36 |

| slow final approach (slow_dist 0.03, slow_speed 0.03 m/s) | 75 % | 83 | **67** | (left +25 pts) |

Screening only (one seed, physics v1). **Physics v2 landed on main (22c0fa1) mid-screening**: everything
is re-measured on v2 with both dev seeds (section "v2" below); the v1 numbers are only a guide. Faster rotation
(`max_rot_step` 0.06) also changes how the arm follows the policy everywhere, so yaw is re-run with the
default cage (align, fit). Full centring on the perceived axis is worse than the tuned partial pull (0.36):
the perceived axis itself is 1–1.4 cm off.

## 4. Franka left arm (running)
Options (controller.py / controller.js, off by default): `grasp_assist.yaw` = align | fit (turn the pinch
target about the vertical so the fingers close across the bread's short axis; "fit" only as far as the
perceived ellipse needs to fit `robots.max_opening` − `yaw_margin`; closing waits for `arrive_rot_deg`),
`lateral_gain` (tuned 0.36 → 1.0: full centring on the perceived axis), `slow_dist` / `slow_speed` (slow
final approach). 2-episode check: yaw reaches 85–90° when the turn is small, but the controller's rotation
cage (`max_rot_step` 0.02 rad / tick) limits it to ~17° per 1.5 s approach, so it is also tried with 0.06.
Data: at closing the left arm's spread axis is 55–83° from the danish's long axis; below ~75° the
14 × 7.4 cm loaf is wider than the 8 cm opening → wrist-yaw alignment is the first thing to try (the
rotation cage, 0.02 rad/tick, means it must start during the descent).

## 5. Release robustness (options in, not measured)
`release.open_time` (gripper opening time at release, was fixed at 0.5 s) and `release.settle` (hold still
before opening), Python + JS.

## Trossen gripper command values (v2)
The real driver (`AriaMimic/inference/RobotArmTrossen.py`) closes with `set_gripper_position(0)` and opens to
`_gripper_max` = 0.045 m (joint limits set to 0 … 0.045); the sim used −0.002 (closed) and 0.044 (open).
A/B on v2 below (`v2realgrip`: closed 0, open 0.045, q normalised by 0.045).

## v2 physics, both dev seeds (`runs/ctrl_cmp/q_<robot>_v2*_7777_8888.json`; old camera height 0.436)

Tuned settings + grasp skill; paired vs base on the same 48 scenes (24 per seed):

| robot | set | per arm | right | left | vs base |
|---|---|---|---|---|---|
| Franka | base (`franka_v3`) | 73 % | 92 | 54 | |
| Franka | slow final approach (3 cm, 0.03 m/s) | 67 % | 83 | 50 | −6.2 pts [−13.5, +1.0], 5 / 11, p = 0.21 |
| Franka | reperceive policy + timeout_near 0.03 | 71 % | 85 | 56 | −2.1 [−11.5, +7.3], 9 / 11, p = 0.82 |
| Franka | grip_check touch (+ one pad, slip 0.5) | 70 % | 92 | 48 | −3.1 [−7.3, 0.0], 0 / 3, p = 0.25 |
| UR5e | base (`ur5e_v3`) | 88 % | 98 | 77 | |
| UR5e | slow final approach | 68 % | 75 | 60 | **−19.8 [−29.2, −11.5], 1 / 20, p = 2e-5** |
| UR5e | reperceive policy + timeout_near 0.03 | 88 % | 98 | 77 | 0.0 (identical outcomes) |
| UR5e | grip_check touch | 91 % | 100 | 81 | +3.1 [0.0, +7.3], 4 / 1, p = 0.38 |

No option gives a significant gain on v2; the slow approach significantly hurts the UR5e. The v1 screening
signals (Franka left: slow +25, reperceive +6, touch +4 on seed 7777) did not hold on v2 with two seeds.

## Hand-over

**On main:** 418f103 parity test + Live fixes; fd4e0b0 the controller options below (off by default,
Python + JS). Final parity (Trossen scene 1, new scene defaults, every option on): sync mode identical,
25 / 25 re-perception calls at the same times, EE targets 0.000 mm, ICT 1e-6; pytest 61 passed.

**Final per-robot profiles: keep ALL of these OFF** (no significant gain on v2; slow approach harmful):

| option | keys | why it exists |
|---|---|---|
| re-perception before a re-grasp | `grasp_assist.reperceive` off / skill / policy, `reperceive_visible`, `reperceive_iou`, `reperceive_timeout` | a dropped or pushed bread is re-located with the released perception (waits until visible and still; a bread found on the plate counts as served) |
| arrival-based approach timeout | `timeout_near`, `approach_max` | retries from high above timed out in mid-air |
| grip checks | `grip_check` q / touch / effort, `touch_min`, `one_pad_empty`, `effort_empty`, `slip_drop`, `slip_time`, `empty_time` | `effort` = what trossen_arm / the 2F-85 current report; `touch` = finger-pad force, a sensor the real grippers do NOT have |
| wrist yaw | `yaw` off / align / fit, `yaw_max_deg`, `yaw_margin`, `arrive_rot_deg`, `robots.max_opening` | fingers across the bread's short axis; in the browser "fit" needs `grasp_assist.max_opening` in the config (the live spec has no robot opening) |
| slow final approach | `slow_dist`, `slow_speed` | |
| release timing | `release.open_time`, `release.home_after`, `release.settle` | defaults now follow the real wrap-up (open 1.0 s; two arms home once open, one arm after 0.5 s); old timing scored the same (2026-10-18) |

Real-gripper sensing (sourced): Trossen reports finger position and motor effort (N); the 2F-85 object
detection (gOBJ), position and current; the Franka Hand width and `is_grasped` (no force). In the sim,
Trossen effort and finger position carry the same information (position servo).

**For the Franka / UR re-tune, search:** `lateral_gain`, `along_max`, `height_max`, `retry_lower`,
`retry_inward`, `approach_timeout`, `close_hold`, the controller's `alpha` / `duration_smooth_param`,
`tcp_offset`, `tool_pitch_deg`. Not worth searching: the options above, `max_rot_step` (0.06 hurt), full
lateral centring (`lateral_gain` 1.0 hurt on v1).

**Open:**
* Trossen gripper command values: the real driver closes to 0 and opens to 0.045 m (sim −0.002 / 0.044,
  `robots.py`); not measured (Trossen re-tune owned by the coordinator), so not changed.
* Browser render parity: the policy-input image differs from MuJoCo's (PSNR 21 dB: viewer backdrop,
  table texture mapping); owned by the web viewer.
* A bread dropped out of reach (IK fails for the commanded orientation) cannot be re-grasped.
