# HumanEgo in Simulation

Closed-loop simulation of the **released HumanEgo `serve_bread` policy** on a physics
replica of the real deployment rig: two Trossen WidowX AI arms and a top-mounted
Intel RealSense D405, placed by the **real hand-eye calibration**. The network is never
touched — the simulator plugs into the same `Camera` / `RobotArm` / `Perception`
contract as [`inference/`](../inference/README.md) and runs the released `ICTPolicy` on
the released checkpoint.

**Interactive page:** <https://humanego-ai.github.io/sim/> — replay pre-computed rollouts
in 3D, or switch to **Live** and run the policy *in your browser* (MuJoCo WebAssembly +
ONNX Runtime Web) from any initial condition you drag into place.

| | |
|---|---|
| Robots | Trossen WidowX AI (real rig) · Franka Panda · UR5e + Robotiq 2F-85 |
| Cameras | RealSense D405 (real rig) · RealSense D435 · ZED 2i · ZED Mini |
| Arms | two (independent bimanual, as in the teaser) · one (as in the paper) |
| Objects | **soft** croissant and sugar danish loaf (MuJoCo flex, they squash in the gripper) modelled after the breads in the real rollouts, light-blue plate, maple butcher-block table; sizes measured from the released dataset |

## Quick start

```bash
conda create -n humanego-sim python=3.11 -y && conda activate humanego-sim
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r simulation/requirements.txt
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints

python simulation/scripts/build_assets.py                 # once: procedural bread / plate / table assets
python simulation/scripts/run_episode.py --video          # one bimanual episode -> simulation/runs/...
```

Other rigs are one flag away:

```bash
python simulation/scripts/fetch_robots.py franka_panda ur5e        # large Menagerie meshes, fetched on demand
python simulation/scripts/fit_robot_poses.py franka_panda ur5e     # start pose = the real Trossen start pose (IK)
python simulation/scripts/run_episode.py --robot franka_panda --camera zed2i --video
python simulation/scripts/run_episode.py --mode single --set control.alpha=0.3
```

Batch success rate over randomized initial conditions (paper protocol):

```bash
python simulation/scripts/evaluate.py --n 40 --workers 8
```

Headless rendering uses `MUJOCO_GL=egl` (set automatically).

## What is simulated, and how faithfully

| Real system | Simulator |
|---|---|
| Trossen WidowX AI ×2, hand-eye calibrated (`T_base_in_cam`) | MuJoCo Menagerie `trossen_wxai`; bases placed from the **real calibration** |
| `trossen_arm` driver: non-blocking Cartesian goals re-planned at 10 Hz with `goal_time` | `SimArm`: quintic re-targeting from the current interpolator state (velocity continuous), DLS IK every physics step, reachability errors, `safe_z_min` floor, EE correction frame |
| RealSense D405, 1280×720 → centre crop → 640×480 | pinhole camera with the same crop/resize and a depth-noise model |
| DINO-SAM masks | ground-truth segmentation (same mask semantics: obj1 = every bread, obj2 = plate) |
| LaMa arm inpainting | render without the robot |
| KptsSelector → multi-frame depth lift → PCA pose → VisualKpts | **the released modules themselves** |
| `Inference.run_bimanual` (mirrored left arm, latching, forced lift, idle-done, wrap-up) | `runner.py`, same logic on a deterministic sim clock |
| `AsyncRobotController` (ensembling, safety cage, EMA/Slerp, look-ahead, stride) | `controller.py`, same logic on a sim clock |

Notes on the modelling choices (all in `configs/serve_bread.yaml`):

* **Scene geometry.** The camera sits 0.436 m above the table: the real driver's
  `safe_z_min = −0.20 m` "protects the table", which puts the table just below the arm
  bases. Objects then lie ~0.6 m from the camera, inside the depth range of the human
  training data. Bread yaw ~90° (long axis left–right in the image) as in the demos.
* **Object sizes and masses** come from the released dataset's triangulated contour
  keypoints (croissant ≈ 13.8 × 6.2 × 5 cm, plate ⌀ 24.5 cm), the real rollout video for
  the danish loaf (≈ 14 × 7.4 × 6 cm, measured against the plate) and real weights
  (croissant 50 g, danish 75 g, plate 450 g).
* **Breads that look like the real ones.** `scripts/bread_gen.py` builds the straight rolled
  croissant of the demos (chevron bands, pale dough in the grooves, glossy egg wash) and the
  laminated danish loaf with its sugar crust as parametric surfaces with painted PBR maps.
  As in the real bimanual run, the right arm gets the croissant and the left arm the danish.
* **Soft bread** (`humanego_sim/softbody.py`). Each bread is a MuJoCo flex: its collision
  surface is interpolated from a 4 × 2 × 2 trilinear grid of point masses with volumetric
  elasticity (E = 20 kPa, ν = 0.25, crumb of a fresh croissant), and the textured mesh is
  bound to the same grid as a skin, so the crust visibly squashes under the fingers and
  springs back. The grip holds by friction alone (μ = 1.0, the Menagerie gripper servo:
  kp 339 N/m, ≈ 11 N on a croissant). Flex elasticity needs MuJoCo's `discrete`
  integrator (implicit in position, stable at the 2 ms step); `impratio = 100` stops a held
  bread from creeping out of the grip. `physics.bread_model: rigid` restores the earlier
  rigid-bread + finger-adhesion proxy.
* **Crumb that lands dead.** Real crumb dissipates almost all impact energy: a dropped bread
  does not bounce. Flex damping 0.03 s and over-damped bread contacts (`solref` damping
  ratio 3) reproduce that. It matters for the second bread: the left arm releases the danish
  ~11 cm above the plate, often onto the croissant already there. In `scripts/drop_test.py`
  (danish dropped 4–9 cm onto a croissant lying on the plate, random offset and tilt) it
  stays on the plate in 90 % of drops, against 13 % with near-elastic crumb.
* **Served = resting on the plate.** An arm succeeds when it released its bread and the bread
  rests on the plate: its centre is over the plate and no part touches the table (the
  plate's floor is 8.5 mm above the table). The earlier rule, centre within R − 2 cm, failed
  breads that lie fully on the plate's rim zone; it is still reported as `success_strict`.
* **Grasp state.** The real loop flags a grasp when the measured gripper opening drops
  below 50 %; in simulation a gripper that has finished closing and stalled on the bread
  also counts (`control.grasp_state_source`).

### Why grasps land above the bread's centre

Measured over 32 grasps (`scripts/grasp_height.py`, `scripts/grasp_test.py`), the
fingertip point closes ≈ 1.6–2.2 cm above the bread's centre (the croissant is 5 cm tall).
The offset adds up from three parts of the real pipeline, all reproduced faithfully:

| | above centre |
|---|---|
| perception: keypoints are lifted from depth on the visible top surface | +0.2–0.6 cm |
| policy: the predicted grasp point relative to those keypoints (the human demos pinch 1.2 cm above the contour centroid) | +0.5 cm |
| execution: the driver re-plans a quintic with `goal_time = 1.2 s` from the current state every 0.1 s (no feed-forward, as `RobotArmTrossen.move_p`), so the arm trails the descending target by ≈ 2 cm; the gripper closes on the `any` grasp signal (0.5 s) and the post-grasp lift starts before the arm catches up | +1.0–1.2 cm |

What does and does not move it (same 16 scenes, soft bread, per-arm success in brackets):

| change | fingertips above centre at closing | per-arm success |
|---|---|---|
| production controller | +1.6 cm | 44 % |
| `look_ahead_offset` 45 → 30 | +2.2 cm | 38 % |
| `duration_smooth_param` 12 → 6 (goal time 0.6 s) | +1.3 cm | 44 % |
| close only if the grasp is ≤ 20 steps ahead (`grasp_k_any` 20) | +1.4 cm | 41 % |
| goal feed-forward velocity (`feedforward_gain` 1, not in the real controller) | +1.8 cm | 34 % |
| real grasp flag only (opening < 50 %, no stall detection) | +1.6 cm | 16 % |

The gripper closes on the early `any` signal while the arm is still descending; the stall
on the bread then counts as a grasp and the lift starts. With friction-only soft bread this matters: a pinch holds when the
fingertips are ≤ ≈ 1.9 cm above the centre and slips out above that (the pads then press
on the rounded top, above the bread's widest section), independently of grid resolution,
crumb stiffness or friction (`scripts/grasp_test.py` replays recorded grasp poses with any physics). The
rigid + adhesion proxy hid this.

## Grasp skill in the low-level controller

The production controller only follows the plan and toggles the gripper on the policy's
grasp signal. An audit of every arm (`scripts/grasp_audit.py`) showed where that loses bread:
the left arm pinched the 7.4 cm danish near its end, high or off to the side and it slipped
out when lifted; some grasps closed on air; a few breads stayed wedged in the opened hand.
`control.grasp_assist` adds a grasp skill that uses only what the real robot has, the
gripper reading and its own perception of the bread at t = 0 (Python and browser):

1. **Refine** the plan's pinch point: pull it onto the perceived bread's long axis, cap its
   offset from the bread's centre along the axis and its height above the centre.
2. **Close on arrival**: keep the gripper open until the arm has reached that point, and hold
   still while the fingers close.
3. **Check the grip**: if the gripper closed on air or the bread slipped out on the way up
   (the gripper reading collapses), reopen and let the policy re-approach; each retry aims
   lower and closer to the bread's middle. The empty check reads the gripper joint, or the
   gap between the pads for the underactuated 2F-85, which closes its driver joint almost
   fully around a bread (the real one reports object detection).

Same 32 held-out scenes, `soft_v5` settings:

| | per arm | right | left | both served |
|---|---|---|---|---|
| production controller behaviour | 70 % | 91 % | 50 % | 50 % |
| + refine & close on arrival & re-grasp | 77 % | 91 % | 63 % | 56 % |
| + adaptive re-grasp (lower, more central) | **80 %** | **94 %** | **66 %** | **63 %** |

Releasing differently (setting the bread down first, lifting or backing the open hand out)
did not help and is left off (`control.release`). Franka (24 scenes): 33 % → 42 %; UR5e:
38 % → 46 % with pad-gap sensing.

## Live mode: the policy in your browser

`web/js/live/` is a line-by-line port of the Python stack to JavaScript:

| | |
|---|---|
| physics | official MuJoCo WebAssembly build (`@mujoco/mujoco` 3.16, same version as Python) on a physics-only MJCF (`live_export.py`) |
| policy | the whole 20-step flow-ODE solve exported as **one ONNX graph** (context + cross-attention K/V computed once, Euler steps unrolled; `scripts/export_onnx.py`, verified against `ICTPolicy.infer` to ~1e-7). WebGPU: fp32, ~0.1 s per re-plan; CPU fallback: fp16 weights, 10 steps |
| perception | three.js renders from the calibrated policy camera: robot-free clean image, object-ID masks, metric depth → the released KptsSelector / lift / PCA / VisualKpts logic |
| control | `driver.js` (quintic re-targeting + DLS IK on `mj_jacSite`), `controller.js`, `runner.js` |

The simulation advances in lock-step with the network (it pauses while a re-plan is
computed), so behaviour does not depend on how fast your machine is.

## Self-tuning loop (`autotune/`)

An agent-in-the-loop population search over simulator and low-level controller
parameters, after the population loop of
[recursive-tamp](https://github.com/ColinYu1/recursive-tamp) (re-implemented here):

```
repeat:
  candidates  <- rank-weighted diagonal CEM (unit cube, per-parameter bounds)
  evaluate every candidate on the SAME randomized scenes (common random numbers)
  evidence    <- success, failure modes (grasp_missed / dropped_in_transit / rim),
                 Spearman(parameter, score), bound saturation
  decision    <- continue_local | retune_search | revise_global | validate_and_stop
  revise_global -> inbox/request_NNN.json; an agent (e.g. Claude Code) answers with
                   inbox/response_NNN.json (new bounds, frozen values, candidates, stop)
```

```bash
python simulation/scripts/autotune.py --name my_rig --pop 8 --scenes 14 --workers 12
```

Searched parameters (`autotune/params.py`): the controller knobs of
`cfg/inference/serve_bread/Control.yaml` (EMA, look-ahead, stride, goal-time factor,
safety cage, grasp threshold and look-ahead window, idle-done release, forced lift),
the TCP offset, and the bread physics within the range of real crumb (Young's modulus,
damping, crust friction, gripper force cap). Then

```bash
python simulation/scripts/autotune_validate.py --name my_rig --top 4 --min_iter 5 --n 32
```

re-evaluates the best candidates and the production settings on the same held-out scenes
and writes `final.json` (what `build_web.py --params` uses).

**Runs `soft_v4` → `soft_v5` (soft bread, two arms; per-arm success on the same 32 held-out scenes):**

| | per-arm | right (croissant) | left (danish) | both served |
|---|---|---|---|---|
| production `Control.yaml`, near-elastic crumb | 28 % | 28 % | 28 % | — |
| `soft_v4` tuned, near-elastic crumb | 58 % | 84 % | 31 % | — |
| production, lossy crumb + physical success | 41 % | 41 % | 41 % | 13 % |
| `soft_v4` tuned, lossy crumb | 66 % | 91 % | 41 % | 41 % |
| **`soft_v5` tuned (adds tool pitch), lossy crumb** | **70 %** | **91 %** | **50 %** | **50 %** |

`soft_v5`: driver goal time 0.37 s (`duration_smooth_param` 3.7), EMA 0.37, look-ahead 40,
tool point 0.9 cm deeper and the gripper held 7.6° more level than the commanded EE pose,
crumb E = 17.5 kPa, μ = 0.95, 13.8 N. Per robot (24 held-out scenes, own tuning runs):

| | production | tuned | calibration found |
|---|---|---|---|
| Franka Panda | 25 % | 33 % | tool point 1.25 cm *shallower* (keeps the palm off the bread), pitch 2.5° |
| UR5e + 2F-85 | 31 % | 38 % | tool point 1.3 cm deeper, pitch 5°, 21 N |

`soft_v4`'s tuned controller changed three things, all pointing at the grasp-height mechanism above:
the driver goal time drops from 1.2 s to 0.49 s (`duration_smooth_param` 12 → 4.9) and the
position EMA from 0.5 to 0.29, so the arm follows the descending plan instead of trailing it,
and the tool point moves 0.9 cm deeper along the (downward-pitched) approach
(`tcp_offset` −0.009). Everything else stays near production (look-ahead 46, grasp
threshold 0.58, `any` closing). The bread physics stays inside real-crumb ranges
(E = 17 kPa, μ = 0.92, 12.7 N). The left arm stays weak: the danish is 7.4 cm wide against
an 8.8 cm maximum gripper opening, and the left arm starts from the real rig's asymmetric
pose and sees a mirrored image. The agent's decisions (`inbox/response_*.json`) record the
reasoning between batches.

**Other robots.** The policy grasps like a human hand: end-on along the bread, fingertips
3–4.5 cm past its centre, approach pitched 20–35° down. The Trossen's 7.9 cm finger plates
(8.4 cm from fingertip to carriage, 6 cm tall) straddle half a croissant. The other hands
are built differently (measured in the EE frame of each model):

| | finger pads (approach × height) | fingertip → palm | what goes wrong with the human-style grasp |
|---|---|---|---|
| Trossen WidowX AI | 7.0 × 6.0 cm plates | 8.4 cm | — |
| Franka Hand | 1.7 × 1.7 cm pads on 5 cm fingers | ~5 cm | the palm housing lands on the bread and squashes it; thin fingers can tunnel into the crumb |
| Robotiq 2F-85 | 3.8 × 2.2 cm pads | 5.5 cm | shallow: a high pinch catches only the crust |

A second, simulation-specific problem: small, sharp-cornered pads (2F-85, Franka) can tunnel
into a surface-only flex while squeezing, and the bread then stays hooked on the opened
gripper and is carried away (most "dropped in transit" failures were this). A volumetric flex
(`physics.soft.dim: 3`, tetrahedra to the bread's centre) keeps the pads out: UR5e 17 % → 54 %
per arm on 12 scenes. It also makes dropped bread bouncier and costs the Trossen rig ~10
points, so it is set for the UR5e only (`per_robot:`); a finer collision skin
(`physics.soft.surface: fine`) helped less.

So each hand gets its own calibration, as when moving a policy to a new gripper:
`control.tool_pitch_deg` holds the gripper more level than the commanded EE pose (about the
spread axis, through the tool point) and `control.tcp_offset` moves the tool point along
the approach; together with the tracking speed and the gripper's servo they are tuned per
robot (`autotune_runs/franka_v1`, `ur5e_v1`). Per-robot values go in `per_robot:` of the
config.

Earlier findings on the rigid-bread proxy: the production grasp timing (`any` over the whole
chunk) is near-optimal there, and most failures were grasps, not placing.

## Web viewer (`web/`)

```bash
python simulation/scripts/export_onnx.py                  # policy for Live mode
python simulation/scripts/build_web.py --configs trossen_wxai:realsense_d405:bimanual:24,franka_panda:realsense_d405:bimanual:10
python simulation/web/tools/serve.py 8791 simulation/web   # http://127.0.0.1:8791 (Range requests, like GitHub Pages)
python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim
python simulation/web/tools/record.py http://127.0.0.1:8791/index.html teaser.mp4 --rollout <id> --clean
```

The viewer replays recorded `qpos` (30 Hz, interpolated to the display rate) through the
MuJoCo kinematic tree in three.js, with the policy's predicted 50-step trajectory, the
perception keypoints riding on the objects, and the exact images the policy saw.

## Layout

```
simulation/
  configs/serve_bread.yaml     rig calibration, scene, physics, controller (real values)
  humanego_sim/
    scene.py robots.py cameras.py   MJCF builder, robot / camera registries
    softbody.py                     soft bread: flex on a trilinear grid + skin
    driver.py env.py                SimArm (Trossen-driver behaviour + IK), physics + rendering
    perception.py controller.py     released perception modules, production controller
    runner.py evaluate.py sampling.py   closed loop, parallel evaluation, initial conditions
    web_export.py live_export.py replay.py glb_tools.py
  autotune/                    population search + decision rules + agent inbox
  scripts/                     build_assets, bread_gen, run_episode, evaluate, autotune, grasp_test,
                               diagnose_episode, fetch_robots, fit_robot_poses, export_onnx,
                               build_web, deploy_web, smoke_test
  web/                         index.html, css/, js/ (viewer.js, app.js, live/), tools/
```

## Credits

Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Trossen WidowX AI, Franka Panda, UR5e, Robotiq 2F-85). Breads, plate and tabletop are
procedural (`scripts/bread_gen.py`, `scripts/build_assets.py`).
Physics: [MuJoCo](https://mujoco.org); in-browser inference: ONNX Runtime Web; rendering: three.js.
