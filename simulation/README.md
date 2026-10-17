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

Shorter look-ahead does not help (it raises the grasp to +2.2 cm); a shorter goal time
lowers it by 0.3 cm. With friction-only soft bread this matters: a pinch holds when the
fingertips are ≤ ≈ 1.9 cm above the centre and slips out above that (the pads then press
on the rounded top, above the bread's widest section), independently of grid resolution,
crumb stiffness or friction (`scripts/grasp_test.py` replays recorded grasp poses with any physics). The
rigid + adhesion proxy hid this.

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
damping, crust friction, gripper force cap). Findings from the loop so far:

* the production grasp timing (`any` over the whole chunk) is near-optimal — closing later
  (`grasp_k_any = 15`) misses more grasps;
* most failures were mislabelled "release" errors: 22 of 25 failed arms never lifted the
  bread, so the search targets grasping, not placing.

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
