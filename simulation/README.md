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
| Objects | photoscanned croissant / roll and a ceramic plate at the sizes measured from the released dataset |

## Quick start

```bash
conda create -n humanego-sim python=3.11 -y && conda activate humanego-sim
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r simulation/requirements.txt
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints

python simulation/scripts/build_assets.py                 # once: object meshes (downloads CC0 photoscans)
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
  keypoints (croissant ≈ 13.8 × 6.2 × 5 cm, plate ⌀ 24.5 cm) and real weights
  (croissant 50 g, roll 65 g, plate 450 g).
* **Grasp state.** The real loop flags a grasp when the measured gripper opening drops
  below 50 % — soft bread squashes well below that. Rigid sim bread cannot squash, so a
  gripper that has finished closing and stalled on an object also counts
  (`control.grasp_state_source`).
* **Soft-bread proxy.** The policy pinches like the humans in the data: ~1–2 cm above the
  bread's centre (measured from the 60 demos: 1.2 cm above the contour centroid, approach
  ~7° below horizontal). Real bread deforms around the fingers and holds; rigid bread is
  squeezed out. Each finger therefore gets a small MuJoCo adhesion actuator that is on
  only while the gripper is closed (`physics.finger_adhesion`, 0 = rigid).

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
the TCP offset, and the soft-bread proxy (adhesion, friction, gripper force) — all within
physically plausible bounds. Findings from the loop so far:

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
    driver.py env.py                SimArm (Trossen-driver behaviour + IK), physics + rendering
    perception.py controller.py     released perception modules, production controller
    runner.py evaluate.py sampling.py   closed loop, parallel evaluation, initial conditions
    web_export.py live_export.py replay.py glb_tools.py
  autotune/                    population search + decision rules + agent inbox
  scripts/                     build_assets, run_episode, evaluate, autotune, diagnose_episode,
                               fetch_robots, fit_robot_poses, export_onnx, build_web, deploy_web
  web/                         index.html, css/, js/ (viewer.js, app.js, live/), tools/
```

## Credits

Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Trossen WidowX AI, Franka Panda, UR5e, Robotiq 2F-85). Bread photoscans:
[Poly Haven](https://polyhaven.com) (CC0). Maple tabletop texture:
[trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco) (BSD-3-Clause).
Physics: [MuJoCo](https://mujoco.org); in-browser inference: ONNX Runtime Web; rendering: three.js.
