# HumanEgo in Simulation

Closed-loop simulation of the **released HumanEgo `serve_bread` policy** on a MuJoCo
replica of the real deployment rig (two Trossen WidowX AI arms + a top-mounted
Intel RealSense D405). The network is never touched: the simulator plugs into the
same `Camera` / `RobotArm` / `Perception` contract as [`inference/`](../inference/README.md)
and runs the released `ICTPolicy` on the released checkpoint.

> Status: work in progress (core loop runs end-to-end; parameter autotuning and the
> interactive web viewer are being built).

## Quick start

```bash
conda create -n humanego-sim python=3.11 -y && conda activate humanego-sim
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r simulation/requirements.txt
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints

python simulation/scripts/build_assets.py            # once: object meshes (downloads CC0 photoscans)
python simulation/scripts/run_episode.py --video     # bimanual serve_bread, writes simulation/runs/...
```

Run headless with `MUJOCO_GL=egl` (set automatically).

## What is simulated, and how faithfully

| Real system | Simulator |
|---|---|
| Trossen WidowX AI ×2, hand-eye calibrated (`T_base_in_cam`) | MuJoCo Menagerie `trossen_wxai`, bases placed from the **real calibration** (`configs/serve_bread.yaml`) |
| `trossen_arm` driver: non-blocking Cartesian goals re-planned at 10 Hz with `goal_time` | `SimArm`: quintic re-targeting from the current interpolator state (velocity continuous), DLS IK every physics step, reachability errors, `safe_z_min` floor, correction frame `C` |
| RealSense D405, 1280×720 → centre-crop → 640×480 | pinhole camera with the same crop/resize, depth noise model |
| DINO-SAM masks | ground-truth segmentation (same mask semantics: obj1 = all bread, obj2 = plate) |
| LaMa arm inpainting | render without the robot |
| KptsSelector → multi-frame depth lift → PCA pose → VisualKpts | **the released modules themselves** |
| `Inference.run_bimanual` (mirrored left arm, latching, forced lift, idle-done, wrap-up) | `runner.py`, same logic on a sim clock |
| `AsyncRobotController` (ensembling, safety cage, EMA/Slerp, look-ahead, stride) | `controller.py`, same logic on a sim clock |

**Scene geometry.** The camera sits 0.436 m above the table: the real driver's
`safe_z_min = −0.20 m` "protects the table", which places the table just below the
arm bases. Objects then lie ~0.6 m from the camera, inside the depth range of the
human training data. Object sizes are measured from the released dataset's
triangulated contour keypoints (croissant ≈ 13.8 × 6 cm, plate ⌀ 24.5 cm).

**Grasp state.** The real loop flags a grasp when the measured gripper opening
drops below 50 % — soft bread squashes to well below that. Rigid sim bread cannot
squash, so a gripper that has finished closing and stalled on an object also counts
(`control.grasp_state_source`).

**Soft-bread proxy.** The policy pinches like the humans in the data: ~1–2 cm above
the bread's centre. Real bread deforms around the fingers and holds; rigid bread is
squeezed out. Each finger therefore gets a small MuJoCo adhesion actuator that is on
only while the gripper is closed (`physics.finger_adhesion`, 0 = rigid).

## Layout

```
simulation/
  configs/serve_bread.yaml     rig calibration, scene, physics, controller (real values)
  humanego_sim/
    scene.py                   MJCF builder (robots from the calibration, objects, camera)
    robots.py  cameras.py      robot / camera registries
    driver.py                  SimArm: Trossen-driver behaviour + IK
    env.py                     physics, rendering, segmentation, depth
    perception.py              released perception modules fed by the simulator
    controller.py              production async controller on a sim clock
    runner.py                  bimanual closed loop around the released ICTPolicy
    replay.py                  episode → MP4
  scripts/                     build_assets.py, run_episode.py, diagnose_episode.py
```

## Credits

Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Trossen WidowX AI, BSD-3-Clause). Bread photoscans: [Poly Haven](https://polyhaven.com)
(CC0). Maple tabletop texture: [trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco) (BSD-3-Clause).
