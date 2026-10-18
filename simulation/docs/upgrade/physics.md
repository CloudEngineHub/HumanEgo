# Physics-fidelity upgrade: experiment log (in progress)

Branch `upgrade/physics`. Policy-free tests: `scripts/physics_bench.py` (scene breads/plate/table with the
episode MJCF, robots stripped), `physics_compress.py`, `physics_pinch.py`, `physics_drop.py`, `physics_gripper.py`.
Bench gotchas: fixtures must be added before stripping the robots (MjSpec.delete breaks later mesh lookups);
fixtures that drag objects must be joint-driven (a teleported mocap body has zero velocity, so friction never acts);
the camera mast stands at the world origin, so tests run at TEST_XY = (0.55, 0).

## Findings so far (current main physics unless stated)
1. **Contact compliance hides the crumb modulus.** 40 mm crumb cube, E = 20 kPa configured: apparent E 6.8 kPa;
   the platen sinks 5.1 of 11.4 mm into the contact (solref [0.004, 3] on ~1 g flex nodes). solref [0.004,1]: 15 kPa
   (1.5 mm); **solref [0.004,3] + solimp [0.95,0.995,0.001,0.5,2] (variant C): 16.3 kPa, 1.5 mm**, drop COR unchanged (0.17),
   danish-on-croissant drop stays on plate 92 % (current 88 %, ζ=1 only 50 %). Variant C is the candidate.
2. **Holding force follows Coulomb with flat pads** (Trossen-size pads, croissant): squeeze 2 N -> slip at 4.5 N load
   (1.1 x 2μN); 6 N -> 15 N. Sub-slip creep 0.1-0.8 mm/s at impratio 100.
3. **Gripper squeeze vs datasheets** (rigid block, physics_gripper.py): Trossen 3.7/5.2/7.9 N per finger at 40/60/74 mm
   (Menagerie servo kp 339, sysid'd per the Menagerie changelog; actuator clamp 12 N binds on the danish), closing
   t90 0.45 s (real video 0.53 ± 0.13 s). Franka 1.0-1.3 N per finger (datasheet 70 N cont.; frankx default 20 N).
   **2F-85 295 N per pad (> 235 N hardware max)** with gripper_force 40; ~9.1 N per pad per unit of tendon force limit.
4. Drop on plate (6 cm, real release height): e ≈ 0.17-0.20, rest 0.3-0.6 s (real: no visible bounce, rest < 0.27 s).
5. Literature/measurement (sources in the session report; to go into docs/physics_params.md): E croissant 15-60 kPa
   (weak), brioche 20 ± 3 kPa; ν 0.17-0.28; tan δ 0.13-0.20 (1-10 Hz); yield ≈ 3 kPa; springiness 0.85-0.93;
   COR est. 0.05-0.15; crust-pad μ 0.8 (0.5-1.1), crust-plate 0.4, crust-table 0.45 (sim: 1.0 on all bread contacts);
   plate rim 2.9 ± 0.4 cm (SLAM 3.5) vs 2.2 cm in sim; croissant in rollouts ~6 cm tall (sim 5 cm, dataset 5 cm).
6. Plasticity is feasible: writing `m.flex_node0` (rotation removed) holds a dent exactly for squash; works in WASM too.

## Baseline before any physics change (runs/physics/before_*.json, compare_params, same scenes)
| robot | seed | tuned (skill on) | production (skill off) |
|---|---|---|---|
| Trossen | 7777 | 92.2 % | 40.6 % |
| Trossen | 8888 | 90.6 % | 34.4 % |
| UR5e | 7777 | 91.7 % | (running) |

## Next steps
- Implement in scene.py/softbody.py/serve_bread.yaml: variant-C solimp; per-surface friction (bread priority 0 +
  2F-85 pads priority 0, bread solmix large so its solref/solimp win, friction by max: bread 0.4, finger pads 0.8,
  plate 0.4, table 0.45); grippers: Trossen clamp 35 N (Menagerie), 2F-85 force ≈ 2.2 (20 N/pad, hardware minimum),
  Franka force-controlled ~20 N/finger (kp_scale ~30, tendon clamp 40); plate rim H 3.0 cm (build_assets.py).
- Plasticity option (yield on node strain, Python; JS port note for the controller/web session).
- Re-check dim-3 UR5e / fine-skin Franka with stiff contacts; impratio creep test at 10/100/1000.
- Policy before/after (both dev seeds, 3 robots, tuned + production), one-at-a-time sensitivity, physics_params.md.
- Controller-session note: robots.py Trossen gripper_closed = -0.002 (real driver closes to 0).
