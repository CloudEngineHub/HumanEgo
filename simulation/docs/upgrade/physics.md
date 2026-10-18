# Physics-fidelity upgrade: experiment log

## Summary (for the report)
Physics now comes from measurements and the literature instead of autotune (sources: `docs/physics_params.md`;
main 22c0fa1, 55047dc and this commit). What changed:
- the bread contact was so soft that the fingers sank into the contact itself (a crumb cube of E = 20 kPa
  responded like 6.8 kPa, contact depth 5 mm): now stiff in steady state (16 kPa, <= 1 mm) and still dead on landing
  (restitution 0.16-0.19);
- friction was 1.0 on every bread contact: now per surface (finger pads 0.8, plate 0.4, table 0.45, crust 0.4);
- grippers to their datasheets: the 2F-85 squeezed 295 N per pad (above its 235 N max), now ~21 N (its minimum
  setting); the Franka ~1 N per finger, now 20 N; the Trossen keeps its system-identified servo (4-8 N per finger);
- plate rim 2.2 -> 3.0 cm (real video 2.9 +- 0.4 cm); autotune no longer searches physics.

Faithful baseline, same scenes, old scene height (cam 0.436 m), both dev seeds pooled, per arm:

| robot | production controller (skill off) before -> after | tuned controller + skill before -> after |
|---|---|---|
| Trossen | 38 -> 26 % | 91 -> 84 % |
| UR5e + 2F-85 | 30 -> 32 % | 86 -> 86 % |
| Franka | 22 -> 24 % | 64 -> 71 % |

Within the sourced ranges, single physics parameters move the Trossen's tuned success by up to ~10 points
(contact softness, pad friction, crumb modulus, squeeze force; none significant alone at n = 32). Plastic set changes
nothing significant (kept off; Python only). The UR5e still needs the volumetric flex and the Franka the fine skin
(about -50 points without). The final numbers at the new scene height are the coordinator's reserved-seed test.


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

## Step 1: physics v2 (main 22c0fa1), policy-free validation (runs/physics/v2/)
Hypothesis: parameters from sources (docs/physics_params.md) change the policy-free behaviour towards the real
measurements. Change: bread solimp (0.9,0.95) -> (0.95,0.995) [solref (4 ms, ζ 3) kept]; friction by surface
(bread priority 0 + solmix 1000; crust 0.4, pads 0.8, plate 0.4, table 0.45; was 1.0 everywhere); Trossen gripper
cap 12 -> 35 N (Menagerie); 2F-85 tendon cap 40 -> 2.2 (295 -> 21 N per pad); Franka force-limited servo
(1.0-1.3 -> 20 N per finger); plate rim 2.2 -> 3.0 cm; optional plastic set (off).

| test | before | v2 | real / literature |
|---|---|---|---|
| crumb cube apparent E (cfg 20 kPa) | 6.8 kPa, contact depth 5.1 mm | 16.3 kPa, 1.5 mm | E is the configured crumb modulus |
| pinch 6 N/finger, croissant: indentation / contact depth | 5.2 / 1.5 mm | 4.6 / 0.7 mm | |
| holding (flat pads, μ_pair 1.0): slip load at 2 / 6 N squeeze | 4 / 15 N | 4 / 11-15 N | Coulomb 2μN = 4 / 12 N |
| creep of a held bread under its weight | 0.1-0.2 mm/s | 0.15-0.2 mm/s | no visible slip |
| drop on plate 3-9 cm: e, rest time | 0.17-0.20, 0.2-0.6 s | 0.16-0.19, 0.1-0.7 s | no bounce, < 0.27 s |
| danish dropped 4-9 cm on the croissant: stays on plate | 88 % | 100 % (3 cm rim) | (real: set down) |
| gripper per finger (40/60/74 mm block) | T 3.7/5.2/7.9, F 1.0/1.3, R 295 N | T 3.7/5.2/8.0, F 20, R 21 N | T ?, F 20 (5-70), R 20-235 |

Numerics (runs/physics/num/): creep at 0.5 N load: impratio 10 / 100 / 1000 -> 0.21 / 0.16 / 0.11 mm/s; at 6 N:
0.94 / 0.27 / 0.04 mm/s; slip load unchanged (15 N). dim 3 under v2: drop e 0.16-0.19 (not bouncier),
danish-on-croissant 88 %, pinch stiffness unchanged.

Plasticity (runs/physics/plastic/): croissant pinched to 30 % of its width: set 15 % (retain 0.3), cycle
energy loss 27 % (elastic: 0 %, 1.6 %); danish 15 %: set 5.5 %. Literature (springiness 0.85-0.93 after 50 %)
suggests ~4-9 % at 30 % -> retain 0.15 (default).

## Step 1 policy check (runs/physics/v2_*.json vs before_*.json, same scenes, per arm)
| robot | controller | seed | before | v2 |
|---|---|---|---|---|
| Trossen | tuned + skill | 7777 / 8888 | 92 / 91 % | 83 / 86 % |
| Trossen | production | 7777 / 8888 | 41 / 34 % | 20 / 31 % |
| UR5e | tuned + skill | 7777 | 92 % | 85 % (drops in transit 2 -> 0) |
| UR5e | production | 7777 | 31 % | 35 % |
Kept: the parameters come from sources; the lower Trossen score is mostly the left arm (danish 91 -> 75 %),
whose pinch with the real fingers now holds up to ~1 cm above its centre instead of ~2 cm (grasp_test.py on a
recorded pose; croissant unchanged at 2.3 cm). Landed on main.

## Baseline before any physics change (runs/physics/before_*.json, compare_params, same scenes)
| robot | seed | tuned (skill on) | production (skill off) |
|---|---|---|---|
| Trossen | 7777 | 92.2 % | 40.6 % |
| Trossen | 8888 | 90.6 % | 34.4 % |
| UR5e | 7777 | 91.7 % | 31.2 % |
| UR5e | 8888 | 81.2 % | 29.2 % |
| Franka | 7777 | 62.5 % | 18.8 % |
| Franka | 8888 | 64.6 % | 25.0 % |

Queued (runs/physics/*.sh): v2 bundle (baseline.sh v2), Trossen full one-at-a-time sensitivity (sens.sh),
UR5e / Franka lite sensitivity + numerical-patch sets (sens2.sh).

## Step 2: faithful baseline, before vs after (runs/physics/{before,v2}_*.json; Wilson 95 % CIs)
Production = `configs/serve_bread.yaml` controller, grasp skill off; tuned = `configs/tuned/<robot>.json` + skill.
Old scene height (cam 0.436 m; main now uses 0.366 m, so these are relative results).

| robot | controller | physics | seed | per arm | right | left | both |
|---|---|---|---|---|---|---|---|
| Trossen | production | before | 7777 | 41 [29, 53] | 41 [26, 58] | 41 [26, 58] | 12 [5, 28] |
| Trossen | production | before | 8888 | 34 [24, 47] | 28 [16, 45] | 41 [26, 58] | 16 [7, 32] |
| Trossen | production | before | pooled | **38 [30, 46]** | | | |
| Trossen | production | v2 | 7777 | 20 [12, 32] | 22 [11, 39] | 19 [9, 35] | 0 [0, 11] |
| Trossen | production | v2 | 8888 | 31 [21, 43] | 28 [16, 45] | 34 [20, 52] | 16 [7, 32] |
| Trossen | production | v2 | pooled | **26 [19, 34]** | | | |
| Trossen | tuned + skill | before | 7777 | 92 [83, 97] | 94 [80, 98] | 91 [76, 97] | 88 [72, 95] |
| Trossen | tuned + skill | before | 8888 | 91 [81, 96] | 94 [80, 98] | 88 [72, 95] | 84 [68, 93] |
| Trossen | tuned + skill | before | pooled | **91 [85, 95]** | | | |
| Trossen | tuned + skill | v2 | 7777 | 83 [72, 90] | 91 [76, 97] | 75 [58, 87] | 66 [48, 80] |
| Trossen | tuned + skill | v2 | 8888 | 86 [75, 92] | 94 [80, 98] | 78 [61, 89] | 72 [55, 84] |
| Trossen | tuned + skill | v2 | pooled | **84 [77, 90]** | | | |
| UR5e | production | before | 7777 | 31 [20, 45] | 25 [12, 45] | 38 [21, 57] | 0 [0, 14] |
| UR5e | production | before | 8888 | 29 [18, 43] | 38 [21, 57] | 21 [9, 40] | 4 [1, 20] |
| UR5e | production | before | pooled | **30 [22, 40]** | | | |
| UR5e | production | v2 | 7777 | 35 [23, 50] | 25 [12, 45] | 46 [28, 65] | 8 [2, 26] |
| UR5e | production | v2 | 8888 | 29 [18, 43] | 33 [18, 53] | 25 [12, 45] | 12 [4, 31] |
| UR5e | production | v2 | pooled | **32 [24, 42]** | | | |
| UR5e | tuned + skill | before | 7777 | 92 [80, 97] | 96 [80, 99] | 88 [69, 96] | 83 [64, 93] |
| UR5e | tuned + skill | before | 8888 | 81 [68, 90] | 96 [80, 99] | 67 [47, 82] | 62 [43, 79] |
| UR5e | tuned + skill | before | pooled | **86 [78, 92]** | | | |
| UR5e | tuned + skill | v2 | 7777 | 85 [73, 93] | 96 [80, 99] | 75 [55, 88] | 71 [51, 85] |
| UR5e | tuned + skill | v2 | 8888 | 88 [75, 94] | 100 [86, 100] | 75 [55, 88] | 75 [55, 88] |
| UR5e | tuned + skill | v2 | pooled | **86 [78, 92]** | | | |
| Franka | production | before | 7777 | 19 [10, 32] | 4 [1, 20] | 33 [18, 53] | 0 [0, 14] |
| Franka | production | before | 8888 | 25 [15, 39] | 21 [9, 40] | 29 [15, 49] | 4 [1, 20] |
| Franka | production | before | pooled | **22 [15, 31]** | | | |
| Franka | production | v2 | 7777 | 31 [20, 45] | 46 [28, 65] | 17 [7, 36] | 8 [2, 26] |
| Franka | production | v2 | 8888 | 17 [9, 30] | 21 [9, 40] | 12 [4, 31] | 0 [0, 14] |
| Franka | production | v2 | pooled | **24 [17, 33]** | | | |
| Franka | tuned + skill | before | 7777 | 62 [48, 75] | 88 [69, 96] | 38 [21, 57] | 29 [15, 49] |
| Franka | tuned + skill | before | 8888 | 65 [50, 77] | 88 [69, 96] | 42 [24, 61] | 42 [24, 61] |
| Franka | tuned + skill | before | pooled | **64 [54, 72]** | | | |
| Franka | tuned + skill | v2 | 7777 | 69 [55, 80] | 100 [86, 100] | 38 [21, 57] | 38 [21, 57] |
| Franka | tuned + skill | v2 | 8888 | 73 [59, 83] | 83 [64, 93] | 62 [43, 79] | 46 [28, 65] |
| Franka | tuned + skill | v2 | pooled | **71 [61, 79]** | | | |

## Step 3: sensitivity (runs/physics/sens3_*.json; tuned + skill, seed 7777, one change at a time from v2,
paired on the same scenes; v2 itself scored 81 % and 80 % in the two Trossen batches: run-to-run noise)

| robot | change (one at a time, from v2) | per arm: v2 -> change | paired diff [95 % CI] | McNemar p |
|---|---|---|---|---|
| Trossen | softer contact (solimp 0.9/0.95, before v2) | 81 -> 92 % | +11 [+2, +22] | 0.092 |
| Trossen | μ 1.0 on every bread contact (before v2) | 81 -> 78 % | -3 [-12, +6] | 0.77 |
| Trossen | pad friction 0.5 | 81 -> 72 % | -9 [-22, +3] | 0.21 |
| Trossen | crust/plate/table friction 0.25/0.25/0.3 | 81 -> 88 % | +6 [-3, +16] | 0.39 |
| Trossen | Trossen squeeze x4 (~15-30 N/finger) | 81 -> 91 % | +9 [-2, +20] | 0.18 |
| Trossen | plastic set on | 81 -> 84 % | +3 [-3, +9] | 0.62 |
| Trossen | volumetric flex (dim 3) | 81 -> 77 % | -5 [-16, +6] | 0.58 |
| Trossen | pad friction 1.1 | 80 -> 91 % | +11 [+2, +22] | 0.12 |
| Trossen | crust/plate/table friction 0.55/0.55/0.6 | 80 -> 80 % | +0 [-8, +8] | 1 |
| Trossen | crumb E 10 kPa | 80 -> 84 % | +5 [-5, +14] | 0.51 |
| Trossen | crumb E 60 kPa | 80 -> 70 % | -9 [-20, +3] | 0.21 |
| Trossen | Trossen squeeze x0.5 (~2-4 N/finger) | 80 -> 72 % | -8 [-17, +0] | 0.27 |
| Trossen | bread mass +30 % | 80 -> 80 % | +0 [-8, +8] | 1 |
| Trossen | impratio 10 | 80 -> 75 % | -5 [-14, +5] | 0.51 |
| UR5e | surface flex (dim 2) instead of dim 3 | 85 -> 33 % | -52 [-62, -42] | 6e-08 |
| UR5e | plastic set on | 85 -> 85 % | +0 [-6, +6] | 1 |
| UR5e | 2F-85 at 235 N | 85 -> 88 % | +2 [-4, +8] | 1 |
| Franka | coarse skin instead of fine | 71 -> 23 % | -48 [-60, -35] | 1.5e-06 |
| Franka | plastic set on | 71 -> 81 % | +10 [+0, +21] | 0.12 |
| Franka | Franka 70 N per finger | 71 -> 69 % | -2 [-15, +10] | 1 |

Reading: the Trossen's grasp of the danish is the sensitive part. Everything that makes the pinch grip more
(softer contact that lets the fingers sink in, higher pad friction, a softer crumb, a harder squeeze) helps it,
by ~10 points each, and the opposite costs about as much; the bread's mass and the hard-surface friction matter
little. So the remaining uncertainty of the physics is worth about +-10 points of Trossen success; the biggest open
measurement is the real gripper's squeeze (position mode, gains undocumented).

## Step 4: plasticity (kept off)
`physics.soft.plastic` (softbody.PlasticSet): every 10 steps, node deviations beyond a yield (0.15 x cell size)
leave a permanent offset (retain 0.15) by rewriting `m.flex_node0`; restored at reset. Policy-free: a 30 %
pinch of the croissant leaves a 15 % set at retain 0.3 (27 % of the cycle energy lost; elastic: ~0); the default retain 0.15 halves the set, towards TPA springiness 0.85-0.93. On the policy:
Trossen +3 [-3, +9], UR5e +0, Franka +10 [+0, +21] (p = 0.12): not significant, so it stays off. It is
portable to the browser (MuJoCo WASM exposes `model.flex_node0` as a writable typed array, re-fetch after memory
growth) but not ported (web/js/live is outside this work).

## Step 5: numerical patches
- `discrete` integrator: required (only it integrates flex elasticity implicitly in MuJoCo 3.16).
- impratio 100: creep of a held bread under its weight 0.16 mm/s (10: 0.21, 1000: 0.11); near slip 0.27 vs
  0.94 mm/s at 10. Policy: impratio 10 costs -5 [-14, +5] points (n.s.). Kept at 100.
- Per-robot collision: with the stiffer contact the patches are still needed: UR5e without the volumetric flex
  -52 points (pads tunnel, the bread stays hooked: 29 drops in transit), Franka without the fine skin -48. The
  volumetric flex on the Trossen costs -5 [-16, +6] (n.s.; the README's earlier -10 was on the old contact).
  No single setting fits all three without a likely Trossen cost, so `per_robot` stays as is.

## Hand-over: recommendation for the final controller-only re-tune
- Freeze physics at the main defaults. `autotune.params.DEFAULT_SPACE` holds no physics (55047dc);
  `configs/tuned/*.json` contain no `physics.*` keys (checked), so nothing needs removing.
- Search `DEFAULT_SPACE` + `GRASP_SKILL` per robot, centred on the shipped tuned values, with:
  - Trossen left arm (danish, 91 -> 75 %): the pinch now holds only up to ~1 cm above the danish's centre (2 cm
    before, real fingers on a recorded pose): `grasp_assist.height_max` -0.014..0.0 (now -0.006), `retry_lower`
    0..0.012, `regrasp_max` 1-4, `along_max` 0.01-0.05, `lateral_gain` 0.4-1.0, `tool_pitch_deg` -15..5,
    `tcp_offset` -0.015..0.01;
  - production-style grasps miss more (the bread slides on the table, crust-wood 0.45): `arrive_tol`,
    `close_hold`, `control.alpha`, `duration_smooth_param`;
  - UR5e (2F-85 now ~21 N per pad, the driver joint closes less around a bread): `grasp_assist.gap_empty`
    (now 0.015) and `q_empty`; Franka (20 N per finger): `q_empty` (now 0.03).
- Report the winner's spread at the ends of the sensitive physics ranges (pad friction 0.5/1.1, E 10/60 kPa,
  Trossen squeeze x0.5/x4) as a robustness check, not as a search knob.
- Controller-side detail: `robots.py` commands the Trossen gripper to -2 mm; the real driver closes to 0.
