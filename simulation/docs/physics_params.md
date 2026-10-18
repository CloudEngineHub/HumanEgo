# Physics parameters: values, ranges, sources

Every physical parameter of the serve_bread simulation, where it comes from and where it is set.
"Range" is the defensible range from the sources; values outside it were moved (column *was*).
Validation tests (no policy): `scripts/physics_compress.py`, `physics_pinch.py`, `physics_drop.py`,
`physics_gripper.py`, `drop_test.py`; their results are in [upgrade/physics.md](upgrade/physics.md).

Paths: `yaml` = `configs/serve_bread.yaml`, `scene` = `humanego_sim/scene.py`, `soft` = `humanego_sim/softbody.py`,
`assets` = `scripts/build_assets.py` / `bread_gen.py`, `robots` = `humanego_sim/robots.py`.

## Bread

| parameter | value | range | was | source | set in |
|---|---|---|---|---|---|
| croissant size | 13.8 × 6.2 × 5.0 cm | L 13–15.5, H 5–6.5 cm | | released dataset contour keypoints (13.6 ± 0.7 cm long after the dataset's ~7 % scale bias); robot rollouts 14–15.5 cm long, 6.0 ± 0.5 cm tall (`serve_bread.mp4` f0, `full_demo.mp4` 67.5 s: probably another croissant) | `bread_gen.py` |
| danish size | 14 × 7.4 × 6 cm | L 13.5–14.3, H 5.4–6.4 cm | | rollout video against the 24.5 cm plate: 13.9 ± 0.4 cm long, 5.9 ± 0.5 cm tall | `bread_gen.py` |
| mass | croissant 50 g, danish 75 g | 0.15–0.33 g/cm³ bulk | | real weights; check: 0.22 and 0.17–0.23 g/cm³ against croissant 3.1–5.3 cm³/g (Steinkellner et al. 2025, *Curr Res Food Sci* 10:101042; Dmitrieva & Andreev 2016) and brioche 0.33 g/cm³ (Assad-Bustillos et al. 2018) | `yaml physics.bread_mass` |
| Young's modulus E | 20 kPa | 10–60 kPa | | brioche 20 ± 3 kPa in compression (Assad-Bustillos, Guessasma, Réguerre, Della Valle 2018, COMSOL Conf.; *Food Hydrocoll.* 91:153, 2019); white crumb 10.5–12.6 kPa in tension (Zghal, Scanlon & Sapirstein 2001, *Cereal Chem* 78:1), 18–20 kPa fresh from G′ at 5 Hz (Persaud, Faubion & Ponte 1990, *Cereal Chem* 67:92); croissant 15–60 kPa extrapolated from firmness (Steinkellner 2025; weak) | `yaml physics.soft.young` |
| Poisson ratio ν | 0.25 | 0–0.33 | | 0.17–0.28 (Guessasma & Nouri 2015, *Food Res Int*, via secondary source); open-cell foams ≈ 1/3 (Gibson & Ashby, *Cellular Solids* 1997) | `yaml physics.soft.poisson` |
| damping (Rayleigh, stiffness-proportional) | 0.03 s | 0.002–0.032 s | | loss tangent 0.13–0.20 at 1–10 Hz (Persaud 1990); tan δ = βω, so 0.03 s is tan δ ≈ 0.19 at 1 Hz, the grasp time scale. Rayleigh damping cannot keep tan δ constant: modes above ~10 Hz are over-damped, which stands in for the plastic crushing that kills impacts | `yaml physics.soft.damping` |
| yield / plastic set | off (option: yield strain 0.15, retain 0.15) | yield 1–6 kPa (strain 0.05–0.3); springiness 0.85–0.93 | | brioche yields at ~3 kPa, elastic–plastic (Assad-Bustillos 2018); TPA springiness 0.85–0.93 (Steinkellner 2025; Wu et al. 2012, *J Food Drug Anal* 20:133); no visible dent > 2–3 mm after the real pinch (`serve_bread.mp4` f5 vs f88) | `yaml physics.soft.plastic`, `soft.PlasticSet` |
| restitution on the plate | e ≈ 0.16–0.19 (emerges) | 0–0.2 (estimate) | | no measured value for any baked good; estimate 0.05–0.15 from impact energy vs crumb plateau stress (lit. review); real croissant release from 6 ± 1.5 cm: no bounce, at rest within 0.27 s (`serve_bread.mp4` f62–f69) | test: `physics_drop.py` |
| grid / skin | trilinear 4×2×2, dim 2 surface, radius 1 mm | numerical | | MuJoCo flex; elasticity is continuum E, ν on the grid cells (MuJoCo `user_flexcomp.cc` ComputeLinearStiffness; corotational, `engine_passive.c`) | `yaml physics.soft` |

## Contacts

MuJoCo takes the larger friction of the two surfaces when their priorities are equal, and the higher-priority
surface's parameters otherwise. The bread now has priority 0 like every geom: its own friction is the floor
(crust on hard surfaces, crust on crust) and each other surface sets its own; a large `solmix` makes the
bread's `solref` / `solimp` win. No published friction exists for bread crust on any material (searched food
engineering, conveyor, food-robotics literature); the values are proxies.

| contact | μ | range | was | source | set in |
|---|---|---|---|---|---|
| crust – silicone / rubber finger pads | 0.8 | 0.5–1.1 (0.3 if buttery) | 1.0 | food on rubber 0.62–0.90 (Saglam & Aktas 2005, *J Agron* 4:308; Pensamiento-Niño et al. 2019, *J Food Sci Technol* 56:763); urethane finger pad 0.69 (Spiers et al. 2020, *IEEE ToH* 13:600). Pads: Trossen "custom molded silicone grip pads" (trossenrobotics.com/widowx-ai), Robotiq silicone / NBR fingertips (2F-85 manual), Franka unspecified | `yaml physics.pad_friction` -> finger collision geoms (`scene._load_robot_spec`) |
| crust – plastic plate | 0.4 | 0.25–0.55 | 1.0 | foods on glass/steel/plywood 0.3–0.5 (same papers); PLA/steel 0.34 (Wang et al. 2021, *Front Robot AI* 8:640805); PS/steel 0.3 | `yaml physics.plate_friction` |
| crust – maple table; plate – table | 0.45 | 0.3–0.6; 0.2–0.5 | 0.8 | food on plywood 0.35–0.58; wood/wood 0.25–0.5 (Engineering ToolBox, weak) | `yaml physics.table_friction` |
| crust – crust | 0.4 | 0.3–0.6 (unmeasured) | 1.0 | generic dry-solid band; sugar on stainless 0.45–0.6 (Duffy & Puri 1996) | `yaml physics.bread_friction[0]` |
| bread contact softness | solref (4 ms, ζ 3), solimp (0.95, 0.995, 1 mm) | numerical | solimp (0.9, 0.95) | with solimp 0.95 the fingers sank up to 5 mm into the contact itself and a 40 mm crumb cube responded like E = 6.8 kPa instead of 20; with 0.995 it is 16 kPa and the penetration ≤ 1 mm; ζ 3 keeps the landing dead (e ≈ 0.17) | `yaml physics.bread_solref/solimp` |
| friction regulariser | impratio 100 | ≥ 30 | | elliptic-cone friction has no position term: under a steady load a held bread slips at a speed ∝ 1/impratio (MuJoCo `engine_core_constraint.c`); measured creep 0.1–0.4 mm/s at 100 (`physics_pinch.py`); real transport: no visible slip (< 1 cm, < 5°; `serve_bread.mp4` f18–f40) | `yaml physics.impratio` |

## Grippers

Squeeze per finger on a rigid block (`physics_gripper.py`), closing command as the real rig (`close_gripper_goal_time` 0.5 s).

| hand | stroke | speed | force (datasheet) | sim | was | source | set in |
|---|---|---|---|---|---|---|---|
| Trossen WidowX AI | 0–44 mm per finger (driver limit) | ≤ 0.25 m/s per finger; real close 0.53 ± 0.13 s (video) | driver effort clamp 100 N (`effort_max`); real position-mode squeeze unknown | Menagerie servo kp 339 N/m (system-identified on the real arm, Menagerie changelog), cap 35 N: 3.7 / 5.2 / 8.0 N per finger at 40 / 60 / 74 mm; t90 0.45 s | cap 12 N (danish 6 N/finger) | docs.trossenrobotics.com/trossen_arm (wxai specs, default config); Menagerie `trossen_wxai` | `yaml physics.gripper_force` |
| Franka Hand | 80 mm | 50 mm/s per finger | 70 N continuous, 140 N max | force-limited stiff servo: 20 N per finger (frankx default grasp force) at every width; t90 0.47 s (driver ramp; the real hand needs ≥ 0.7 s) | Menagerie servo: 1.0–1.3 N per finger | Panda datasheet; franka.de/franka-hand; frankx | `yaml physics.robot_gripper.franka_panda` |
| Robotiq 2F-85 | 85 mm | 20–150 mm/s | 20–235 N (force setting) | tendon cap 2.2: 21 N per pad, the gripper's minimum force setting | cap 40: 295 N per pad, above the hardware maximum | Robotiq 2F-85/140 manual §4.3, §4.5, Table 5-1 | `yaml physics.robot_gripper.ur5e` |

The Trossen's real squeeze is the least certain number here: the driver closes in position mode towards 0 with
an effort clamp of 100 N, but its gains' units are undocumented. A squeeze near the clamp (~50 N per finger) would
crush the croissant (> 50 % compression at its 3 kPa yield) and leave a deep dent; the rollout videos show none
(> 2–3 mm), which favours the system-identified servo (≈ 4–8 N). The sensitivity sweep covers 2–25 N.
The Trossen's commanded closed position is −2 mm in `robots.py` (the real driver closes to 0; controller session).

## Plate and table

| parameter | value | range | was | source | set in |
|---|---|---|---|---|---|
| plate rim diameter | 24.5 cm | 24.2–24.8 | | circle fit to the Aria SLAM points of 42 demo episodes: 24.5 ± 0.3 cm | `assets build_plate`, `plate_meta.json` |
| plate rim height | 3.0 cm | 2.5–3.9 | 2.2 cm | three rollout views (16°, 21°, 27° elevation): 2.9 ± 0.4 cm; SLAM profile 3.5 cm (3.1–3.9) | `assets build_plate` (H) |
| flat well radius | 8 cm | 7.1–8.3 | | ringed slope starts at r ≈ 7.7 ± 0.6 cm (video) | `plate_meta.json` |
| plate mass | 450 g | | | weighed (README) | `yaml physics.plate_mass` |
| table | maple butcher block | | | video | `scene` |

## Simulator settings (numerical, not physical)

| setting | value | why |
|---|---|---|
| timestep | 2 ms | |
| integrator | `discrete` | the only MuJoCo 3.16 integrator that integrates flex elasticity implicitly (`implicit`/`implicitfast` raise an error, Euler/RK4 blow up at 2 ms) |
| cone | elliptic | |
| per-robot collision skin | UR5e dim 3, Franka fine surface + 2 mm shell | see [upgrade/physics.md](upgrade/physics.md), numerical patches |
