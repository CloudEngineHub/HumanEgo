# Upgrade log: sim-to-real validation (session "real")

Branch `upgrade/real`, worktree `/home/leo/leo/HumanEgo-wt/real` (from origin/main a9773c9). Results doc:
`simulation/docs/real_vs_sim.md`. Notes with citations: `docs/upgrade/real_notes/`.

## 2026-10-17

* **Inventory.** A subagent read the paper.
  * `AriaMimic/paper.txt` / `paper.pdf` are AlphaNIST (NVIDIA), not HumanEgo; the paper numbers come from the `.tex` sources and the figure scripts.
  * Serve Bread real: 95 % (38/40, single right arm, croissant).
  * Only the released checkpoint exists.
* **Videos.** A subagent inventoried them; I verified the events frame by frame.
  * The x8 / x16 labels contradict code-fixed durations: wrap-up = open 1.0 s + `go_home` 3.0 s, unchanged since March 2026 (checked in AriaMimic history).
  * Self-calibration per rollout gives A ≈ x22 and B5/B6 ≈ x47–54.
  * Tooling: `scripts/real/real_video_annotations.json` + `video_timing.py`.
* **Sim rollouts, dev seed 7777, Trossen bimanual, 32 scenes, logged** (`runs/real_dev7777/`):
  * production per arm 40.6 % (R 41, L 41);
  * tuned + skill 92.2 % (R 94, L 91).
  * `scripts/real/sim_phases.py` gives phases, grasp point, release height, lift and smoothness.
* **Findings vs video.**
  * Real sets the bread down; sim drops it from 5–8 cm (all sim releases are idle-done).
  * Real grasps near the bread's end (along ≈ 0.7); sim production 0.4–0.5.
  * Production lift (R 26 / L 15 cm) matches real (≈25 / 20 cm); tuned lifts 40 cm and is jerky (16 speed peaks vs 1–2).
  * Transport time matches production; production reach is ~1.5–2x slower than real.
* **AriaMimic `sim-logging`** (b06fa8f, pushed to that branch only):
  * `inference/SimLogger.py` + hooks; the selftest passes;
  * `pyflakes` is clean; the hooks were checked against the variables in scope.
* **`scripts/real/replay_real.py`** (scene / kinematic / closed / policy) and the `sim_as_real.py` fixture.
  * Fixed: `np.linalg.svd` with `full_matrices=True` on ~40k points allocated 12.8 GB and got OOM-killed. Now `full_matrices=False`.
  * Fixed: `MUJOCO_GL=egl` must be set before `import mujoco` (EGL context error).
  * Results on `fake_real_s5` (production, both arms failed): see real_vs_sim.md §5.
  * Added a scene refinement loop (perception fixed point) after this test.
* **Single-arm paper protocol, seed 7777, 40 scenes:**
  * **production 12.5 % (5/40)**, 35 grasp misses;
  * tuned 82.5 %;
  * portable params (Control.yaml knobs only): single 45 %, bimanual 62.5 % per arm.
* **Cross-condition, single arm, 32 scenes, production / tuned:**
  * Franka 21.9 / 75 %;
  * UR5e 12.5 / 75 %;
  * ZED 2i 12.5 / 81 %;
  * table 10 cm higher (`rig.cam_height` 0.336): **75 / 94 %**.

## 2026-10-18

* Pushed `sim-logging`; committed WIP 6824c9c on `upgrade/real`.
* **Table height measured from the real video.**
  * In two full-res frames with the arms folded at home (A end, B start), measured camera-above-base / base-above-table = 1.47 (398/271 and 330/225 px, same depth).
  * The calibration puts the camera 0.221 m above the bases, so the real bases are ≈0.15 m above the table (camera ≈0.37 m). The sim has 0.215 m (0.436 m), an inference from a `safe_z_min` comment.
  * Sweep queued: cam_height 0.366 (single / bimanual, production / tuned), 0.346, 0.396.
* Height sweep, seed 7777: single-arm production at cam_height 0.366 → **67.5 %** (27/40; 9 grasp misses, 3 rim, 1 drop) vs 12.5 % at 0.436. (Box load ~120 from other jobs; runs are slow.)
* Height sweep at cam_height 0.366, seed 7777:
  * single production 67.5 %;
  * single tuned 85.0 %;
  * bimanual production 56.2 % per arm (R 71.9, L 40.6; both served 28 %).
  * Bimanual tuned, 0.396 and 0.346 are still running.
  * Landed on main 12ec02b (via worktree HumanEgo-wt/real-land; checks: pyflakes, video_timing, sim_phases, replay scene + kinematic on the fixture).
* Sweep finished, single production by cam_height: 0.436 → 12.5 %, 0.396 → 40 %, 0.366 → 67.5 %, 0.346 → 75 %, 0.336 → 75 %.
  * Bimanual tuned at 0.366: 79.7 % per arm (R 81, L 78) vs 92.2 % at 0.436.
  * Conclusion: table height is the single largest sim-to-real lever found. Next: E1 (real depth gives the camera height directly), then re-tune.
* Human-data agent finished (60 episodes): croissant 14.5 × 7.1 × 5.4 cm; at t0 the human bread is 36 ± 4 cm forward of the camera foot (sim 39–55) and 14.5° below the optical axis (sim 4.3°); only 2/60 human layouts are inside the sampler on all axes. In-distribution set proposed in docs/proposed_scene.json (bread x 0.29–0.45, gap 0.21–0.31 m, yaw 87 ± 10°).
