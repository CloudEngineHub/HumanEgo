# Simulation vs the real robot (serve_bread): status 2026-10-17

The run was cut short by the usage limit. Sections 3 (human-data object statistics) and 4
(cross-condition / single-arm sim runs) are still pending: their jobs were running when it stopped.
Raw notes with file:line citations are in `docs/upgrade/real_notes/`.

## 1. Real evidence that exists

* **Paper.** `AriaMimic/paper.txt` / `paper.pdf` are a different paper (AlphaNIST, NVIDIA). Use
  `paper/example.tex` (CoRL with appendix), `arxiv_v3/main.tex`, `icra_2026/main.tex` and the
  plotting scripts. Serve Bread is evaluated **single-arm (right arm), one croissant, 40 trials**:
  HumanEgo-30 95 % (38/40), HumanEgo-15 82.5 %, ACT 52.5 %, SPOT 62.5 %.
  * Success means the croissant rests on the plate, in any orientation.
  * Protocol: bread–plate horizontal offset U[0, 50] cm, independent depths, "arbitrary" start pose.
  * Failure modes are defined (off the plate, slip in transport) but never counted.
  * Data efficiency, hand tracker, co-training and cross-condition numbers are in `real_notes/paper_real_numbers.md`.
* **Checkpoints.** Only the released one exists (HF `Leo-TX/HumanEgo`, = `Baselines/01_HumanEgo`,
  60 demos, camera frame, no aux, the 95 % model). No ablation or data-efficiency weights exist
  anywhere on this machine or on HF.
* **Videos.** Third-person only, sped-up edits:
  * Clip A (labelled x8, bimanual): `website/static/videos/serve_bread.mp4`.
  * x16 montage (B1–B6, C, D): `cross_condition_3.mp4`.
  * All other files are copies or sub-clips of these. No failures are shown.
  * **The speed labels contradict the code.** The wrap-up after a release is open 1.0 s (blocking) + `go_home` 3.0 s; that has been fixed since March 2026. Clip A shows it in 9.5 frames (x8 → 1.3 s), B5/B6 in 4–4.5 frames, B1 in 2.
  * Calibrating each rollout with code-fixed motions (wrap-up, `go_initial` 2.5 s, left delay 6 s) gives A ≈ x22 and B5/B6 ≈ x47–54. B1's rulers are only 2 frames long, too coarse to use.
  * Annotations: `scripts/real/real_video_annotations.json`. Calibration: `scripts/real/video_timing.py`.
* **Real logs, per-trial outcomes, real-time video: none exist.** The real loop saves `raw.mp4` +
  overlays in `results/policy/<stamp>/` on the robot PC, but none are on this machine.

## 2. Real (video) vs sim, bimanual Trossen, dev seed 7777, 32 scenes

Sim production per arm: 41 % (R 41, L 41). Sim tuned + grasp skill: 92 % (R 94, L 91).
Sim columns are medians over served arms.

| quantity | real (video) | sim production | sim tuned | verdict |
|---|---|---|---|---|
| transport, lift→release (s) | A 15–22, B5/B6 8–12 | R 11.8, L 16.1 | 8.0 / 8.3 | prod inside the real range; tuned at its lower edge |
| reach, first motion→grasp (s) | 1.0–3.7 (A-L, B5, B6) | 4.2–4.8 | 2.4–2.8 | **sim production reach ~1.5–2x slow** |
| transport / wrap-up (scale-free) | 2.6–5.4 | 3.3–4.5 | 2.2–2.3 | prod matches |
| grasp point along bread (0 = middle, 1 = tip) | ≈0.7: fingers over the outer ~30 % (A, B5) | 0.39 R / 0.47 L | 0.56 / 0.50 | **real grasps nearer the end** |
| release | bread **set down** on plate / croissant, then opened | bread bottom 7.9 cm (R) / 4.9 cm (L) above plate; all releases by idle-done | 4.4 / 3.2 cm | **sim drops, real places** |
| peak lift above table | R ≈25 cm, L ≈20 cm (plate = 24.5 cm scale) | R 26, L 15 | **40–41** | prod matches; tuned lifts 2x too high |
| smoothness (speed peaks in transport) | continuous, no visible stops | 1–2 | **16–17** (0.33 m/s peaks) | tuned motion is jerky, unlike real |
| L/R timing | left starts ≥6 s after right; right releases first | Δclose 6.0 s, Δopen 8.4 s | 6.7 / 7.2 s | consistent |
| failure modes | none shown (curated); paper: slip / off-plate, not counted | prod: slipped_on_lift 29/64 arms | 8/64 arms fail | — |

Discrepancies, ranked by likely impact:
1. **Success gap.** Real 95 % (single arm) vs sim production 41 % per arm; production fails mostly by slipping at lift. Grasp physics or grasp location in sim is too unforgiving.
2. **Release: placing vs dropping.** Every sim release is idle-done at ~13 cm from the plate; the real robot sets the bread down. Bounce / roll-off failures (the danish landing on the croissant) may be a sim artefact.
   * Hypothesis: the real ICT grasp flag (gripper opening < 50 %) does not fire on a thick bread, so idle-done never starts and the release comes from the policy or the operator. The sim adds a stall rule that the real loop lacks. The real log's `done_reason` and `g_pos` settle this.
3. **Grasp location** (end vs nearer the middle); it interacts with 1 and 2.
4. **Tuned controller realism.** It lifts 2x too high and moves jerkily, so its 92 % is not evidence about the real robot. Recommend a smoothness term in autotune (speed peaks / LDLJ, from `sim_phases.py`).
5. **Reach speed** (production too slow). Possibly the real policy loop rate differs from the sim's 0.3 s assumption; the logger records `loop_dt`.
6. **Minor.**
   * The sim D405 intrinsics are nominal (fx_raw 660); the real K is unknown and now logged.
   * The real wrap-up opens 1.0 s blocking, the sim 0.8 s.
   * The real arm goes idle (limp) at home.
   * The paper's protocol (single arm, offset 0–50 cm, arbitrary start) is wider than `sampling.py` single (gap 16–38 cm, fixed start).

## 3. Initial conditions and objects

Pending; the human-data agent was still measuring.
* Partial finding: all 61 demos use the same croissant and plate (no danish), plate left / bread right.
* Partial finding: ep 060 has wrong intrinsics files (1920x1080 re-extraction); ep 033 has a bad bread triangulation.
* Results will be in `/tmp/.../scratchpad/notes/human_object_stats.{md,json}`.

`docs/proposed_portable_controller.json` holds the tuned Trossen values restricted to knobs that exist in the real `Control.yaml`. A sim prediction for it was queued (see 4).

## 4. Sim–real correlation

* A checkpoint-level correlation is impossible: one checkpoint.
* Needed from the robot / training PC (`runs/serve_bread/...`):
  * `ScalingLaw/01_HumanEgo_HumanData/N{1,3,6,12,24,36,48}` (real 0 / 2.5 / 12.5 / 37.5 / 67.5 / 85 / 92.5 %, 40 trials each);
  * the HumanEgo-15 checkpoint (82.5 %);
  * `HandTracking/0{1,2,3}` (0 / 45 / 32.5 %);
  * the co-training mixes.
  * `XFrame/01_AnchorFrame` also needs anchor-frame support in `runner.py`.
* Alternative: retrain the N-series locally (data are local; 400 epochs each).
* Condition-level, with the released checkpoint: the paper's cross-condition means are Trossen 91.25, Franka 88.75, UR10 87.5, ZED 90, height 86.25 (Serve Bread + Downstack averaged).
  * Queued in sim (single arm, 32 scenes, production + tuned): Franka, UR5e, ZED 2i, table 10 cm higher, plus single-arm Trossen and the portable params.
  * Results land in `/tmp/claude-1000/-home-leo-leo/07eae8a5-6f4d-4727-9051-cae6319cdd15/scratchpad/logs/{single7777_*,xc_*,portable_*}.json`.

## 5. Logging and replay (done, tested)

* **AriaMimic branch `sim-logging`** (`inference/SimLogger.py`, hooks in `Inference.py` / `InferenceController.py`, `inference/SIM_LOGGING.md`).
  * Logs calibration, the t = 0 perception, RGB-D + policy images per step, plans / grasp / done outputs, every controller command, events with the done reason, and 50 Hz joints / gripper / EE.
  * `python -m inference.SimLogger selftest <dir>` passes without hardware.
* **`scripts/real/replay_real.py <log>`** (tested on `scripts/real/sim_as_real.py` logs with known truth):
  * scene: table plane recovered to 0.4 cm / 0.07°; objects within 1.0–2.1 cm and 1–12° before refinement (keypoint bias towards the camera); the refinement loop was added after this test and still needs a rerun;
  * kinematic replay: outcome agrees 2/2, FK residual median 0.3–0.45 cm;
  * closed loop: phases within 0.1–1.6 s; EE path DTW 3–4 cm;
  * policy replay: plan error 0.2 cm, the same as the noise floor; sim-image plan difference 0.0 cm on sim logs, so it isolates the visual gap on real logs.
  * Known issue: `real_phases` misreads the left arm's release when the gripper never reopens.
* `scripts/real/sim_phases.py`: phase / grasp / release / smoothness metrics of sim runs.

## 6. Real experiments to run (≈2.5 h of robot time)

* **E0.** Selftest + 1 logged rollout: check files and `loop_dt`.
* **E1.** 5 static scenes with breads at tape-measured positions, plus a 5-pose joint sweep. Validates scene reconstruction and FK.
* **E2.** Paper protocol, single right arm, croissant, released checkpoint, production `Control.yaml`: 20 logged trials spanning offsets 15–50 cm and yaw 60–120°, filmed at real time (60 fps phone).
* **E3.** Bimanual (croissant R, danish L), production: 20 logged trials.
* **E4.** E3 with `docs/proposed_portable_controller.json` values in `Control.yaml`. Tests whether the sim's predicted gain transfers.
* **E5.** If found or retrained: N12 and N24 checkpoints, 10 trials each on the E2 protocol (correlation).

Then run `replay_real.py` on every log. Compare the outcome, `done_reason`, gripper `g_pos` while holding, release height and phase timings with the sim on the same rebuilt scenes.
