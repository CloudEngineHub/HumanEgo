# HumanEgo: real-robot numbers for serve_bread, with sources (for the sim-to-real validation report)

Compiled 2026-10-17. I only read files. Abbreviations used for paths:

| tag | file |
|---|---|
| **CoRL** | `/home/leo/leo/AriaMimic/paper/example.tex` (CoRL submission, with appendix) |
| **arXiv-v1** | `/home/leo/leo/AriaMimic/paper/arxiv.tex`: the same text as CoRL. `diff` shows only the author block, the preprint option, the code-URL sentence and the acknowledgments differ. |
| **v3** | `/home/leo/leo/AriaMimic/paper/arxiv_v3/main.tex` (Sep 14; the most complete and most corrected version) |
| **ICRA** | `/home/leo/leo/AriaMimic/paper/icra_2026/main.tex` (no appendix) |
| **REB-v6 / REB-v1** | `/home/leo/leo/AriaMimic/paper/corl_rebuttal/corl-rebuttal-template_humanego/rebuttal_template_v6.tex` / `rebuttal_template.tex` |
| **ANALYSIS** | `.../corl-rebuttal-template_humanego/analysis.md` (internal pre-rebuttal audit) |
| **figs/** | `/home/leo/leo/AriaMimic/paper/figs/` (plotting scripts with the data hard-coded) |
| **cfg/** | `/home/leo/leo/AriaMimic/cfg/` |

> **WARNING: `paper.txt` is not HumanEgo.** `/home/leo/leo/AriaMimic/paper.txt`, `paper.pdf` and `arxiv_abs.html` (all dated Oct 16) are **"AlphaNIST: Emergent Assembly Skills from Binary Task Rewards" (arXiv:2610.13354, NVIDIA)**. See paper.txt:1 and the `<title>` in arxiv_abs.html. Nothing in this report comes from that file. Every HumanEgo number below is from the .tex sources and the figure scripts.

---

## 1. Real-robot success rates for serve_bread

### 1.1 Main result (Fig. "Overall Real-World Evaluation", `fig:baselines`)

The per-task bars come from `figs/baselines_plot.py:96-105` (the `DATA` dict). The "Overall" bar is computed as the mean of the 4 tasks (`baselines_plot.py:363-366`). The text in `figs/baselines.pdf` matches these values.

| Method (training data) | **Serve Bread** | k/40 (implied) | Downstack | Water Fl. | Adjust T. | 4-task mean |
|---|---|---|---|---|---|---|
| HumanEgo-30 (30 min human) | **95.0** | 38/40 | 87.5 | 95.0 | 92.5 | 92.5 |
| HumanEgo-15 (15 min human) | **82.5** | 33/40 | 67.5 | 75.0 | 75.0 | 75.0 |
| ACT (30 min robot teleop, wrist and top cameras) | **52.5** | 21/40 | 45.0 | 45.0 | 62.5 | 51.25 ("51.2") |
| SPOT | 62.5 | 25/40 | 35.0 | 45.0 | 37.5 | 45.0 |
| ZeroMimic | 47.5 | 19/40 | 27.5 | 30.0 | 0.0 | 26.25 |
| Track2Act | 35.0 | 14/40 | 12.5 | 17.5 | 10.0 | 18.75 |
| PointPolicy | 12.5 | 5/40 | 2.5 | 7.5 | 5.0 | 6.875 |
| EgoZero | 5.0 | 2/40 | 0.0 | 2.5 | 0.0 | 1.875 ("1.9") |

How the text reports these numbers:
- **n = 40 trials per task.** CoRL:575-579 says "we report success rate (%) over 40 trials per task with randomized initial object positions". The same wording is at v3:660-661 and ICRA:722-723.
- CoRL:618-620 (= v3:684-686, ICRA:751-753): "HumanEgo with only 30 minutes of human data reaches **92.5%** average success rate across all four tasks." This is a **4-task average**, not a Serve Bread number. The Serve Bread value (95%) appears only in the figure and its script.
- CoRL:630-631 (= v3:700-701, ICRA:767-768): "with only 15 minutes of human data already reaches 75.0%, surpassing ACT trained on 30 minutes of robot teleoperation (51.2%)". Both numbers are 4-task averages.
- Serve Bread's own ACT number appears in the text only once: v3:777-779 / ICRA:829-831 / CoRL:703-705, "ACT trained on 30 minutes of robot teleoperation (52.5% on this task)".
- All five human-video baselines are "trained on the same 30 minutes of human data per task, are reimplemented in our codebase, and receive the same perception frontend, controller, and post-processing as HumanEgo, including the grasp latching" (v3:669-673, ICRA:737-741). CoRL:612 says only "trained on the same 30 minutes".

### 1.2 What counts as a trial and what counts as success

- **Single-arm, per episode.** CoRL:968-984 = v3:1140-1156: "*Scene.* A croissant and a dinner plate sit on a tabletop, plate on the left and bread on the right. ... The right arm starts from an arbitrary, non-aligned pose above the bread. *Target behavior.* Grasp the croissant from above, lift it clear of the table, transport it to a release pose above the center of the plate, and open the gripper. *Success.* The croissant comes to rest on the plate, in any orientation. *Failure.* (a) the bread ends up outside the plate (on the table, draped over the rim, or knocked off); or (b) the bread slips from the gripper during transport."
- Each trial has one croissant, so the success rate is per episode (equivalently per bread). The training config is `single_hand: True`, `single_hand_side: "right"` (`cfg/training/serve_bread/Baselines/01_HumanEgo.yaml:42-43`).
- **Partial-credit or stage-wise scores** (grasp success vs place success) are **not reported** for Serve Bread in any version. Only the binary success above exists.
- v3:1662-1667 (= ICRA:581-587): latching "keeps the object token correct under occlusion, but a physical slip out of the gripper still ends the episode in failure."

### 1.3 Other real-robot numbers that include Serve Bread
These are covered in §4: data efficiency (Serve Bread only), the hand-tracker study (Serve Bread), the co-training study (Serve Bread), the reference-frame study (CoRL only; the task is not named but is almost certainly Serve Bread), and the cross-condition study (mean of Serve Bread and Downstack Cups).

---

## 2. Initial-condition protocol for serve_bread

- **Randomization.** CoRL:971-976 = v3:1143-1148: "we independently randomize (i) the horizontal offset between the two objects, sampled from **[0, 50] cm**, and (ii) their depths along the table's front–back axis, so the pair is generally *not* colinear. The right arm starts from an arbitrary, non-aligned pose above the bread."
- Number of discrete positions: **not specified.** The offset is described as continuous [0, 50] cm. The depth range is not given.
- **Orientation of the bread or plate: not specified.**
- **Bread: a croissant.** Only "croissant" appears (CoRL:588, 969, 977). **No "danish" or other bread** appears in any paper version or in the rebuttal. Perception prompts are generic: `obj1: "a piece of bread ."`, `obj2: "a plate ."` (`cfg/inference/serve_bread/ObjectCentric.yaml:12-16`; training preprocessing uses `"piece of bread ."` in `cfg/preprocess/tasks/serve_bread.yaml:5-8`). Novel bread instances appear only in the OOD "Novel Objects" condition.
- **Distractors, lighting, table:** not specified for the in-distribution evaluation. They are varied only in the OOD study (§4.6). REB-v1:90 says the conditions use "lighting, background, **three to eight distractors**, and object instances differing in shape, size, and colour", and that "All nine conditions vary one factor at a time".
- Placement robustness claim: CoRL:752-756: "On Serve Bread, it delivers the bread to the plate across arbitrary absolute and relative positions on the table and even at novel heights". v3:844-845 softens "arbitrary" to "a wide range of".
- Hardware: CoRL:1460-1472 = v3:1712-1724: "two Trossen WidowX AI arms mounted side-by-side ... a single Intel RealSense D405 mounted top-down above the workspace". Arm extrinsic for the right arm: `cfg/inference/serve_bread/RobotArmTrossenRight.yaml:14-19`. Initial joints: `:25`.
- Object poses are estimated **once per episode** from 10 RGB-D frames (`ObjectCentric.yaml:26`), with `pose_method: "pca1"` (`:22`). v3:1648-1650: "Detection, segmentation, and orientation estimation run once per episode".

---

## 3. Failure modes

| What | Source | Count? |
|---|---|---|
| Serve Bread failure categories: (a) bread off the plate, on the table, over the rim or knocked off; (b) slip during transport | CoRL:982-984, v3:1154-1156 | **No counts given** |
| "The lowest scores come from the point-based methods ... their failures concentrate at the grasp, where the gripper arrives offset from the randomized object and the episode cannot recover." | v3:690-694, ICRA:757-761 (not in CoRL) | qualitative |
| "the dominant failure is detection or segmentation, at a measured rate below 5%" | REB-v6:59 (also REB-v1:78) | **<5%**, applies to all tasks, not specific to Serve Bread |
| Latching "assumes it does not move in the hand; in-hand rotation or re-grasp after slip would break this, which we did not observe in these tasks" | REB-v6:59 | — |
| Physical slip ends the episode in failure | v3:1666-1667 | — |
| Safety cage: "we did not observe any safety-cage clamp during normal rollouts" | CoRL:1517-1518, v3:1769-1770 | 0 |
| Monocular hand trackers: "5–11 cm systematic depth offset ... so the policy never learns a consistent grasp" | CoRL:1594-1596, v3:1848-1850 | — |
| Limitations: "few-shot learning plateaus at ~1 cm precision" | CoRL:869, v3:949 | — |

The rebuttal planning notes asked for per-trial failure logs ("Per-trial failure logs for the baselines, especially EgoZero on all four", `rebuttal_outline.md:302`) and Wilson confidence intervals (`rebuttal_outline.md:170`). Neither appears in any paper version.

---

## 4. Ablation and variant results with real-robot numbers

I checked whether each variant's weights exist. **None exist except the released main checkpoint** (§7).

### 4.1 Data efficiency (Fig. `fig:scaling`, Serve Bread only, per task)

Source data: `figs/data_efficiency_plot.py:107-113`. The figure has no printed values.

| Collection time (min, x-axis) | 1 | 2 | 4 | 8 | 12 | 18 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|
| HumanEgo (human data, with aux) | 0 | 10 | 27.5 | 57.5 | 72.5 | 85 | 92.5 | **95** |
| HumanEgo without aux (human data) | 0 | 2.5 | 12.5 | 37.5 | 67.5 | 85 | 92.5 | **95** |
| HumanEgo (robot teleop data) | 0 | 0 | 5 | 12.5 | 27.5 | 37.5 | 50 | **65** |
| ACT (robot teleop data) | 0 | 0 | 0 | 7.5 | 20 | 32.5 | 40 | **52.5** |

How the text reports it:
- CoRL:671-673: "as a function of collection time on Serve Bread". CoRL:676-678: "With only ~7 minutes ... already reaches 50% success rate ..., reaching 95% at 30 minutes".
- CoRL:685-688: "Between 2 and 12 minutes, HumanEgo with auxiliary losses consistently outperforms the variant without them, with the largest gain at 8 min (57.5% vs. 37.5%). Beyond 18 minutes both variants converge, reaching 95%". v3:759-760 changes this to "At 18 minutes and beyond the two variants coincide".
- CoRL:703-706: "At 8 minutes ... (57.5%) already surpasses ACT trained on 30 minutes ... (52.5%)---a 3.75× reduction".
- v3:781-784 only: "the same policy trained on 30 min of robot teleoperation reaches 65% versus 95% on human video (+30 pp)".
- The number of trials per point is not stated separately. The global 40-trials rule applies, and every value is a multiple of 2.5, which is consistent with 40 trials.

**The x-axis was relabelled.** `data_efficiency_plot.py:107` reads `TIME_MINS = np.array([1, 2, 4, 8, 12, 18, 24, 30])# [1, 2, 4, 8, 16, 24, 32, 40]`. The commented-out older axis is the "40-min" accounting. Only the last four points were rescaled by 0.75.

Configs that would reproduce each curve (none have weights):
- **Without aux:** `cfg/training/serve_bread/ScalingLaw/01_HumanEgo_HumanData/N{1,3,6,12,24,36,48,60}.yaml`. All use `data_sources: {aria: 60}`. The file name and the `data_num` inside disagree: N1→1, N3→**5**, N6→**3**, N12→10, N24→20, N36→30, N48→40, N60→50. `data_num` means "the first N training sessions" (`HumanEgo/training/FlowMatchingTrainer.py:757-759`), not a random subset. The directory names match the old axis at 1.5 demos per minute (N60 = 40 min).
- **With aux:** `ScalingLaw/02_HumanEgoWAux_HumanData/N*.yaml` is **byte-identical to 01** (checked with `diff`), and no aux flag is set. So the with-aux curve cannot be reproduced from these files. The actual with-aux configs appear to be `AuxTraining/01_AuxObjVisCont_DataNum{1,3,5,10,20}.yaml` and `01_AuxObjVisCont.yaml` (60 demos). These set `use_aux_obj_dynamics`, `use_aux_visual_foresight` and `use_aux_temporal_contrastive` to True and use `data_sources aria: N`.
- **HumanEgo on robot data:** `ScalingLaw/03_HumanEgo_RobotData/N*.yaml` all set `teleop: 3`, which is wrong. The 30-min point (65%) matches `DataSource/01_Teleop.yaml` (`teleop: 40`, same architecture).
- **ACT:** `ScalingLaw/04_ACT_RobotData/N*.yaml` uses teleop 3,3,3,5,10,20,30,40. `RobotBaselines/ACT_N{1,2,4,8,16,24,32,40}_JointSpace.yaml` mostly have 100 epochs. N40 has v1, v2 and v3 at 400, 300 and 100 epochs. `cfg/inference/serve_bread/commands.md:13` runs `ACT_N40_JointSpace_v3`, which is 100 epochs. The paper's hyperparameter table says ACT is trained for 400 epochs (v3:2112).
- `cfg/inference/serve_bread/Policy.yaml:37-44` lists checkpoints named `ScalingLaw/0X_DataNumY`, a third naming scheme. None of these run directories exist.

### 4.2 Hand-tracking study (Serve Bread, per task, n = 40 each)

- CoRL:1577-1579 = v3:1831-1833: "train HumanEgo from scratch, and evaluate 40 real-world trials on Serve Bread".
- CoRL:1552-1555: "the same 45 demonstrations (30 min total)". Caption CoRL:1530-1531: "(45 demonstrations, ~45 k frames)".
- CoRL:1584-1586: "collapsing by 95 → 45 → 32.5 → 0 %". Source data: `figs/hand_tracking_results_plot.py:62-67`.

| Tracker | SR | k/40 | Config |
|---|---|---|---|
| Aria-MPS (default) | 95 | 38 | `Baselines/01_HumanEgo.yaml`. **This is the released checkpoint.** |
| WiLoR | 45 | 18 | `HandTracking/02_WiLoR.yaml` (aria: 60, no aux) |
| HaMeR | 32.5 | 13 | `HandTracking/03_HaMeR.yaml` |
| MediaPipe | 0 | 0 | `HandTracking/01_MediaPipe.yaml` |

Offline metrics in CoRL:1600-1609: keypoint error 1.4 cm (HaMeR and WiLoR) vs 2.2 cm (MediaPipe); detection rate 86.9% (WiLoR) vs 66.5% (MediaPipe).

### 4.3 Human–robot co-training (Serve Bread, per task, n = 40 per condition)

- CoRL:1649-1659 = v3:1903-1913: total collection time fixed at 30 min; "from 0% (pure robot teleoperation, ~45 teleop episodes) to 100% (pure human egocentric video, ~45 egocentric demos) in 25-pp steps ... We evaluate 40 real-world trials on Serve Bread per condition."
- CoRL:1670-1672: "65 → 72.5 → 77.5 → 90 → 95 % for human ratios of 0/25/50/75/100 %". Source: `figs/cotraining_plot.py:60-66`.

Matching configs:
- 0% human: `DataSource/01_Teleop.yaml` (teleop: 40).
- 100% human: probably the released 60-demo checkpoint (95%). `DataSource/00_Human.yaml` has aria: **30**.
- The only mixed config is `DataSource/04_CoTraining_TeleopAndHuman.yaml` (teleop 20 + aria 30). **There are no configs for the 25/50/75 splits.**

### 4.4 Reference-frame study (CoRL appendix only; dropped in v3 and ICRA)

- The text (CoRL:1711-1773) gives no numbers and does not name the task. The figure is `figs/coordinate_frame.pdf` (CoRL:1717).
- The **two plotting scripts disagree at the 4-minute point:**
  - `figs/coordinate_frame_plot.py:68-91` (the file CoRL includes): 4 min: anchor 27.5 / camera 22.5. 40 min: anchor 87.5 / camera **95**.
  - `figs/reference_frame_plot.py:63-66`: "4 min (6 demos)": camera 27.5 / anchor 32.5. "40 min (60 demos)": camera **95** / anchor 87.5.
- "40 min = 60 demos" and camera-frame 95% match the released checkpoint (camera_frame, 60 demos). The anchor-frame variant is `XFrame/01_AnchorFrame.yaml` (aria 60, anchor_frame).

### 4.5 Representation and auxiliary-loss studies (the paper says Water Flowers, not Serve Bread)

- CoRL:779-781: "We ablate HumanEgo's two core design choices on **Water Flowers**". v3:885-888 adds "at 30 minutes" (representation) and "at 15 minutes" (aux).
- Representation results (`figs/representation_plot.py:62-69`):

  | Input | SR |
  |---|---|
  | Human RGB | 7.5 |
  | + Kpts & Inpaint | 20 |
  | Robot RGB | 32.5 |
  | ICT only | 75 |
  | Human RGB + ICT | 85 |
  | Full | 95 |

  The CoRL caption (CoRL:824) says "**five** input configurations ... +52.5 pp". v3:872 and ICRA:848 say "**six** ... +62.5 pp". The "ICT only = 75%" bar comes from the rebuttal (REB-v6:68; v3:901-903).
- Auxiliary-loss results at 15 min (`figs/aux_training_plot.py:60-67`):

  | Variant | SR |
  |---|---|
  | No aux | 50 |
  | + Object Motion | 67.5 |
  | + 2D Trace | 55 |
  | + Latent Consistency | 62.5 |
  | Full | 75 |

  The script comment says "**20 trials each**". That cannot be right: 67.5% and 62.5% are impossible with 20 trials, so the data must be from 40 trials. ANALYSIS:126 (item C6) flags the same 40-vs-20 trial-count discrepancy.
- **Config mismatch.** The representation ablation configs exist only for **serve_bread**: `Representation/01_rgb_WoEST`, `02_rgbWoArmWArmObjKpts_WoEST`, `03_..._RobotData` (teleop 40), `04_rgb_WEST` and `05_..._WEST` (**both 0-byte files**), and `VisualInput/10_NoImg` (ICT only). `water_flowers/` has only aux configs 00–07, all with `data_num: 100` and `anchor_frame`, and no 15-minute limit. So either the paper mislabels the task, or the Water Flowers representation configs were never committed. Either way, **these are not Serve Bread numbers in the paper.**

### 4.6 Zero-shot cross-condition generalization (`fig:generalization`, **mean of Serve Bread and Downstack Cups**)

- CoRL:716-719: "on Serve Bread and Downstack Cups across 9 out-of-distribution conditions ... (40 trials each for each task)". That gives n = 80 per condition.
- v3:822-826 / ICRA:863-867: "each trained on 30 minutes of human data ... Trossen is the in-distribution reference ... report the **mean success rate over the two tasks**".
- Values (`figs/generalization_plot.py:99-126`, which match the text in `generalization.pdf`):

  | Condition | Mean SR |
  |---|---|
  | Trossen (in-distribution reference) | 91.25 |
  | UR10 | 87.5 |
  | Franka | 88.75 |
  | Novel Camera (ZED) | 90.0 |
  | Novel Height | 86.25 |
  | Novel Viewpoint | 87.5 |
  | Novel Lighting | 91.25 |
  | Novel Background | 91.25 |
  | Novel Objects | 85.0 |
  | Distractors | 91.25 |

- Check: Trossen 91.25 = (95 + 87.5)/2, the two main-table numbers. **No per-task Serve Bread value is published for any OOD condition.**
- Shift magnitudes from the rebuttal: Novel Camera and Novel Viewpoint are "up to 20 cm and 30° from the nominal mount" (REB-v6:62); distractors are 3–8 objects (REB-v1:90). The template CSV `figs/generalization/generalization.csv:13` says "Novel Height: e.g. +/- 10cm".
- Checkpoint: "each cross-condition result comes from the same checkpoint as its main-table task" (REB-v6:62). That means the Serve Bread half would use `Baselines/01_HumanEgo`, the released checkpoint.
- `generalization.csv` is an **empty template**: `n_success` is blank and `n_trials` is 20 per task. It also names the setup "Trossen ViperX + Realsense D435" and "UR5", which contradicts the paper's WidowX AI, D405 and UR10.

### 4.7 Other real-world comparison
- Matched-architecture data-source comparison: 65% on robot teleop data vs 95% on human video, Serve Bread, 30 min (v3:781-784; REB-v6:68).

---

## 5. Robot control details stated in the paper

| Item | Paper value | Source | Value in the inference config / code (`cfg/inference/serve_bread/Control.yaml`, `inference/Inference.py`) |
|---|---|---|---|
| Re-plan / control loop | 10 Hz, synchronous, ~100 ms old frames, RTX 4090 | CoRL:1497, v3:1684-1687, ICRA:633-636 | `control_freq: 10` (:8) |
| ODE solver | Euler, 20 steps, Δt = 1/20 | CoRL:1483-1489 | `Policy.yaml:63` 20 |
| Action chunk | K = 50 | CoRL:1488, v3:2080 | `exec_horizon: 50` (:48) |
| Stride and executed rate | stride 2, so 5 Hz | CoRL:1499-1500 | `step_stride: 2` (:10) |
| Look-ahead offset | **25 steps** | CoRL:1500-1502, v3:2084 | **`look_ahead_offset: 45`** (:11). **Mismatch.** |
| Gripper close rule | `any`-over-horizon, p > 0.6 | CoRL:1503-1505 | `should_grasp_method: 'any'`, `grasp_threshold: 0.6` (:28, :39) |
| Grasp latch | "optional" (CoRL:1505-1507); "enabled for all methods, ours included" (REB-v6:68) | | `grasp_latch: False` (:84), then overridden by **`grasp_latch: True`** (:101). In YAML the last duplicate key wins. |
| Smoothing | position EMA α = 0.5, quaternion SLERP on rotation, overlap blend 12 | CoRL:1510-1514 | `alpha: 0.5`, **`alpha_rot: 0.0`**, `duration_smooth_param: 12.0` (:16, :17, :24) |
| Safety cage | ≤0.08 m and ≤0.02 rad per cycle | CoRL:1515-1518 | :65-68 (identical) |
| Gripper | WidowX AI 6-DoF parallel-jaw, ~1.5 kg payload, ±1 mm | CoRL:1467-1469 | open/close goal time 0.5 s (:91-92) |
| Camera | single top-down RealSense D405, RGB only into the policy (depth feeds the perception frontend) | CoRL:1470-1472; corrected in REB-v6:59 | `CamRS.yaml`: 30 fps, 1280×720 raw resized to 640×480. Policy input 240×320. |
| ACT baseline | top and wrist cameras, 7-DoF joint positions, teleop 200 Hz control, 30 Hz recording | CoRL:1473-1477, 1904-1911 | |

Behaviour that exists only in the code and is **not described in any paper version** (important for matching the sim):
- **Forced lift after grasp.** A +5 cm base-frame Z bias over 15 inference frames with a 5-frame ramp is applied by default, because the serve_bread Control.yaml does not override it (`Inference.py:126-130`, `:1083-1085`).
- **Idle-done auto-termination.** After grasping for ≥1.5 s, if the end effector is within 0.13 m of a non-anchor object (the plate), the episode is marked done (`Control.yaml:60-62`; `Inference.py:1106-1114`). On done, the code opens the gripper and sends the arm home (`Inference.py:1287-1290`, `1308-1313`). With the grasp latch on, **the release onto the plate is triggered by this done logic, not by the policy's own gripper output.**
- Done from the policy's done head requires probability > 0.99 and a grasp having occurred (`Control.yaml:75`, `:89`; `Inference.py:1168-1172`).
- `bimanual: True`, "shared_context", with a 6.0 s start delay is set even though serve_bread is a single-arm task (`Control.yaml:96-99`).
- The released simplified inference config (`/home/leo/leo/HumanEgo/cfg/inference/example_dualarm.yaml:20-28`) uses different values: `exec_horizon 8`, `done_threshold 0.80`, `max_pos_step 0.05`, `grasp_threshold 0.50`, `alpha_rot 0.5`. The released `inference/*.py` has no forced lift, idle-done or grasp latch.

## 6. Episode duration and timing

- **No paper version gives a rollout duration, time limit or timeout.** I found no timeout in the inference code either. Episodes end through done, idle-done, or a manual key press.
- Demonstration times:
  - CoRL:1059-1063 = v3:1231-1235: "a single human egocentric demonstration takes approximately 30–40 seconds, while a matched teleoperated demonstration ... takes 60–70 seconds".
  - v3:677-680 / ICRA:744-747: "A human demonstration takes about 40 s, beginning with a ~1 s scene sweep ..., whereas an experienced operator needs about 60 s per teleoperated demonstration."
  - REB-v6:68: "Each demonstration lasts at most 40 s including a ~1 s sweep".
- **Measured from the released data** (`/home/leo/leo/HumanEgo/data/serve_bread/aria/*/preprocess/aria_phases_results.json`, field `duration_s`):
  - 61 sessions: mean 30.0 s, range 18.1–41.4 s.
  - The 60 training sessions (001–060) total **29.95 min**.
  - The first 45 sessions total 24.1 min. The first 30 total 17.8 min.

## 7. Checkpoints: what exists and what is missing

**Hugging Face** (checked with `huggingface_hub`, nothing downloaded):
- Model repo `Leo-TX/HumanEgo` (sha 9d14cb0, 2 commits) contains only:
  - `serve_bread/config.json` (5.3 KB)
  - `serve_bread/dataset_stats.json`
  - `serve_bread/latest.pt` (60.7 MB, inference weights)
  - `serve_bread/latest_full.pt` (182 MB)
- Commit message: "Add serve_bread HumanEgo checkpoint ... with **migrated** config.json (object_centric + camera_frame + max_ict)". The config was edited after training.
- Dataset repos `Leo-TX/HumanEgo` (serve_bread/aria and water_flowers/aria) and `Leo-TX/AriaMimic` contain no checkpoint files. Other model repos (`Leo-TX/hamer`, `Leo-TX/mediapipe-hand`) are unrelated.

**Local copy:** `/home/leo/leo/HumanEgo/checkpoints/serve_bread/{latest.pt, config.json, dataset_stats.json}` is the same HF snapshot. The `HumanEgo-wt/*/checkpoints` entries are symlinks to it.

The released config.json says:
- `out_dir ./runs/serve_bread/Baselines/01_HumanEgo`
- 60 training sessions (001–060) and 1 eval session (000); `data_num 100`, so no truncation
- `object_centric`, `camera_frame`, `rgb_WoArm_WArmObjKpts.png`, `single_hand right`
- **all three aux losses False**, `use_state_noise False`
- 400 epochs, seed 7

The state_dict has only `head_v` and `head_done`, with no aux heads. It has 15.17 M parameters, `state_proj` input dimension 20 (single-hand ICT), and `action_proj` 10.

**Which paper numbers this checkpoint corresponds to:**
- Most likely: the Serve Bread **HumanEgo-30 = 95%** (main table).
- The same 95% entry appears as "Aria-MPS" in the hand-tracking study, "Pure Human" in co-training, the no-aux 30-min point in data efficiency, and camera-frame "40 min (60 demos)" in the reference-frame study.
- It is also the Serve Bread half of the cross-condition study (REB-v6:62).
- Caveat: the paper says the main model uses aux losses. This checkpoint has none. The data-efficiency figure shows aux and no-aux both at 95% at 30 min, so the number is the same, but the variant is the no-aux one.

**Missing everywhere** (no `runs/` directory exists under AriaMimic; I searched all of `/home/leo/leo` for .pt/.pth/.safetensors/.ckpt files larger than 1 MB, excluding HumanEgo/data, third_party, node_modules and .git):
- HumanEgo-15 (82.5%): no dedicated config. REB-v6:68 says it is "a random half", but `data_num` takes the first N sessions.
- All data-efficiency points
- ACT
- The five human-video baselines
- WiLoR, HaMeR, MediaPipe
- Co-training mixes
- Anchor frame (`XFrame/01_AnchorFrame`)
- The VisualInput, Representation and AuxTraining variants
- Downstack Cups, Water Flowers and Adjust Table checkpoints

Files that look related but are not paper variants: `/home/leo/leo/HumanWAM/HumanEgoDex/out/runs*/{ict,ego,state,blind,ict_kp2}/{best,last}.pt`. These are a separate HumanWAM-project model with offline hold-out sessions. `/home/leo/leo/HumanWAM/HumanWAM8/out/ft_bread_*` are world-model fine-tunes. Neither is the paper's FlowMatching policy.

## 8. Inconsistencies

1. **How many demos, and how many minutes, per task:**
   - CoRL:369: "around 30 minutes".
   - CoRL:1805-1806 (hyperparameter table): "60 demos / 40 min". v3:1993-1994 corrects this to "45 / 30 min".
   - CoRL:1419 and v3:1592: "~40 min" (not fixed in v3).
   - CoRL:1553 and v3:1798: "45 demonstrations (30 min)".
   - REB-v6:68: 45 × ≤40 s = 30 min, and 60/40 was "an earlier configuration".
   - **However:** every serve_bread training config uses `aria: 60`, the released checkpoint uses 60 demos, and the 60 released demos measure **30.0 min**. So "30 min" most plausibly means 60 demos of about 30 s each, and the "45 demos" story in v3 and the rebuttal looks wrong.
2. **How many teleop demos for "30 min" of ACT data:**
   - "~45 teleop episodes" (CoRL:1651, v3:1905).
   - "30 demonstrations" (REB-v6:68).
   - Configs use `teleop: 40` (`DataSource/01_Teleop`, `RobotBaselines/ACT_N40*`, `ScalingLaw/04_ACT_RobotData/N40`).
   - At 60–70 s per demo, 40 demos would be 40–47 min.
3. **Trials per condition:** 40 in the paper; "20 trials each" in the aux_training_plot.py comment; 20 in generalization.csv. Values like 67.5 imply 40. ANALYSIS:126 flags this as item C6.
4. **The 41-point gap:** "+41%" in CoRL:119 vs "41 percentage points" in v3 and ICRA. The true difference is 92.5 − 51.25 = 41.25 pp.
5. **Representation figure:** "five configurations, +52.5 pp" in CoRL vs "six, +62.5 pp" in v3 and ICRA. `paper/figs/representation.pdf` was regenerated on Sep 14 with 6 bars, so recompiling CoRL would pair a 6-bar figure with a "five" caption.
6. **Ablation task label:** the representation and aux ablations are labelled Water Flowers, but the representation configs exist only under serve_bread, and the water_flowers aux configs have no 15-min limit.
7. **Data-efficiency x-axis:** relabelled from [1,2,4,8,16,24,32,40] to [1,2,4,8,12,18,24,30], so only the upper half was rescaled. The ScalingLaw directory names follow the old axis. The `data_num` values inside do not match the names (N3 has 5, N6 has 3). The "WAux" scaling configs are identical to the no-aux ones.
8. **Inference config vs paper:** look-ahead 45 in config vs 25 in the paper; `alpha_rot` 0 vs "SLERP"; grasp latch on despite "optional"; forced lift and idle-done termination are undocumented.
9. **Stale draft note:** CoRL:50-54 defines a note saying "experiments for Overall Real-World Evaluation (Fig. baselines), Auxiliary Training Study (Fig. aux_training), and Zero-Shot Generalization (Fig. generalization) are still in progress. Many data points are currently placeholders". The macro is defined but **never invoked** in CoRL or arXiv-v1. v3:136 and ICRA:54 disable it as "stale". It is still the authors' own record that these three figures' data were placeholders at some point.
10. **Reference-frame study:** two scripts give different 4-minute numbers (coordinate_frame: anchor 27.5 / camera 22.5; reference_frame: camera 27.5 / anchor 32.5). The study was dropped in v3.
11. **Frame mode:** the paper says one static camera frame (CoRL:457-458). ANALYSIS:546 and the configs show `frame_mode` is set per task: camera_frame for serve_bread and downstack_cups, anchor_frame for water_flowers and adjust_table. That is consistent with the released Serve Bread checkpoint, which uses camera_frame.
12. **Generalization CSV template** names the setup as ViperX, D435 and UR5, vs WidowX AI, D405 and UR10 in the paper.
