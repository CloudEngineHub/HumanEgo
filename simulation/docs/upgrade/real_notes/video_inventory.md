# Real-robot serve_bread footage: inventory (HumanEgo, bimanual Trossen WidowX AI)

Generated 2026-10-17. All source files were only read; nothing under /home/leo/leo/AriaMimic was modified.
Frames: `scratchpad/real_frames/<video_stem>/`. Scripts: `scratchpad/scripts/` (sheet.py, dump.py, match.py, cuts.py, diffs.py, mindiff.py).

## 0. Key facts (read this first)

* **Two edits hold all the real serve_bread footage.** (1) **A**: one bimanual rollout at **x8**. (2) **B/C/D**: an **x16** montage. Every other file is a re-encode or a sub-clip of `full_demo.mp4`, matched frame-for-frame (see §2).
* **Frame index map ("fd" = frame index in website `full_demo.mp4`, 60 fps, fd = round(t*60)):**
  * `full_demo.mp4` (website) == `Synced/video/full_demo.mp4`. They are the same file (md5 `b2c524400cc36ec548311730bc36ef11`).
  * `HumanEgo_for_ICRA_compressed.mp4`: identical cut list and frame numbers in the bread section (fd 3830-4545 pixel MAD 3.3/255, i.e. only re-encode noise). It is anonymised elsewhere (blurred faces, no author list) and is 9 frames shorter at the end.
  * `serve_bread.mp4` frame k = fd 3837+k (k = 0..116). It has the x8 label and **no caption**.
  * `cross_condition_3.mp4` frame k = fd 3956+k (k = 0..580). It has the x16 label and **no caption**.
  * `HumanEgo_clip_2_twitter.mp4` frame k = fd 3386+k (k = 0..1150). It has the label and the caption.
  * `Synced/gifs/serve_bread.gif` and `humanego_release/assets/infer_serve_bread.gif`: gif frame k = serve_bread.mp4 frame 4k+1 = fd 3838+4k (k = 0..28). Frames last 0.06/0.07 s, so playback is about 15 fps (the same x8 speed). The two gifs differ only in size (720 vs 520 px).
  * `Synced/gifs/cross_setup.gif` frame k is roughly cross_condition_3 frame 4.27k+2 (136 frames, about 14.3 fps). gif 0-98 = B, 99-105 = C, 106-135 = D.
* **Speed labels are burned in** at the top-left corner ("x8" for A; "x16" for B, C and D). I checked them at full resolution on fd 3840, 3960, 4000, 4255, 4380, 4400, 4415, 4440, 4460, 4505 and 4525. No file has an on-screen timer or clock. The output is a constant 60 fps.
* **Caption:** `full_demo`, `ICRA` and `twitter` carry an orange/white "Zero-Shot Human-Robot Transfer" caption. It fades in during A (about fd 3845) and stays until fd 4536. It covers the bottom ~15% of the frame, which is where the breads and plate sit in the front views. **For B/C/D use `cross_condition_3.mp4`, and for A use `serve_bread.mp4`, because neither has the caption.**
* **Camera:** a static tripod third-person view. Phase correlation on the gantry drifts ≤2 px across A and across B. The view looks at the robot from the front, so **the robot's RIGHT arm appears on the IMAGE LEFT**. No robot/policy-camera views and no keypoint or trajectory overlays exist for the real robot. The only overlay videos are the human Aria HUD clips (see §1.7).
* **Arm/bread assignment:** in the canonical setup (croissant left of the plate in the image, sugar-danish loaf right, light-blue plate), the robot's right arm (image-left) takes the croissant and the left arm (image-right) takes the danish. This holds in A, B5, C and D1-D6. B6 swaps the sides (loaf image-left, croissant image-right, plate raised on a cardboard box), so the right arm takes the danish and the left arm the croissant. B1-B4 are single-arm runs: the arm on the bread's side picks it.
* **No failures, retries, regrasps or drops are visible anywhere.** Every rollout shown ends with the bread(s) on the plate (curated montage).
* **The speed labels disagree with each other** (§4). In real time, lift-to-release takes 6.3-6.9 s in A ("x8") and 2.7-4.3 s in every "x16" rollout. Either A was a ~2x slower run or one label is wrong. Phase *fractions* are safe to use; absolute seconds carry a factor-of-2 uncertainty.

## 1. Per-video tables

### 1.1 `website/static/videos/full_demo.mp4` (== `Synced/video/full_demo.mp4`)
1920x1080, 60 fps (avg_frame_rate 120060000/2001001 ≈ 60.0), 5336 frames, 88.93 s. Hard cuts in the bread section (full-frame MAD>40): 3589, 3835, 3955, 4379, 4412, 4537. D also has internal cuts at 4435, 4455, 4479, 4501 and 4521.

| seg | fd start-end | t (s) | label | view | arms | breads | cuts inside | outcome |
|---|---|---|---|---|---|---|---|---|
| A | 3835-3954 (120 fr) | 63.917-65.900 | **x8** | front 3rd-person, slightly tighter/darker grade than B, static | **both, simultaneous** | R arm croissant, L arm danish | none. The clip starts mid-rollout (both arms already down at the breads, gripping) and ends with both arms home, idle ~0.1 s | success: danish lies on top of croissant on plate |
| B | 3955-4378 (424 fr) | 65.917-72.967 | **x16** | front 3rd-person (same rig/camera position as A, brighter grade), static | 6 rollouts (below) | see below | **hidden jump cuts between rollouts** (resets partly skipped). Confirmed at fd 4284/4285: plate and breads vanish in one frame with no human visible. Each rollout looks continuous | all success |
| C | 4379-4411 (33 fr) | 72.983-73.517 | **x16** | oblique/closer 3rd-person, cluttered table (cups, bottle, tape, charger) | both | R croissant, L danish | none inside. Starts mid-reach, cut before arms return home | success |
| D | 4412-4536 (125 fr) | 73.533-75.600 | **x16** | front view from a lower camera, whiter background, same lab | both | R croissant, L danish | **jump-cut montage of 6 sub-rollouts**: D1 4412-4434, D2 4435-4454, D3 4455-4478, D4 4479-4500, D5 4501-4520, D6 4521-4536. Each starts at the "ready" pose and ends around release, never home. Frozen/duplicated frames at 4466-4468, 4484-4485, 4502-4505 | all appear successful (D6 is truncated during placing) |

Rollouts inside B (fd frames; at x16 each frame = 16/60 = 0.267 s real; events are ±1-2 frames):

| roll | arms | bread → plate | idle before | arm leaves home | "ready" hold (dup frames) | grasp / lift-off | release | back home | after |
|---|---|---|---|---|---|---|---|---|---|
| B1 | R only | croissant (img-left of plate) | 3970-3986 | 3987 | 3988-3991 | ~3997 / 3999-4000 | ~4014-4015 | 4018 | human enters 4019 |
| B2 | L only | croissant (img-right) | ~4032-4049 | ~4050 | 4052-4062 (slow) | ~4067 / 4068-4070 | ~4080-4081 | ~4083 | human 4085 |
| B3 | R only | danish loaf (img-left) | ~4106-4118 | 4119 | 4120-4132 | ~4136 / 4137-4138 | ~4147-4148 | ~4150 | idle to 4157, plate removed 4158+ |
| B4 | R only | pale round bun, **dark-blue plate** (novel object/plate) | ~4176-4185 | 4186 | 4187-4190 | ~4196 / 4197 | ~4206-4207 | ~4208-4209 | human 4211 |
| **B5** | **both** | **canonical**: R croissant, L danish, light-blue plate | **4230-4248** | L 4249, R 4251 | 4252-4256 (dups 4254-4256) | R ~4261 / 4262; L ~4265 / 4266-4267 | R ~4272-4273; L ~4279-4281 | R ~4277-4278; L 4284 (**jump cut 4284/4285**) | table emptied by the cut |
| **B6** | **both** | **swapped**: R danish (img-left), L croissant (img-right); **plate raised on cardboard box** | ~4318-4338 | L 4339, R 4341-4342 | 4344-4347 | R ~4349-4350 / 4351; L ~4355-4356 / 4357 | R ~4362-4363; L ~4368-4370 | both ~4372 | idle 4372-4378, then hard cut to C |

C (fd): R descending at 4379-4384, R grasp ~4385 / lift 4386-4387, L grasp ~4390-4392, R release ~4397, L release ~4409-4411. The cut comes before either arm reaches home.
D (fd, approx.): D1 R lift ~4416, R release ~4423, L release ~4430-4432 · D2 R lift ~4439, L lift ~4443, R release ~4445, L release ~4450-4452 · D3 R lift ~4459-4461, freeze 4466-4468, releases ~4472-4477 · D4 R lift ~4487, releases ~4492-4499 · D5 freeze 4502-4505, R lift ~4510, releases ~4514-4518 · D6 R lift ~4525, L lift ~4529, ends 4536 mid-place.

The rest of full_demo has no robot bread content. Egocentric human-demo visualisations (4-panel Aria HUD, some show a human placing a croissant on a plate) run at 31.43-38.40 s (fd 1886-2303). These are human data, not robot.

### 1.2 `HumanEgo_for_ICRA_compressed.mp4`
1920x1080, 60 fps, 5327 frames, 88.78 s, md5 `ac8743c1…`. Same segments and frame numbers as §1.1 (A 3835-3954, B 3955-4378, C 4379-4411, D 4412-4536). Same x8/x16 labels and caption. It adds no information beyond `full_demo`.

### 1.3 `website/static/videos/serve_bread.mp4`: **cleanest source for A**
1920x1080, 60 fps, 117 frames, 1.95 s, md5 `06bf6a2f…`. k = fd−3837.

| seg | frames | t (s) | label | view | arms | breads | cuts | outcome |
|---|---|---|---|---|---|---|---|---|
| A | 0-116 (fd 3837-3953) | 0-1.933 | **x8** (no caption) | front 3rd-person, static | both | R croissant, L danish | none (2 frames shorter at each end than the full_demo A) | success |

Events in A (serve_bread.mp4 frame k; fd = k+3837; x8 means 1 frame = 8/60 = 0.133 s real):

| event | right arm (img-left, croissant) | left arm (img-right, danish) |
|---|---|---|
| clip start (k=0) | already down at croissant, closing/closed | already down at danish |
| lift-off | k≈19-21 | k≈44-46 |
| over plate | k≈60-64 | k≈90-95 |
| release | k≈66-68 (dup frame at k=68) | k≈96-98 (dup at k=98) |
| back home | k≈78-79 | k≈107-110 |
| idle until | 116 | 116 |

The reach onset and the home→ready motion were cut off by the editor before k=0.

### 1.4 `website/static/videos/cross_condition_3.mp4`: **cleanest source for B, C and D**
1920x1080, 60 fps, 581 frames, 9.683 s. k = fd−3956. B = k 0-422, C = k 423-455, D = k 456-580 (cuts at k 423, 456, 479, 499, 523, 545, 565). x16 label, no caption. Content is identical to §1.1 B/C/D, and the rollout tables apply with k = fd−3956: B1 31-62, B2 94-127, B3 163-194, B4 230-253, **B5 293-328** (cut after 328), **B6 383-416** (idle to 422).

### 1.5 `website/static/videos/HumanEgo_clip_2_twitter.mp4`
1920x1080, 60 fps, 1151 frames, 19.183 s. k = fd−3386. Contents: cross-embodiment cups (k 0-202), interaction-centric sink 4-panel (203-448), **A 449-568 (x8)**, **B 569-992**, **C 993-1025**, **D 1026-1150** (x16). Label and caption are both present. It adds nothing beyond `full_demo`.

### 1.6 GIFs
| file | size | frames | dur | content |
|---|---|---|---|---|
| `Synced/gifs/serve_bread.gif` | 720x405 | 29 | 1.94 s (0.06/0.07 s per frame) | A, every 4th frame of serve_bread.mp4 (fd 3838+4k), x8 label, no caption. 0.533 s real per gif frame |
| `humanego_release/assets/infer_serve_bread.gif` | 520x293 | 29 | 1.94 s | the same frames as above, smaller |
| `Synced/gifs/cross_setup.gif` | 720x405 | 136 | 9.71 s | = cross_condition_3 subsampled about every 4.27 frames (B 0-98, C 99-105, D 106-135) |
| `humanego_release/assets/aria_vis_serve_bread.gif` | 460x307 | 330 | 33 s @10 fps | **human** Aria egocentric demo (no robot) with HUD: "FRAME n / 01002 \| FPS 30 \| PHASE …", hand open/closed state. Real time (every 3rd frame of 30 fps). The right hand closes on the croissant ~frame 320 and opens over the plate ~frame 960-990. Useful only as a human-timing reference |
| other Synced/gifs and assets gifs | | | | no serve_bread robot footage: cross_embodiment = cross_condition_1 (cups, UR10 vs Trossen), cross_environment = cross_condition_2 (sink), teaser/architecture/baselines/generalization/representation/data_efficiency = slides/charts, other task gifs |

### 1.7 Other checked files
* `cross_condition_1.mp4` (3.38 s, 203 fr) shows cup down-stacking, Trossen and UR side by side, x8. **No bread.**
* `cross_condition_2.mp4` (4.07 s, 244 fr) shows a 4-panel sink/strawberry "interaction-centric" clip, x8. **No bread.**
* Elsewhere on disk: `/home/leo/leo/human2sim2robot/inputs/serve_bread_040{,_mirror}/*.mp4` are human input videos for DexAgent, not real-robot rollouts. They were not inventoried.

## 2. Equivalence checks done
* Cut lists of full_demo and ICRA match exactly (20 hard cuts at identical frames). The bread section matches pixel-wise.
* I ran a best-match search with downscaled MAD and skipped the label and caption rows. Short clips match at a constant offset with error 1.0-1.8/255; the runner-up is 3-10. serve_bread.mp4 is offset +3837, cross_condition_3 +3956, twitter +3386, the gifs 4k+1.
* A and B5 are **different** rollouts even though their initial layouts are identical. Final bread poses on the plate differ: in A the danish lies diagonally on the croissant; in B5 the danish lies flat over the croissant.

## 3. Duplicated/frozen frames (codec-level MAD < 0.7 during motion)
* A: fd 3841, 3905, 3916, 3935, 3947. These sit at release moments and right after, so they are plausibly the arm genuinely pausing while the gripper opens. They could also be single-frame holds. Either way they add ≤1 frame of jitter.
* B: the dups cluster at each rollout's "ready" pose (B1 3990-3991, B2 4054/4057/4062, B3 4122-4132, B4 4189-4190, B5 4254-4256, B6 4344/4346). This is consistent with the robot holding the start pose before the first action chunk, so it is real behaviour, not editing.
* D: freezes of 2-4 frames at 4466-4468, 4484-4485 and 4502-4505. These are probably editor holds, which is one more reason not to time D.

## 4. Playback-speed evidence and the x8 vs x16 inconsistency
* The only speed evidence is the burned-in label. No timer is visible and no human-paced reference is reliable at x16.
* Lift-off to release, converted to real seconds with the labelled factor:
  * A (x8): R 47 fr → 6.3 s; L ~52 fr → 6.9 s
  * B1 16 fr → 4.3 s · B2 12 → 3.2 s · B3 11 → 2.9 s · B4 10 → 2.7 s
  * B5: R 10 → 2.7 s; L ~13 → 3.5 s · B6: R 11 → 2.9 s; L ~12 → 3.2 s
  * C: R 10 → 2.7 s · D2: R ~6 → 1.6 s
* Whole rollouts at x16 (leave home → back home) take 23-35 frames, i.e. **6.1-9.3 s real** (B5 9.3 s, B6 8.8 s). A covers only part of its rollout yet already spans 117 frames = 15.6 s real.
* So A is about 2x slower in real time than every x16 clip. The x16 clips agree with each other.
* Sanity check on x16: B5's right arm goes from folded-home to the extended ready pose in ~2 frames. At x16 that is ≈0.5 s real (fast but feasible). At x8 it would be ≈0.27 s, which is implausible. So the x16 label is, if anything, not overstated.
* Conclusion: either A was a slower run (different controller settings or recording day), or A's "x8" label is wrong (A would then be ~x16-x20).
* **Recommendation:** treat absolute real-time durations as uncertain by up to 2x. Compare sim and real with normalised phase fractions, or anchor to the controller's known rate. The deployed Control.yaml runs at 10 Hz with goal_time 1.2 s, and the B rollouts' 6-9 s total is the more mutually consistent anchor.

## 5. Ranked segments for frame-level phase timing (reach / grasp / lift / transport / release / home)

1. **B5 (canonical bimanual): cross_condition_3 k 290-330 (fd 4246-4286), x16.**
   * Continuous from idle and arm departure through both releases to home. A jump cut at fd 4284/4285 lands just as the left arm reaches home, so the left-arm "home" time is bounded at ≤4284.
   * Same layout as the policy's real task. All frames: `real_frames/cross_condition_3/allframes_B5/`.
   * Weakness: 1 frame = 0.267 s real, so phase boundaries are only ±0.3 s.
2. **A (canonical bimanual): serve_bread.mp4 k 0-116 (fd 3837-3953), x8.**
   * Best temporal resolution (1 frame = 0.133 s real), no caption, uncut, ends at home/idle.
   * Weaknesses: it starts mid-grasp, so there is no reach-onset or home→ready phase, and its x8 label is inconsistent with the x16 clips (§4). Use it for grasp→lift→transport→release→home ordering and fractions.
   * All frames: `real_frames/serve_bread/allframes_A/`.
3. **B6 (bimanual, swapped sides, raised plate): cc3 k 379-422 (fd 4335-4378), x16.**
   * Fully continuous idle → home → idle with no cut inside. It is the only bimanual clip with both endpoints visible.
   * The condition is non-canonical (danish via the right arm, croissant via the left, plate elevated on a box). `allframes_B6/`.
4. **B1-B4 (single-arm): cc3 k 28-66 / 90-130 / 160-198 / 226-257, x16.**
   * Each is continuous from idle to home. Good for single-arm phase durations: B1 croissant/right, B2 croissant/left, B3 danish/right; B4 is a novel bun and dark plate.
   * `allframes_B1..B4/`.
5. **C: cc3 k 423-455 (fd 4379-4411), x16.** Uncut but partial (starts mid-reach, ends before home) and in a cluttered, different view. Lift→release only.
6. **Not usable for timing:**
   * D: a jump-cut montage with freezes; each sub-clip is ~20 frames and starts and ends mid-rollout.
   * The gifs: 4x temporally subsampled copies of A/B/C/D.
   * The full_demo, ICRA and twitter copies of A-D: identical content, but the caption occludes the breads and plate. Use the clean files instead.

## 6. Saved frames
`/tmp/claude-1000/-home-leo-leo/07eae8a5-6f4d-4727-9051-cae6319cdd15/scratchpad/real_frames/`
* `<stem>/<SEG>_t<sec>s_f<frame>[_fd<fd>].jpg`: one 960-px keyframe every 0.25 s for each segment. `<SEG>_START|END_fullres_...png` is the full-resolution first and last frame.
  * stems: `full_demo`, `HumanEgo_for_ICRA_compressed`, `HumanEgo_clip_2_twitter`, `cross_condition_3`, `serve_bread`, `serve_bread_gif`, `infer_serve_bread_gif`, `cross_setup_gif`
* `serve_bread/allframes_A/`: all 117 frames of A, 960 px.
* `cross_condition_3/allframes_{B1..B6,C}/`: every frame of each rollout, 960 px, file names carry the fd index.
* `serve_bread_gif/allframes_A`, `infer_serve_bread_gif/allframes_A`: all 29 gif frames.
