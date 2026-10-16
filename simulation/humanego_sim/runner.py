# -*- coding: utf-8 -*-
"""
EpisodeRunner — HumanEgo's real bimanual inference loop, closed in simulation.

This is `AriaMimic inference/Inference.py: run_bimanual` (independent bimanual:
one single-hand model drives both arms; the left arm sees a horizontally
flipped world) with the hardware replaced by `ServeBreadSim`, and wall-clock
threads replaced by a deterministic sim-time schedule:

    every policy_period  : observe -> clean image + ICT -> released ICTPolicy
                           -> decode -> SimController.update_target
    every 1/control_freq : SimController.tick()   (production async controller)
    every physics step   : SimArm.update()        (driver interpolation + IK)

The policy is the *released* `inference/policy.py: ICTPolicy` loading the
released checkpoint — the simulator never touches the network.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .controller import SimController
from .driver import T_to_p
from .env import REPO_ROOT, ServeBreadSim
from .perception import SideObjects, SimPerception, mirror_pose

for _p in (os.path.join(REPO_ROOT, "inference"), REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from interfaces import ObjectState  # noqa: E402  (released inference template)
from policy import ICTPolicy  # noqa: E402

T_ALIGN_ARIA = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def _fresh_hand_state():
    return {"is_grasping": False, "latched_obj": None, "T_lock_h2obj": None,
            "has_released": False, "has_ever_grasped": False}


@dataclass
class SideCtx:
    side: str
    is_flip: bool
    objs: SideObjects
    hand: dict = field(default_factory=_fresh_hand_state)
    is_active: bool = False
    is_done: bool = False
    wrap_phase: Optional[str] = None      # None | "open" | "home" | "finished"
    wrap_t0: float = 0.0
    grasp_start_time: Optional[float] = None
    forced_lift_counter: int = 0
    forced_lift_anchor_p_base: Optional[np.ndarray] = None
    pred_done_prob: float = 0.0
    done_reason: str = ""


@dataclass
class EpisodeResult:
    success: bool
    per_side: Dict[str, dict]
    duration: float
    n_policy_steps: int
    wall_time: float
    log_dir: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


def load_policy(cfg: dict, device: str = "cuda") -> ICTPolicy:
    pc = cfg["policy"]
    ckpt = pc["ckpt"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.normpath(os.path.join(REPO_ROOT, ckpt))
    torch.manual_seed(int(pc.get("seed", 0)))
    return ICTPolicy({"ckpt": ckpt, "num_inference_steps": int(pc.get("num_inference_steps", 20))}, device=device)


class EpisodeRunner:
    def __init__(self, cfg: dict, policy: Optional[ICTPolicy] = None, device: str = "cuda",
                 env: Optional[ServeBreadSim] = None):
        self.cfg = cfg
        self.device = device
        self.env = env or ServeBreadSim(cfg)
        self.policy = policy or load_policy(cfg, device)
        if self.policy.frame_mode != "camera_frame" or self.policy.action_mode != "absolute":
            raise NotImplementedError("sim runner currently supports camera_frame + absolute checkpoints")
        cc = cfg["control"]
        self.cc = cc
        self.T_align = T_ALIGN_ARIA
        self.anchor_key = "obj1"
        self.forced_lift_height = float(cc.get("forced_lift_height", 0.05))
        self.forced_lift_frames = int(cc.get("forced_lift_frames", 15))
        self.forced_lift_ramp = max(1, int(cc.get("forced_lift_ramp_frames", 5)))
        self.latch_dist = float(cfg.get("perception", {}).get("latch_dist_thresh", 0.20))
        # How the gripper reading becomes the ICT grasp flag. The real loop uses
        # measured gq < 0.5; soft bread squashes so that also fires on a held bread.
        # Rigid sim bread cannot squash, so "measured_or_stalled" also counts a gripper
        # that finished closing and stalled on an object (see README, "Grasp state").
        self.grasp_source = cc.get("grasp_state_source", "measured_or_stalled")

    # ------------------------------------------------------------------
    def _seed_policy_noise(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.policy.noise = torch.randn(self.policy.noise.shape, generator=g).to(self.policy.noise.device)

    def _grasp_val(self, arm) -> float:
        if self.grasp_source == "commanded":
            return 1.0 if arm.gripper_cmd_closed() else 0.0
        closed = arm.get_gripper_q() < 0.5
        if self.grasp_source == "measured_or_stalled":
            closed = closed or arm.gripper_settled_closed()
        return 1.0 if closed else 0.0

    def _apply_forced_lift(self, pt_cam, arm, counter, anchor_p_base):
        if counter <= 0:
            return pt_cam
        elapsed = self.forced_lift_frames - counter + 1
        bias = self.forced_lift_height * min(1.0, elapsed / float(self.forced_lift_ramp))
        pb = arm.p_in_cam_to_p_in_base(pt_cam)
        if self.cc.get("forced_lift_lock_xy", False) and anchor_p_base is not None:
            pb[0], pb[1], pb[2] = anchor_p_base[0], anchor_p_base[1], anchor_p_base[2] + bias
        else:
            pb[2] += bias
        return arm.p_in_base_to_p_in_cam(pb)

    # ------------------------------------------------------------------
    def _run_side(self, ctx: SideCtx, clean_bg: np.ndarray, t: float):
        """Port of Inference._run_side for camera_frame / absolute checkpoints."""
        env, pol, arm = self.env, self.policy, self.env.arms[ctx.side]
        so, st = ctx.objs, ctx.hand
        Tr = arm.get_T_ee_in_cam() @ self.T_align
        Tv = mirror_pose(Tr) if ctx.is_flip else Tr
        gv = self._grasp_val(arm)
        is_grasp = gv > 0.5

        # ---- kinematic latching state machine ----
        if not is_grasp:
            st["has_released"] = True
        if is_grasp and not st["is_grasping"] and st["has_released"]:
            st["is_grasping"] = True
            st["has_ever_grasped"] = True
            ctx.forced_lift_counter = self.forced_lift_frames
            ctx.forced_lift_anchor_p_base = arm.get_p()[:3].copy()
            best, bd = None, float("inf")
            for ok, To in so.dynamic_obj_poses.items():
                dd = float(np.linalg.norm(Tv[:3, 3] - To[:3, 3]))
                if dd < bd:
                    best, bd = ok, dd
            if best and bd < self.latch_dist:
                st["latched_obj"] = best
                st["T_lock_h2obj"] = np.linalg.inv(Tv) @ so.dynamic_obj_poses[best]
            else:
                st["latched_obj"] = None
        elif not is_grasp and st["is_grasping"]:
            st["is_grasping"] = False
            st["latched_obj"] = None
            st["T_lock_h2obj"] = None
        if st["is_grasping"] and st["latched_obj"]:
            so.dynamic_obj_poses[st["latched_obj"]] = Tv @ st["T_lock_h2obj"]

        # ---- idle-done (release over the plate) ----
        if self.cc.get("use_idle_done", False) and st["is_grasping"]:
            if ctx.grasp_start_time is None:
                ctx.grasp_start_time = t
            if t - ctx.grasp_start_time >= float(self.cc.get("idle_time_limit", 3.0)):
                for ok, To in so.dynamic_obj_poses.items():
                    if ok == self.anchor_key:
                        continue
                    if float(np.linalg.norm(Tv[:3, 3] - To[:3, 3])) < float(self.cc.get("idle_dist_threshold", 0.05)):
                        ctx.is_done = True
                        ctx.done_reason = "idle_over_plate"
                        break
        elif not st["is_grasping"]:
            ctx.grasp_start_time = None

        # ---- policy input image ----
        vis = self.perception.make_side_image(ctx.side, so, clean_bg, arm, is_grasp)

        # ---- ICT + inference (released ICTPolicy) ----
        hs = pol.single_hand_side
        objs = {k: ObjectState(v.copy()) for k, v in so.dynamic_obj_poses.items()}
        x_ict, mask = pol.build_ict({hs: Tv}, {hs: gv}, objs, self.anchor_key)
        x_rgb = pol.prepare_image(vis)
        active_K = self.perception.K_flip if ctx.is_flip else self.perception.K
        a_uv = pol.compute_anchor_uv(ObjectState(so.dynamic_obj_poses.get(self.anchor_key, so.T_anchor_in_cam_static)),
                                     active_K, env.W, env.H)
        traj, done_prob = pol.infer(x_rgb, x_ict, mask, a_uv)
        ctx.pred_done_prob = done_prob
        if done_prob > float(self.cc.get("done_threshold", 0.8)):
            if not self.cc.get("done_requires_grasp", False) or st["has_ever_grasped"]:
                ctx.is_done = True
                ctx.done_reason = "policy_done"

        # ---- decode (camera_frame, absolute) ----
        he = min(int(self.cc.get("exec_horizon", pol.pred_horizon)), pol.pred_horizon)
        pos, o6d, grasp = traj[hs]
        raw, pT_world = [], []
        for k in range(he):
            Tv_k = pol.decode_ee_in_cam(pos[k], o6d[k], None, self.T_align)   # hand(ref=cam) @ inv(T_align)
            Treal = mirror_pose(Tv_k) if ctx.is_flip else Tv_k
            pt = T_to_p(Treal)
            pt = self._apply_forced_lift(pt, arm, ctx.forced_lift_counter, ctx.forced_lift_anchor_p_base)
            if not np.any(np.isnan(pt)):
                raw.append(pt)
                pT_world.append((env.T_cam_in_world @ np.append(pt[:3], 1.0))[:3])
        if ctx.forced_lift_counter > 0:
            ctx.forced_lift_counter -= 1
        return raw, grasp, vis, np.array(pT_world), float(done_prob)

    # ------------------------------------------------------------------
    def run(self, scene: Optional[dict] = None, seed: int = 0, log_dir: Optional[str] = None,
            record_hz: float = 30.0, save_policy_images: bool = True, verbose: bool = True) -> EpisodeResult:
        wall0 = time.time()
        env, cc = self.env, self.cc
        env.rng = np.random.default_rng(seed)
        self._seed_policy_noise(int(self.cfg["policy"].get("seed", 0)) + seed)
        env.reset(scene)
        sides = env.sides
        self.perception = SimPerception(env, split_x_ratio=float(cc.get("split_x_ratio", 0.5)),
                                        n_init_frames=int(self.cfg.get("perception", {}).get("n_init_frames", 10)),
                                        depth_noise=bool(self.cfg.get("perception", {}).get("depth_noise", True)))
        side_objs = self.perception.estimate_objects(sides, shared_context=True)
        ctxs = {s: SideCtx(side=s, is_flip=(s == "left"), objs=side_objs[s]) for s in sides}
        ctxs["right"].is_active = True
        controller = SimController(env.arms, cc)
        start_delay = float(cc.get("bimanual_start_delay", 6.0))

        P = float(cc.get("policy_period", 0.3))
        Tc = 1.0 / float(cc.get("control_freq", 10.0))
        T_max = float(cc.get("max_episode_time", 40.0))
        rec_dt = 1.0 / record_hz
        next_pol, next_ctl, next_rec = 0.0, 0.0, 0.0
        pending = None          # (deliver_time, payload) — inference latency
        frames, policy_log = [], []
        img_dir = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if save_policy_images:
                img_dir = os.path.join(log_dir, "policy_view")
                os.makedirs(img_dir, exist_ok=True)
        n_pol = 0

        while env.t < T_max:
            t = env.t
            # ---- deliver the previous inference, then observe again ----
            if t >= next_pol - 1e-9:
                if pending is not None:
                    controller.update_target(*pending)
                    pending = None
                if not ctxs["right"].is_active:
                    ctxs["right"].is_active = True
                if len(sides) > 1 and not ctxs["left"].is_active and t >= start_delay:
                    ctxs["left"].is_active = True
                clean = self.perception.clean_background()
                all_rt, all_g = {}, {}
                step_log = {"t": t, "sides": {}}
                for s in ("right", "left"):
                    if s not in ctxs:
                        continue
                    ctx = ctxs[s]
                    if not ctx.is_active or ctx.wrap_phase is not None:
                        continue
                    rt, g, vis, pW, dp = self._run_side(ctx, clean, t)
                    all_rt[s], all_g[s] = rt, g
                    entry = {"traj_world": np.round(pW, 4).tolist(), "grasp": np.round(np.asarray(g).ravel(), 3).tolist(),
                             "done": round(dp, 4), "grasping": ctx.hand["is_grasping"],
                             "latched": ctx.hand["latched_obj"]}
                    if img_dir is not None:
                        fn = f"{n_pol:04d}_{s}.jpg"
                        cv2.imwrite(os.path.join(img_dir, fn), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        entry["image"] = fn
                    step_log["sides"][s] = entry
                pending = (all_rt, all_g)
                policy_log.append(step_log)
                n_pol += 1
                next_pol += P
                # ---- per-side wrap-up trigger ----
                for s, ctx in ctxs.items():
                    if ctx.is_active and ctx.is_done and ctx.wrap_phase is None:
                        ctx.wrap_phase = "open"
                        ctx.wrap_t0 = t
                        controller.muted[s] = True
                        controller.clear(s)
                        arm = env.arms[s]
                        arm.move_p(arm.get_p(), duration=0.05)      # halt
                        arm.open_gripper(goal_time=0.5)
                        if verbose:
                            print(f"[sim] {s} done ({ctx.done_reason}) at t={t:.1f}s -> open + home")
            # ---- wrap-up sequencing (open -> home) ----
            for s, ctx in ctxs.items():
                if ctx.wrap_phase == "open" and t - ctx.wrap_t0 >= 0.8:
                    ctx.wrap_phase = "home"
                    ctx.wrap_t0 = t
                    env.arms[s].go_home(duration=3.0)
                elif ctx.wrap_phase == "home" and t - ctx.wrap_t0 >= 3.1:
                    ctx.wrap_phase = "finished"
            # ---- controller ----
            if t >= next_ctl - 1e-9:
                controller.tick()
                next_ctl += Tc
            # ---- record ----
            if t >= next_rec - 1e-9:
                snap = env.snapshot()
                snap["grip_cmd"] = {s: float(env.arms[s].g_cmd) for s in sides}
                frames.append(snap)
                next_rec += rec_dt
            env.step()
            if all(c.wrap_phase == "finished" for c in ctxs.values()):
                # keep recording a short tail so the viewer can ease out
                tail_end = env.t + 0.6
                while env.t < tail_end:
                    if env.t >= next_rec - 1e-9:
                        snap = env.snapshot()
                        snap["grip_cmd"] = {s: float(env.arms[s].g_cmd) for s in sides}
                        frames.append(snap)
                        next_rec += rec_dt
                    env.step()
                break

        per_side = self.evaluate(ctxs)
        success = all(v["success"] for v in per_side.values())
        res = EpisodeResult(success=success, per_side=per_side, duration=env.t, n_policy_steps=n_pol,
                            wall_time=time.time() - wall0, log_dir=log_dir,
                            metrics={"ik_fail": float(sum(controller.n_ik_fail.values()))})
        if log_dir:
            self._save(log_dir, frames, policy_log, res, scene, side_objs)
        if verbose:
            print(f"[sim] episode: success={success} {json.dumps(per_side)} t={env.t:.1f}s "
                  f"policy_steps={n_pol} wall={res.wall_time:.1f}s")
        return res

    # ------------------------------------------------------------------
    def evaluate(self, ctxs: Dict[str, SideCtx]) -> Dict[str, dict]:
        """Success = the bread rests inside the plate rim, released, and the arm finished."""
        env = self.env
        Tp = env.body_T_world("plate")
        R_plate = env.meta["objects"]["plate"]["R"]
        out = {}
        for s, ctx in ctxs.items():
            Tb = env.body_T_world(f"bread_{s}")
            dxy = float(np.linalg.norm(Tb[:2, 3] - Tp[:2, 3]))
            on_plate = dxy < R_plate - 0.02 and Tb[2, 3] > 0.004 and Tb[2, 3] < 0.12
            released = not env.arms[s].gripper_cmd_closed()
            out[s] = {"success": bool(on_plate and released), "dist_to_plate_center": round(dxy, 4),
                      "bread_z": round(float(Tb[2, 3]), 4), "grasped": bool(ctx.hand["has_ever_grasped"]),
                      "done_reason": ctx.done_reason or "timeout"}
        return out

    def _save(self, log_dir, frames, policy_log, res, scene, side_objs):
        env = self.env
        meta = dict(env.meta)
        meta["scene_init"] = scene or env.default_scene()
        meta["result"] = {"success": res.success, "per_side": res.per_side, "duration": res.duration}
        meta["perception"] = {s: {k: {"T_obj_in_cam_static": np.round(v["T_obj_in_cam_static"], 5).tolist(),
                                      "kpts_local": np.round(v["kpts_local"], 5).tolist()}
                                  for k, v in so.objs_meta.items()} for s, so in side_objs.items()}
        with open(os.path.join(log_dir, "episode.json"), "w") as f:
            json.dump({"meta": meta, "frames": frames, "policy": policy_log}, f)
