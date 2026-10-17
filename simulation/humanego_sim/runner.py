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

    def _perceived_geometry(self, side: str, so) -> dict:
        """What the robot knows about the scene from its own perception at t = 0 (world frame):
        bread centre / long axis / extent from its keypoints, and a place height over the plate."""
        env, out = self.env, {}
        Tcw = env.T_cam_in_world
        pts = {}
        for key, m in so.objs_meta.items():
            T, kl = np.asarray(m["T_obj_in_cam_static"]), np.asarray(m["kpts_local"]).reshape(-1, 3)
            if side == "left":                       # perception of the left arm lives in the mirrored image
                T, kl = mirror_pose(T), kl * np.array([-1.0, 1.0, 1.0])
            pc = (T[:3, :3] @ kl.T).T + T[:3, 3]
            pts[key] = (Tcw[:3, :3] @ pc.T).T + Tcw[:3, 3]
        b = pts.get(self.anchor_key)
        if b is not None and len(b) >= 3:
            c = b.mean(0)
            xy = b[:, :2] - c[:2]
            ev, evec = np.linalg.eigh(xy.T @ xy)
            a2 = evec[:, np.argmax(ev)]
            a = np.array([a2[0], a2[1], 0.0])
            w = np.array([-a[1], a[0], 0.0])
            out["bread"] = {"c": c.tolist(), "a": a.tolist(), "half_len": float(np.abs(xy @ a2).max()),
                            "half_wid": float(np.abs(xy @ w[:2]).max())}
            plate = pts.get("obj2")
            # place height: bread centre height on the table + the plate's floor (rim keypoints - 1.35 cm)
            floor = float(plate[:, 2].mean()) - 0.0135 if plate is not None else 0.0085
            out["place_z"] = float(c[2]) + max(floor, 0.0)
        return out

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
    def _intro_drop(self, rng, record_hz: float, drop_h=(0.05, 0.09), t_total: float = 1.6) -> list:
        """Objects appear in mid-air one after another and fall onto the table under physics.
        Recorded with negative time stamps, before perception runs at t = 0."""
        env = self.env
        names = ["plate"] + sorted(n for n in env.obj_names if n != "plate")
        rest = {n: np.asarray(env.object_pose7(n)) for n in names}
        release = {n: 0.15 + 0.22 * i for i, n in enumerate(names)}
        air = {}
        for n in names:
            q = rest[n].copy()
            q[2] += rng.uniform(*drop_h) if n != "plate" else 0.035
            if n != "plate":   # a little tilt so the landing looks natural
                from scipy.spatial.transform import Rotation as R
                r0 = R.from_quat([q[4], q[5], q[6], q[3]])
                r1 = R.from_euler("xy", rng.uniform(-12, 12, 2), degrees=True) * r0
                x, y, z, w = r1.as_quat()
                q[3:7] = [w, x, y, z]
            air[n] = q
        frames, t, nxt = [], 0.0, 0.0
        dt = env.dt
        while t < t_total:
            for n in names:
                if t < release[n]:
                    env.place_object(n, air[n])
            if t >= nxt - 1e-9:
                snap = env.snapshot()
                snap["t"] = round(t - t_total, 4)
                snap["grip_cmd"] = {s: float(env.arms[s].g_cmd) for s in env.sides}
                snap["hidden"] = [n for n in names if t < release[n] - 0.12]
                frames.append(snap)
                nxt += 1.0 / record_hz
            env.step()
            t += dt
        env.t = 0.0
        for arm in env.arms.values():
            arm.t = 0.0
            arm._g_t0 = 0.0
        return frames

    def run(self, scene: Optional[dict] = None, seed: int = 0, log_dir: Optional[str] = None,
            record_hz: float = 30.0, save_policy_images: bool = True, verbose: bool = True,
            intro: bool = True) -> EpisodeResult:
        wall0 = time.time()
        env, cc = self.env, self.cc
        env.rng = np.random.default_rng(seed)
        self._seed_policy_noise(int(self.cfg["policy"].get("seed", 0)) + seed)
        env.reset(scene, settle=0.0 if intro else 0.4)
        intro_frames = self._intro_drop(np.random.default_rng(seed + 99), record_hz) if intro else []
        sides = env.sides
        self.perception = SimPerception(env, split_x_ratio=float(cc.get("split_x_ratio", 0.5)),
                                        n_init_frames=int(self.cfg.get("perception", {}).get("n_init_frames", 10)),
                                        depth_noise=bool(self.cfg.get("perception", {}).get("depth_noise", True)))
        side_objs = self.perception.estimate_objects(sides, shared_context=True)
        ctxs = {s: SideCtx(side=s, is_flip=(s == "left"), objs=side_objs[s]) for s in sides}
        ctxs["right"].is_active = True
        controller = SimController(env.arms, cc)
        geoms = {s: self._perceived_geometry(s, side_objs[s]) for s in sides}
        for s in sides:
            if geoms[s].get("bread"):
                controller.set_geometry(s, geoms[s]["bread"], env.T_cam_in_world)
        self._controller = controller
        start_delay = float(cc.get("bimanual_start_delay", 6.0))

        P = float(cc.get("policy_period", 0.3))
        Tc = 1.0 / float(cc.get("control_freq", 10.0))
        T_max = float(cc.get("max_episode_time", 40.0))
        rec_dt = 1.0 / record_hz
        next_pol, next_ctl, next_rec = 0.0, 0.0, 0.0
        pending = None          # (deliver_time, payload) — inference latency
        frames, policy_log = list(intro_frames), []
        self._bread_max_z = {}
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
                        ctx.wrap_t0 = t
                        controller.muted[s] = True
                        controller.clear(s)
                        arm = env.arms[s]
                        arm.move_p(arm.get_p(), duration=0.05)      # halt
                        rel = cc.get("release", {}) or {}
                        if rel.get("place_down") and arm.gripper_cmd_closed() and geoms[s].get("place_z") is not None:
                            # lower the bread onto the plate before opening (instead of dropping it)
                            T = arm.T_base_in_world @ arm.get_T_ee_in_base()
                            z_goal = geoms[s]["place_z"] + float(rel.get("margin", 0.012))
                            if T[2, 3] > z_goal + 0.005:
                                T[2, 3] = z_goal
                                Tb = np.linalg.inv(arm.T_base_in_world) @ T
                                arm.move_p(T_to_p(Tb), duration=float(rel.get("place_time", 0.9)),
                                           safe_z_min=controller.safe_z_min)
                                ctx.wrap_phase = "place"
                        if ctx.wrap_phase is None:
                            ctx.wrap_phase = "open"
                            arm.open_gripper(goal_time=0.5)
                        if verbose:
                            print(f"[sim] {s} done ({ctx.done_reason}) at t={t:.1f}s -> open + home")
            # ---- wrap-up sequencing (open -> home) ----
            for s, ctx in ctxs.items():
                rel = cc.get("release", {}) or {}
                if ctx.wrap_phase == "place" and t - ctx.wrap_t0 >= float(rel.get("place_time", 0.9)) + 0.15:
                    ctx.wrap_phase = "open"
                    ctx.wrap_t0 = t
                    env.arms[s].open_gripper(goal_time=0.5)
                elif ctx.wrap_phase == "open" and t - ctx.wrap_t0 >= 0.8:
                    ctx.wrap_t0 = t
                    d_back = float(rel.get("retreat", 0.0))
                    d_up = float(rel.get("lift", 0.0))
                    if d_up > 0 and d_back <= 0:
                        # lift the open hand straight up first: a bread pinned under a finger is
                        # freed instead of being dragged along by the move home
                        arm = env.arms[s]
                        T = arm.T_base_in_world @ arm.get_T_ee_in_base()
                        T[2, 3] += d_up
                        Tb = np.linalg.inv(arm.T_base_in_world) @ T
                        arm.move_p(T_to_p(Tb), duration=0.8, safe_z_min=controller.safe_z_min)
                        ctx.wrap_phase = "retreat"
                    elif d_back > 0:
                        # back the open fingers out along the approach before going home, so a bread
                        # resting on the plate is not dragged along
                        arm = env.arms[s]
                        T = arm.T_base_in_world @ arm.get_T_ee_in_base()
                        T[:3, 3] -= d_back * T[:3, 2]
                        T[2, 3] += float(rel.get("retreat_up", 0.02))
                        Tb = np.linalg.inv(arm.T_base_in_world) @ T
                        arm.move_p(T_to_p(Tb), duration=0.8, safe_z_min=controller.safe_z_min)
                        ctx.wrap_phase = "retreat"
                    else:
                        ctx.wrap_phase = "home"
                        env.arms[s].go_home(duration=3.0)
                elif ctx.wrap_phase == "retreat" and t - ctx.wrap_t0 >= 0.9:
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
            for s in sides:
                bz = env.object_pos(f"bread_{s}")[2]
                if bz > self._bread_max_z.get(s, 0.0):
                    self._bread_max_z[s] = float(bz)
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
        for s_ in per_side:
            if controller.events.get(s_):
                per_side[s_]["events"] = controller.events[s_][:24]
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
        """Success = the bread was released and rests on the plate: its centre is over the plate and
        no part of it touches the table (the plate's floor sits 8.5 mm above the table top)."""
        env = self.env
        Tp = env.body_T_world("plate")
        R_plate = env.meta["objects"]["plate"]["R"]
        out = {}
        for s, ctx in ctxs.items():
            Tb = env.body_T_world(f"bread_{s}")
            dxy = float(np.linalg.norm(Tb[:2, 3] - Tp[:2, 3]))
            low = env.object_lowest_z(f"bread_{s}")
            on_plate = dxy < R_plate and low > 0.003 and Tb[2, 3] < 0.12
            released = not env.arms[s].gripper_cmd_closed()
            out[s] = {"success": bool(on_plate and released), "dist_to_plate_center": round(dxy, 4),
                      "lowest_z": round(low, 4),
                      "success_strict": bool(dxy < R_plate - 0.02 and 0.004 < Tb[2, 3] < 0.12 and released),
                      "bread_z": round(float(Tb[2, 3]), 4), "grasped": bool(ctx.hand["has_ever_grasped"]),
                      "bread_max_z": round(getattr(self, "_bread_max_z", {}).get(s, 0.0), 4),
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
