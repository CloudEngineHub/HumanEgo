# -*- coding: utf-8 -*-
"""
SimController — the production `AsyncRobotController` (AriaMimic
inference/InferenceController.py) on a simulation clock.

The real controller runs in its own thread at `control_freq`; here `tick()` is
called by the runner every 1/control_freq seconds of *sim* time, so the same
behaviour is reproduced deterministically:

  update_target():  should-grasp vote over the chunk, temporal ensembling of
                    the last `max_history_len` chunks (exp(-0.25*offset) weights,
                    Markley quaternion averaging), optional grasp latch.
  tick():           pick the target at `look_ahead_offset` (advancing by
                    `step_stride` each tick), safety cage around the *measured*
                    pose, EMA + Slerp against the previous command, then a
                    non-blocking `move_p_in_cam(duration = dt * duration_smooth_param)`
                    and the gripper open/close command.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class SimController:
    def __init__(self, arms: Dict[str, object], cfg: dict):
        self.arms = arms
        c = cfg
        self.control_freq = float(c.get("control_freq", 10.0))
        self.max_history_len = int(c.get("max_history_len", 1))
        self.step_stride = int(c.get("step_stride", 1))
        self.look_ahead_offset = int(c.get("look_ahead_offset", 30))
        self.alpha_pos = float(c.get("alpha", 0.8))
        self.alpha_rot = float(c.get("alpha_rot", self.alpha_pos))
        self.duration_smooth_param = float(c.get("duration_smooth_param", 1.0))
        self.grasp_threshold = float(c.get("grasp_threshold", 0.5))
        self.should_grasp_method = c.get("should_grasp_method", "first")
        self.grasp_k = {"first_k": int(c.get("grasp_k_first", 1)), "last_k": int(c.get("grasp_k_last", 1)),
                        "sum_k": int(c.get("grasp_k_sum", 3)), "any_k": int(c.get("grasp_k_any", 50))}
        self.use_safety_cage = bool(c.get("use_safety_cage", True))
        self.max_pos_step = float(c.get("max_pos_step", 0.018))
        self.max_rot_step = float(c.get("max_rot_step", 0.10))
        self.safe_z_min = float(c.get("safe_z_min", -0.20))
        self.close_goal_time = float(c.get("close_gripper_goal_time", 1.0))
        self.open_goal_time = float(c.get("open_gripper_goal_time", 1.0))
        self.grasp_latch = bool(c.get("grasp_latch", False))
        self.fix_rotation = bool(c.get("fix_rotation", False))

        sides = list(arms)
        self.shared_traj = {s: [] for s in sides}
        self.shared_grip = {s: 0.0 for s in sides}
        self.history = {s: [] for s in sides}
        self.active_traj = {s: [] for s in sides}
        self.traj_idx = {s: self.look_ahead_offset for s in sides}
        self.last_cmd = {s: None for s in sides}
        self.muted = {s: False for s in sides}
        self._latch_ever_open = {s: False for s in sides}
        self._latch_locked = {s: False for s in sides}
        self.fixed_rot = {}
        self.n_ik_fail = {s: 0 for s in sides}

    # ------------------------------------------------------------------
    @staticmethod
    def _avg_quat(quats, weights):
        Q = np.zeros((4, 4))
        ws = float(np.sum(weights))
        for q, w in zip(quats, weights):
            q = np.asarray(q)
            Q += (w / ws) * np.outer(q, q)
        evals, evecs = np.linalg.eigh(Q)
        return evecs[:, np.argmax(evals)]

    def _should_grasp(self, g):
        g = np.asarray(g).flatten()
        th = self.grasp_threshold
        m = self.should_grasp_method
        if m == "any":
            return bool(np.any(g > th))
        if m == "any_first_k":     # grasp intent anywhere in the next k predicted steps (k = horizon == "any")
            return bool(np.any(g[: max(1, self.grasp_k["any_k"])] > th))
        if m == "average":
            return bool(np.mean(g) > th)
        if m == "last":
            return bool(g[-1] > th)
        if m == "first_k":
            return bool(np.mean(g[: min(self.grasp_k["first_k"], len(g))]) > th)
        if m == "last_k":
            return bool(np.mean(g[-min(self.grasp_k["last_k"], len(g)):]) > th)
        if m == "sum_k":
            return bool(np.sum(g > th) >= self.grasp_k["sum_k"])
        return bool(g[0] > th)

    def update_target(self, traj_dict: Dict[str, List[np.ndarray]], grasp_dict: Dict[str, np.ndarray]):
        for side, chunk in traj_dict.items():
            if side not in self.arms or len(chunk) == 0:
                continue
            should_grasp = self._should_grasp(grasp_dict.get(side, np.zeros(len(chunk))))
            hb = self.history[side]
            hb.append(chunk)
            if len(hb) > self.max_history_len:
                hb.pop(0)
            ens = []
            for k in range(len(chunk)):
                sp, quats, ws = np.zeros(3), [], []
                for hi, ht in enumerate(hb):
                    off = len(hb) - 1 - hi
                    kk = k + off
                    if kk < len(ht):
                        w = np.exp(-0.25 * off)
                        pt = ht[kk]
                        sp += pt[:3] * w
                        quats.append(R.from_rotvec(pt[3:]).as_quat())
                        ws.append(w)
                if not ws:
                    break
                q = self._avg_quat(quats, ws)
                ens.append(np.concatenate([sp / np.sum(ws), R.from_quat(q).as_rotvec()]))
            if self.grasp_latch:
                if not should_grasp:
                    self._latch_ever_open[side] = True
                elif self._latch_ever_open[side]:
                    self._latch_locked[side] = True
                if self._latch_locked[side]:
                    should_grasp = True
            if not self.muted[side]:
                self.shared_traj[side] = ens
                self.shared_grip[side] = 1.0 if should_grasp else 0.0

    def clear(self, side: str):
        self.shared_traj[side] = []
        self.history[side] = []
        self.active_traj[side] = []
        self.last_cmd[side] = None

    # ------------------------------------------------------------------
    def tick(self):
        """One control step (call every 1/control_freq s of sim time)."""
        dt = 1.0 / self.control_freq
        for side, arm in self.arms.items():
            if len(self.shared_traj[side]) > 0:
                self.active_traj[side] = self.shared_traj[side]
                self.traj_idx[side] = min(self.look_ahead_offset, len(self.active_traj[side]) - 1)
                self.shared_traj[side] = []
            if self.muted[side]:
                self.active_traj[side] = []
                self.last_cmd[side] = None
                continue
            traj = self.active_traj[side]
            idx = self.traj_idx[side]
            if not (len(traj) > 0 and idx < len(traj)):
                continue
            target = np.array(traj[idx], dtype=np.float64)
            actual = arm.get_p_in_cam()
            if self.use_safety_cage:
                dp = target[:3] - actual[:3]
                dist = np.linalg.norm(dp)
                if dist > self.max_pos_step:
                    target[:3] = actual[:3] + dp / dist * self.max_pos_step
                ra, rt = R.from_rotvec(actual[3:]), R.from_rotvec(target[3:])
                ang = (rt * ra.inv()).magnitude()
                if ang > self.max_rot_step:
                    sl = Slerp([0, 1], R.from_quat([ra.as_quat(), rt.as_quat()]))
                    target[3:] = sl(self.max_rot_step / ang).as_rotvec()
            last = self.last_cmd[side]
            if last is None:
                cmd = target.copy()
            else:
                pos = self.alpha_pos * last[:3] + (1 - self.alpha_pos) * target[:3]
                sl = Slerp([0, 1], R.from_quat([R.from_rotvec(last[3:]).as_quat(),
                                                R.from_rotvec(target[3:]).as_quat()]))
                cmd = np.concatenate([pos, sl(1.0 - self.alpha_rot).as_rotvec()])
            if self.fix_rotation:
                if side not in self.fixed_rot:
                    self.fixed_rot[side] = actual[3:].copy()
                cmd[3:] = self.fixed_rot[side]
            self.last_cmd[side] = cmd.copy()
            ok = arm.move_p_in_cam(cmd, duration=dt * self.duration_smooth_param, blocking=False,
                                   safe_z_min=self.safe_z_min)
            if not ok:
                self.n_ik_fail[side] += 1
                self.shared_traj[side] = []
                self.history[side] = []
                self.last_cmd[side] = actual.copy()
            if self.shared_grip[side] > self.grasp_threshold:
                arm.close_gripper(goal_time=self.close_goal_time)
            else:
                arm.open_gripper(goal_time=self.open_goal_time)
            self.traj_idx[side] += self.step_stride
