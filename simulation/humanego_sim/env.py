# -*- coding: utf-8 -*-
"""
ServeBreadSim — the MuJoCo world: physics, the two simulated arms, and the
RGB-D camera the policy looks through.

Nothing here knows about the policy; `runner.py` wires the released HumanEgo
inference logic on top of it.
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .cameras import CameraSpec, get_camera
from .driver import SimArm, T_from
from .robots import RobotSpec, get_robot
from .scene import G_ROBOT, build_scene, yaw_quat

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    path = path or os.path.join(SIM_ROOT, "configs", "serve_bread.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ServeBreadSim:
    def __init__(self, cfg: dict, seed: int = 0, render: bool = True):
        self.cfg = cfg
        self.robot: RobotSpec = get_robot(cfg["robot"])
        self.cam: CameraSpec = get_camera(cfg["camera"])
        self.xml_path, self.meta = build_scene(cfg, self.robot, self.cam)
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.sides: List[str] = self.meta["sides"]
        self.dt = float(self.m.opt.timestep)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        self.T_cam_in_world = np.array(self.meta["T_cam_in_world"])
        self.T_world_in_cam = np.linalg.inv(self.T_cam_in_world)
        self.K = self.cam.K.copy()
        self.H, self.W = self.cam.height, self.cam.width

        # object bodies
        self.obj_bodies = {name: self.m.body(name).id for name in self.meta["objects"]}
        self.obj_free_qadr = {name: self.m.jnt_qposadr[self.m.body_jntadr[bid]] for name, bid in self.obj_bodies.items()}
        self.obj_free_dadr = {name: self.m.jnt_dofadr[self.m.body_jntadr[bid]] for name, bid in self.obj_bodies.items()}
        # geom -> segmentation label
        self.seg_label = {}
        for name, bid in self.obj_bodies.items():
            for gi in range(self.m.ngeom):
                if self.m.geom_bodyid[gi] == bid:
                    self.seg_label[gi] = name

        self._init_arms()
        self.renderer = mujoco.Renderer(self.m, self.H, self.W) if render else None
        self._opt_clean = mujoco.MjvOption()
        self._opt_clean.geomgroup[G_ROBOT] = 0          # hide the robot (perfect "inpainting")

    # ------------------------------------------------------------------ arms
    def _init_arms(self):
        self.arms: Dict[str, SimArm] = {}
        for s in self.sides:
            self.arms[s] = SimArm(self.m, self.d, s, self.robot,
                                  np.array(self.meta["T_base_in_world"][s]),
                                  np.array(self.meta["T_base_in_cam"][s]),
                                  tcp_offset=float(self.cfg.get("control", {}).get("tcp_offset", 0.0)))

    # ----------------------------------------------------------------- reset
    def reset(self, scene: Optional[dict] = None, settle: float = 0.4, arm_q: Optional[dict] = None):
        """Reset physics. `scene` overrides object poses:
            {"plate": [x, y, yaw], "bread_right": [x, y, yaw], "bread_left": [...]}
        """
        mujoco.mj_resetData(self.m, self.d)
        self.t = 0.0
        sc = self.default_scene()
        if scene:
            sc.update(scene)
        for name, pose in sc.items():
            if name not in self.obj_bodies:
                continue
            self.set_object_pose(name, *pose)
        for s, arm in self.arms.items():
            q = (arm_q or {}).get(s, self.robot.initial_joints[s])
            arm.set_j_instant(q, gripper=self.robot.gripper_open)
        mujoco.mj_forward(self.m, self.d)
        # let objects settle under gravity with the arms holding position
        n = int(settle / self.dt)
        for _ in range(n):
            self.step()
        self.t = 0.0
        for arm in self.arms.values():
            arm.t = 0.0
            arm._g_t0 = 0.0

    def default_scene(self) -> dict:
        sc = self.cfg["scene"]
        if self.cfg.get("mode") == "single":
            return {"plate": list(sc["single"]["plate"]), "bread_right": list(sc["single"]["bread"]["pose"])}
        out = {"plate": list(sc["plate"])}
        for s in self.sides:
            out[f"bread_{s}"] = list(sc["bread"][s]["pose"])
        return out

    def set_object_pose(self, name: str, x: float, y: float, yaw_deg: float):
        qa = self.obj_free_qadr[name]
        if name == "plate":
            z = 0.0005
        else:
            z = self.meta["objects"][name]["extents"][2] / 2 + 0.0008
        self.d.qpos[qa:qa + 3] = [x, y, z]
        self.d.qpos[qa + 3:qa + 7] = yaw_quat(yaw_deg)
        da = self.obj_free_dadr[name]
        self.d.qvel[da:da + 6] = 0.0

    # ------------------------------------------------------------------ step
    def step(self, n: int = 1):
        for _ in range(n):
            for arm in self.arms.values():
                arm.update(self.t)
            mujoco.mj_step(self.m, self.d)
            self.t += self.dt

    # ------------------------------------------------------------- geometry
    def body_T_world(self, name: str) -> np.ndarray:
        bid = self.m.body(name).id
        return T_from(self.d.xmat[bid].reshape(3, 3), self.d.xpos[bid])

    def body_T_cam(self, name: str) -> np.ndarray:
        return self.T_world_in_cam @ self.body_T_world(name)

    # ------------------------------------------------------------ rendering
    def render_rgb(self, camera: str = "policy_cam", hide_robot: bool = False) -> np.ndarray:
        """BGR uint8 (OpenCV order, like CamRS)."""
        r = self.renderer
        r.update_scene(self.d, camera=camera, scene_option=self._opt_clean if hide_robot else None)
        return np.ascontiguousarray(r.render()[:, :, ::-1])

    def render_depth(self, noise: bool = True) -> np.ndarray:
        r = self.renderer
        r.enable_depth_rendering()
        r.update_scene(self.d, camera="policy_cam")
        z = r.render().astype(np.float32)
        r.disable_depth_rendering()
        if noise:
            sig = self.cam.depth_noise_a + self.cam.depth_noise_b * z ** 2
            z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * sig
            z[(z < self.cam.depth_min) | (z > self.cam.depth_max)] = 0.0
        return z

    def render_segmentation(self) -> Dict[str, np.ndarray]:
        """Ground-truth object masks (uint8 0/255) — stands in for DINO-SAM."""
        r = self.renderer
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera="policy_cam")
        seg = r.render()
        r.disable_segmentation_rendering()
        geom_ids = seg[:, :, 0]
        types = seg[:, :, 1]
        masks = {name: np.zeros((self.H, self.W), np.uint8) for name in self.obj_bodies}
        valid = types == mujoco.mjtObj.mjOBJ_GEOM
        for gi, name in self.seg_label.items():
            masks[name][valid & (geom_ids == gi)] = 255
        return masks

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Compact state for the web recorder."""
        out = {"t": self.t, "qpos": {}, "objects": {}}
        for s, arm in self.arms.items():
            q = self.d.qpos[arm.qadr].tolist()
            out["qpos"][s] = q + [float(self.d.qpos[arm.g_qadr])]
        out["qfull"] = np.round(self.d.qpos, 5).tolist()
        for name in self.obj_bodies:
            bid = self.obj_bodies[name]
            out["objects"][name] = self.d.xpos[bid].tolist() + self.d.xquat[bid].tolist()
        return out
