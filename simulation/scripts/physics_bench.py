# -*- coding: utf-8 -*-
"""
Policy-free physics bench: the serve_bread scene's breads, plate and table with exactly the
episode physics (same generated MJCF, options and contact parameters), the robots removed and
test fixtures (platens, pads, grippers on mocap bodies) added by each `physics_*.py` script.

    from physics_bench import Bench
    b = Bench({"physics": {"soft": {"young": 1.5e4}}}, fixtures=add_platen)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one core per test process
    os.environ.setdefault(_k, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from humanego_sim.cameras import get_camera  # noqa: E402
from humanego_sim.env import load_config  # noqa: E402
from humanego_sim.robots import ASSETS, get_robot  # noqa: E402
from humanego_sim.scene import build_scene  # noqa: E402
from humanego_sim.softbody import SoftBody  # noqa: E402


TEST_XY = (0.55, 0.0)     # free table area (the plate's usual spot; the camera mast stands at x = -0.045)


def parse_sets(sets):
    ov = {}
    for s in sets:
        key, val = s.split("=", 1)
        cur = ov
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
    return ov


def _robot_items(spec: mujoco.MjSpec) -> list:
    """Everything of the scene that refers to the arms (all of it belongs to the robots)."""
    return [x for coll in (spec.actuators, spec.sensors, spec.equalities, spec.tendons, spec.excludes,
                           spec.pairs, spec.keys) for x in coll]


def strip_robots(spec: mujoco.MjSpec, items: Optional[list] = None) -> None:
    """Remove both arms (bodies prefixed right/ left/) and everything that refers to them."""
    for x in (items if items is not None else _robot_items(spec)):
        spec.delete(x)
    for b in list(spec.worldbody.bodies):
        if b.name.startswith(("right/", "left/")):
            spec.delete(b)


class Bench:
    """Compiled bench model + handles. `fixtures(spec)` adds test bodies before compiling."""

    def __init__(self, overrides: Optional[dict] = None, fixtures: Optional[Callable] = None,
                 keep_robots: bool = False):
        self.cfg = load_config(None, overrides or {})
        robot = get_robot(self.cfg["robot"])
        cam = get_camera(self.cfg["camera"])
        xml, self.meta = build_scene(self.cfg, robot, cam)
        spec = mujoco.MjSpec.from_file(xml)
        spec.modelfiledir = ASSETS
        robot_items = _robot_items(spec)
        if fixtures is not None:          # before stripping: MjSpec.delete breaks later mesh lookups
            fixtures(spec)
        if not keep_robots:
            strip_robots(spec, robot_items)
        self.spec = spec
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)
        self.soft: Dict[str, SoftBody] = {n: SoftBody(self.m, n, o["soft"])
                                          for n, o in self.meta["objects"].items() if "soft" in o}
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ objects
    def extents(self, name: str) -> np.ndarray:
        return np.asarray(self.meta["objects"][name]["extents"], float)

    def place(self, name: str, pos, yaw_deg: float = 0.0, euler_xyz=None) -> None:
        """Rigidly place a soft bread (bounding-box centre) or the plate."""
        rot = R.from_euler("xyz", euler_xyz, degrees=True) if euler_xyz is not None else R.from_euler("z", yaw_deg, degrees=True)
        x, y, z, w = rot.as_quat()
        if name in self.soft:
            self.soft[name].place(self.d, pos, [w, x, y, z])
        else:
            j = self.m.body(name).jntadr[0]
            qa, da = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
            self.d.qpos[qa:qa + 7] = list(pos) + [w, x, y, z]
            self.d.qvel[da:da + 6] = 0.0

    def park(self, name: str, xy=(2.0, 2.0)) -> None:
        """Move an object far away (onto the floor area, out of every test)."""
        ext = self.extents(name) if name in self.soft else np.zeros(3)
        self.place(name, [xy[0], xy[1], -0.76 + ext[2] / 2 + 0.002])

    def verts(self, name: str) -> np.ndarray:
        fid = self.soft[name].flex_id
        a, n = self.m.flex_vertadr[fid], self.m.flex_vertnum[fid]
        return self.d.flexvert_xpos[a:a + n].copy()

    def centre(self, name: str) -> np.ndarray:
        return self.soft[name].T_world(self.d)[:3, 3].copy()

    def node_mass(self, name: str) -> np.ndarray:
        return self.m.body_mass[self.soft[name].bodies]

    def push_nodes(self, name: str, force_world) -> None:
        """Spread an external force over the bread's nodes in proportion to their mass."""
        b = self.soft[name].bodies
        w = self.node_mass(name) / self.node_mass(name).sum()
        self.d.xfrc_applied[b, :3] = np.outer(w, np.asarray(force_world, float))

    # ----------------------------------------------------------------- contacts
    def contact_force(self, geom_ids, flex_id: Optional[int] = None) -> np.ndarray:
        """Total world-frame force exerted ON the given geoms by contacts (optionally only with one flex)."""
        gs = set(int(g) for g in np.atleast_1d(geom_ids))
        f6 = np.zeros(6)
        tot = np.zeros(3)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 not in gs and g2 not in gs:
                continue
            if flex_id is not None and flex_id not in (int(c.flex[0]), int(c.flex[1])):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]      # contact frame rows -> world
            # the contact force acts on geom2/flex2 along +normal (normal points from 1 to 2)
            tot += fw if g2 in gs else -fw
        return tot

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)


def add_mocap_box(spec, name, size, pos, rgba=(0.3, 0.3, 0.3, 1), friction=None, euler=None):
    b = spec.worldbody.add_body()
    b.name = name
    b.mocap = True
    b.pos = list(pos)
    if euler is not None:
        x, y, z, w = R.from_euler("xyz", euler, degrees=True).as_quat()
        b.quat = [w, x, y, z]
    g = b.add_geom()
    g.name = f"{name}_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(size)
    g.rgba = list(rgba)
    if friction is not None:
        g.friction = list(friction)
    return b
