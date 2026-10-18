# -*- coding: utf-8 -*-
"""
Gripper closing speed and squeeze force of each simulated hand vs its datasheet (no policy).

The arm holds its start pose; the gripper is commanded exactly as the controller does
(SimArm.close_gripper, `control.close_gripper_goal_time`). Measured:

  air    closing on nothing: time to 90 % of the stroke, peak finger speed
  block  closing on a rigid block of the given width at the pinch point: steady normal force
         per finger (the squeeze a held object feels)

    python simulation/scripts/physics_gripper.py --robot trossen_wxai --widths 0.04,0.06,0.074
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one core per test process
    os.environ.setdefault(_k, "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from physics_bench import Bench, parse_sets  # noqa: E402
from humanego_sim.driver import SimArm  # noqa: E402

# datasheets (simulation/docs/physics_params.md): per-finger stroke (m), speed (m/s), force (N)
DATASHEET = {
    "trossen_wxai": {"stroke": 0.044, "speed_max": 0.25, "force": "driver effort clamp 100 N total (~50 N per finger)"},
    "franka_panda": {"stroke": 0.040, "speed_max": 0.05, "force": "70 N continuous, 140 N max"},
    "ur5e": {"stroke": 0.0425, "speed_max": 0.15, "force": "20-235 N (force setting)"},
}


def block_fixture(spec):
    b = spec.worldbody.add_body()
    b.name = "test_block"
    b.pos = [3.0, 3.0, 0.5]
    b.gravcomp = 1.0                         # floats where it is put and centres itself between the fingers
    j = b.add_freejoint()
    j.name = "test_block_free"
    g = b.add_geom()
    g.mass = 0.05
    g.name = "test_block_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.02, 0.012, 0.012]
    g.rgba = [0.2, 0.8, 0.2, 1]


def setup(ov):
    b = Bench(ov, fixtures=block_fixture, keep_robots=True)
    side = "right"
    from humanego_sim.robots import get_robot
    robot = get_robot(b.cfg["robot"])
    arm = SimArm(b.m, b.d, side, robot, np.array(b.meta["T_base_in_world"][side]),
                 np.array(b.meta["T_base_in_cam"][side]))
    return b, arm, robot


def reset(b, arm, robot, width=None):
    m, d = b.m, b.d
    b.reset()
    for o in list(b.soft) + ["plate"]:
        b.park(o, xy=(2.0, 2.0 + 0.3 * len(o)))
    arm.set_j_instant(robot.initial_joints["right"], gripper=robot.gripper_open)
    arm.t = 0.0
    arm._g_t0 = 0.0
    mujoco.mj_forward(m, d)
    jq = m.joint("test_block_free").qposadr[0]
    if width is None:
        d.qpos[jq:jq + 3] = [3.0, 3.0, 0.5]
    else:
        # block at the pinch point, its long side along the finger-opening direction
        site = m.site(arm.prefix + robot.tool_site).id
        p = d.site_xpos[site].copy()
        Rs = d.site_xmat[site].reshape(3, 3)
        # EE frame columns in world: R_ee = R_site @ R_tool_to_ee  (X = spread, Z = approach)
        R_ee = Rs @ robot.R_tool_to_ee
        x, z = R_ee[:, 0], R_ee[:, 2]
        y = np.cross(z, x)
        Rb = np.stack([x, y, z], 1)
        from scipy.spatial.transform import Rotation as R
        qx, qy, qz, qw = R.from_matrix(Rb).as_quat()
        d.qpos[jq:jq + 7] = list(p - 0.005 * z) + [qw, qx, qy, qz]
        gid = m.geom("test_block_geom").id
        m.geom_size[gid] = [width / 2, 0.012, 0.012]
    mujoco.mj_forward(m, d)
    return d


def finger_geoms(b, arm):
    m = b.m
    out = {}
    for fb in arm.robot.finger_bodies:
        bid = m.body(arm.prefix + fb).id
        # the finger body and everything below it (pads on child bodies)
        ids = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        sub = [g for g in ids if _is_descendant(m, m.geom_bodyid[g], bid)]
        out[fb] = sub
    return out


def _is_descendant(m, b, anc):
    while b > 0:
        if b == anc:
            return True
        b = m.body_parentid[b]
    return False


def run(ov, widths, goal_time):
    b, arm, robot = setup(ov)
    m, d = b.m, b.d
    res = {"robot": robot.name, "goal_time_s": goal_time, "datasheet": DATASHEET.get(robot.name)}
    # --- air
    reset(b, arm, robot, None)
    t, qs = 0.0, []
    arm.close_gripper(goal_time=goal_time)
    for k in range(int((goal_time + 1.0) / b.dt)):
        arm.update(t)
        mujoco.mj_step(m, d)
        t += b.dt
        arm.t = t
        qs.append(arm.get_gripper_q())
    qs = np.array(qs)
    ts = np.arange(len(qs)) * b.dt
    i90 = int(np.argmax(qs <= 0.1)) if np.any(qs <= 0.1) else None
    res["air_t90_s"] = round(float(ts[i90]), 3) if i90 is not None else None
    stroke = abs(robot.gripper_joint_open - robot.gripper_joint_closed)
    gv = np.abs(np.gradient(qs, b.dt)) * stroke
    if robot.name == "ur5e":
        res["air_note"] = "2F-85 driver joint (rad); speed in rad/s"
    res["air_peak_speed"] = round(float(gv.max()), 3)
    # --- rigid blocks
    fg = finger_geoms(b, arm)
    gblk = m.geom("test_block_geom").id
    for w in widths:
        reset(b, arm, robot, w)
        t = 0.0
        arm.close_gripper(goal_time=goal_time)
        f6 = np.zeros(6)
        for k in range(int((goal_time + 1.0) / b.dt)):
            arm.update(t)
            mujoco.mj_step(m, d)
            t += b.dt
            arm.t = t
        forces = {}
        for fb, gs in fg.items():
            tot = np.zeros(3)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom[0]), int(c.geom[1])
                if gblk not in (g1, g2) or (g1 not in gs and g2 not in gs):
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                tot += fw if g2 == gblk else -fw          # force on the block
            forces[fb] = round(float(np.linalg.norm(tot)), 2)
        res[f"block_{w * 1000:.0f}mm_N_per_finger"] = forces
        res[f"block_{w * 1000:.0f}mm_actuator_N"] = round(float(d.actuator_force[arm.gid]), 2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="trossen_wxai")
    ap.add_argument("--widths", default="0.04,0.06,0.074")
    ap.add_argument("--goal_time", type=float, default=None, help="default: control.close_gripper_goal_time")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    ov = parse_sets(args.set)
    ov["robot"] = args.robot
    from humanego_sim.env import load_config
    gt = args.goal_time if args.goal_time is not None else float(load_config(None, ov)["control"]["close_gripper_goal_time"])
    print(json.dumps(run(ov, [float(x) for x in args.widths.split(",")], gt)))
