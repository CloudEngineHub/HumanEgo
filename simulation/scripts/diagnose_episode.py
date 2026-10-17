# -*- coding: utf-8 -*-
"""Grasp-moment diagnostics for a recorded episode.

For every side: when the gripper command switches to closed, report the tool
point relative to the bread (height above its centre, offset along / across the
finger axis), the approach pitch, and what happened afterwards.

    python simulation/scripts/diagnose_episode.py simulation/runs/<name>
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402


def load_state(env, fr):
    m, d = env.m, env.d
    if "qfull" in fr:
        d.qpos[:] = fr["qfull"]
    else:
        for s, q in fr["qpos"].items():
            arm = env.arms[s]
            d.qpos[arm.qadr] = q[:len(arm.qadr)]
            d.qpos[arm.g_qadr] = q[len(arm.qadr)]
        for name, pq in fr["objects"].items():
            qa = env.obj_free_qadr[name]
            d.qpos[qa:qa + 7] = pq
    mujoco.mj_forward(m, d)


def diagnose(log_dir, cfg_over=None):
    ep = json.load(open(os.path.join(log_dir, "episode.json")))
    meta = ep["meta"]
    cfg = load_config(None, dict(mode=meta["mode"], robot=meta["robot"], camera=meta["camera"], **(cfg_over or {})))
    env = ServeBreadSim(cfg, render=False)
    frames = ep["frames"]
    out = {}
    for s in meta["sides"]:
        arm = env.arms[s]
        prev = None
        events = []
        for i, fr in enumerate(frames):
            g = fr["grip_cmd"][s]
            closed = abs(g - arm.robot.gripper_closed) < abs(g - arm.robot.gripper_open)
            if prev is not None and closed and not prev:
                # the command started closing at this frame; look 0.6 s later (fingers settled)
                j = min(i + 18, len(frames) - 1)
                load_state(env, frames[i])
                Tw = arm.T_base_in_world @ arm.get_T_ee_in_base()
                b = env.body_T_world(f"bread_{s}")
                rel = Tw[:3, 3] - b[:3, 3]
                spread = Tw[:3, 0]
                approach = Tw[:3, 2]
                long_axis = b[:3, 0]
                load_state(env, frames[j])
                gq = arm.get_gripper_q()
                # did the bread follow?
                k = min(i + 90, len(frames) - 1)
                bz_later = frames[k]["objects"][f"bread_{s}"][2]
                events.append({
                    "t": round(fr["t"], 2),
                    "dz_above_center_cm": round(100 * rel[2], 1),
                    "offset_along_spread_cm": round(100 * float(rel @ spread), 1),
                    "offset_horiz_cm": round(100 * float(np.linalg.norm(rel[:2])), 1),
                    "approach_pitch_deg": round(float(np.degrees(np.arcsin(-approach[2]))), 0),
                    "spread_vs_bread_long_axis_deg": round(float(np.degrees(np.arccos(abs(spread @ long_axis)))), 0),
                    "gq_after_close": round(gq, 2),
                    "bread_z_3s_later_cm": round(100 * bz_later, 1),
                })
            prev = closed
        out[s] = events
    return out


if __name__ == "__main__":
    print(json.dumps(diagnose(sys.argv[1]), indent=1))
