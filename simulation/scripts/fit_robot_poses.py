# -*- coding: utf-8 -*-
"""
Give every non-Trossen robot the same starting hand pose as the real rig:
IK-fit its arm so the HumanEgo EE frame matches the Trossen WidowX AI's EE
frame at the real `initial_joints`. Writes humanego_sim/assets/robots/robot_poses.json.

    python simulation/scripts/fit_robot_poses.py franka_panda ur5e
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402
from humanego_sim.robots import ASSETS  # noqa: E402

ref = ServeBreadSim(load_config(None, {"robot": "trossen_wxai"}), render=False)
ref.reset(settle=0.0)
targets = {s: a.T_base_in_world @ a.get_T_ee_in_base() for s, a in ref.arms.items()}
out_p = os.path.join(ASSETS, "robots", "robot_poses.json")
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
for name in sys.argv[1:]:
    env = ServeBreadSim(load_config(None, {"robot": name}), render=False)
    res = {}
    for s, arm in env.arms.items():
        T_ee_base = arm.T_world_in_base @ targets[s]
        T_raw = T_ee_base @ arm.C_inv
        best = None
        rng = np.random.default_rng(0)
        seeds = [np.array(env.robot.home_joints)] + [np.array(env.robot.home_joints) + rng.normal(0, 0.6, len(arm.qadr))
                                                    for _ in range(40)]
        for q0 in seeds:
            q, err = arm.ik(T_raw, np.clip(q0, arm.qlo, arm.qhi), iters=300)
            # prefer solutions far from joint limits (smooth policy tracking afterwards)
            margin = np.min(np.minimum(q - arm.qlo, arm.qhi - q))
            score = err - 0.002 * margin + 0.004 * float(np.linalg.norm(q - np.array(env.robot.home_joints)))
            if best is None or score < best[0]:
                best = (score, q, err, margin)
        res[s] = [round(float(x), 4) for x in best[1]]
        print(f"{name} {s}: err={best[2]*1000:.2f} mm margin={best[3]:.2f} rad q={res[s]}")
    out[name] = {"initial": res}
with open(out_p, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", out_p)
