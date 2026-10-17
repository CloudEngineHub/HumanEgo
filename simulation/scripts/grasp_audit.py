# -*- coding: utf-8 -*-
"""
Per-arm grasp audit of logged episodes (evaluate.py / compare_params.py --log).

For every arm: how many times the gripper closed, where the fingertip point was relative to
the bread when it first closed, how far the fingers closed (did they close on air?), whether
the bread came up, when it separated from the hand, and how the episode ended:

  served            released and resting on the plate
  closed_on_air     the fingers closed (almost) completely: nothing between them
  slipped_on_lift   the fingers closed on the bread but it never left the table
  dropped_lift      the bread came up but fell within ~1 s of the grasp
  dropped_transit   the bread fell later, before the release
  stuck             the bread stayed with the hand after the gripper opened
  released_off      released, but not resting on the plate
  never_closed      the gripper never closed

    python simulation/scripts/grasp_audit.py simulation/runs/robust_r0/v5
"""
import glob
import json
import os
import sys
from collections import Counter

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402


def audit_episode(env, ep, side):
    meta = ep["meta"]
    arm = env.arms[side]
    r = env.robot
    is_closed = lambda g: abs(g - r.gripper_closed) < abs(g - r.gripper_open)
    frames = [f for f in ep["frames"] if f["t"] >= 0]
    res = meta["result"]["per_side"][side]
    name = f"bread_{side}"
    b0 = np.array(frames[0]["objects"][name][:3])
    closes, opens = [], []
    prev = False
    for f in frames:
        c = is_closed(f["grip_cmd"][side])
        if c and not prev:
            closes.append(f["t"])
        if prev and not c:
            opens.append(f["t"])
        prev = c
    out = {"closes": len(closes), "served": bool(res["success"])}
    if not closes:
        out["outcome"] = "never_closed"
        return out
    t_c = closes[0]

    def state_at(t):
        f = min(frames, key=lambda x: abs(x["t"] - t))
        env.d.qpos[:] = f["qfull"]
        mujoco.mj_forward(env.m, env.d)
        T = arm.T_base_in_world @ arm.get_T_ee_in_base()
        return f, T

    f, T = state_at(t_c)
    Tb = env.body_T_world(name)
    rel = np.linalg.inv(Tb) @ T
    out["tcp_above_centre_cm"] = round(100 * (T[2, 3] - Tb[2, 3]), 1)
    out["tcp_in_bread_cm"] = [round(100 * v, 1) for v in rel[:3, 3]]
    _, _ = state_at(t_c + 0.8)
    out["grip_q_after"] = round(arm.get_gripper_q(), 2)
    out["squash_mm"] = round(1000 * env.soft[name].strain(env.d), 1) if name in env.soft else 0.0
    # bread height / hand distance over time after the first close
    t_open = opens[0] if opens else frames[-1]["t"]
    lifted, t_sep = False, None
    for f in frames:
        if f["t"] < t_c or f["t"] > t_open:
            continue
        bz = f["objects"][name][2] - b0[2]
        if bz > 0.03:
            lifted = True
        if lifted and t_sep is None:
            env.d.qpos[:] = f["qfull"]
            mujoco.mj_forward(env.m, env.d)
            hand = (arm.T_base_in_world @ arm.get_T_ee_in_base())[:3, 3]
            if np.linalg.norm(hand - np.array(f["objects"][name][:3])) > 0.12 and bz < 0.03:
                t_sep = f["t"]
    out["lifted"] = lifted
    if res["success"]:
        out["outcome"] = "served"
    elif not lifted:
        out["outcome"] = "closed_on_air" if out["grip_q_after"] < 0.12 else "slipped_on_lift"
    elif t_sep is not None:
        out["outcome"] = "dropped_lift" if t_sep - t_c < 1.5 else "dropped_transit"
        out["t_drop_after_grasp"] = round(t_sep - t_c, 1)
    else:
        # was the bread still at the hand 1 s after the gripper opened?
        if opens:
            f2 = min(frames, key=lambda x: abs(x["t"] - (opens[0] + 1.0)))
            env.d.qpos[:] = f2["qfull"]
            mujoco.mj_forward(env.m, env.d)
            hand = (arm.T_base_in_world @ arm.get_T_ee_in_base())[:3, 3]
            if np.linalg.norm(hand - np.array(f2["objects"][name][:3])) < 0.08 and f2["objects"][name][2] > 0.06:
                out["outcome"] = "stuck"
                return out
        out["outcome"] = "released_off"
    out["dist_cm"] = round(100 * res["dist_to_plate_center"], 1)
    return out


def main(run_dir):
    env, rows = None, []
    for p in sorted(glob.glob(os.path.join(run_dir, "ep*", "episode.json"))):
        ep = json.load(open(p))
        meta = ep["meta"]
        if env is None:
            env = ServeBreadSim(load_config(None, dict(mode=meta["mode"], robot=meta["robot"], camera=meta["camera"])),
                                render=False)
        for s in meta["sides"]:
            a = audit_episode(env, ep, s)
            a.update({"ep": os.path.basename(os.path.dirname(p)), "side": s})
            rows.append(a)
    for s in ("right", "left"):
        rr = [r for r in rows if r["side"] == s]
        if not rr:
            continue
        c = Counter(r["outcome"] for r in rr)
        print(f"{s:5s} n={len(rr)}  " + "  ".join(f"{k} {v}" for k, v in c.most_common()))
        for k in ("tcp_above_centre_cm", "grip_q_after", "squash_mm"):
            for outc in ("served", "closed_on_air", "slipped_on_lift", "dropped_lift", "dropped_transit"):
                v = [r[k] for r in rr if r["outcome"] == outc and k in r]
                if v:
                    print(f"      {k:20s} {outc:16s} mean {np.mean(v):5.2f}  (n={len(v)})")
    json.dump(rows, open(os.path.join(run_dir, "grasp_audit.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
