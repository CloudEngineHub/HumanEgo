# -*- coding: utf-8 -*-
"""
Controlled pinch-and-lift test of the bread physics, independent of the policy.

Replays the gripper pose of real policy grasps (taken from recorded episodes,
relative to the bread) in a fresh scene, shifted up/down by `dz`: approach from
6 cm behind, close the gripper exactly like the real driver (position command,
0.5 s), lift 12 cm, hold 1 s. Reports whether the bread came along, how far the
fingers sank in, and the crumb deformation.

    python simulation/scripts/grasp_test.py simulation/runs/soft_E20 --dz -0.01,0,0.01 \\
        --set physics.soft.young=1.5e4
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402
from evaluate import parse_sets  # noqa: E402  (scripts/evaluate.py)


def grasp_poses(run_dir: str, side: str, n: int):
    """(T_ee_in_bread, bread pose7) at the moment the gripper starts closing, from recorded episodes."""
    out = []
    env = None
    for d in sorted(glob.glob(os.path.join(run_dir, "ep*"))):
        ep = json.load(open(os.path.join(d, "episode.json")))
        meta = ep["meta"]
        if env is None:
            env = ServeBreadSim(load_config(None, dict(mode=meta["mode"], robot=meta["robot"], camera=meta["camera"])),
                                render=False)
        arm = env.arms[side]
        prev = None
        for fr in ep["frames"]:
            g = fr["grip_cmd"][side]
            closed = abs(g - arm.robot.gripper_closed) < abs(g - arm.robot.gripper_open)
            if prev is not None and closed and not prev:
                env.d.qpos[:] = fr["qfull"]
                mujoco.mj_forward(env.m, env.d)
                Tw = arm.T_base_in_world @ arm.get_T_ee_in_base()
                Tb = env.body_T_world(f"bread_{side}")
                out.append((np.linalg.inv(Tb) @ Tw, env.object_pose7(f"bread_{side}"), os.path.basename(d)))
                break
            prev = closed
        if len(out) >= n:
            break
    return out


SAFE_Z = -0.20     # real driver floor in the arm-base frame (control.safe_z_min)


def run_test(env, side, T_ee_in_bread, bread7, dz, lift=0.12, settle=0.3, back=0.0):
    env.reset()
    name = f"bread_{side}"
    env.place_object(name, bread7)
    for o in env.obj_names:                 # keep the other bread out of the way
        if o.startswith("bread_") and o != name:
            p = env.object_pose7(o)
            p[1] = -p[1] if abs(p[1]) > 0.1 else p[1] + 0.4
            env.place_object(o, p)
    env.step(int(settle / env.dt))
    arm = env.arms[side]
    Tb = env.body_T_world(name)
    Tg = Tb @ T_ee_in_bread
    Tg[2, 3] += dz
    approach = Tg[:3, 2]
    Tg[:3, 3] -= back * approach            # shallower grasp: fingertips stop `back` m earlier
    Tpre = Tg.copy()
    Tpre[:3, 3] -= 0.06 * approach
    Tpre[2, 3] += 0.02
    to_cam = lambda T: arm.T_base_in_cam @ np.linalg.inv(arm.T_base_in_world) @ T
    t0 = env.t
    def go(T, dur):
        arm.move_p_in_cam(arm.T_to_p(to_cam(T)), duration=dur, safe_z_min=SAFE_Z)
        env.step(int((dur + 0.15) / env.dt))
    go(Tpre, 1.6)
    go(Tg, 1.0)
    z_tcp = (arm.T_base_in_world @ arm.get_T_ee_in_base())[2, 3] - Tb[2, 3]
    arm.close_gripper(goal_time=0.5)
    env.step(int(0.9 / env.dt))
    sb = env.soft.get(name)
    squash = sb.strain(env.d) if sb else 0.0
    gq = arm.get_gripper_q()
    Tl = Tg.copy()
    Tl[2, 3] += lift
    go(Tl, 1.2)
    env.step(int(1.0 / env.dt))
    bz = env.object_pos(name)[2]
    held = bz > Tb[2, 3] + 0.6 * lift
    return {"dz_cm": round(dz * 100, 1), "tcp_above_centre_cm": round(z_tcp * 100, 1), "held": bool(held),
            "bread_rise_cm": round((bz - Tb[2, 3]) * 100, 1), "gripper_q": round(gq, 2),
            "squash_mm": round(squash * 1000, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dz", default="-0.01,0,0.01")
    ap.add_argument("--back", default="0", help="comma list: pull the grasp back along the approach (m)")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    poses = grasp_poses(args.run_dir, args.side, args.n)
    cfg = load_config(None, parse_sets(args.set))
    env = ServeBreadSim(cfg, render=False)
    rows = []
    for T, b7, ep in poses:
        for dz in [float(x) for x in args.dz.split(",")]:
          for bk in [float(x) for x in args.back.split(",")]:
            r = run_test(env, args.side, T, b7, dz, back=bk)
            r["episode"] = ep
            r["back_cm"] = round(bk * 100, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)
    for dz, bk in sorted({(r["dz_cm"], r["back_cm"]) for r in rows}):
        rr = [r for r in rows if r["dz_cm"] == dz and r["back_cm"] == bk]
        print(f"dz {dz:+.1f} cm back {bk:.1f} cm: held {sum(r['held'] for r in rr)}/{len(rr)}  "
              f"tcp above centre {np.mean([r['tcp_above_centre_cm'] for r in rr]):.1f} cm  "
              f"squash {np.mean([r['squash_mm'] for r in rr]):.1f} mm")
