# -*- coding: utf-8 -*-
"""
Phase timings, grasp location, release height and motion quality of logged simulation
episodes, measured the way they can be read off a real rollout video (see
simulation/docs/real_vs_sim.md, "Quantitative comparison").

Per arm:
  t_move     the tool point first moves faster than 3 cm/s (start of the reach)
  t_close    the gripper is commanded closed (first time)
  t_lift     the bread rises 2 cm above its resting height
  t_open     the gripper is commanded open again (release)
  t_rest     the arm is back at rest after the release (speed < 1 cm/s for 0.3 s)
  phases     reach = t_close - t_move, grasp = t_lift - t_close, transport = t_open - t_lift,
             home = t_rest - t_open
  grasp      tool point in the bread frame once the fingers have closed (t_close + 0.5 s):
             along-axis offset / half length (0 = middle, 1 = tip), height above the centre
  release    bread bottom and tool point above the plate floor when the gripper opens
  motion     peak / mean tool speed in reach and transport, path length, number of speed
             peaks (sub-movements) and log dimensionless jerk (LDLJ, higher = smoother)
  outcome    grasp_audit.py classification (served / slipped_on_lift / dropped_transit / ...)

    python simulation/scripts/real/sim_phases.py simulation/runs/real_dev7777/production --out phases.json
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede `import mujoco`
import glob
import json
import sys

import mujoco
import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "scripts"))
from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402
from grasp_audit import audit_episode  # noqa: E402

PLATE_FLOOR = 0.0085   # plate floor above the table top (runner.evaluate)


def _smooth(x, k=5):
    if len(x) < k:
        return x
    w = np.ones(k) / k
    pad = np.pad(x, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], w, mode="valid") for j in range(x.shape[1])], 1)


def ldlj(t, p):
    """Log dimensionless jerk of a position trace (Hogan & Sternad 2009). Higher = smoother."""
    if len(t) < 8:
        return None
    dt = np.diff(t).mean()
    v = np.gradient(p, dt, axis=0)
    j = np.gradient(np.gradient(v, dt, axis=0), dt, axis=0)
    D = t[-1] - t[0]
    vpk = np.linalg.norm(v, axis=1).max()
    if vpk < 1e-4:
        return None
    return float(-np.log(D ** 3 / vpk ** 2 * np.trapezoid((j ** 2).sum(1), t)))


def n_peaks(speed, prom=0.02):
    """Number of speed maxima with a prominence of at least `prom` m/s (sub-movements)."""
    return int(len(find_peaks(speed, prominence=prom)[0]))


def seg_stats(t, p, a, b):
    if a is None or b is None or b <= a:
        return {}
    m = (t >= a) & (t <= b)
    if m.sum() < 4:
        return {}
    tt, pp = t[m], p[m]
    v = np.linalg.norm(np.gradient(pp, tt, axis=0), axis=1)
    return {"peak_speed": round(float(v.max()), 3), "mean_speed": round(float(v.mean()), 3),
            "path": round(float(np.linalg.norm(np.diff(pp, axis=0), axis=1).sum()), 3),
            "n_peaks": n_peaks(v), "ldlj": None if ldlj(tt, pp) is None else round(ldlj(tt, pp), 2)}


def analyse_side(env, ep, side):
    arm, r = env.arms[side], env.robot
    is_closed = lambda g: abs(g - r.gripper_closed) < abs(g - r.gripper_open)
    frames = [f for f in ep["frames"] if f["t"] >= 0]
    t = np.array([f["t"] for f in frames])
    name = f"bread_{side}"
    tcp = np.zeros((len(frames), 3))
    for i, f in enumerate(frames):
        env.d.qpos[:] = f["qfull"]
        mujoco.mj_kinematics(env.m, env.d)
        tcp[i] = (arm.T_base_in_world @ arm.get_T_ee_in_base())[:3, 3]
    tcp = _smooth(tcp)
    speed = np.linalg.norm(np.gradient(tcp, t, axis=0), axis=1)
    bz = np.array([f["objects"][name][2] for f in frames])
    closed = np.array([is_closed(f["grip_cmd"][side]) for f in frames])

    pol = [p for p in ep["policy"] if side in p["sides"]]
    t_active = pol[0]["t"] if pol else None
    out = {"side": side, "t_active": t_active}
    i_mv = np.where((speed > 0.03) & (t >= (t_active or 0)))[0]
    out["t_move"] = float(t[i_mv[0]]) if len(i_mv) else None
    i_c = np.where(closed)[0]
    t_close = float(t[i_c[0]]) if len(i_c) else None
    out["t_close"] = t_close
    out["n_close"] = int(np.sum(np.diff(closed.astype(int)) == 1) + (1 if closed[0] else 0))
    t_lift = t_open = t_rest = None
    if t_close is not None:
        z0 = bz[t <= t_close].min()
        il = np.where((t > t_close) & (bz > z0 + 0.02))[0]
        t_lift = float(t[il[0]]) if len(il) else None
        # release = the LAST opening after the bread was lifted (re-grasps open and close again)
        opens = [float(t[i + 1]) for i in range(len(t) - 1) if closed[i] and not closed[i + 1]]
        after = [o for o in opens if t_lift is not None and o > t_lift]
        t_open = after[-1] if after else (opens[-1] if opens else None)
        if t_open is not None:
            still = speed < 0.01
            for i in np.where(t > t_open + 0.8)[0]:
                j = np.searchsorted(t, t[i] + 0.3)
                if j < len(t) and still[i:j].all():
                    t_rest = float(t[i])
                    break
    out.update(t_lift=t_lift, t_open=t_open, t_rest=t_rest)
    ph = {}
    if out["t_move"] is not None and t_close is not None:
        ph["reach"] = t_close - out["t_move"]
    if t_lift is not None and t_close is not None:
        ph["grasp"] = t_lift - t_close
    if t_open is not None and t_lift is not None:
        ph["transport"] = t_open - t_lift
    if t_rest is not None and t_open is not None:
        ph["home"] = t_rest - t_open
    if out["t_move"] is not None and t_rest is not None:
        ph["total"] = t_rest - out["t_move"]
    out["phases"] = {k: round(v, 2) for k, v in ph.items()}

    if t_close is not None:
        k = int(np.argmin(np.abs(t - (t_close + 0.5))))
        env.d.qpos[:] = frames[k]["qfull"]
        mujoco.mj_forward(env.m, env.d)
        T = arm.T_base_in_world @ arm.get_T_ee_in_base()
        Tb = env.body_T_world(name)
        rel = (np.linalg.inv(Tb) @ T)[:3, 3]
        ext = env.meta["objects"][name]["extents"]
        along = abs(rel[0]) / (ext[0] / 2)
        out["grasp"] = {"along_frac": round(float(along), 2), "along_cm": round(100 * float(rel[0]), 1),
                        "lateral_cm": round(100 * float(rel[1]), 1),
                        "above_centre_cm": round(100 * float(T[2, 3] - Tb[2, 3]), 1),
                        "where": "end" if along > 0.5 else "centre",
                        "approach_pitch_deg": round(float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1)))), 1)}
    if t_open is not None:
        k = int(np.argmin(np.abs(t - t_open)))
        env.d.qpos[:] = frames[k]["qfull"]
        mujoco.mj_forward(env.m, env.d)
        T = arm.T_base_in_world @ arm.get_T_ee_in_base()
        out["release"] = {"bread_bottom_above_plate_cm": round(100 * (env.object_lowest_z(name) - PLATE_FLOOR), 1),
                          "bread_centre_above_table_cm": round(100 * float(bz[k]), 1),
                          "tcp_above_table_cm": round(100 * float(T[2, 3]), 1)}
    out["lift_max_cm"] = round(100 * float(bz.max() - bz[0]), 1)
    out["motion"] = {"reach": seg_stats(t, tcp, out["t_move"], t_close),
                     "transport": seg_stats(t, tcp, t_lift, t_open),
                     "home": seg_stats(t, tcp, t_open, t_rest)}
    out["peak_speed_all"] = round(float(speed.max()), 3)
    try:
        out["outcome"] = audit_episode(env, ep, side).get("outcome")
    except Exception as e:  # noqa: BLE001
        out["outcome"] = f"audit_error:{e}"
    out["served"] = bool(ep["meta"]["result"]["per_side"][side]["success"])
    return out


def analyse_dir(paths):
    env, rows = None, []
    for p in paths:
        ep = json.load(open(p))
        meta = ep["meta"]
        if env is None:
            env = ServeBreadSim(load_config(None, dict(mode=meta["mode"], robot=meta["robot"], camera=meta["camera"])),
                                render=False)
        for s in meta["sides"]:
            row = analyse_side(env, ep, s)
            row["ep"] = os.path.relpath(os.path.dirname(p))
            row["duration"] = meta["result"]["duration"]
            rows.append(row)
    return rows


def summarise(rows):
    def q(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        return {"n": len(v), "median": round(float(np.median(v)), 2), "p25": round(float(np.percentile(v, 25)), 2),
                "p75": round(float(np.percentile(v, 75)), 2), "mean": round(float(np.mean(v)), 2)}
    out = {}
    for side in ("right", "left", "all"):
        rr = [r for r in rows if side == "all" or r["side"] == side]
        if not rr:
            continue
        sv = [r for r in rr if r["served"]]
        s = {"n": len(rr), "served": len(sv)}
        for ph in ("reach", "grasp", "transport", "home", "total"):
            s[f"phase_{ph}_s"] = q([r["phases"].get(ph) for r in sv])
        s["t_close_s"] = q([r["t_close"] for r in sv])
        s["grasp_along_frac"] = q([r.get("grasp", {}).get("along_frac") for r in rr])
        s["grasp_at_end_frac"] = round(float(np.mean([r["grasp"]["where"] == "end" for r in rr if "grasp" in r])), 2) \
            if any("grasp" in r for r in rr) else None
        s["grasp_above_centre_cm"] = q([r.get("grasp", {}).get("above_centre_cm") for r in rr])
        s["approach_pitch_deg"] = q([r.get("grasp", {}).get("approach_pitch_deg") for r in rr])
        s["release_bottom_above_plate_cm"] = q([r.get("release", {}).get("bread_bottom_above_plate_cm") for r in sv])
        s["lift_max_cm"] = q([r["lift_max_cm"] for r in sv])
        for seg in ("reach", "transport", "home"):
            for k in ("peak_speed", "mean_speed", "n_peaks", "ldlj"):
                s[f"{seg}_{k}"] = q([r["motion"][seg].get(k) for r in sv])
        s["n_close"] = q([r["n_close"] for r in rr])
        oc = {}
        for r in rr:
            oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
        s["outcomes"] = oc
        out[side] = s
    # left/right timing within the same episode
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["ep"], {})[r["side"]] = r
    d_close = [e["left"]["t_close"] - e["right"]["t_close"] for e in by_ep.values()
               if "left" in e and "right" in e and e["left"]["t_close"] is not None and e["right"]["t_close"] is not None]
    d_open = [e["left"]["t_open"] - e["right"]["t_open"] for e in by_ep.values()
              if "left" in e and "right" in e and e["left"]["t_open"] is not None and e["right"]["t_open"] is not None]
    out["left_minus_right"] = {"t_close_s": q(d_close), "t_open_s": q(d_open)}
    out["episode_duration_s"] = q([r["duration"] for r in rows if r["side"] == "right"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir with ep*/episode.json, or one episode.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = [args.run] if args.run.endswith(".json") else sorted(glob.glob(os.path.join(args.run, "ep*", "episode.json")))
    if not paths and os.path.exists(os.path.join(args.run, "episode.json")):
        paths = [os.path.join(args.run, "episode.json")]
    rows = analyse_dir(paths)
    summ = summarise(rows)
    print(json.dumps(summ, indent=1))
    if args.out:
        json.dump({"rows": rows, "summary": summ}, open(args.out, "w"), indent=1)
