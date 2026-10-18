# -*- coding: utf-8 -*-
"""
Grasp audit (scripts/grasp_audit.py) of logged robustness runs, with each episode replayed on the
model it ran on (scaled breads, shifted table) and the perception error of that episode:
perceived bread centre / long axis at t = 0 (in the robot's calibrated world frame) vs the truth.

    python simulation/scripts/robust_audit.py trossen_dev7777 all clean
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from grasp_audit import audit_episode  # noqa: E402
from humanego_sim.env import ServeBreadSim, deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import variant_overrides  # noqa: E402
from humanego_sim.perception import mirror_pose  # noqa: E402


def perception_error(ep, side, T_belief):
    """Perceived bread (keypoint centroid, PCA long axis) vs its true pose at t = 0."""
    meta = ep["meta"]
    m = meta["perception"][side]["obj1"]
    T, kl = np.asarray(m["T_obj_in_cam_static"]), np.asarray(m["kpts_local"]).reshape(-1, 3)
    if side == "left":
        T, kl = mirror_pose(T), kl * np.array([-1.0, 1.0, 1.0])
    if len(kl) < 3:
        return {"perc_fail": True}
    pc = (T[:3, :3] @ kl.T).T + T[:3, 3]
    pw = (T_belief[:3, :3] @ pc.T).T + T_belief[:3, 3]
    f0 = next(f for f in ep["frames"] if f["t"] >= 0)
    q = f0["objects"][f"bread_{side}"]
    c_gt = np.array(q[:3])
    a_gt = R.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()[:, 0]
    xy = pw[:, :2] - pw[:, :2].mean(0)
    ev, evec = np.linalg.eigh(xy.T @ xy)
    a = evec[:, np.argmax(ev)]
    ang = np.degrees(np.arccos(min(1.0, abs(float(a @ a_gt[:2]) / max(np.linalg.norm(a_gt[:2]), 1e-9)))))
    return {"perc_dxy_cm": round(100 * float(np.linalg.norm(pw[:, :2].mean(0) - c_gt[:2])), 2),
            "perc_axis_deg": round(float(ang), 1)}


def audit_run(run_dir, base_ov):
    envs, rows = {}, []
    for p in sorted(glob.glob(os.path.join(run_dir, "ep*", "episode.json"))):
        ep = json.load(open(p))
        meta = ep["meta"]
        scene = meta.get("scene_init") or {}
        key, ov = variant_overrides(scene, base_ov)
        if key not in envs:
            envs[key] = ServeBreadSim(load_config(None, deep_update(base_ov, ov) if ov else base_ov), render=False)
        env = envs[key]
        T_belief = np.array(meta["T_cam_in_world"], float)
        T_belief[2, 3] += float((scene.get("_var") or {}).get("table_dz", 0.0))
        for s in meta["sides"]:
            a = audit_episode(env, ep, s)
            a.update(perception_error(ep, s, T_belief))
            a.update({"ep": os.path.basename(os.path.dirname(p)), "side": s,
                      "robustness": meta.get("robustness"), "var": scene.get("_var")})
            rows.append(a)
    return rows


def report(rows, title):
    print(f"\n== {title}: {len(rows)} arms")
    for s in ("right", "left"):
        rr = [r for r in rows if r["side"] == s]
        c = Counter(r["outcome"] for r in rr)
        print(f"  {s:5s} " + "  ".join(f"{k} {v}" for k, v in c.most_common()))
    by = defaultdict(list)
    for r in rows:
        by[r["outcome"]].append(r)
    for k, rr in sorted(by.items(), key=lambda kv: -len(kv[1])):
        d = [r["perc_dxy_cm"] for r in rr if "perc_dxy_cm" in r]
        ax = [r["perc_axis_deg"] for r in rr if "perc_axis_deg" in r]
        h = [r["tcp_above_centre_cm"] for r in rr if "tcp_above_centre_cm" in r]
        print(f"  {k:16s} n={len(rr):3d}  perceived centre err {np.median(d) if d else float('nan'):4.1f} cm (med)  "
              f"axis err {np.median(ax) if ax else float('nan'):4.1f} deg  tcp above centre {np.mean(h) if h else float('nan'):4.1f} cm")


if __name__ == "__main__":
    name, conds = sys.argv[1], sys.argv[2:]
    rep = json.load(open(os.path.join(SIM, "runs", "robust_report", name, "meta.json")))
    for c in conds:
        run_dir = os.path.join(SIM, "runs", "robust_logs", name, c)
        rows = audit_run(run_dir, rep["base_overrides"])
        json.dump(rows, open(os.path.join(run_dir, "robust_audit.json"), "w"), indent=1)
        report(rows, f"{name}/{c}")
