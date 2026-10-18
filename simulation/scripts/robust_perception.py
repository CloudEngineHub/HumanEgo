# -*- coding: utf-8 -*-
"""
Open-loop perception error under each robustness condition (no policy, no rollout): reset the
scene, run the perception exactly as at t = 0 of an episode, and compare what the robot believes
(runner._perceived_geometry, in the calibrated world frame) with the ground truth.

    python simulation/scripts/robust_perception.py --n 32 --seed 7777 --conditions clean masks depth calib
    python simulation/scripts/robust_perception.py --n 4 --conditions all --dump robust_dump   # debug images

Per bread: centre error (xy, z), long-axis error, perceived length; plate: centre error; and the
share of breads whose perception failed (no keypoints / > 5 cm off).
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from evaluate import parse_sets  # noqa: E402
from humanego_sim.env import ServeBreadSim, deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import apply_runtime_variation, variant_overrides  # noqa: E402
from humanego_sim.perception import SimPerception  # noqa: E402
from humanego_sim.robustness import condition  # noqa: E402
from humanego_sim.runner import EpisodeRunner  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed  # noqa: E402


class _EnvOnly:
    """What apply_scene_variation needs from a runner, without loading the policy."""
    def __init__(self, cfg):
        self.cfg, self.env, self.policy = cfg, ServeBreadSim(cfg), None


def _rebuild(holder, scene, base_ov, key):
    """apply_scene_variation for an env without a policy."""
    k, ov = variant_overrides(scene, base_ov)
    if k != key:
        holder.env.renderer.close()
        holder = _EnvOnly(load_config(None, deep_update(base_ov, ov) if ov else base_ov))
    apply_runtime_variation(holder.env, scene)
    return holder, k


def gt_bread(env, side):
    p = env.object_pose7(f"bread_{side}")
    Rm = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    a = Rm[:, 0].copy()
    a[2] = 0
    return np.asarray(p[:3]), a / max(np.linalg.norm(a), 1e-9)


def run_condition(name, args, base_ov):
    c = condition(name)
    ov = deep_update(base_ov, c.get("set", {}))
    cfg = load_config(None, ov)
    scenes = sample_scenes(args.n, args.seed, cfg.get("mode", "bimanual"), variation=c.get("variation") or None)
    holder, key = _EnvOnly(cfg), ""
    rows = []
    for i, sc in enumerate(scenes):
        holder, key = _rebuild(holder, sc, ov, key)
        env = holder.env
        env.rng = np.random.default_rng(args.seed * 1000 + i)        # the episode seed of evaluate()
        env.reset(sc, settle=0.4)
        perc = SimPerception(env, split_x_ratio=float(cfg["control"].get("split_x_ratio", 0.5)),
                             n_init_frames=int(cfg.get("perception", {}).get("n_init_frames", 10)),
                             depth_noise=bool(cfg.get("perception", {}).get("depth_noise", True)))
        so = perc.estimate_objects(env.sides)
        stub = SimpleNamespace(env=env, cfg=cfg, anchor_key="obj1")
        Tp = env.body_T_world("plate")
        for s in env.sides:
            g = EpisodeRunner._perceived_geometry(stub, s, so[s])
            c_gt, a_gt = gt_bread(env, s)
            row = {"scene": i, "side": s}
            if "bread" not in g:
                row["fail"] = True
            else:
                b = g["bread"]
                cp, ap = np.asarray(b["c"]), np.asarray(b["a"])
                row.update(c=cp.tolist(), a=ap.tolist(),
                           dxy=float(np.linalg.norm(cp[:2] - c_gt[:2])), dz=float(cp[2] - c_gt[2]),
                           axis_deg=float(np.degrees(np.arccos(min(1.0, abs(float(ap @ a_gt)))))),
                           half_len=float(b["half_len"]), fail=bool(np.linalg.norm(cp[:2] - c_gt[:2]) > 0.05))
            pm = so[s].objs_meta.get("obj2")
            if pm is not None and len(pm["kpts_local"]):
                T = np.asarray(pm["T_obj_in_cam_static"])
                if s == "left":
                    from humanego_sim.perception import mirror_pose
                    T = mirror_pose(T)
                pw = env.T_cam_in_world @ T
                row["plate_dxy"] = float(np.linalg.norm(pw[:2, 3] - Tp[:2, 3]))
            rows.append(row)
        if args.dump and i < args.dump_n:
            d = os.path.join(os.path.dirname(HERE), "runs", args.dump, name)
            os.makedirs(d, exist_ok=True)
            m = perc.detect_masks()
            rgb = env.render_rgb()
            ov_img = rgb.copy()
            ov_img[m["obj1"] > 0] = (0.5 * ov_img[m["obj1"] > 0] + [0, 0, 127]).astype(np.uint8)
            ov_img[m["obj2"] > 0] = (0.5 * ov_img[m["obj2"] > 0] + [127, 0, 0]).astype(np.uint8)
            z = env.render_depth(noise=True, artifacts=perc.depth_artifacts)
            zc = cv2.applyColorMap(np.clip((z - 0.4) / 0.4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            zc[z <= 0] = 0
            cv2.imwrite(os.path.join(d, f"s{i:02d}_masks.jpg"), ov_img)
            cv2.imwrite(os.path.join(d, f"s{i:02d}_depth.png"), zc)
            cv2.imwrite(os.path.join(d, f"s{i:02d}_clean.jpg"), perc.clean_background())
    holder.env.renderer.close()
    ok = [r for r in rows if not r.get("fail")]
    q = lambda k, f=np.median: float(f([abs(r[k]) for r in ok])) if ok else float("nan")  # noqa: E731
    summ = {"n_breads": len(rows), "fail": sum(r.get("fail", False) for r in rows),
            "dxy_med_cm": 100 * q("dxy"), "dxy_p90_cm": 100 * q("dxy", lambda v: np.quantile(v, 0.9)),
            "dz_med_cm": 100 * float(np.median([r["dz"] for r in ok])) if ok else float("nan"),
            "axis_med_deg": q("axis_deg"), "axis_p90_deg": q("axis_deg", lambda v: np.quantile(v, 0.9)),
            "plate_dxy_med_cm": 100 * float(np.median([r["plate_dxy"] for r in rows if "plate_dxy" in r] or [np.nan]))}
    return summ, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dump", default=None, help="write debug images under simulation/runs/<dump>/<condition>/")
    ap.add_argument("--dump_n", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_seed([args.seed], args.final)
    base_ov = parse_sets(args.set)
    res = {}
    names = ["clean"] + [c for c in args.conditions if c != "clean"]
    for name in names:
        summ, rows = run_condition(name, args, base_ov)
        if name != "clean":
            # shift of the perceived bread relative to the clean perception of the same scene
            ref = {(r["scene"], r["side"]): r for r in res["clean"]["rows"]}
            sh, ax = [], []
            for r in rows:
                q = ref.get((r["scene"], r["side"]))
                if q is None or r.get("fail") or q.get("fail") or "c" not in r or "c" not in q:
                    continue
                sh.append(float(np.linalg.norm(np.subtract(r["c"], q["c"]))))
                ax.append(float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(r["a"], q["a"]))))))))
            if sh:
                summ.update(shift_med_cm=100 * float(np.median(sh)), shift_p90_cm=100 * float(np.quantile(sh, 0.9)),
                            shift_axis_p90_deg=float(np.quantile(ax, 0.9)))
        res[name] = {"summary": summ, "rows": rows}
        print(f"[perception] {name:14s} " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                   for k, v in summ.items()), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
