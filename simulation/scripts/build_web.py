# -*- coding: utf-8 -*-
"""
Build the web viewer's data: models, rollouts, manifest.

    python simulation/scripts/build_web.py --configs trossen_wxai:realsense_d405:bimanual --n 24 --workers 6
    python simulation/scripts/build_web.py --manifest-only

Rollouts come from the released policy on randomized initial conditions
(paper protocol, fixed seeds) with the physical drop-in intro recorded.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
os.environ.setdefault("MUJOCO_GL", "egl")

from humanego_sim.cameras import CAMERAS  # noqa: E402
from humanego_sim.env import ServeBreadSim, deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.robots import ROBOTS  # noqa: E402
from humanego_sim.sampling import sample_scene  # noqa: E402
from humanego_sim.web_export import export_model, export_rollout  # noqa: E402
from humanego_sim.live_export import export_live  # noqa: E402

WEB = os.path.join(SIM, "web")


def mirror_pose(T):
    S = np.diag([-1.0, 1.0, 1.0, 1.0])
    return S @ T @ S


def T_free(q):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, 3] = q[:3]
    T[:3, :3] = R.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
    return T


def rollout_extras(env, log_dir, cfg):
    """Keypoints / PCA frames in object body frames, appear times, look-ahead."""
    ep = json.load(open(os.path.join(log_dir, "episode.json")))
    meta, frames = ep["meta"], ep["frames"]
    f0 = next(f for f in frames if f["t"] >= 0)
    T_cw = np.array(meta["T_cam_in_world"])
    kpts = {}
    colors = {"obj1": "#22d3ee", "obj2": "#3ddc97"}
    for side, objs in meta["perception"].items():
        for k, v in objs.items():
            body = f"bread_{side}" if k == "obj1" else "plate"
            if body in kpts:
                continue
            Ts = np.array(v["T_obj_in_cam_static"])
            kl = np.array(v["kpts_local"]).reshape(-1, 3)
            if side == "left":
                Ts = mirror_pose(Ts)
                kl = kl * np.array([-1.0, 1.0, 1.0])   # local coords mirror with the frame
            pc = (Ts[:3, :3] @ kl.T).T + Ts[:3, 3]
            pw = (T_cw[:3, :3] @ pc.T).T + T_cw[:3, 3]
            Tb = T_free(np.array(f0["objects"][body]))
            Tbi = np.linalg.inv(Tb)
            pb = (Tbi[:3, :3] @ pw.T).T + Tbi[:3, 3]
            Tf = Tbi @ T_cw @ Ts
            kpts[body] = {"pts": np.round(pb, 4).tolist(), "color": colors[k], "frame": np.round(Tf, 4).ravel().tolist()}
    appear = {}
    for f in frames:
        hid = set(f.get("hidden", []))
        for name in f["objects"]:
            if name not in hid and name not in appear:
                appear[name] = f["t"]
    return {"kpts": kpts, "appear": appear, "look_ahead": int(cfg["control"]["look_ahead_offset"])}


def build_config(robot, camera, mode, n, workers, seed, overrides):
    key = f"{robot}_{camera}_{mode}"
    ov = deep_update(overrides, {"robot": robot, "camera": camera, "mode": mode})
    cfg = load_config(None, ov)
    env = ServeBreadSim(cfg)
    model_key = key
    export_model(env, WEB, model_key)
    export_live(env, WEB, model_key)          # physics-only MJCF for the in-browser Live mode
    rng = np.random.default_rng(seed)
    scenes = [sample_scene(rng, mode) for _ in range(n)]
    log_root = os.path.join(SIM, "runs", "web", key)
    shutil.rmtree(log_root, ignore_errors=True)
    st = evaluate(ov, scenes=scenes, workers=workers, seed=seed, log_root=log_root, intro=True)
    rolls = []
    for r in st["results"]:
        if "error" in r:
            print("  episode error:", r["error"])
            continue
        ld = os.path.join(log_root, f"ep{r['idx']:03d}")
        rid = f"{key}_{r['idx']:03d}"
        ro = export_rollout(env, ld, WEB, rid)
        ro.update(rollout_extras(env, ld, cfg))
        ro["model"] = model_key
        with open(os.path.join(WEB, "data", "rollouts", f"{rid}.json"), "w") as f:
            json.dump(ro, f, separators=(",", ":"))
        rolls.append({"id": rid, "scene": ro["scene"], "per_side": {s: {"success": v["success"]}
                      for s, v in ro["result"]["per_side"].items()}, "success": ro["result"]["success"]})
    return {"robot": robot, "camera": camera, "mode": mode, "model": model_key, "rollouts": rolls}


def tuning_summary(name):
    d = os.path.join(SIM, "autotune_runs", name, "iterations")
    if not os.path.isdir(d):
        return None
    batches = []
    for p in sorted(glob.glob(os.path.join(d, "iter_*.json"))):
        it = json.load(open(p))
        batches.append({"scores": [round(c["stats"]["per_side_success_rate"], 3) for c in it["candidates"]],
                        "action": it.get("decision", {}).get("action")})
    best = None
    for fn in ("final.json", "best.json"):          # final.json: winner re-validated on held-out scenes
        fp = os.path.join(SIM, "autotune_runs", name, fn)
        if os.path.exists(fp):
            best = json.load(open(fp))
            break
    note = ""
    if best and best.get("trials"):         # final.json from autotune_validate.py
        prod = next((t for t in best["trials"] if t["name"] == "production"), None)
        v = best["validation"]
        note = (f"Grey dots: candidates in each batch (per-arm success on shared random scenes); orange: best so far. "
                f"On {best['n_scenes']} held-out scenes the tuned settings serve <b>{v['per_side_success_rate']:.0%}</b> "
                f"per arm" + (f" vs <b>{prod['validation']['per_side_success_rate']:.0%}</b> with the production "
                              f"controller." if prod else "."))
        return {"batches": batches, "note": note, "best": best}
    if best:
        v = best.get("validation", {})
        note = (f"Grey dots: candidates in each batch (per-arm success on shared random scenes); orange: best so far. "
                f"Best per-arm success <b>{best['stats']['per_side_success_rate']:.0%}</b>"
                + (f", validated at <b>{v['per_side_success_rate']:.0%}</b> on {v['n']} fresh scenes." if v else "."))
    return {"batches": batches, "note": note, "best": best}


TUNING_NOTE = None     # set from --tuning_note


def write_manifest(configs, tuning_name):
    man_p = os.path.join(WEB, "data", "manifest.json")
    old = json.load(open(man_p)) if os.path.exists(man_p) else {"configs": []}
    keep = {(c["robot"], c["camera"], c["mode"]): c for c in old.get("configs", [])}
    for c in configs:
        keep[(c["robot"], c["camera"], c["mode"])] = c
    cfgs = sorted(keep.values(), key=lambda c: (list(ROBOTS).index(c["robot"]) if c["robot"] in ROBOTS else 9,
                                                list(CAMERAS).index(c["camera"]) if c["camera"] in CAMERAS else 9,
                                                c["mode"] != "bimanual"))
    summary = []
    for c in cfgs:
        c["live"] = os.path.exists(os.path.join(WEB, "assets", "live", c["model"], "spec.json")) and \
            os.path.exists(os.path.join(WEB, "assets", "policy", "serve_bread_meta.json"))
        rs = c["rollouts"]
        side = lambda s: (np.mean([r["per_side"][s]["success"] for r in rs if s in r["per_side"]])
                          if any(s in r["per_side"] for r in rs) else None)
        summary.append({"robot": c["robot"], "camera": c["camera"], "mode": c["mode"], "n": len(rs),
                        "right": side("right"), "left": side("left"),
                        "both": float(np.mean([r["success"] for r in rs])) if rs else None})
        # feature a fully successful rollout
        for r in rs:
            r.pop("featured", None)
        good = [r for r in rs if r["success"]]
        if good:
            good[0]["featured"] = True
    man = {
        "robots": {k: v.label for k, v in ROBOTS.items()},
        "cameras": {k: v.label for k, v in CAMERAS.items()},
        "configs": cfgs,
        "default": {"robot": "trossen_wxai", "camera": "realsense_d405", "mode": "bimanual"},
        "summary": summary,
        "summary_note": ("Every rollout is a closed-loop run of the released checkpoint from a randomized initial "
                         "condition (bread–plate offset, depth and bread yaw as in the paper's protocol), with "
                         "<b>soft bread</b> held by friction alone and the controller settings found by the "
                         "self-tuning loop on the Trossen rig. <b>Two arms</b>: one single-hand model drives both "
                         "arms; the left arm sees a mirrored image, exactly as on the real robot. <b>Served</b> = "
                         "released and resting on the plate (not touching the table). The left arm carries the "
                         "7.4 cm-wide danish (8.8 cm max. gripper opening). Franka and UR5e run their own tuned "
                         "calibration, but their hands suit the policy's deep end-on pinch less (short fingers, "
                         "small pads)."),
        "tuning": tuning_summary(tuning_name),
    }
    if TUNING_NOTE and man["tuning"]:
        man["tuning"]["note"] = TUNING_NOTE
    with open(man_p, "w") as f:
        json.dump(man, f, separators=(",", ":"))
    print(f"[web] manifest: {len(cfgs)} configs, {sum(len(c['rollouts']) for c in cfgs)} rollouts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="trossen_wxai:realsense_d405:bimanual")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--params", default=None, help="tuned parameters JSON (best.json) to apply")
    ap.add_argument("--tuning", default="bimanual_v1")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="extra override, e.g. control.grasp_assist={\"enabled\": true}")
    ap.add_argument("--tuning_note", default=None, help="HTML note under the tuning plot")
    args = ap.parse_args()
    TUNING_NOTE = args.tuning_note
    overrides = {}
    if args.params:
        from autotune.params import params_to_overrides
        best = json.load(open(args.params))
        overrides = params_to_overrides(best["params"], load_config())
    if args.set:
        from evaluate import parse_sets
        overrides = deep_update(overrides, parse_sets(args.set))
    built = []
    if not args.manifest_only:
        for spec in args.configs.split(","):
            parts = spec.split(":")
            robot, camera, mode = parts[:3]
            n = int(parts[3]) if len(parts) > 3 else args.n
            print(f"[web] building {robot} / {camera} / {mode} ({n} rollouts)", flush=True)
            built.append(build_config(robot, camera, mode, n, args.workers, args.seed, overrides))
            write_manifest(built, args.tuning)         # usable after every config
    write_manifest(built, args.tuning)
