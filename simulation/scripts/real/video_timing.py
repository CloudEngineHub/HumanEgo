# -*- coding: utf-8 -*-
"""
Real rollout phase timings from the frame annotations of the released videos
(real_video_annotations.json), calibrated per rollout with motions whose duration is fixed in
the real code, and compared with simulated episodes (sim_phases.py output).

The videos are sped-up edits whose burned-in labels (x8 / x16) are inconsistent with the code:
e.g. the wrap-up after a release (open_gripper 1.0 s blocking + go_home 3.0 s) spans 9.5 frames
in clip A, which is 1.3 s at the labelled x8. So every rollout gets its own speed factor
F = ruler seconds * 60 / ruler frames, and the comparison is also given scale-free, as phase
durations divided by the same rollout's wrap-up (release -> home) duration.

    python simulation/scripts/real/video_timing.py --sim phases_prod.json phases_tuned.json
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def speed_factor(r, ruler_s):
    vals = []
    for name, (a, b) in r.get("rulers", {}).items():
        kind = "wrapup" if name.startswith("wrapup") else name
        lo, hi = ruler_s[kind]
        n = b - a
        if n > 0:
            vals.append((lo * 60 / n, hi * 60 / n, name, n))
    if not vals:
        return None
    lo = float(np.median([v[0] for v in vals]))
    hi = float(np.median([v[1] for v in vals]))
    return {"lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "rulers": {v[2]: f"{v[3]:g} fr -> x{v[0]:.0f}-{v[1]:.0f}" for v in vals}}


def real_rows(ann):
    out = []
    rs = ann["ruler_seconds"]
    for rid, r in ann["rollouts"].items():
        F = speed_factor(r, rs)
        for side in ("right", "left"):
            e = r.get(side)
            if not e:
                continue
            row = {"rollout": rid, "side": side, "layout": r["layout"], "label": r["label"],
                   "F": None if F is None else round(F["mid"], 1), "F_range": None if F is None else [round(F["lo"]), round(F["hi"])],
                   "rulers": None if F is None else F["rulers"]}
            fr = {}
            if e.get("move") is not None and e.get("grasp") is not None:
                fr["reach"] = e["grasp"] - e["move"]
            if e.get("lift") is not None and e.get("grasp") is not None:
                fr["grasp"] = e["lift"] - e["grasp"]
            if e.get("release") is not None and e.get("lift") is not None:
                fr["transport"] = e["release"] - e["lift"]
            if e.get("home") is not None and e.get("release") is not None:
                fr["home"] = e["home"] - e["release"]
            row["frames"] = fr
            if F is not None:
                row["seconds"] = {k: [round(v * F["lo"] / 60, 1), round(v * F["hi"] / 60, 1)] for k, v in fr.items()}
                row["labelled_seconds"] = {k: round(v * float(r["label"].strip("x")) / 60, 1) for k, v in fr.items()}
            if fr.get("home"):
                row["per_wrapup"] = {k: round(v / fr["home"], 2) for k, v in fr.items() if k != "home"}
            out.append(row)
    return out


def sim_rows(path):
    d = json.load(open(path))
    S = d["summary"]
    out = {}
    for side in ("right", "left"):
        s = S.get(side)
        if not s:
            continue
        ph = {k: (s.get(f"phase_{k}_s") or {}).get("median") for k in ("reach", "grasp", "transport", "home")}
        row = {"seconds": ph, "served": f"{s['served']}/{s['n']}"}
        if ph.get("home"):
            row["per_wrapup"] = {k: round(v / ph["home"], 2) for k, v in ph.items() if v is not None and k != "home"}
        out[side] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=os.path.join(HERE, "real_video_annotations.json"))
    ap.add_argument("--sim", nargs="*", default=[], help="sim_phases.py --out json files (name=path or path)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ann = json.load(open(args.ann))
    real = real_rows(ann)
    print("REAL (per rollout; seconds = range over the ruler uncertainty; per_wrapup = phase / (release -> home))")
    for r in real:
        print(f"  {r['rollout']:3s} {r['side']:5s} F={r['F']} {r['F_range']}  frames={r['frames']}\n"
              f"      seconds={r.get('seconds')}  labelled-speed seconds={r.get('labelled_seconds')}\n"
              f"      per_wrapup={r.get('per_wrapup')}  rulers={r['rulers']}")
    sims = {}
    for spec in args.sim:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        sims[name] = sim_rows(path)
        print(f"SIM {name}")
        for side, s in sims[name].items():
            print(f"  {side:5s} served {s['served']}  seconds={s['seconds']}  per_wrapup={s.get('per_wrapup')}")
    if args.out:
        json.dump({"real": real, "sim": sims}, open(args.out, "w"), indent=1)
