# -*- coding: utf-8 -*-
"""
Drop / bounce test of the soft bread on the plate (no robot, no policy).

The bread is released at rest with its underside `--h` above the plate floor (real release:
the croissant's underside 6 +- 1.5 cm above the floor, no visible bounce, at rest within
~0.27 s of landing: serve_bread.mp4 f62-f69), with a random tilt and yaw, and lands on the
plate's flat well. Reports the rebound (highest centre rise after the first impact, as a
coefficient of restitution e = sqrt(rebound / drop)), the time until the bread is at rest,
how far it travelled after landing, and whether it stayed on the plate.

    python simulation/scripts/physics_drop.py --h 0.03,0.06,0.09 --n 8
    python simulation/scripts/physics_drop.py --kind bun --set 'physics.bread_solref=[0.004,1.0]'
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

from physics_bench import TEST_XY, Bench, parse_sets  # noqa: E402

PLATE_FLOOR = 0.0085      # plate floor above the table (plate_meta foot_h + t)


def drop_once(b: Bench, name: str, h: float, rng, tilt_deg: float = 10.0):
    m, d = b.m, b.d
    b.reset()
    for o in b.soft:
        if o != name:
            b.park(o, xy=(2.0, 2.0 + 0.3 * len(o)))
    b.place("plate", [TEST_XY[0], TEST_XY[1], 0.0005])
    mujoco.mj_forward(m, d)
    b.step(int(0.2 / b.dt))
    ext = b.extents(name)
    eul = [rng.uniform(-tilt_deg, tilt_deg), rng.uniform(-tilt_deg, tilt_deg), rng.uniform(0, 180)]
    off = rng.uniform(-0.02, 0.02, 2)
    z0 = PLATE_FLOOR + h + ext[2] / 2
    b.place(name, [TEST_XY[0] + off[0], TEST_XY[1] + off[1], z0], euler_xyz=eul)
    mujoco.mj_forward(m, d)
    zs, xy = [], []
    T = 2.0
    for k in range(int(T / b.dt)):
        b.step()
        c = b.centre(name)
        zs.append(c[2])
        xy.append(c[:2].copy())
    zs = np.array(zs)
    xy = np.array(xy)
    t = np.arange(len(zs)) * b.dt
    # first impact: the centre stops falling
    vz = np.gradient(zs, b.dt)
    falling = np.nonzero(vz < -0.1)[0]
    start = int(falling[0]) if len(falling) else 0
    i_min = start + int(np.argmax(vz[start:] >= 0.0))
    rebound = float(zs[i_min:].max() - zs[i_min])
    drop = float(z0 - zs[i_min])
    e = float(np.sqrt(max(rebound, 0.0) / max(drop, 1e-6)))
    # at rest: centre speed below 5 mm/s for good
    sp = np.linalg.norm(np.gradient(np.c_[xy, zs], b.dt, axis=0), axis=1)
    moving = np.nonzero(sp > 0.005)[0]
    t_rest = float(t[moving[-1]] - t[i_min]) if len(moving) else 0.0
    travel = float(np.linalg.norm(xy[-1] - xy[i_min]))
    V = b.verts(name)
    plate_xy = np.asarray(TEST_XY)
    on_plate = bool(V[:, 2].min() > 0.003 and np.linalg.norm(xy[-1] - plate_xy) < 0.1225)
    return {"h_cm": round(h * 100, 1), "rebound_mm": round(rebound * 1000, 2), "e": round(e, 3),
            "t_rest_s": round(t_rest, 3), "travel_mm": round(travel * 1000, 1), "on_plate": on_plate}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="croissant", choices=["croissant", "bun"])
    ap.add_argument("--h", default="0.03,0.06,0.09", help="underside above the plate floor, m")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ov = parse_sets(args.set)
    if args.kind == "bun":
        ov.setdefault("scene", {}).setdefault("bread", {}).setdefault("left", {})["type"] = "bun"
    name = "bread_right" if args.kind == "croissant" else "bread_left"
    b = Bench(ov)
    rng = np.random.default_rng(0)
    rows = []
    for h in [float(x) for x in args.h.split(",")]:
        rr = [drop_once(b, name, h, rng) for _ in range(args.n)]
        rows += rr
        print(json.dumps({"kind": args.kind, "h_cm": h * 100, "e_mean": round(float(np.mean([r["e"] for r in rr])), 3),
                          "e_max": round(float(np.max([r["e"] for r in rr])), 3),
                          "rebound_mm_mean": round(float(np.mean([r["rebound_mm"] for r in rr])), 2),
                          "t_rest_s_mean": round(float(np.mean([r["t_rest_s"] for r in rr])), 3),
                          "travel_mm_mean": round(float(np.mean([r["travel_mm"] for r in rr])), 1),
                          "on_plate": round(float(np.mean([r["on_plate"] for r in rr])), 2)}), flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
