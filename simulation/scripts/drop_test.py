# -*- coding: utf-8 -*-
"""
Second-bread drop test: the croissant lies on the plate, the danish is released above it
(height, offset and tilt like the policy's releases) and settles. Reports how often it stays
on the plate (no table contact) and how far it rolls - a probe of the crumb's energy loss.

    python simulation/scripts/drop_test.py --set physics.soft.damping=0.02
"""
import argparse
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one core per test process
    os.environ.setdefault(_k, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from evaluate import parse_sets  # noqa: E402
from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402


def on_plate(env, name, plate_xy, R_plate):
    """Served = no part of the bread touches the table and its centre is over the plate."""
    sb = env.soft[name]
    fv = env.d.flexvert_xpos[env.m.flex_vertadr[sb.flex_id]: env.m.flex_vertadr[sb.flex_id] + env.m.flex_vertnum[sb.flex_id]]
    c = env.object_pos(name)
    return bool(fv[:, 2].min() > 0.003 and np.linalg.norm(c[:2] - plate_xy) < R_plate), float(np.linalg.norm(c[:2] - plate_xy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    env = ServeBreadSim(load_config(None, parse_sets(args.set)), render=False)
    rng = np.random.default_rng(0)
    R_plate = env.meta["objects"]["plate"]["R"]
    res = []
    for i in range(args.n):
        env.reset(scene={"plate": [0.47, 0.0, 0.0], "bread_right": [0.47, -0.30, 90.0], "bread_left": [0.47, 0.30, 90.0]},
                  settle=0.3)
        plate_xy = env.object_pos("plate")[:2]
        # croissant resting on the plate's floor (8.5 mm above the table), as after the first serve
        h_c = env.meta["objects"]["bread_right"]["extents"][2]
        q = R.from_euler("z", rng.uniform(60, 120), degrees=True).as_quat()
        env.place_object("bread_right", [plate_xy[0] + rng.uniform(-0.03, 0.03), plate_xy[1] + rng.uniform(-0.03, 0.03),
                                         0.0085 + h_c / 2 + 0.003, q[3], q[0], q[1], q[2]])
        env.step(int(0.8 / env.dt))
        h = rng.uniform(0.04, 0.09)            # release height of the danish bottom above the croissant top
        off = rng.uniform(-0.035, 0.035, 2)
        tilt = R.from_euler("xyz", [rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(60, 120)], degrees=True)
        x, y, z, w = tilt.as_quat()
        top = env.object_pos("bread_right")[2] + env.meta["objects"]["bread_right"]["extents"][2] / 2
        z0 = top + h + env.meta["objects"]["bread_left"]["extents"][2] / 2
        env.place_object("bread_left", [plate_xy[0] + off[0], plate_xy[1] + off[1], z0, w, x, y, z])
        for arm in env.arms.values():          # arms out of the way
            arm.set_j_instant(env.initial_joints(arm.side) if hasattr(arm, "side") else arm.get_j())
        env.step(int(2.5 / env.dt))
        ok, dist = on_plate(env, "bread_left", plate_xy, R_plate)
        ok2, _ = on_plate(env, "bread_right", plate_xy, R_plate)
        res.append((ok, dist, ok2))
    a = np.array(res, dtype=float)
    print(f"danish stays on plate {a[:, 0].mean():.0%}  mean dist {a[:, 1].mean() * 100:.1f} cm  "
          f"croissant still on plate {a[:, 2].mean():.0%}  ({args.n} drops, {' '.join(args.set) or 'defaults'})")
