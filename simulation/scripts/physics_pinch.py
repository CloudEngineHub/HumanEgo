# -*- coding: utf-8 -*-
"""
Pinch-and-lift test of the soft-bread physics with idealised fingers (no robot, no policy).

Two flat pads on a mocap "hand" straddle the bread's width, the long side of each pad along the
bread's long axis (the end-on human-style grasp of the demos). Each pad is driven by a force
actuator, so the squeeze is set directly in newtons per finger. Sequence:

  1. squeeze: ramp to `--force` N per finger in 0.3 s, hold 0.5 s  -> indentation, contact depth
  2. lift the hand 10 cm in 1 s, hold 0.5 s                         -> held?
  3. slip: pull the bread down with an extra load ramped at 2 N/s   -> holding force at slip onset
     (bread moves > 3 mm relative to the hand); creep rate at half of that load

    python simulation/scripts/physics_pinch.py --pad trossen --force 6 --dz 0,0.01,0.02
    python simulation/scripts/physics_pinch.py --kind bun --force 2,6,20 --set 'physics.bread_friction=[0.6,0.01,0.001]'
"""
from __future__ import annotations

import argparse
import itertools
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

# finger pads (length along the approach = along the bread, height), measured on each hand's MJCF
PADS = {"trossen": (0.070, 0.060), "franka": (0.017, 0.017), "robotiq": (0.038, 0.022)}
PAD_T = 0.004        # pad half-thickness


def build(ov, pad, pad_friction):
    L, H = PADS[pad]

    def fixtures(spec):
        # the hand rides on three actuated slides: friction in MuJoCo acts on contact velocity, and a
        # mocap body moved by teleporting has none (its pads would slide off anything they hold)
        hand = spec.worldbody.add_body()
        hand.name = "hand"
        hand.pos = [0.0, 0.0, 0.0]
        hand.gravcomp = 1.0
        for ax, vec in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
            j = hand.add_joint()
            j.name = f"hand_{ax}"
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = vec
            j.armature = 0.1
            a = spec.add_actuator()
            a.name = f"hand_{ax}"
            a.target = f"hand_{ax}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_position(kp=2e4, kv=300.0)
        hg = hand.add_geom()
        hg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        hg.size = [0.005, 0, 0]
        hg.mass = 0.2
        hg.contype = hg.conaffinity = 0
        for side, sgn in (("L", 1), ("R", -1)):
            pb = hand.add_body()
            pb.name = f"pad{side}"
            pb.pos = [0.0, sgn * 0.06, 0.0]
            pb.gravcomp = 1.0
            j = pb.add_joint()
            j.name = f"pad{side}_slide"
            j.type = mujoco.mjtJoint.mjJNT_SLIDE
            j.axis = [0.0, -sgn * 1.0, 0.0]          # +q = closing
            j.range = [0.0, 0.06]
            j.damping = [2.0, 0.0, 0.0]
            j.armature = 0.01
            g = pb.add_geom()
            g.name = f"pad{side}_geom"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [L / 2, PAD_T, H / 2]
            g.mass = 0.05
            if pad_friction is not None:
                g.friction = [pad_friction, 0.005, 0.0001]
            a = spec.add_actuator()
            a.name = f"pad{side}_force"
            a.target = f"pad{side}_slide"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_motor()
            a.ctrlrange = [0.0, 200.0]
            ex = spec.add_exclude()                  # idealised fingers: they may pass the table top
            ex.bodyname1, ex.bodyname2 = f"pad{side}", "table"

    return Bench(ov, fixtures=fixtures)


def run_one(b: Bench, name: str, force: float, dz: float, along: float, slip_max: float = 40.0, lift_time: float = 1.0):
    m, d = b.m, b.d
    mujoco.mj_resetData(m, d)
    for o in list(b.soft) + ["plate"]:
        if o != name:
            b.park(o, xy=(2.0, 2.0 + 0.3 * len(o)))
    ext = b.extents(name)
    # bread long axis along world x (the hand's pads lie along x, squeeze along y)
    b.place(name, [TEST_XY[0], TEST_XY[1], ext[2] / 2 + 0.001], yaw_deg=0.0)     # mesh long axis = x
    hq = [m.joint(f"hand_{a}").qposadr[0] for a in "xyz"]
    ha = [m.actuator(f"hand_{a}").id for a in "xyz"]

    def hand_to(p, teleport=False):
        d.ctrl[ha] = p
        if teleport:
            d.qpos[hq] = p
            d.qvel[[m.joint(f"hand_{a}").dofadr[0] for a in "xyz"]] = 0.0

    def hand_pos():
        return d.qpos[hq].copy()

    hand_to([TEST_XY[0], TEST_XY[1], 0.4], teleport=True)
    mujoco.mj_forward(m, d)
    b.step(int(0.5 / b.dt))
    c0 = b.centre(name)
    V = b.verts(name)
    width = V[:, 1].max() - V[:, 1].min()
    # pads centred `dz` above the bread's centre, shifted `along` the bread from its middle
    zc = c0[2] + dz
    hand_to([c0[0] + along, c0[1], zc], teleport=True)
    qL, qR = m.joint("padL_slide").qposadr[0], m.joint("padR_slide").qposadr[0]
    open_q = 0.06 - (width / 2 + PAD_T + 0.004)          # pads start 4 mm off the bread
    d.qpos[qL] = d.qpos[qR] = open_q
    d.qvel[:] = 0
    aL, aR = m.actuator("padL_force").id, m.actuator("padR_force").id
    gL, gR = m.geom("padL_geom").id, m.geom("padR_geom").id
    fid = b.soft[name].flex_id
    mujoco.mj_forward(m, d)
    # 1. squeeze (approach at a small constant force first, then ramp)
    T = 0.3
    for k in range(int((T + 0.5) / b.dt)):
        f = force * min(1.0, k * b.dt / T)
        d.ctrl[aL] = d.ctrl[aR] = max(f, 0.3)
        b.step()
    contact_q = 0.06 - (width / 2 + PAD_T)                # pad touching the undeformed surface
    indent = 0.5 * ((d.qpos[qL] - contact_q) + (d.qpos[qR] - contact_q))
    Fn = 0.5 * (abs(b.contact_force([gL], fid)[1]) + abs(b.contact_force([gR], fid)[1]))
    pen = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if fid in (int(c.flex[0]), int(c.flex[1])) and (int(c.geom[0]) in (gL, gR) or int(c.geom[1]) in (gL, gR)):
            pen = max(pen, -float(c.dist))
    strain = b.soft[name].strain(d)
    # 2. lift 10 cm
    p0 = hand_pos()
    for k in range(int(lift_time / b.dt)):
        s = k * b.dt / lift_time
        hand_to(p0 + [0, 0, 0.10 * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)])
        b.step()
    b.step(int(0.5 / b.dt))
    rel0 = b.centre(name) - hand_pos()
    held = bool(b.centre(name)[2] > c0[2] + 0.06)
    out = {"force_N": force, "dz_cm": round(dz * 100, 1), "along_cm": round(along * 100, 1), "held": held,
           "Fn_contact_N": round(Fn, 2), "indent_mm": round(indent * 1000, 1), "contact_depth_mm": round(pen * 1000, 2),
           "squash_mm": round(strain * 1000, 1)}
    if not held:
        out["hold_N"] = 0.0
        return out
    # 3. slip: extra downward load in steps held 0.6 s each; the creep speed over the last 0.3 s of a
    #    step separates true slip from the crumb's elastic stretch (bounded displacement)
    W = b.node_mass(name).sum() * 9.81
    slip, creep = None, {}
    hold_target = hand_pos()
    for load in [0.25, 0.5, 1, 2, 3, 4, 6, 8, 11, 15, 20, 30, 40]:
        b.push_nodes(name, [0, 0, -load])
        n = int(0.6 / b.dt)
        zs = []
        for k in range(n):
            hand_to(hold_target)
            b.step()
            if k >= n // 2:
                zs.append((b.centre(name) - hand_pos())[2])
        v = (zs[-1] - zs[0]) / (0.3 - b.dt)
        creep[load] = round(-v * 1000, 2)                   # mm/s downward
        drop = (b.centre(name) - hand_pos() - rel0)[2]
        if -v > 0.005 or drop < -0.02:                   # gross slip: > 5 mm/s or lost 2 cm
            slip = load
            break
        out[f"stretch_mm_at_{load:g}N"] = round(-drop * 1000, 1)
    b.push_nodes(name, [0, 0, 0])
    out["slip_load_N"] = slip
    out["hold_N"] = round(W + (slip if slip is not None else 40.0), 2)   # weight + extra load at slip onset
    out["creep_mm_s"] = creep
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="croissant", choices=["croissant", "bun"])
    ap.add_argument("--pad", default="trossen", choices=list(PADS))
    ap.add_argument("--pad_friction", type=float, default=None, help="pad geom friction (bread's wins at priority 1)")
    ap.add_argument("--force", default="6", help="N per finger, comma list")
    ap.add_argument("--dz", default="0", help="pad centre above the bread centre, m, comma list")
    ap.add_argument("--along", default="0.02", help="pad centre offset along the bread from its middle, m")
    ap.add_argument("--lift_time", type=float, default=1.0, help="s for the 10 cm lift")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ov = parse_sets(args.set)
    if args.kind == "bun":
        ov.setdefault("scene", {}).setdefault("bread", {}).setdefault("left", {})["type"] = "bun"
    name = "bread_right" if args.kind == "croissant" else "bread_left"
    b = build(ov, args.pad, args.pad_friction)
    mu = b.cfg["physics"]["bread_friction"][0]
    rows = []
    for F, dz, al in itertools.product([float(x) for x in args.force.split(",")], [float(x) for x in args.dz.split(",")],
                                       [float(x) for x in args.along.split(",")]):
        r = run_one(b, name, F, dz, al, lift_time=args.lift_time)
        r["hold_over_2muFn"] = round(r["hold_N"] / max(2 * mu * r["Fn_contact_N"], 1e-6), 2)
        r.update({"kind": args.kind, "pad": args.pad, "mu": mu})
        print(json.dumps(r), flush=True)
        rows.append(r)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
