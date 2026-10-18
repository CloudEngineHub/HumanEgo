# -*- coding: utf-8 -*-
"""
Compression tests of the soft-bread model, independent of the policy (scripts/physics_bench.py).

  cube   a 40 mm crumb cube (same flex model: trilinear grid, E, nu, damping) between the table
         and a flat platen: checks that the simulated stress-strain slope is the configured E,
         as in the crumb-cube tests of the food-science literature
  flat   the whole croissant / danish lying on the table, pressed from the top by a flat platen
  pinch  the whole bread squeezed across its width between two finger-sized plates (a pinch),
         force per finger vs indentation per finger - what the gripper feels

Each test loads at `--speed` to `--strain` (fraction of the loaded dimension), holds `--hold` s,
unloads at the same speed and waits; it reports force at fixed strains, the apparent modulus,
the hysteresis (energy lost per cycle) and the residual set after unloading.

    python simulation/scripts/physics_compress.py --mode cube
    python simulation/scripts/physics_compress.py --mode pinch --kind bun --set physics.soft.young=1.2e4
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

from physics_bench import TEST_XY, Bench, add_mocap_box, parse_sets  # noqa: E402

CUBE = 0.04
PLATEN_H = 0.02          # platen half-thickness


def add_cube(spec, cfg, cells=(4, 4, 4)):
    """A crumb cube flex built like the breads (softbody.add_soft_bread), no skin."""
    import trimesh
    h = CUBE / 2
    # surface resolution like the bread's flex mesh (~5 mm), so loads spread over the face
    box = trimesh.creation.box([CUBE] * 3)
    for _ in range(3):
        box = box.subdivide()
    V, F = np.asarray(box.vertices), np.asarray(box.faces)
    me = spec.add_mesh()
    me.name = "cube_flex"
    rad = float(cfg["physics"]["soft"].get("radius", 0.001))
    me.uservert = (V * (CUBE - 2 * rad) / CUBE).ravel().tolist()
    me.userface = np.asarray(F).ravel().tolist()
    sp = cfg["physics"]["soft"]
    dim = int(sp.get("dim", 2))
    kw = {"origin": [0.0, 0.0, 0.0]} if dim == 3 else {}
    fl = spec.worldbody.make_flex("cube", type="mesh", dim=dim, dof=sp.get("dof", "trilinear"),
                                  cellcount=list(cells), mass=0.064 * 0.25,   # 64 cm3 of crumb at 0.25 g/cm3
                                  radius=rad, pos=[TEST_XY[0], TEST_XY[1], h + 0.0005], mesh="cube_flex", **kw)
    fl.young, fl.poisson, fl.damping = float(sp["young"]), float(sp["poisson"]), float(sp["damping"])
    fl.elastic2d = 0
    fl.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE
    ph = cfg["physics"]
    fl.friction = list(ph["bread_friction"])
    fl.condim = int(ph.get("bread_condim", 4))
    fl.solref = list(ph["bread_solref"])
    fl.solimp = list(ph["bread_solimp"])
    fl.priority = 1


def run(args):
    ov = parse_sets(args.set)
    name = "cube" if args.mode == "cube" else f"bread_{'right' if args.kind == 'croissant' else 'left'}"
    if args.kind == "bun":
        ov.setdefault("scene", {}).setdefault("bread", {}).setdefault("left", {})["type"] = "bun"

    def fixtures(spec):
        cfg = b_cfg[0]
        if args.mode == "cube":
            add_cube(spec, cfg)
        if args.mode == "pinch":
            pw = PADS[args.pad]
            for s in (-1, 1):
                add_mocap_box(spec, f"pad{'L' if s > 0 else 'R'}", [pw[0] / 2, 0.003, pw[1] / 2], [0.0, s * 0.2, 0.2])
        else:
            add_mocap_box(spec, "platen", [0.06, 0.06, PLATEN_H], [0.0, 0.0, 0.3])   # thick: cannot be tunnelled

    from humanego_sim.env import load_config
    b_cfg = [load_config(None, ov)]
    b = Bench(ov, fixtures=fixtures)
    m, d = b.m, b.d
    # everything else out of the way; the test object at the origin on the table
    for o in list(b.soft) + ["plate"]:
        if o != name:
            b.park(o, xy=(2.0, 2.0 + 0.3 * len(o)))
    if name == "cube":
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "cube")
        a, n = m.flex_vertadr[fid], m.flex_vertnum[fid]
        nodes = [m.flex_nodebodyid[i] for i in range(m.flex_nodeadr[fid], m.flex_nodeadr[fid] + m.flex_nodenum[fid])]
        ext = np.array([CUBE] * 3)
    else:
        fid = b.soft[name].flex_id
        a, n = m.flex_vertadr[fid], m.flex_vertnum[fid]
        ext = b.extents(name)
        b.place(name, [TEST_XY[0], TEST_XY[1], ext[2] / 2 + 0.001], yaw_deg=0.0)   # long axis along world y
    mujoco.mj_forward(m, d)
    b.step(int(0.6 / b.dt))                                           # settle on the table
    V = d.flexvert_xpos[a:a + n].copy()
    lo, hi = V.min(0), V.max(0)
    size = hi - lo
    centre = 0.5 * (lo + hi)

    if args.mode == "pinch":
        # squeeze across the bread's width (its shortest horizontal extent), pads centred at mid-height
        ax = int(np.argmin(size[:2]))
        L0 = size[ax]
        pads = [m.body("padL").mocapid[0], m.body("padR").mocapid[0]]
        pg = [m.geom("padL_geom").id, m.geom("padR_geom").id]
        zc = centre[2] + args.pinch_dz
        # pads face the bread along axis `ax`; long side of the pad along the bread's long axis
        from scipy.spatial.transform import Rotation as R
        q = R.from_euler("z", 90 if ax == 0 else 0, degrees=True).as_quat()
        for i, sgn in enumerate((1, -1)):
            d.mocap_quat[pads[i]] = [q[3], q[0], q[1], q[2]]
        def set_gap(gap):
            for i, sgn in enumerate((1, -1)):
                p = centre.copy()
                p[2] = zc
                p[ax] = centre[ax] + sgn * (gap / 2 + 0.003)
                d.mocap_pos[pads[i]] = p
        load_axis = ax
        x0 = L0
        set_pos = set_gap
        force_of = lambda: 0.5 * (abs(b.contact_force([pg[0]])[ax]) + abs(b.contact_force([pg[1]])[ax]))
    else:
        pl = m.body("platen").mocapid[0]
        pg = m.geom("platen_geom").id
        L0 = size[2]
        x0 = hi[2]
        def set_pos(top):
            d.mocap_pos[pl] = [centre[0], centre[1], top + PLATEN_H]
        load_axis = 2
        force_of = lambda: abs(b.contact_force([pg])[2])
        d.mocap_pos[pl] = [centre[0], centre[1], hi[2] + PLATEN_H + 0.002]
    test_geoms = set(int(g) for g in (pg if isinstance(pg, list) else [pg]))

    def penetration():
        """Deepest contact penetration of the test fixture into the flex surface (m, >0)."""
        p = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (int(c.geom[0]) in test_geoms or int(c.geom[1]) in test_geoms) and fid in (int(c.flex[0]), int(c.flex[1])):
                p = max(p, -float(c.dist))
        return p

    # footprint area for nominal stress: the cube's face; for breads the bounding box face (upper bound)
    A = {"cube": CUBE * CUBE}.get(name, None)
    if A is None:
        A = size[0] * size[1] if args.mode == "flat" else size[[i for i in (0, 1) if i != load_axis][0]] * size[2]

    rows = []
    v = args.speed
    T_load = args.strain * L0 / v
    t = 0.0
    # approach: start 2 mm out of contact
    set_pos(x0 + 0.002 if args.mode != "pinch" else x0 + 0.004)
    b.step(int(0.3 / b.dt))
    def disp_at(t):
        if t < 0.002 / v:                  # free travel into contact
            return -0.002 + v * t
        return min(v * t - 0.002, args.strain * L0)
    phases = [("load", T_load + 0.002 / v), ("hold", args.hold), ("unload", T_load + 0.002 / v), ("rest", args.rest)]
    t_phase0 = 0.0
    dmax = args.strain * L0
    for ph, T in phases:
        nst = int(T / b.dt)
        for k in range(nst):
            tt = k * b.dt
            if ph == "load":
                delta = disp_at(tt)
            elif ph == "hold":
                delta = dmax
            elif ph == "unload":
                delta = dmax - v * tt
            else:
                delta = -0.002 - v * 0.0
            if args.mode == "pinch":
                set_pos(L0 - 2 * max(delta, -0.002))       # delta is per-side indentation... symmetric squeeze
            else:
                set_pos(x0 - delta)
            b.step()
            if k % 5 == 0:
                rows.append((ph, t_phase0 + tt, float(delta), force_of(), penetration()))
        t_phase0 += T
    # residual set: free height / width after unloading and resting
    Vr = d.flexvert_xpos[a:a + n]
    size_r = Vr.max(0) - Vr.min(0)
    res = {"mode": args.mode, "kind": "cube" if name == "cube" else args.kind, "pad": args.pad if args.mode == "pinch" else None,
           "L0_mm": round(L0 * 1000, 1), "A_cm2": round(A * 1e4, 1), "speed_mm_s": v * 1000,
           "young_cfg_kPa": b.cfg["physics"]["soft"]["young"] / 1e3, "poisson": b.cfg["physics"]["soft"]["poisson"],
           "damping": b.cfg["physics"]["soft"]["damping"]}
    arr = [(r[2], r[3]) for r in rows if r[0] == "load" and r[2] >= 0]
    dd = np.array([x[0] for x in arr])
    ff = np.array([x[1] for x in arr])
    # pinch: delta = indentation per side, strain = 2*delta / width
    strain = (2 * dd if args.mode == "pinch" else dd) / L0
    for e in (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
        if strain.max() >= e - 1e-6:
            i = int(np.argmin(abs(strain - e)))
            res[f"F_at_{int(e * 100)}pct_N"] = round(float(ff[i]), 3)
            res[f"stress_at_{int(e * 100)}pct_kPa"] = round(float(ff[i]) / A / 1e3, 3)
    sel = (strain > 0.02) & (strain < 0.10)
    if sel.sum() > 3:
        k = np.polyfit(strain[sel], ff[sel] / A, 1)[0]
        res["E_apparent_kPa"] = round(k / 1e3, 2)
    if args.mode == "pinch":
        for F in (2.0, 4.0, 6.0, 10.0, 20.0):
            if ff.max() >= F:
                res[f"indent_per_side_at_{F:g}N_mm"] = round(float(dd[np.argmax(ff >= F)]) * 1000, 1)
    # hysteresis: energy in on loading vs returned on unloading
    lo_ = [(r[2], r[3]) for r in rows if r[0] == "load" and r[2] >= 0]
    un_ = [(r[2], r[3]) for r in rows if r[0] == "unload" and r[2] >= 0]
    W_in = np.trapezoid([x[1] for x in lo_], [x[0] for x in lo_]) if len(lo_) > 2 else 0
    W_out = -np.trapezoid([x[1] for x in un_], [x[0] for x in un_]) if len(un_) > 2 else 0
    res["energy_loss_frac"] = round(float(1 - W_out / W_in), 3) if W_in > 0 else None
    pen = [r[4] for r in rows if r[0] == "load" and r[2] >= 0]
    res["penetration_at_peak_mm"] = round(float(pen[-1]) * 1000, 2) if pen else None
    res["peak_F_N"] = round(float(ff[-1]), 3) if len(ff) else None
    res["peak_disp_mm"] = round(float(dd[-1]) * 1000, 2) if len(dd) else None
    hold = [r[3] for r in rows if r[0] == "hold"]
    if hold:
        res["relaxation_frac_during_hold"] = round(float(1 - hold[-1] / max(hold[0], 1e-9)), 3)
    res["residual_set_frac"] = round(float(1 - size_r[load_axis] / size[load_axis]), 4)
    res["_curve"] = [(r[0], round(r[1], 3), round(r[2] * 1000, 2), round(r[3], 3), round(r[4] * 1000, 2)) for r in rows[::4]]
    return res


PADS = {"trossen": (0.070, 0.060), "franka": (0.017, 0.017), "robotiq": (0.038, 0.022), "tpa": (0.036, 0.036)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cube", choices=["cube", "flat", "pinch"])
    ap.add_argument("--kind", default="croissant", choices=["croissant", "bun"])
    ap.add_argument("--pad", default="trossen", choices=list(PADS))
    ap.add_argument("--pinch_dz", type=float, default=0.0, help="pad centre above the bread's centre, m")
    ap.add_argument("--strain", type=float, default=0.4)
    ap.add_argument("--speed", type=float, default=0.002, help="m/s (TPA tests: 1-2 mm/s)")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--rest", type=float, default=2.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args)
    curve = res.pop("_curve")
    print(json.dumps(res))
    if args.out:
        json.dump({**res, "curve": curve}, open(args.out, "w"))
