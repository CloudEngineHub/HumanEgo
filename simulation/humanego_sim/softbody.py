# -*- coding: utf-8 -*-
"""
Soft bread: a MuJoCo flex with a trilinear background grid plus a textured skin.

Physics. The bread's collision surface is a closed triangle mesh (``<kind>_flex.obj``)
turned into a ``flexcomp`` with ``dof="trilinear"``: its degrees of freedom are the
corners of a ``cells[0] x cells[1] x cells[2]`` grid over the bread's bounding box
(each corner is a point mass on three world-aligned slide joints), and every surface
vertex is trilinearly interpolated from the corners of its cell. Volumetric elasticity
(Young's modulus, Poisson ratio, damping) acts on that grid, so pinching the bread
squashes it locally and the fingers sink in. MuJoCo's ``discrete`` integrator treats
the elasticity implicitly in position, which keeps a soft, light object stable at the
scene's 2 ms step.

Rendering. The dense textured mesh is bound to the same grid as a MuJoCo ``skin``
whose bone weights are the trilinear weights, so the visible crust deforms exactly
like the collision surface. The web viewer applies the identical interpolation in
three.js (``web/js/viewer.js``), driven by the recorded corner positions.

Pose. A deformable body has no single frame; ``SoftBody.pose`` returns the
least-squares rigid fit (Kabsch) of the current corner positions to their rest
layout, which is what success checks, logging and the intro animation use.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def grid_nodes(lo, hi, cells) -> np.ndarray:
    """Rest positions of the grid corners, in flexcomp order (i, j, k with k fastest)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ci, cj, ck = (int(c) for c in cells)
    out = []
    for i in range(ci + 1):
        for j in range(cj + 1):
            for k in range(ck + 1):
                out.append(lo + (hi - lo) * np.array([i / ci, j / cj, k / ck]))
    return np.asarray(out)


def trilinear_binding(V, lo, hi, cells) -> Tuple[np.ndarray, np.ndarray]:
    """For points V (N, 3) in the grid's frame: (N, 8) corner indices and (N, 8) weights."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.asarray(cells, int)
    u = np.clip((np.asarray(V) - lo) / (hi - lo), 0.0, 1.0) * c
    cell = np.minimum(np.floor(u).astype(int), c - 1)
    f = u - cell
    nj, nk = c[1] + 1, c[2] + 1
    idx = np.zeros((len(V), 8), np.int64)
    w = np.zeros((len(V), 8))
    n = 0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                idx[:, n] = ((cell[:, 0] + di) * nj + (cell[:, 1] + dj)) * nk + (cell[:, 2] + dk)
                w[:, n] = ((f[:, 0] if di else 1 - f[:, 0]) * (f[:, 1] if dj else 1 - f[:, 1])
                           * (f[:, 2] if dk else 1 - f[:, 2]))
                n += 1
    return idx, w


def add_soft_bread(spec: mujoco.MjSpec, name: str, kind: str, pos, mass: float, vis: tuple,
                   flex_mesh: str, material: str, soft: dict, contact: dict,
                   flex_group: int, skin_group: int) -> dict:
    """Add the flex + skin for one bread at `pos` (bounding-box centre, no rotation).

    vis = (V, F, UV) of the visual mesh in the bread frame (bounding-box centred).
    Returns the meta entry needed at runtime (grid, rest layout).
    """
    V, F, UV = vis
    cells = [int(c) for c in soft["cells"]]
    # dim 3: tetrahedra from every surface triangle to the bread's centre, so a finger that gets
    # inside the crumb is still in contact and is pushed out (a dim-2 surface can be tunnelled)
    dim = int(soft.get("dim", 2))
    kw = {"origin": [0.0, 0.0, 0.0]} if dim == 3 else {}
    fl = spec.worldbody.make_flex(name, type="mesh", dim=dim, dof=soft.get("dof", "trilinear"),
                                  cellcount=cells, mass=float(mass), radius=float(soft.get("radius", 0.001)),
                                  pos=[float(p) for p in pos], mesh=flex_mesh, **kw)
    fl.young = float(soft["young"])
    fl.poisson = float(soft["poisson"])
    fl.damping = float(soft["damping"])
    fl.elastic2d = 0                      # elasticity comes from the volumetric background grid
    fl.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE
    fl.friction = list(contact["friction"])
    fl.condim = int(contact["condim"])
    fl.solref = list(contact["solref"])
    fl.solimp = list(contact["solimp"])
    fl.priority = 1                       # the bread's contact parameters win
    fl.group = flex_group                 # collision surface: never rendered
    fl.rgba = [0.9, 0.6, 0.2, 0.0]

    # grid = bounding box of the flex mesh vertices (the visual mesh shares the same box)
    me = next(m for m in spec.meshes if m.name == flex_mesh)
    Vf = np.asarray(me.uservert).reshape(-1, 3)
    lo, hi = Vf.min(0), Vf.max(0)
    nodes_local = grid_nodes(lo, hi, cells)
    names = [f"{name}_{i}_{j}_{k}" for i in range(cells[0] + 1) for j in range(cells[1] + 1)
             for k in range(cells[2] + 1)]

    # skin: dense textured mesh, bone weights = trilinear weights
    idx, w = trilinear_binding(V, lo, hi, cells)
    sk = spec.add_skin()
    sk.name = f"{name}_skin"
    sk.material = material
    sk.group = skin_group
    pos = np.asarray(pos, float)
    sk.vert = (np.asarray(V) + pos).ravel().tolist()
    sk.face = np.asarray(F).ravel().tolist()
    sk.texcoord = np.asarray(UV).ravel().tolist()
    vid: List[List[int]] = [[] for _ in names]
    vw: List[List[float]] = [[] for _ in names]
    for c in range(8):
        for vi in np.nonzero(w[:, c] > 1e-9)[0]:
            vid[idx[vi, c]].append(int(vi))
            vw[idx[vi, c]].append(float(w[vi, c]))
    keep = [n for n in range(len(names)) if vid[n] and any(wt > 1e-6 for wt in vw[n])]
    sk.bodyname = [names[n] for n in keep]
    sk.bindpos = np.concatenate([nodes_local[n] + pos for n in keep]).tolist()
    sk.bindquat = [1.0, 0.0, 0.0, 0.0] * len(keep)
    sk.vertid = [vid[n] for n in keep]
    sk.vertweight = [vw[n] for n in keep]
    return {"cells": cells, "lo": lo.tolist(), "hi": hi.tolist(), "pos0": pos.tolist(), "node_names": names}


class SoftBody:
    """Runtime handle of one soft bread (corner bodies, their joints, rest layout)."""

    def __init__(self, m: mujoco.MjModel, name: str, meta: dict):
        self.name = name
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, name)
        assert fid >= 0, f"no flex named {name}"
        a, n = int(m.flex_nodeadr[fid]), int(m.flex_nodenum[fid])
        bodies = np.asarray(m.flex_nodebodyid[a:a + n])
        # MuJoCo pins corners that no surface vertex depends on (empty corner cells around the
        # tapered tips) to the world; only the moving corners carry state
        self.valid = bodies != 0
        self.bodies = bodies[self.valid]
        self.local = grid_nodes(meta["lo"], meta["hi"], meta["cells"])[self.valid]
        self.pos0 = np.asarray(meta["pos0"], float)
        self.p0 = self.local + self.pos0                       # compile-time corner positions
        assert np.allclose(m.body_pos[self.bodies], self.p0, atol=1e-6), "flex grid layout mismatch"
        ja = m.body_jntadr[self.bodies]
        self.qadr = np.asarray(m.jnt_qposadr[ja])               # 3 consecutive slides (x, y, z)
        self.dadr = np.asarray(m.jnt_dofadr[ja])
        self.qidx = (self.qadr[:, None] + np.arange(3)[None]).ravel()
        self.didx = (self.dadr[:, None] + np.arange(3)[None]).ravel()
        self.centroid_local = self.local.mean(0)
        self.flex_id = fid

    # --------------------------------------------------------------- state
    def positions(self, d: mujoco.MjData) -> np.ndarray:
        return d.xpos[self.bodies].copy()

    def place(self, d: mujoco.MjData, pos, quat_wxyz) -> None:
        """Rigidly place the (undeformed) bread: bounding-box centre at pos, orientation quat."""
        w, x, y, z = quat_wxyz
        Rm = R.from_quat([x, y, z, w]).as_matrix()
        target = self.local @ Rm.T + np.asarray(pos, float)
        d.qpos[self.qidx] = (target - self.p0).ravel()
        d.qvel[self.didx] = 0.0

    def fit(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Least-squares rigid transform local -> P: returns (R, t) with P ~ local @ R.T + t."""
        A = self.local - self.centroid_local
        c = P.mean(0)
        H = A.T @ (P - c)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rm = Vt.T @ D @ U.T
        t = c - Rm @ self.centroid_local
        return Rm, t

    def T_world(self, d: mujoco.MjData) -> np.ndarray:
        Rm, t = self.fit(d.xpos[self.bodies])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, t
        return T

    def pose7(self, d: mujoco.MjData) -> List[float]:
        Rm, t = self.fit(d.xpos[self.bodies])
        x, y, z, w = R.from_matrix(Rm).as_quat()
        return [float(v) for v in t] + [float(w), float(x), float(y), float(z)]

    def strain(self, d: mujoco.MjData) -> float:
        """Largest corner deviation from the best rigid fit (m): how squashed the bread is."""
        P = d.xpos[self.bodies]
        Rm, t = self.fit(P)
        return float(np.abs(P - (self.local @ Rm.T + t)).max())
