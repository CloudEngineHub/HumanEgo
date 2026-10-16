# -*- coding: utf-8 -*-
"""
Export simulator models and recorded rollouts for the three.js web viewer.

model JSON   kinematic tree (bodies, joints, visual geoms, materials, cameras)
             so the browser can run forward kinematics from recorded qpos;
mesh GLB     geometry of every visual mesh, taken from the *compiled* MuJoCo
             model so mesh frames match the simulator exactly;
rollout JSON qpos at 30 Hz + the policy log (predicted 50-step EE trajectory,
             grasp / done probabilities) + the result;
policy MP4   exactly the images the policy saw (one tile per arm).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh

from .robots import ASSETS

JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GT = {mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_BOX: "box",
      mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_MESH: "mesh",
      mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}

# web materials (PBR) keyed by MuJoCo material name suffix
WEB_MATERIALS = {
    "trossen_material": {"color": [1, 1, 1], "map": "assets/textures/trossen_BaseColor.png", "roughness": 0.45, "metalness": 0.15},
    "maple": {"color": [1, 1, 1], "map": "assets/textures/maple.jpg", "roughness": 0.55, "metalness": 0.0, "repeat": [2, 2]},
    "frame_black": {"color": [0.05, 0.05, 0.055], "roughness": 0.35, "metalness": 0.6},
    "floor": {"color": [0.72, 0.72, 0.70], "roughness": 0.9, "metalness": 0.0},
    "plate_mat": {"color": [0.67, 0.78, 0.90], "roughness": 0.12, "metalness": 0.0, "clearcoat": 1.0},
}


def _r(x, n=5):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


def export_mesh_glb(m: mujoco.MjModel, mesh_ids: List[int], out_path: str, strip_prefix: str = ""):
    """Write one GLB whose nodes are the given MuJoCo meshes (geometry only)."""
    scene = trimesh.Scene()
    for mid in mesh_ids:
        name = m.mesh(mid).name
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[va:va + vn]
        F = m.mesh_face[fa:fa + fn]
        # unroll faces so per-face texcoords / normals are kept (MuJoCo stores them per face corner)
        pos = V[F.reshape(-1)]
        na = m.mesh_normaladr[mid]
        nrm = m.mesh_normal[na + m.mesh_facenormal[fa:fa + fn].reshape(-1)]
        uv = None
        if m.mesh_texcoordnum[mid] > 0:
            ta = m.mesh_texcoordadr[mid]
            uv = m.mesh_texcoord[ta + m.mesh_facetexcoord[fa:fa + fn].reshape(-1)].copy()
        faces = np.arange(len(pos)).reshape(-1, 3)
        vis = trimesh.visual.TextureVisuals(uv=uv) if uv is not None else None
        tm = trimesh.Trimesh(vertices=pos, faces=faces, vertex_normals=nrm, visual=vis, process=False)
        scene.add_geometry(tm, node_name=name, geom_name=name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.export(out_path)


def export_model(env, out_root: str, key: str) -> Dict:
    """Kinematic tree + visual geoms for the browser. `key` names the model, e.g. trossen_wxai_bimanual."""
    m = env.m
    robot = env.robot.name
    # ---- meshes: one GLB per robot (shared by both arms), objects use the Poly Haven GLBs ----
    robot_mesh_ids, prefix = [], f"{env.sides[0]}/"
    for g in range(m.ngeom):
        if m.geom_group[g] == 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            if m.body(m.geom_bodyid[g]).name.startswith(prefix):
                robot_mesh_ids.append(int(m.geom_dataid[g]))
    glb = os.path.join(out_root, "assets", "robots", f"{robot}.glb")
    export_mesh_glb(m, sorted(set(robot_mesh_ids)), glb, strip_prefix=prefix)

    bodies = []
    for b in range(m.nbody):
        jnts = []
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            if j < 0:
                continue
            jnts.append({"type": JT[int(m.jnt_type[j])], "axis": _r(m.jnt_axis[j]), "pos": _r(m.jnt_pos[j]),
                         "qadr": int(m.jnt_qposadr[j])})
        bodies.append({"name": m.body(b).name, "parent": int(m.body_parentid[b]), "pos": _r(m.body_pos[b]),
                       "quat": _r(m.body_quat[b]), "joints": jnts})
    geoms = []
    for g in range(m.ngeom):
        grp = int(m.geom_group[g])
        if grp not in (0, 1, 2) or m.geom_rgba[g][3] == 0:
            continue
        bname = m.body(m.geom_bodyid[g]).name
        gt = GT.get(int(m.geom_type[g]))
        if gt is None:
            continue
        mat = m.mat(m.geom_matid[g]).name if m.geom_matid[g] >= 0 else None
        if mat and "/" in mat:
            mat = mat.split("/", 1)[1]
        entry = {"body": int(m.geom_bodyid[g]), "type": gt, "group": grp, "size": _r(m.geom_size[g]),
                 "pos": _r(m.geom_pos[g]), "quat": _r(m.geom_quat[g]), "material": mat,
                 "rgba": _r(m.geom_rgba[g], 3), "name": m.geom(g).name}
        if gt == "mesh":
            if bname.startswith("bread_"):
                kind = env.meta["objects"][bname]["kind"]
                entry.update({"asset": f"assets/objects/{kind}.glb", "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                              "yup": True})
            elif bname == "plate":
                entry.update({"asset": "assets/objects/plate.glb", "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                              "yup": True})
            else:
                mname = m.mesh(m.geom_dataid[g]).name
                for s in env.sides:
                    if mname.startswith(f"{s}/"):
                        mname = mname[len(s) + 1:]
                entry.update({"asset": f"assets/robots/{robot}.glb", "node": mname})
        geoms.append(entry)

    # textures used by the web materials
    tex_dir = os.path.join(out_root, "assets", "textures")
    os.makedirs(tex_dir, exist_ok=True)
    tsrc = os.path.join(ASSETS, "robots", "trossen_wxai", "assets", "materials", "1k_textures", "trossen_BaseColor.png")
    if os.path.exists(tsrc):
        shutil.copy(tsrc, os.path.join(tex_dir, "trossen_BaseColor.png"))

    tool_sites = {}
    for s in env.sides:
        sid = m.site(f"{s}/{env.robot.tool_site}").id
        tool_sites[s] = {"body": int(m.site_bodyid[sid]), "pos": _r(m.site_pos[sid]), "quat": _r(m.site_quat[sid])}
    model = {
        "key": key, "robot": robot, "robot_label": env.robot.label, "camera": env.cam.name,
        "camera_label": env.cam.label, "mode": env.cfg.get("mode"), "sides": env.sides, "nq": int(m.nq),
        "bodies": bodies, "geoms": geoms, "materials": WEB_MATERIALS, "tool_sites": tool_sites,
        "policy_camera": {"T_cam_in_world": _r(env.T_cam_in_world), "K": _r(env.K), "width": env.W, "height": env.H},
        "objects": {k: {"qadr": int(env.obj_free_qadr[k])} for k in env.obj_bodies},
    }
    p = os.path.join(out_root, "data", "models", f"{key}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    return model


def export_rollout(env, log_dir: str, out_root: str, rid: str, scene_meta: Optional[dict] = None) -> Dict:
    """Recorded episode -> data/rollouts/<rid>.json (+ policy-view mp4)."""
    with open(os.path.join(log_dir, "episode.json")) as f:
        ep = json.load(f)
    m, d = env.m, env.d
    frames = []
    for fr in ep["frames"]:
        if "qfull" in fr:
            q = fr["qfull"]
            frames.append([round(fr["t"], 3)] + [round(float(v), 4) for v in q])
            continue
        q = np.array(d.qpos)  # template
        for s, qs in fr["qpos"].items():
            arm = env.arms[s]
            q[arm.qadr] = qs[:6]
            for j in range(m.njnt):
                nm = m.joint(j).name
                if nm.startswith(f"{s}/") and "carriage" in nm:
                    q[m.jnt_qposadr[j]] = qs[6]
        for name, pq in fr["objects"].items():
            qa = env.obj_free_qadr[name]
            q[qa:qa + 7] = pq
        frames.append([round(fr["t"], 3)] + [round(float(v), 4) for v in q])
    pol = []
    for st in ep["policy"]:
        e = {"t": round(st["t"], 3), "sides": {}}
        for s, v in st["sides"].items():
            e["sides"][s] = {"traj": [[round(x, 3) for x in p] for p in v["traj_world"]],
                             "grasp": v["grasp"], "done": v["done"], "grasping": v["grasping"]}
        pol.append(e)
    # policy-view video: one tile per side, side-by-side, one frame per policy step
    vid_rel = None
    img_dir = os.path.join(log_dir, "policy_view")
    if os.path.isdir(img_dir) and ep["policy"]:
        vid_rel = f"data/rollouts/{rid}_policy.mp4"
        vp = os.path.join(out_root, vid_rel)
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        sides = env.sides
        P = ep["policy"][1]["t"] - ep["policy"][0]["t"] if len(ep["policy"]) > 1 else 0.3
        w = imageio.get_writer(vp, fps=round(1.0 / P, 4), codec="libx264", quality=6, macro_block_size=8,
                               ffmpeg_params=["-g", "6", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        last = {s: np.zeros((240, 320, 3), np.uint8) for s in sides}
        for st in ep["policy"]:
            for s in sides:
                e = st["sides"].get(s)
                if e and e.get("image"):
                    im = cv2.imread(os.path.join(img_dir, e["image"]))
                    if im is not None:
                        last[s] = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
            tile = np.hstack([last[s] for s in sides])
            w.append_data(tile[:, :, ::-1])
        w.close()
    meta = ep["meta"]
    out = {
        "id": rid, "model": None, "fps": 30, "frames": frames, "policy": pol, "policy_video": vid_rel,
        "policy_period": round(P, 4) if ep["policy"] else 0.3,
        "result": meta["result"], "scene": meta["scene_init"], "extra": scene_meta or {},
    }
    p = os.path.join(out_root, "data", "rollouts", f"{rid}.json")
    with open(p, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out
