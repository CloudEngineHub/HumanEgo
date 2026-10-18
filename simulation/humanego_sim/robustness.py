# -*- coding: utf-8 -*-
"""
Sensor and scene perturbations for robustness evaluation (README, "Robustness").

Everything here is off unless a config key / scene variation asks for it:

    perception.mask_noise       DINO-SAM-like segmentation errors on the ground-truth masks
    perception.depth_artifacts  RealSense-like flying pixels at depth edges and holes
    perception.calib_error      true camera pose != the hand-eye calibration the robot uses
    perception.latency          camera frame age / jitter / dropped frames for the policy image
    scene["_var"]               lights, visual distractors, table height (sampling.apply_variation)

Each perturbation draws from its own random stream, derived from the episode seed, so switching
one on does not change the noise of another (and the clean depth noise stays bit-identical).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

# ----------------------------------------------------------------------------- defaults
MASK_NOISE = {"enabled": False,
              "bias_px": 2.0,          # per-episode, per-object erosion (-) / dilation (+), uniform +- [px]
              "jitter_px": 1.5,        # per-frame boundary jitter, std of a smooth displacement field [px]
              "jitter_scale_px": 16.0,  # correlation length of the jitter along the boundary [px]
              "p_miss": 0.05,          # per-frame, per-object probability of an empty detection
              "p_blob": 0.10,          # per-frame probability of a spurious blob in the bread mask
              "blob_px": [6, 18],      # blob radius range [px]
              "p_blob_attached": 0.5}  # share of blobs grown out of an object (shadow, plate rim, crumbs)
# per-episode probability that a distractor is segmented as bread, by type: the real DINO-SAM on
# 32 cluttered renders (scripts/robust_mask_calib.py) took the banana for bread 3 / 10 times, the
# yellow box 1 / 13, never the others
DISTRACTOR_CONFUSION = {"enabled": False, "p": {"banana": 0.3, "box": 0.08}}
DEPTH_ARTIFACTS = {"enabled": False,
                   "edge_jump": 0.008,  # depth step [m] between neighbours that makes an edge
                   "edge_px": 2,        # width of the artefact band around an edge [px]
                   "p_flying": 0.5,     # flying pixel in the band: depth between foreground and background
                   "p_edge_hole": 0.25,  # invalid (0) pixel in the band
                   "hole_frac": 0.03,   # share of pixels in random hole blobs
                   "hole_px": 10.0,     # hole blob size scale [px]
                   "hole_persist": 0.7,  # how much of the hole pattern is fixed over the episode (0..1)
                   "scale_std": 0.005}  # per-episode depth scale error (std; D405 spec: <= 2 % at 50 cm)
CALIB_ERROR = {"enabled": False,
               "rot_deg": 0.3,          # per-axis std of the camera rotation error [deg]
               "trans_m": 0.003}        # per-axis std of the camera translation error [m]
LATENCY = {"enabled": False,
           "image_ms": 66.0,            # age of the camera frame behind the policy image [ms]
           "jitter_ms": 33.0,           # + uniform [0, jitter] [ms]
           "p_drop": 0.05}              # frame not refreshed: the policy sees the previous image


def params(cfg: dict, key: str, defaults: dict) -> dict:
    """perception.<key> merged over its defaults; {} or missing -> disabled."""
    p = dict(defaults)
    p.update((cfg.get("perception", {}) or {}).get(key, {}) or {})
    return p


def child_rng(rng: np.random.Generator, tag: int) -> np.random.Generator:
    """An independent stream derived from the episode rng's seed WITHOUT consuming its draws."""
    ss = rng.bit_generator.seed_seq
    return np.random.default_rng([int(x) for x in np.atleast_1d(ss.entropy)] + list(ss.spawn_key) + [7919, int(tag)])


def smooth_field(rng: np.random.Generator, shape, scale_px: float) -> np.ndarray:
    """Zero-mean, unit-std Gaussian random field with correlation length ~scale_px."""
    h, w = shape
    f = max(1.0, scale_px / 2.0)
    hs, ws = max(2, int(math.ceil(h / f))), max(2, int(math.ceil(w / f)))
    g = rng.standard_normal((hs, ws)).astype(np.float32)
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=max(0.5, scale_px / 4.0))
    s = float(g.std())
    return (g - float(g.mean())) / (s if s > 1e-6 else 1.0)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask boundary [px], positive inside."""
    m = (mask > 0).astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return din - dout


# ----------------------------------------------------------------------------- masks
class MaskNoise:
    """Per-episode segmentation error model. Persistent part (bias per object, which distractors are
    taken for bread) is drawn once; transient part (jitter, misses, blobs) on every detector frame."""

    def __init__(self, p: dict, rng: np.random.Generator, objects: List[str]):
        self.p = p
        self.rng = rng
        self.bias = {o: float(rng.uniform(-1, 1)) * float(p["bias_px"]) for o in objects}

    def instance(self, name: str, gt: np.ndarray) -> np.ndarray:
        """One detector frame's mask of one object instance."""
        p, rng = self.p, self.rng
        if gt.max() == 0 or rng.random() < float(p["p_miss"]):
            return np.zeros_like(gt)
        sd = signed_distance(gt)
        level = sd + self.bias.get(name, 0.0)
        if float(p["jitter_px"]) > 0:
            level = level + float(p["jitter_px"]) * smooth_field(rng, gt.shape, float(p["jitter_scale_px"]))
        out = (level > 0).astype(np.uint8) * 255
        # keep it one blob, as SAM returns for one box
        n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n > 2:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            out = (lab == k).astype(np.uint8) * 255
        return out

    def blob(self, shape, table_rows: int, anchors: List[np.ndarray]) -> Optional[np.ndarray]:
        """A spurious blob (or None): grown out of an object, or somewhere on the table."""
        p, rng = self.p, self.rng
        if rng.random() >= float(p["p_blob"]):
            return None
        h, w = shape
        r = float(rng.uniform(*p["blob_px"]))
        anchors = [a for a in anchors if a.max() > 0]
        if anchors and rng.random() < float(p["p_blob_attached"]):
            a = anchors[int(rng.integers(len(anchors)))]
            cs, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea).reshape(-1, 2)
            u, v = c[int(rng.integers(len(c)))]
            m = cv2.moments(a, binaryImage=True)
            cu, cv_ = m["m10"] / max(m["m00"], 1), m["m01"] / max(m["m00"], 1)
            d = np.array([u - cu, v - cv_], float)
            d /= max(np.linalg.norm(d), 1e-6)
            cen = np.array([u, v], float) + 0.6 * r * d
        else:
            cen = np.array([rng.uniform(0.05, 0.95) * w, rng.uniform(table_rows, h - 10)], float)
        out = np.zeros(shape, np.uint8)
        axes = (max(2, int(r)), max(2, int(r * rng.uniform(0.4, 1.0))))
        cv2.ellipse(out, (int(cen[0]), int(cen[1])), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
        return out


# ----------------------------------------------------------------------------- depth
class DepthArtifacts:
    def __init__(self, p: dict, rng: np.random.Generator, shape):
        self.p = p
        self.rng = rng
        self.scale = 1.0 + float(rng.normal(0.0, float(p.get("scale_std", 0.0))))
        self.persist = smooth_field(rng, shape, float(p["hole_px"])) if float(p["hole_frac"]) > 0 else None

    def __call__(self, z: np.ndarray, z_clean: np.ndarray) -> np.ndarray:
        """z: noisy depth [m] (0 = invalid); z_clean: the noise-free render it came from (finds the edges)."""
        p, rng = self.p, self.rng
        z = z * np.float32(self.scale)
        z_clean = z_clean * np.float32(self.scale)
        k = np.ones((3, 3), np.uint8)
        zmax = cv2.dilate(z_clean, k)
        zmin = cv2.erode(z_clean, k)
        edge = ((zmax - zmin) > float(p["edge_jump"])).astype(np.uint8)
        if int(p["edge_px"]) > 1:
            edge = cv2.dilate(edge, np.ones((2 * int(p["edge_px"]) - 1,) * 2, np.uint8))
        band = edge > 0
        u = rng.random(z.shape, dtype=np.float32)
        fly = band & (u < float(p["p_flying"]))
        hole = band & (u >= float(p["p_flying"])) & (u < float(p["p_flying"]) + float(p["p_edge_hole"]))
        lo, hi = cv2.erode(z_clean, np.ones((5, 5), np.uint8)), cv2.dilate(z_clean, np.ones((5, 5), np.uint8))
        a = rng.random(z.shape, dtype=np.float32)
        z[fly] = (lo + a * (hi - lo))[fly]
        z[hole] = 0.0
        if self.persist is not None:
            w = float(p["hole_persist"])
            f = w * self.persist + (1 - w) * smooth_field(rng, z.shape, float(p["hole_px"]))
            f /= max(float(f.std()), 1e-6)
            thr = float(np.quantile(f, 1.0 - float(p["hole_frac"])))
            z[f > thr] = 0.0
        return z


# ----------------------------------------------------------------------------- calibration
def camera_error(rng: np.random.Generator, p: dict) -> np.ndarray:
    """Rigid error of the true camera relative to the calibrated one, in the (OpenCV) camera frame."""
    dT = np.eye(4)
    dT[:3, :3] = R.from_rotvec(np.radians(rng.normal(0.0, float(p["rot_deg"]), 3))).as_matrix()
    dT[:3, 3] = rng.normal(0.0, float(p["trans_m"]), 3)
    return dT


def set_policy_camera(env, dT: Optional[np.ndarray]) -> None:
    """Move the rendering camera to calibrated pose (+) dT; the robot keeps using the calibration
    (arm T_base_in_cam, env.T_cam_in_world). dT None restores the calibrated pose."""
    m = env.m
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "policy_cam")
    if not hasattr(env, "_cam_nominal"):
        if dT is None:
            return
        env._cam_nominal = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
    pos0, quat0 = env._cam_nominal
    if dT is None:
        m.cam_pos[cid], m.cam_quat[cid] = pos0, quat0
    else:
        D = np.diag([1.0, -1.0, -1.0])
        Rm0 = R.from_quat(quat0[[1, 2, 3, 0]]).as_matrix()          # MuJoCo camera frame in world
        Rcv = Rm0 @ D                                               # OpenCV camera frame in world
        Rm = Rcv @ dT[:3, :3] @ D
        x, y, z, w = R.from_matrix(Rm).as_quat()
        m.cam_pos[cid] = pos0 + Rcv @ dT[:3, 3]
        m.cam_quat[cid] = [w, x, y, z]
    mujoco.mj_camlight(m, env.d)


# ----------------------------------------------------------------------------- scene variation (runtime)
def set_lights(env, light: Optional[dict]) -> None:
    """Scale / tint the scene lights and tilt the key light (light None restores the built scene)."""
    m = env.m
    if not hasattr(env, "_light_nominal"):
        if not light:
            return
        env._light_nominal = (m.light_diffuse.copy(), m.light_specular.copy(), m.light_dir.copy(),
                              m.light_pos.copy(), m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
    dif, spec, ldir, lpos, hd, ha = env._light_nominal
    m.light_diffuse[:], m.light_specular[:], m.light_dir[:], m.light_pos[:] = dif, spec, ldir, lpos
    m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = hd, ha
    if light:
        s, t = float(light.get("intensity", 1.0)), float(light.get("tint", 0.0))
        tint = np.array([1.0 + t, 1.0, 1.0 - t])                   # warm (+) / cool (-)
        m.light_diffuse[:] = np.clip(dif * s * tint, 0, 1)
        m.light_specular[:] = np.clip(spec * s, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(hd * s * tint, 0, 1)
        m.vis.headlight.ambient[:] = np.clip(ha * s * tint, 0, 1)
        if float(light.get("dir_deg", 0.0)) > 0 and m.nlight > 0:
            az = math.radians(float(light.get("dir_azim", 0.0)))
            axis = np.array([math.cos(az), math.sin(az), 0.0])
            rot = R.from_rotvec(axis * math.radians(float(light["dir_deg"])))
            m.light_dir[0] = rot.apply(ldir[0])
            m.light_pos[0] = rot.apply(lpos[0])
    mujoco.mj_camlight(m, env.d)


def add_distractor_geoms(scn, distractors: List[dict]) -> None:
    """Append visual-only distractor geoms to an mjvScene (after update_scene). They show in RGB and
    depth renders; in segmentation renders they carry no object label (objid -1)."""
    for d in distractors or []:
        if scn.ngeom >= scn.maxgeom:
            return
        gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                 "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[d["geom"]]
        size = list(d["size"]) + [0.0] * (3 - len(d["size"]))
        if d["geom"] == "sphere":
            size = [size[0]] * 3
            zc = size[0]
        elif d["geom"] == "cylinder":
            zc = size[1]
            size = [size[0], size[1], 0.0]
        else:
            zc = size[2]
        mat = R.from_euler("z", float(d["yaw"]), degrees=True).as_matrix()
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.array([d["xy"][0], d["xy"][1], zc + 0.0005]),
                            mat.reshape(-1), np.asarray(d["rgba"], np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_STATIC
        g.segid = scn.ngeom
        scn.ngeom += 1


def distractor_masks(env, distractors: List[dict]) -> List[np.ndarray]:
    """Image masks of the distractors (for the 'distractor segmented as bread' error)."""
    out = []
    for d in distractors or []:
        r = env.renderer
        r.enable_segmentation_rendering()
        r.update_scene(env.d, camera="policy_cam")
        n0 = r.scene.ngeom
        add_distractor_geoms(r.scene, [d])
        r.scene.geoms[n0].objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        r.scene.geoms[n0].objid = 999999
        seg = r.render()
        r.disable_segmentation_rendering()
        out.append(((seg[:, :, 0] == 999999)).astype(np.uint8) * 255)
    return out


# ----------------------------------------------------------------------------- report conditions
# "Realistic" levels used by scripts/robust_report.py (README, "Robustness" explains each choice).
# set: perception overrides (--set), variation: scene variation (sampling.apply_variation).
# Masks are fused over perception.mask_frames frames (default 5); add --set perception.mask_frames=1
# for the released single-frame pipeline.
CONDITIONS = {
    "clean": {},
    "masks": {"set": {"perception": {"mask_noise": {"enabled": True}}}},
    "dinosam": {"set": {"perception": {"mask_source": "dinosam"}}},
    "depth": {"set": {"perception": {"depth_artifacts": {"enabled": True}}}},
    "calib": {"set": {"perception": {"calib_error": {"enabled": True}}}},
    "latency": {"set": {"perception": {"latency": {"enabled": True}}}},
    "bread_size": {"variation": {"bread_scale": 0.12}},
    "layout": {"variation": {"bread_yaw": 15, "plate_xy": 0.03}},
    "table_height": {"variation": {"table_dz": 0.01}},
    "lighting": {"variation": {"light": {"intensity": 0.35, "tint": 0.12, "dir_deg": 25}}},
    "distractors": {"variation": {"distractors": {"n": [1, 3]}},
                    "set": {"perception": {"distractor_confusion": {"enabled": True}}}},
}


def combined_condition(names=None) -> dict:
    """All perturbations of `names` (default: every non-clean condition) at once."""
    from .env import deep_update
    out = {"set": {}, "variation": {}}
    for n in names or [k for k in CONDITIONS if k not in ("clean", "dinosam")]:
        c = CONDITIONS[n]
        out["set"] = deep_update(out["set"], c.get("set", {}))
        out["variation"] = deep_update(out["variation"], c.get("variation", {}))
    return out


def condition(name: str) -> dict:
    if name == "all":
        return combined_condition()
    if "+" in name:
        return combined_condition(name.split("+"))
    return CONDITIONS[name]
