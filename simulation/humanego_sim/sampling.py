# -*- coding: utf-8 -*-
"""
Initial-condition sampling, following the paper's serve_bread protocol
(App. "Task Details"): plate and bread are placed independently, their
horizontal offset spans [0, 50] cm, and their depths along the table's
front-back axis vary so the pair is generally not colinear.

World frame: x forward (depth), y left. Bread yaw ~90 deg = long axis
left-right in the camera image, as in the human demonstrations.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# workspace limits (world frame, metres) that keep objects in the camera view
# and inside the WidowX AI's reach from the calibrated bases
X_RANGE = (0.36, 0.56)


def sample_scene(rng: np.random.Generator, mode: str = "bimanual", difficulty: float = 1.0) -> Dict[str, List[float]]:
    """difficulty in [0, 1] scales the randomization around the nominal layout."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    if mode == "single":
        plate_y = 0.11 + rng.uniform(-0.06, 0.06) * d
        gap = 0.26 + rng.uniform(-0.10, 0.12) * d            # centre-to-centre, ~16-38 cm
        bread_y = plate_y - gap
        return {
            "plate": [0.47 + rng.uniform(-0.07, 0.07) * d, plate_y, rng.uniform(0, 360)],
            "bread_right": [0.47 + rng.uniform(-0.08, 0.08) * d, bread_y, 90 + rng.uniform(-35, 35) * d],
        }
    plate = [0.47 + rng.uniform(-0.06, 0.06) * d, rng.uniform(-0.05, 0.05) * d, rng.uniform(0, 360)]
    out = {"plate": plate}
    for s, sign in (("right", -1.0), ("left", 1.0)):
        gap = 0.27 + rng.uniform(-0.07, 0.08) * d
        out[f"bread_{s}"] = [0.46 + rng.uniform(-0.08, 0.08) * d, plate[1] + sign * gap,
                             90 + rng.uniform(-35, 35) * d]
    return out


def scene_grid(mode: str = "bimanual"):
    """Deterministic grid of initial conditions used by the web viewer."""
    out = []
    if mode == "single":
        for bx in (0.40, 0.47, 0.54):
            for gap in (0.18, 0.26, 0.34):
                for yaw in (70, 90, 110):
                    out.append({"plate": [0.47, 0.11, 0.0], "bread_right": [bx, 0.11 - gap, yaw]})
        return out
    for bx in (0.40, 0.47, 0.54):
        for gap in (0.22, 0.28, 0.34):
            for yaw in (75, 105):
                out.append({"plate": [0.47, 0.0, 0.0],
                            "bread_right": [bx, -gap, yaw],
                            "bread_left": [bx, gap, 180 - yaw]})
    return out


# ----------------------------------------------------------------------------- scene variation
# Off unless asked for. Every variation draws from its own random stream (seed, scene, kind), so
# the base layout of scene i is identical with and without variation, and enabling one kind of
# variation never changes the draws of another: conditions stay paired scene by scene.
#
#   bread_scale  0.12             each bread uniformly scaled by [1 - s, 1 + s] (scene.bread.<side>.scale)
#   bread_yaw    15               extra bread yaw, deg, uniform +-
#   plate_xy     0.03             extra plate offset per axis, m, uniform +-
#   table_dz     0.01             table top height relative to the rig, m, uniform +- (the robot assumes 0)
#   light        {"intensity": 0.35, "tint": 0.12, "dir_deg": 25}
#                                 light intensity x [1 - i, 1 + i], warm/cool tint, key-light tilt
#   distractors  {"n": [1, 3]}    visual-only clutter on the free table behind the task area
VAR_KINDS = ("bread_scale", "bread_yaw", "plate_xy", "table_dz", "light", "distractors")

DISTRACTOR_TYPES = {
    # type: (geom, size (MuJoCo half-sizes, m), candidate colours)
    "mug": ("cylinder", [0.040, 0.048], [[0.92, 0.92, 0.90], [0.20, 0.35, 0.60], [0.75, 0.20, 0.15]]),
    "apple": ("sphere", [0.038], [[0.75, 0.10, 0.08], [0.55, 0.75, 0.20]]),
    "banana": ("ellipsoid", [0.085, 0.020, 0.018], [[0.95, 0.80, 0.25]]),
    "knife": ("box", [0.095, 0.012, 0.003], [[0.70, 0.70, 0.72]]),
    "napkin": ("box", [0.075, 0.075, 0.002], [[0.95, 0.95, 0.93], [0.85, 0.30, 0.30]]),
    "box": ("box", [0.045, 0.030, 0.035], [[0.30, 0.55, 0.30], [0.85, 0.65, 0.20]]),
}


def _var_rng(seed: int, i: int, kind: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), 991, int(i), VAR_KINDS.index(kind)])


def _sample_distractors(rng, spec: dict, scene: dict) -> List[dict]:
    lo, hi = spec.get("n", [1, 3])
    n = int(rng.integers(lo, hi + 1))
    types = spec.get("types", list(DISTRACTOR_TYPES))
    keep_out = [(np.asarray(v[:2]), 0.20 if k == "plate" else 0.15) for k, v in scene.items() if not k.startswith("_")]
    out = []
    for _ in range(200):
        if len(out) >= n:
            break
        x, y = rng.uniform(0.64, 0.86), rng.uniform(-0.45, 0.45)
        p = np.array([x, y])
        if any(np.linalg.norm(p - c) < r for c, r in keep_out):
            continue
        if any(np.linalg.norm(p - np.array(d["xy"])) < 0.12 for d in out):
            continue
        t = types[int(rng.integers(len(types)))]
        g, size, cols = DISTRACTOR_TYPES[t]
        out.append({"type": t, "geom": g, "size": list(size), "xy": [float(x), float(y)],
                    "yaw": float(rng.uniform(0, 360)), "rgba": list(cols[int(rng.integers(len(cols)))]) + [1.0]})
    return out


def apply_variation(scene: dict, seed: int, i: int, variation: dict, mode: str = "bimanual") -> dict:
    """Scene i of scene set `seed` with the requested variation (see VAR_KINDS). Returns a new dict;
    non-pose settings go under scene["_var"] (env.reset ignores keys that are not objects)."""
    sc = {k: list(v) if isinstance(v, list) else v for k, v in scene.items()}
    var = {}
    breads = sorted(k for k in sc if k.startswith("bread_"))
    if variation.get("bread_scale"):
        r, s = _var_rng(seed, i, "bread_scale"), float(variation["bread_scale"])
        scales = {b[6:]: round(float(r.uniform(1 - s, 1 + s)), 2) for b in ("bread_right", "bread_left")}
        if mode == "single":
            var["cfg"] = {"scene": {"single": {"bread": {"scale": scales["right"]}}}}
        else:
            var["cfg"] = {"scene": {"bread": {side: {"scale": scales[side]} for side in ("right", "left")}}}
        var["bread_scale"] = {side: scales[side] for side in ([b[6:] for b in breads])}
    if variation.get("bread_yaw"):
        r = _var_rng(seed, i, "bread_yaw")
        d = [float(r.uniform(-1, 1)) * float(variation["bread_yaw"]) for _ in range(2)]
        for j, b in enumerate(("bread_right", "bread_left")):
            if b in sc:
                sc[b][2] = float(sc[b][2]) + d[j]
    if variation.get("plate_xy"):
        r, a = _var_rng(seed, i, "plate_xy"), float(variation["plate_xy"])
        dx, dy = r.uniform(-a, a, 2)
        sc["plate"][0] = float(sc["plate"][0]) + float(dx)
        sc["plate"][1] = float(sc["plate"][1]) + float(dy)
    if variation.get("table_dz"):
        r, a = _var_rng(seed, i, "table_dz"), float(variation["table_dz"])
        var["table_dz"] = round(float(r.uniform(-a, a)), 3)          # 1 mm steps: few distinct scenes to build
    if variation.get("light"):
        r, L = _var_rng(seed, i, "light"), variation["light"]
        var["light"] = {"intensity": float(1.0 + r.uniform(-1, 1) * float(L.get("intensity", 0.0))),
                        "tint": float(r.uniform(-1, 1) * float(L.get("tint", 0.0))),
                        "dir_deg": float(r.uniform(0, 1) * float(L.get("dir_deg", 0.0))),
                        "dir_azim": float(r.uniform(0, 360))}
    if variation.get("distractors"):
        spec = variation["distractors"] if isinstance(variation["distractors"], dict) else {}
        var["distractors"] = _sample_distractors(_var_rng(seed, i, "distractors"), spec, sc)
    if var:
        sc["_var"] = var
    return sc


def sample_scenes(n: int, seed: int, mode: str = "bimanual", difficulty: float = 1.0,
                  variation: dict = None) -> List[dict]:
    """The n scenes of scene set `seed` (what compare_params / autotune_validate evaluate on),
    optionally with scene variation. Without variation this is exactly the historical sampling."""
    rng = np.random.default_rng(seed)
    scenes = [sample_scene(rng, mode, difficulty) for _ in range(n)]
    if variation:
        scenes = [apply_variation(sc, seed, i, variation, mode) for i, sc in enumerate(scenes)]
    return scenes
