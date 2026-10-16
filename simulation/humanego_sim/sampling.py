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
