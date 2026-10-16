# -*- coding: utf-8 -*-
"""
Tunable parameters of the simulator and of the low-level controller.

Each `ParamSpec` maps a scalar to a dotted config path. Bounds are kept inside
physically / operationally plausible ranges: the tuner may make the controller
better or the soft-bread proxy stronger, but it may not make bread weightless
or teleport the arm.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ParamSpec:
    name: str              # dotted config path, e.g. "control.alpha"
    lo: float
    hi: float
    init: float
    kind: str = "float"    # float | int | log
    group: str = "control"  # control | physics
    note: str = ""

    def to_unit(self, v: float) -> float:
        if self.kind == "log":
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "log":
            v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + u * (self.hi - self.lo)
        if self.kind == "int":
            v = int(round(v))
        return v


DEFAULT_SPACE: List[ParamSpec] = [
    # ---- low-level controller (cfg/inference/serve_bread/Control.yaml knobs) ----
    ParamSpec("control.alpha", 0.0, 0.85, 0.5, note="position EMA (higher = smoother/laggier)"),
    ParamSpec("control.alpha_rot", 0.0, 0.85, 0.0, note="rotation Slerp smoothing"),
    ParamSpec("control.look_ahead_offset", 10, 49, 45, "int", note="chunk index the servo starts from"),
    ParamSpec("control.step_stride", 1, 4, 2, "int", note="chunk steps advanced per control tick"),
    ParamSpec("control.duration_smooth_param", 2.0, 16.0, 12.0, note="driver goal_time = dt * this"),
    ParamSpec("control.max_pos_step", 0.02, 0.15, 0.08, "log", note="safety cage, m per tick"),
    ParamSpec("control.max_rot_step", 0.01, 0.30, 0.02, "log", note="safety cage, rad per tick"),
    ParamSpec("control.grasp_threshold", 0.35, 0.72, 0.6, note="grasp prob -> close (policy peaks ~0.7)"),
    ParamSpec("control.idle_time_limit", 0.5, 3.5, 1.5, note="s holding near the plate before release"),
    ParamSpec("control.idle_dist_threshold", 0.04, 0.18, 0.13, note="hand-to-plate distance for release"),
    ParamSpec("control.forced_lift_height", 0.0, 0.10, 0.05, note="post-grasp lift bias, m"),
    # ---- simulator physics (soft-bread proxy, contact) ----
    ParamSpec("physics.finger_adhesion", 0.0, 15.0, 6.0, group="physics", note="N per finger while closed"),
    ParamSpec("physics.bread_friction.0", 0.8, 3.0, 2.0, group="physics", note="effective sliding friction"),
    ParamSpec("physics.gripper_force", 6.0, 35.0, 12.0, group="physics", note="N"),
]


def space_from_names(names: Optional[List[str]] = None) -> List[ParamSpec]:
    if not names:
        return list(DEFAULT_SPACE)
    by = {p.name: p for p in DEFAULT_SPACE}
    return [by[n] for n in names]


def params_to_overrides(params: Dict[str, float], base_cfg: dict) -> dict:
    """{'control.alpha': 0.3, 'physics.bread_friction.0': 1.5} -> nested overrides dict."""
    ov: dict = {}
    for k, v in params.items():
        parts = k.split(".")
        if parts[-1].isdigit():                     # list element, e.g. bread_friction.0
            idx = int(parts[-1])
            cur_base = base_cfg
            for p in parts[:-1]:
                cur_base = cur_base[p]
            lst = list(cur_base)
            lst[idx] = v
            parts, v = parts[:-1], lst
        cur = ov
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return ov


def spec_dict(space: List[ParamSpec]) -> List[dict]:
    return [asdict(p) for p in space]
