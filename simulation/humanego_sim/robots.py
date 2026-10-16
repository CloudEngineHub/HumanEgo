# -*- coding: utf-8 -*-
"""
Robot registry for the HumanEgo simulator.

Every robot is described by a `RobotSpec`. The simulator only relies on:
  * a MuJoCo MJCF (from MuJoCo Menagerie) whose root body is attached at the
    calibrated base pose,
  * the 6 (or 7) arm joints + their position actuators,
  * a gripper actuator with its open / closed control values,
  * a tool site at the pinch point, and
  * `R_tool_to_ee`: the rotation from that site's frame to the HumanEgo
    end-effector convention used by the real Trossen driver
    (`RobotArmTrossen.get_p`): +Z = approach, +X = finger-opening direction.

With that convention fixed, `T_align` (EE -> Aria-hand frame) is shared by every
robot, which is exactly the hardware-agnostic contract of the released
`inference/` template.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@dataclass
class RobotSpec:
    name: str
    label: str
    xml: str                       # relative to ASSETS
    root_body: str                 # body to attach at the base pose
    arm_joints: List[str]
    arm_actuators: List[str]
    gripper_actuator: str
    gripper_open: float            # actuator ctrl for fully open
    gripper_closed: float          # actuator ctrl for fully closed
    tool_site: str                 # pinch point
    R_tool_to_ee: np.ndarray       # (3,3) site frame -> HumanEgo EE frame (Z approach, X spread)
    initial_joints: Dict[str, List[float]] = field(default_factory=dict)
    home_joints: Optional[List[float]] = None
    # base mount offset applied on top of the calibrated base pose (e.g. a
    # pedestal for arms that are taller than the WidowX AI)
    mount_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gripper_force: float = 20.0    # N, clamp on the gripper actuator force
    finger_bodies: List[str] = field(default_factory=list)   # bodies that get the soft-bread adhesion proxy
    reach: float = 0.70            # m, nominal reach (for sampling / UI)
    web_name: str = ""


# Real Trossen driver: T_new = T_raw @ C  with  X_new=Y_raw, Y_new=Z_raw, Z_new=X_raw
# (see AriaMimic inference/RobotArmTrossen._get_correction_matrix). The MuJoCo
# `ee_gripper_link` site has link_6's orientation, i.e. the driver's raw frame.
_TROSSEN_C = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

ROBOTS: Dict[str, RobotSpec] = {
    "trossen_wxai": RobotSpec(
        name="trossen_wxai",
        label="Trossen WidowX AI",
        xml="robots/trossen_wxai/wxai_follower.xml",
        root_body="base_link",
        arm_joints=[f"joint_{i}" for i in range(6)],
        arm_actuators=[f"joint_{i}" for i in range(6)],
        gripper_actuator="right_carriage_joint",
        gripper_open=0.044,
        gripper_closed=-0.002,
        tool_site="ee_gripper_link",
        R_tool_to_ee=_TROSSEN_C,
        # cfg/inference/serve_bread/RobotArmTrossen{Left,Right}.yaml (real rig)
        initial_joints={
            "right": [-0.622, 2.511, 1.697, -1.526, -0.936, -2.526],
            "left": [0.470, 2.340, 1.589, -1.283, 0.812, 2.300],
        },
        home_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_force=12.0,
        reach=0.70,
        finger_bodies=["gripper_left", "gripper_right"],
        web_name="trossen_wxai",
    ),
}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise KeyError(f"unknown robot '{name}'. Available: {list(ROBOTS)}")
    return ROBOTS[name]
