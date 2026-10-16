# -*- coding: utf-8 -*-
"""
Camera registry. Each entry reproduces what the real inference stack hands to
the policy: a 640x480 RGB-D frame + pinhole K, after `CamRS`'s centre-crop to
4:3 and resize (1280x720 -> 960x720 crop -> 640x480).

Intrinsics are nominal values derived from each sensor's datasheet field of
view (the released code reads the factory calibration at run time; the sim uses
the same numbers every run so results are reproducible).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CameraSpec:
    name: str
    label: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # simple depth-noise model (metres): sigma = a + b * z^2  (structured light / stereo)
    depth_noise_a: float = 0.0005
    depth_noise_b: float = 0.0015
    depth_min: float = 0.07
    depth_max: float = 3.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def fovy_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / 2.0 / self.fy)))


def _crop_resize(fx_raw, w_raw=1280, h_raw=720, w=640, h=480):
    """Mirror CamRS._get_crop_parameters: centre-crop to w:h aspect, then resize."""
    aspect = w / h
    crop_w = h_raw * aspect if w_raw / h_raw > aspect else w_raw
    scale = w / crop_w
    return fx_raw * scale


CAMERAS: Dict[str, CameraSpec] = {
    # Intel RealSense D405 (paper hardware): ~87 deg HFOV at 1280x720
    "realsense_d405": CameraSpec("realsense_d405", "Intel RealSense D405", 640, 480,
                                 *(2 * [_crop_resize(660.0)]), 320.0, 240.0,
                                 depth_noise_a=0.0003, depth_noise_b=0.0010, depth_min=0.07),
    # Intel RealSense D435 colour: ~69 deg HFOV at 1280x720
    "realsense_d435": CameraSpec("realsense_d435", "Intel RealSense D435", 640, 480,
                                 *(2 * [_crop_resize(910.0)]), 320.0, 240.0,
                                 depth_noise_a=0.0008, depth_noise_b=0.0025, depth_min=0.20),
    # Stereolabs ZED 2i (2.1 mm lens), rectified HD720
    "zed2i": CameraSpec("zed2i", "Stereolabs ZED 2i", 640, 480,
                        *(2 * [_crop_resize(527.0)]), 320.0, 240.0,
                        depth_noise_a=0.0010, depth_noise_b=0.0040, depth_min=0.30),
    # Stereolabs ZED Mini, rectified HD720
    "zed_mini": CameraSpec("zed_mini", "Stereolabs ZED Mini", 640, 480,
                           *(2 * [_crop_resize(700.0)]), 320.0, 240.0,
                           depth_noise_a=0.0010, depth_noise_b=0.0040, depth_min=0.10),
}


def get_camera(name: str) -> CameraSpec:
    if name not in CAMERAS:
        raise KeyError(f"unknown camera '{name}'. Available: {list(CAMERAS)}")
    return CAMERAS[name]
