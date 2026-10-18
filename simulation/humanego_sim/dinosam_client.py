# -*- coding: utf-8 -*-
"""Client for scripts/dinosam_server.py: the real DINO-SAM segmenter on rendered frames
(perception.mask_source: dinosam). One server process per simulator process, started lazily."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT)
DEFAULT_PY = os.path.expanduser("~/miniconda3/envs/v2d/bin/python")
# the serve_bread task prompts of the released pipeline (cfg/preprocess/tasks/serve_bread.yaml)
PROMPTS = {"obj1": "piece of bread .", "obj2": "a plate ."}

_CLIENT: Optional["DinoSamClient"] = None


class DinoSamClient:
    def __init__(self, python: Optional[str] = None, log: Optional[str] = None):
        py = python or os.environ.get("HUMANEGO_DINOSAM_PYTHON", DEFAULT_PY)
        self.log = open(log or os.devnull, "ab")
        self.proc = subprocess.Popen([py, os.path.join(SIM_ROOT, "scripts", "dinosam_server.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=REPO_ROOT)
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError(f"DINO-SAM server failed to start ({py}); see HUMANEGO_DINOSAM_PYTHON")
        atexit.register(self.close)

    def segment(self, bgr: np.ndarray, prompts: Dict[str, str] = None) -> Dict[str, np.ndarray]:
        prompts = prompts or PROMPTS
        keys = list(prompts)
        h, w, c = bgr.shape
        hdr = {"shape": [h, w, c], "prompts": [prompts[k] for k in keys]}
        self.proc.stdin.write((json.dumps(hdr) + "\n").encode())
        self.proc.stdin.write(np.ascontiguousarray(bgr, np.uint8).tobytes())
        self.proc.stdin.flush()
        meta = json.loads(self.proc.stdout.readline())
        out = {}
        for k in keys:
            out[k] = np.frombuffer(self.proc.stdout.read(h * w), np.uint8).reshape(h, w).copy()
        self.last_meta = dict(zip(keys, meta["masks"]))
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def client() -> DinoSamClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = DinoSamClient()
    return _CLIENT
