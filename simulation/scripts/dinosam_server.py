# -*- coding: utf-8 -*-
"""
DINO-SAM mask server: the released preprocess/DINOSAM.py engine (Grounding-DINO tiny + SAM2 tiny,
cfg/preprocess/base/DINOSAM.yaml) behind a pipe, so the simulator can use the real open-vocabulary
segmenter while its own environment lacks transformers / sam2. Started by
humanego_sim/dinosam_client.py in an environment that has them (HUMANEGO_DINOSAM_PYTHON,
default ~/miniconda3/envs/v2d/bin/python).

Protocol on stdin / stdout: a JSON header line, then raw bytes.
    request   {"shape": [H, W, 3], "prompts": ["piece of bread .", ...]} + H*W*3 uint8 (BGR)
    response  {"masks": [{"conf": c, "n_boxes": k}, ...]} + one H*W uint8 mask (0/255) per prompt
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

proto_out = os.fdopen(os.dup(1), "wb")      # protocol channel; everything printed goes to stderr
os.dup2(2, 1)
proto_in = sys.stdin.buffer

try:
    import box  # noqa: F401
except ImportError:                          # utils.utils_io only needs ConfigBox's attribute access
    class _ConfigBox(dict):
        def __getattr__(self, k):
            v = self[k]
            return _ConfigBox(v) if isinstance(v, dict) else v
    sys.modules["box"] = types.SimpleNamespace(ConfigBox=_ConfigBox)

import numpy as np  # noqa: E402

from preprocess.DINOSAM import DINOSAMEngine  # noqa: E402
from utils.utils_io import load_cfg  # noqa: E402


def main():
    cfg = load_cfg(os.path.join(REPO, "cfg", "preprocess", "base", "DINOSAM.yaml"))
    if os.environ.get("HUMANEGO_DINOSAM_BOX_THRESHOLD"):
        cfg["box_threshold"] = float(os.environ["HUMANEGO_DINOSAM_BOX_THRESHOLD"])
    eng = DINOSAMEngine(cfg)
    proto_out.write(b'{"ready": true}\n')
    proto_out.flush()
    while True:
        line = proto_in.readline()
        if not line:
            return
        req = json.loads(line)
        h, w, c = req["shape"]
        buf = proto_in.read(h * w * c)
        img = np.frombuffer(buf, np.uint8).reshape(h, w, c)
        eng.predictor.set_image(img)
        metas, outs = [], []
        for pr in req["prompts"]:
            m, conf, boxes, _ = eng.predict_frame_internal(img, pr)
            m = np.zeros((h, w), np.uint8) if m is None else m.astype(np.uint8)
            metas.append({"conf": float(conf), "n_boxes": 0 if boxes is None else int(len(boxes))})
            outs.append(np.ascontiguousarray(m).tobytes())
        proto_out.write((json.dumps({"masks": metas}) + "\n").encode())
        for o in outs:
            proto_out.write(o)
        proto_out.flush()


if __name__ == "__main__":
    main()
