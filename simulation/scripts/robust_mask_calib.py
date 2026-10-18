# -*- coding: utf-8 -*-
"""
How far the real DINO-SAM (scripts/dinosam_server.py) is from the rendered ground-truth masks, per
object instance: IoU, signed boundary offset (mean = dilation bias, std = boundary jitter), missed
instances and false-positive blobs. Used to set the synthetic perception.mask_noise levels.

    python simulation/scripts/robust_mask_calib.py --n 32 --seed 7777 --conditions clean lighting distractors
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from robust_perception import _EnvOnly, _rebuild  # noqa: E402
from evaluate import parse_sets  # noqa: E402
from humanego_sim.dinosam_client import client  # noqa: E402
from humanego_sim.env import deep_update, load_config  # noqa: E402
from humanego_sim.robustness import condition, signed_distance  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed  # noqa: E402


def instance_stats(det: np.ndarray, gt: np.ndarray, others: np.ndarray) -> dict:
    """det: detector mask (may hold several objects); gt: this instance; others: the other GT instances."""
    sd = signed_distance(gt)
    win = (sd > -25) & ~((others > 0) & (sd < -2))          # this instance's neighbourhood
    d = (det > 0) & win
    g = gt > 0
    inter, union = int((d & g).sum()), int((d | g).sum())
    cover = inter / max(int(g.sum()), 1)
    if cover < 0.3:
        return {"miss": True, "iou": inter / max(union, 1)}
    cs, _ = cv2.findContours(d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    # + = detector boundary outside the GT (dilation); a boundary pixel of the GT itself has sd = 1
    off = 1.0 - sd[c[:, 1], c[:, 0]]
    return {"miss": False, "iou": inter / max(union, 1), "bias_px": float(np.median(off)),
            "jitter_px": float(np.std(off - np.median(off))), "area_ratio": float(d.sum() / max(g.sum(), 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_seed([args.seed], args.final)
    base_ov = parse_sets(args.set)
    res = {}
    for name in args.conditions:
        c = condition(name)
        ov = deep_update(base_ov, c.get("set", {}))
        cfg = load_config(None, ov)
        scenes = sample_scenes(args.n, args.seed, cfg.get("mode", "bimanual"), variation=c.get("variation") or None)
        holder, key = _EnvOnly(cfg), ""
        rows, fps = [], 0
        for i, sc in enumerate(scenes):
            holder, key = _rebuild(holder, sc, ov, key)
            env = holder.env
            env.reset(sc, settle=0.4)
            gt = env.render_segmentation()
            rgb = env.render_rgb()
            det = client().segment(rgb)
            breads = [k for k in gt if k.startswith("bread")]
            allb = np.zeros_like(gt["plate"])
            for b in breads:
                allb |= gt[b]
            for b in breads:
                others = allb & ~gt[b]
                rows.append({"scene": i, "obj": b, **instance_stats(det["obj1"], gt[b], others)})
            rows.append({"scene": i, "obj": "plate", **instance_stats(det["obj2"], gt["plate"], allb)})
            # false positives: detector components that touch no GT bread
            n, lab, st, _ = cv2.connectedComponentsWithStats((det["obj1"] > 0).astype(np.uint8), connectivity=8)
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] >= 50 and not ((lab == k) & (allb > 0)).any():
                    fps += 1
            if args.dump and i < 6:
                d = os.path.join(os.path.dirname(HERE), "runs", args.dump, name)
                os.makedirs(d, exist_ok=True)
                o = rgb.copy()
                o[det["obj1"] > 0] = (0.5 * o[det["obj1"] > 0] + [0, 0, 127]).astype(np.uint8)
                o[det["obj2"] > 0] = (0.5 * o[det["obj2"] > 0] + [127, 0, 0]).astype(np.uint8)
                for m in list(gt.values()):
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(o, cs, -1, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(d, f"s{i:02d}.jpg"), o)
        holder.env.renderer.close()
        summ = {}
        for kind in ("bread", "plate"):
            r = [x for x in rows if x["obj"].startswith(kind)]
            ok = [x for x in r if not x["miss"]]
            summ[kind] = {"n": len(r), "miss": sum(x["miss"] for x in r),
                          "iou_med": float(np.median([x["iou"] for x in ok])) if ok else None,
                          "iou_p10": float(np.quantile([x["iou"] for x in ok], 0.1)) if ok else None,
                          "bias_med_px": float(np.median([x["bias_px"] for x in ok])) if ok else None,
                          "bias_absp90_px": float(np.quantile([abs(x["bias_px"]) for x in ok], 0.9)) if ok else None,
                          "bias_std_px": float(np.std([x["bias_px"] for x in ok])) if ok else None,
                          "jitter_med_px": float(np.median([x["jitter_px"] for x in ok])) if ok else None}
        summ["false_blobs_per_frame"] = fps / max(len(scenes), 1)
        res[name] = {"summary": summ, "rows": rows}
        print(f"[mask-calib] {name}: {json.dumps(summ)}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
