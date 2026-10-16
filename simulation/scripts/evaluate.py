# -*- coding: utf-8 -*-
"""
Batch success rate of the released policy in simulation.

    python simulation/scripts/evaluate.py --n 32 --workers 8
    python simulation/scripts/evaluate.py --mode single --n 40 --set control.alpha=0.3
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.evaluate import evaluate  # noqa: E402


def parse_sets(sets):
    ov = {}
    for s in sets:
        key, val = s.split("=", 1)
        cur = ov
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
    return ov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--log", default=None, help="save every episode under simulation/runs/<log>/")
    args = ap.parse_args()
    ov = parse_sets(args.set)
    for k in ("mode", "robot", "camera"):
        if getattr(args, k):
            ov[k] = getattr(args, k)
    log_root = os.path.join(os.path.dirname(HERE), "runs", args.log) if args.log else None
    st = evaluate(ov, n_episodes=args.n, workers=args.workers, seed=args.seed, difficulty=args.difficulty,
                  log_root=log_root)
    st.pop("results", None)
    print(json.dumps(st, indent=1))
