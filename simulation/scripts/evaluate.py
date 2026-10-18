# -*- coding: utf-8 -*-
"""
Batch success rate of the released policy in simulation.

    python simulation/scripts/evaluate.py --n 32 --workers 8
    python simulation/scripts/evaluate.py --mode single --n 40 --set control.alpha=0.3

Prints per-arm / per-side / both-served success with Wilson 95 % intervals. Scene sets: --seed
7777 / 8888 for development, 9001-9099 only with --final (README, "Evaluation protocol").
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed  # noqa: E402


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
    ap.add_argument("--variation", default=None,
                    help="scene variation JSON (sampling.sample_scenes), e.g. '{\"light\": {\"intensity\": 0.35}}'")
    ap.add_argument("--final", action="store_true", help="allow the reserved seeds 9001-9099 (final reports only)")
    args = ap.parse_args()
    check_seed([args.seed], args.final)
    ov = parse_sets(args.set)
    for k in ("mode", "robot", "camera"):
        if getattr(args, k):
            ov[k] = getattr(args, k)
    log_root = os.path.join(os.path.dirname(HERE), "runs", args.log) if args.log else None
    scenes = None
    if args.variation:
        from humanego_sim.env import load_config
        scenes = sample_scenes(args.n, args.seed, load_config(None, ov).get("mode", "bimanual"), args.difficulty,
                               variation=json.loads(args.variation))
    st = evaluate(ov, n_episodes=args.n, workers=args.workers, seed=args.seed, difficulty=args.difficulty,
                  log_root=log_root, scenes=scenes)
    st.pop("results", None)
    print(json.dumps(st, indent=1))
