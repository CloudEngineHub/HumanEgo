# -*- coding: utf-8 -*-
"""
Self-improving search over simulator + low-level controller parameters.

    python simulation/scripts/autotune.py --name bimanual_v1 --pop 8 --scenes 10 --workers 12

Watch simulation/autotune_runs/<name>/: iterations/iter_NNN.json per batch,
population_memory.json, best.json. When the search plateaus with no usable
signal it writes inbox/request_NNN.json and prints `AGENT_REQUEST <path>`;
answer with inbox/response_NNN.json (format in autotune/loop.py) — e.g. from
Claude Code — or let it time out and fall back to a heuristic.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from autotune.loop import AutoTuner  # noqa: E402
from autotune.params import space_from_names  # noqa: E402
from humanego_sim.profiles import add_profile_arg  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bimanual_v1")
    ap.add_argument("--mode", default="bimanual")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--params", default=None, help="comma-separated subset of parameter names")
    ap.add_argument("--agent_timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="params JSON (e.g. a previous final.json) to start the search from")
    ap.add_argument("--set", action="append", default=[], help="fixed base override, e.g. physics.soft.damping=0.03")
    add_profile_arg(ap)    # candidates are searched on top of this profile
    args = ap.parse_args()
    base = {"mode": args.mode, "profile": args.profile}
    if args.robot:
        base["robot"] = args.robot
    if args.camera:
        base["camera"] = args.camera
    space = space_from_names(args.params.split(",") if args.params else None)
    if args.set:
        from evaluate import parse_sets
        from humanego_sim.env import deep_update
        base = deep_update(base, parse_sets(args.set))
    if args.init:
        init = json.load(open(args.init))["params"]
        for p in space:
            if p.name in init:
                p.init = float(min(max(init[p.name], p.lo), p.hi))
    tuner = AutoTuner(args.name, base, space, pop=args.pop, scenes_per_candidate=args.scenes, workers=args.workers,
                      seed=args.seed, target=args.target, max_batches=args.batches, agent_timeout=args.agent_timeout)
    best = tuner.run()
    print(json.dumps(best, indent=1, default=float))
