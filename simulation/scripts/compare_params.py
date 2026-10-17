# -*- coding: utf-8 -*-
"""
Evaluate several parameter sets on the SAME held-out scenes (common random numbers).

    python simulation/scripts/compare_params.py --sets production=- tuned=simulation/autotune_runs/soft_v4/final.json \\
        --drop physics.soft.damping --n 32 --seed 7777 --set robot='"franka_panda"'
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from autotune.params import params_to_overrides  # noqa: E402
from evaluate import parse_sets  # noqa: E402
from humanego_sim.env import deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.sampling import sample_scene  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="name=params.json (or name=- for production)")
    ap.add_argument("--drop", nargs="*", default=[], help="parameter names to ignore from the files")
    ap.add_argument("--set", action="append", default=[], help="extra overrides for every set")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    extra = parse_sets(args.set)
    base = load_config(None, extra)
    mode = base.get("mode", "bimanual")
    rng = np.random.default_rng(args.seed)
    scenes = [sample_scene(rng, mode) for _ in range(args.n)]
    rows = []
    for spec in args.sets:
        name, path = spec.split("=", 1)
        params = {} if path == "-" else {k: v for k, v in json.load(open(path))["params"].items() if k not in args.drop}
        ov = deep_update(params_to_overrides(params, base) if params else {}, extra)
        st = evaluate(ov, scenes=scenes, workers=args.workers, seed=args.seed)
        res = st.pop("results", [])
        strict = [v.get("success_strict") for r in res if "per_side" in r for v in r["per_side"].values()]
        st["per_side_success_strict"] = float(np.mean(strict)) if strict else None
        print(f"[compare] {name:12s} per-arm {st['per_side_success_rate']:.3f} (strict {st['per_side_success_strict']:.3f})  "
              f"sides {st['side_success']}  both {st['success_rate']:.3f}  modes {st['failure_modes']}", flush=True)
        rows.append({"name": name, "params": params, "stats": st})
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
