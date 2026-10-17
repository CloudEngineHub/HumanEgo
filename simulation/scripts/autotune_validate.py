# -*- coding: utf-8 -*-
"""
Re-evaluate the best candidates of an autotune run on the SAME fresh scenes (common random
numbers), together with the production settings, and write the winner to final.json.

Batch scores are noisy (each batch draws new scenes, 14 scenes x 2 arms); this picks the
parameters that hold up on a larger held-out set.

    python simulation/scripts/autotune_validate.py --name soft_v4 --top 3 --n 32 --workers 20
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)

import numpy as np  # noqa: E402

from autotune.params import params_to_overrides  # noqa: E402
from humanego_sim.env import load_config  # noqa: E402
from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.sampling import sample_scene  # noqa: E402


def candidates(run_dir, top, min_iter):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "iterations", "iter_*.json"))):
        it = json.load(open(p))
        if it["iteration"] < min_iter:
            continue
        for c in it["candidates"]:
            rows.append((c["stats"]["score"], c["params"], it["iteration"]))
    rows.sort(key=lambda r: -r[0])
    return rows[:top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--min_iter", type=int, default=0)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--mode", default="bimanual")
    ap.add_argument("--robot", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(SIM, "autotune_runs", args.name)
    fixed = {"mode": args.mode, **({"robot": args.robot} if args.robot else {})}
    base = load_config(None, fixed)
    rng = np.random.default_rng(args.seed)
    scenes = [sample_scene(rng, args.mode) for _ in range(args.n)]
    trials = [("production", {}, None)] + [(f"batch{it}_{s:.3f}", p, s) for s, p, it in
                                          candidates(run_dir, args.top, args.min_iter)]
    results = []
    for name, params, batch_score in trials:
        ov = params_to_overrides(params, base) if params else {}
        ov = {**ov, **fixed}
        st = evaluate(ov, scenes=scenes, workers=args.workers, seed=args.seed)
        st.pop("results", None)
        print(f"[validate] {name:22s} per-arm {st['per_side_success_rate']:.3f}  sides {st['side_success']}  "
              f"modes {st['failure_modes']}", flush=True)
        results.append({"name": name, "params": params, "batch_score": batch_score, "validation": st})
    best = max(results[1:], key=lambda r: r["validation"]["per_side_success_rate"])
    out = {"params": best["params"], "stats": best["validation"], "validation": best["validation"],
           "chosen": best["name"], "trials": results, "n_scenes": args.n, "seed": args.seed}
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[validate] chosen {best['name']}: {best['validation']['per_side_success_rate']:.3f} per arm "
          f"(production {results[0]['validation']['per_side_success_rate']:.3f}) -> {run_dir}/final.json")
