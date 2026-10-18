# -*- coding: utf-8 -*-
"""
Evaluate several parameter sets on the SAME held-out scenes (common random numbers), with
95 % CIs and a paired test of every set against the first one.

    python simulation/scripts/compare_params.py --sets production=- tuned=simulation/autotune_runs/soft_v4/final.json \\
        --drop physics.soft.damping --n 32 --seed 7777 8888 --set robot='"franka_panda"'
    python simulation/scripts/compare_params.py --sets tuned=- --profile tuned --set robot='"ur5e"'

Protocol (README, "Evaluation protocol"): development seeds 7777 / 8888 select parameters;
seeds 9001-9099 are reserved for final reports (--final) and must never be used to pick them.
A set may add raw overrides to a params file, or be raw overrides alone:
    name=simulation/configs/tuned/ur5e.json@'{"perception": {"mask_frames": 5}}'   name=@'{...}'.
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
from humanego_sim.profiles import add_profile_arg  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed, fmt_paired, fmt_rate, paired, summarize_ci  # noqa: E402


def set_overrides(path: str, base: dict, drop) -> tuple:
    """(params dict, overrides) for one --sets entry: '-', 'params.json', '@{json}' or 'params.json@{json}'."""
    path, _, raw = path.partition("@")
    extra = json.loads(raw) if raw else {}
    if path in ("", "-"):
        return {}, extra
    params = {k: v for k, v in json.load(open(path))["params"].items() if k not in drop}
    ov = params_to_overrides(params, base) if params else {}
    return params, deep_update(ov, extra) if extra else ov


def line(name: str, ci: dict) -> str:
    pa, bo = ci["per_arm"], ci["both"]
    sides = "  ".join(f"{s} {fmt_rate(v['k'], v['n'])}" for s, v in sorted(ci["sides"].items(), reverse=True))
    lo, hi = pa["scene_boot95"]
    return (f"{name:14s} per-arm {fmt_rate(pa['k'], pa['n'])} (scene bootstrap [{100 * lo:.0f}, {100 * hi:.0f}])  "
            f"{sides}  both {fmt_rate(bo['k'], bo['n'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="name=params.json, name=- (production), name=@<JSON overrides> or name=params.json@<JSON>")
    ap.add_argument("--drop", nargs="*", default=[], help="parameter names to ignore from the files")
    ap.add_argument("--set", action="append", default=[], help="extra overrides for every set")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, nargs="+", default=[7777], help="scene-set seed(s); each gives n scenes")
    ap.add_argument("--final", action="store_true", help="allow the reserved seeds 9001-9099 (final reports only)")
    ap.add_argument("--variation", default=None,
                    help="scene variation JSON (sampling.sample_scenes), e.g. '{\"bread_scale\": 0.12}'")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None, help="keep every episode under simulation/runs/<log>/<set name>/")
    add_profile_arg(ap)    # base profile of every set; a set's params file applies on top
    args = ap.parse_args()
    check_seed(args.seed, args.final)
    extra = parse_sets(args.set)
    extra.setdefault("profile", args.profile)
    base = load_config(None, extra)
    mode = base.get("mode", "bimanual")
    variation = json.loads(args.variation) if args.variation else None
    rows, per_set = [], {}
    for spec in args.sets:
        name, path = spec.split("=", 1)
        params, ov = set_overrides(path, base, args.drop)
        ov = deep_update(ov, extra) if ov else dict(extra)
        res_all = []
        for seed in args.seed:
            scenes = sample_scenes(args.n, seed, mode, variation=variation)
            log_root = None
            if args.log:
                log_root = os.path.join(os.path.dirname(HERE), "runs", args.log, name,
                                        f"seed{seed}" if len(args.seed) > 1 else "")
            st = evaluate(ov, scenes=scenes, workers=args.workers, seed=seed, log_root=log_root, verbose=False)
            res = st.pop("results", [])
            for r in res:                      # scene ids unique across seeds for the paired tests
                r["seed_set"] = seed
                r["idx"] = seed * 100000 + r["idx"]
            res_all += res
            if len(args.seed) > 1:
                print(line(f"{name}@{seed}", st["ci"]), flush=True)
        ci = summarize_ci(res_all)
        strict = [v.get("success_strict") for r in res_all if "per_side" in r for v in r["per_side"].values()]
        modes = {}
        for r in res_all:
            if "error" in r:
                modes["error"] = modes.get("error", 0) + 1
        from humanego_sim.evaluate import summarize
        st_all = summarize(res_all)
        print(line(name, ci) + f"  strict {np.mean(strict):.2f}  modes {st_all['failure_modes']}", flush=True)
        per_set[name] = res_all
        rows.append({"name": name, "params": params, "overrides": ov, "seeds": args.seed, "n": args.n,
                     "stats": {k: v for k, v in st_all.items() if k != "results"},
                     "outcomes": [{"idx": r["idx"], "seed_set": r["seed_set"], "scene": r["scene"],
                                   "per_side": {s: bool(v["success"]) for s, v in r.get("per_side", {}).items()},
                                   **({"error": r["error"]} if "error" in r else {})} for r in res_all]})
    names = [s.split("=", 1)[0] for s in args.sets]
    if len(names) > 1:
        print(f"[compare] paired vs '{names[0]}' (same scenes):")
        for nm in names[1:]:
            p = paired(per_set[nm], per_set[names[0]])
            print("  " + fmt_paired(p, nm, names[0]), flush=True)
            next(r for r in rows if r["name"] == nm)["paired_vs_first"] = p
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
