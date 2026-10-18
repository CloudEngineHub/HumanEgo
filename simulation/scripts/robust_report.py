# -*- coding: utf-8 -*-
"""
Robustness report: one robot, its tuned settings (configs/tuned/<robot>.json + grasp skill), every
perturbation of robustness.CONDITIONS alone and all of them together, on the same scenes, with 95 %
intervals and a paired test against the clean condition.

    python simulation/scripts/robust_report.py --robot trossen_wxai --seed 9001 --n 100 --final
    python simulation/scripts/robust_report.py --robot ur5e --seed 7777 --n 24 --conditions clean masks all \\
        --name ur5e_fused --set 'perception.mask_frames=5'
    python simulation/scripts/robust_report.py --table trossen_wxai_9001 franka_panda_9001 ur5e_9001

Each condition's results go to simulation/runs/robust_report/<name>/<condition>.json (an existing file
is reused, so an interrupted report resumes); --table prints the README table from them.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)                  # before scripts/: the autotune package, not scripts/autotune.py

from autotune.params import params_to_overrides  # noqa: E402
from evaluate import parse_sets  # noqa: E402
from humanego_sim.env import deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import evaluate, summarize  # noqa: E402
from humanego_sim.robustness import CONDITIONS, condition  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed, paired, summarize_ci  # noqa: E402

ROOT = os.path.join(SIM, "runs", "robust_report")
DEFAULT_CONDITIONS = [c for c in CONDITIONS] + ["all"]


def tuned_overrides(robot: str) -> dict:
    """configs/tuned/<robot>.json with the grasp skill it requires, as config overrides."""
    path = os.path.join(SIM, "configs", "tuned", f"{robot}.json")
    tuned = json.load(open(path))
    base = load_config(None, {"robot": robot})
    ov = params_to_overrides(tuned["params"], base)
    ov = deep_update(ov, {"robot": robot, "control": {"grasp_assist": {"enabled": True}}})
    return ov


def run(args):
    check_seed([args.seed], args.final)
    name = args.name or f"{args.robot}_{args.seed}"
    out_dir = os.path.join(ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    base = deep_update(tuned_overrides(args.robot), parse_sets(args.set))
    mode = load_config(None, base).get("mode", "bimanual")
    meta = {"robot": args.robot, "seed": args.seed, "n": args.n, "base_overrides": base,
            "commit": os.popen(f"git -C {SIM} rev-parse --short HEAD").read().strip()}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    for cname in args.conditions:
        f = os.path.join(out_dir, f"{cname}.json")
        if os.path.exists(f):
            print(f"[robust] {name}/{cname}: reusing {f}", flush=True)
            continue
        c = condition(cname)
        ov = deep_update(base, c.get("set", {}))
        scenes = sample_scenes(args.n, args.seed, mode, variation=c.get("variation") or None)
        log_root = os.path.join(SIM, "runs", "robust_logs", name, cname) if args.log else None
        st = evaluate(ov, scenes=scenes, workers=args.workers, seed=args.seed, log_root=log_root,
                      verbose=False, log_images=False)
        res = st.pop("results")
        json.dump({"condition": cname, "overrides": ov, "variation": c.get("variation"), "stats": st,
                   "results": [{k: v for k, v in r.items() if k != "trace"} for r in res]}, open(f, "w"))
        ci = st["ci"]["per_arm"]
        print(f"[robust] {name}/{cname}: per-arm {100 * ci['rate']:.0f} % "
              f"[{100 * ci['wilson95'][0]:.0f}, {100 * ci['wilson95'][1]:.0f}]  modes {st['failure_modes']}  "
              f"wall {st['wall']:.0f}s", flush=True)
    table([name])


def load(name):
    d = os.path.join(ROOT, name)
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and f != "meta.json":
            r = json.load(open(os.path.join(d, f)))
            out[r["condition"]] = r["results"]
    return out, json.load(open(os.path.join(d, "meta.json")))


def table(names, conditions=None):
    """Markdown: per condition, per-arm rate [Wilson 95 %], right / left / both, paired diff vs clean."""
    for name in names:
        res, meta = load(name)
        conds = [c for c in (conditions or DEFAULT_CONDITIONS) if c in res] + \
                [c for c in res if c not in (conditions or DEFAULT_CONDITIONS)]
        print(f"\n**{meta['robot']}**, scene set {meta['seed']}, {meta['n']} scenes (commit {meta.get('commit', '?')})\n")
        print("| condition | per arm [95 % CI] | right | left | both | vs clean (paired, 95 % CI) | McNemar p | failures |")
        print("|---|---|---|---|---|---|---|---|")
        for c in conds:
            ci = summarize_ci(res[c])
            pa, sd, bo = ci["per_arm"], ci["sides"], ci["both"]
            modes = summarize(res[c])["failure_modes"]
            modes = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(modes.items(), key=lambda kv: -kv[1])
                              if k != "success")
            if c != "clean" and "clean" in res:
                p = paired(res[c], res["clean"])
                lo, hi = p["boot95"]
                vs = f"{100 * p['diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"
                mp = f"{p['mcnemar_p']:.2g}"
            else:
                vs, mp = "—", "—"
            side = lambda s: f"{100 * sd[s]['rate']:.0f} %" if s in sd else "—"  # noqa: E731
            print(f"| {c} | **{100 * pa['rate']:.0f} %** [{100 * pa['wilson95'][0]:.0f}, {100 * pa['wilson95'][1]:.0f}] "
                  f"| {side('right')} | {side('left')} | {100 * bo['rate']:.0f} % | {vs} | {mp} | {modes} |")


def compare(pairs, conditions):
    """Paired test of report B against report A, per condition, pooled over the given report pairs
    (e.g. both development seeds): pairs = ["A1:B1", "A2:B2", ...]."""
    print("| condition | A per arm | B per arm | B − A (paired, 95 % CI) | discordant B+/A+ | McNemar p | scenes |")
    print("|---|---|---|---|---|---|---|")
    for c in conditions:
        ra, rb = [], []
        for k, pr in enumerate(pairs):
            a, b = pr.split(":")
            fa, fb = (os.path.join(ROOT, x, f"{c}.json") for x in (a, b))
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            for src, dst in ((fa, ra), (fb, rb)):
                for r in json.load(open(src))["results"]:
                    dst.append(dict(r, idx=k * 100000 + r["idx"]))
        if not ra:
            continue
        p = paired(rb, ra)
        lo, hi = p["boot95"]
        # paired(rb, ra): rate_a is B's, rate_b is A's; a_only = arms only B served
        print(f"| {c} | {100 * p['rate_b']:.0f} % | {100 * p['rate_a']:.0f} % | "
              f"{100 * p['diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}] | {p['a_only']} / {p['b_only']} | "
              f"{p['mcnemar_p']:.2g} | {p['n_scenes']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="trossen_wxai")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--final", action="store_true", help="allow the reserved seeds 9001-9099 (final reports only)")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--set", action="append", default=[], help="extra overrides for every condition (e.g. a mitigation)")
    ap.add_argument("--name", default=None, help="report name (default <robot>_<seed>)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--log", action="store_true", help="keep every episode (for scripts/grasp_audit.py)")
    ap.add_argument("--table", nargs="*", default=None, help="only print the tables of these reports")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="paired test B vs A per condition, pooled over pairs 'A:B' (e.g. two seeds)")
    a = ap.parse_args()
    if a.compare is not None:
        compare(a.compare, a.conditions)
    elif a.table is not None:
        table(a.table)
    else:
        run(a)
