# -*- coding: utf-8 -*-
"""
Markdown table of baseline runs (runs/physics/<tag>_<robot>_<seed>_<prod|tuned>.json from compare_params.py):
per arm, right, left, both served, with Wilson 95 % intervals (humanego_sim/stats.py).

    python simulation/scripts/physics_report.py before v2
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.stats import wilson  # noqa: E402

ROBOTS = [("trossen_wxai", "Trossen"), ("ur5e", "UR5e"), ("franka_panda", "Franka")]


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def cells(path):
    st = json.load(open(path))[0]["stats"]
    n = st["n"]
    ss = st["side_success"]
    k_r, k_l = round(ss["right"] * n), round(ss["left"] * n)
    k_b = round(st["success_rate"] * n)
    return [rate(k_r + k_l, 2 * n), rate(k_r, n), rate(k_l, n), rate(k_b, n)], (k_r + k_l, 2 * n)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(HERE), "runs", "physics")
    tags = sys.argv[1:] or ["v2"]
    print("| robot | controller | physics | seed | per arm | right | left | both |")
    print("|---|---|---|---|---|---|---|---|")
    for robot, label in ROBOTS:
        for ctl, cname in (("prod", "production"), ("tuned", "tuned + skill")):
            for tag in tags:
                pooled = [0, 0]
                for seed in (7777, 8888):
                    p = os.path.join(root, f"{tag}_{robot}_{seed}_{ctl}.json")
                    if not os.path.exists(p):
                        continue
                    c, (k, n) = cells(p)
                    pooled[0] += k
                    pooled[1] += n
                    print(f"| {label} | {cname} | {tag} | {seed} | " + " | ".join(c) + " |")
                if pooled[1]:
                    print(f"| {label} | {cname} | {tag} | pooled | **{rate(*pooled)}** | | | |")
