# -*- coding: utf-8 -*-
"""
Copy the built web viewer (simulation/web) into a static site, e.g. the
HumanEgo project page:

    python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim

Copies index.html, css/, js/, assets/, data/ (not tools/). Stale rollouts in the
target are removed so the site mirrors the current build exactly.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), "web")


def main(dst):
    os.makedirs(dst, exist_ok=True)
    for item in ("index.html", "css", "js", "assets", "data"):
        src = os.path.join(WEB, item)
        out = os.path.join(dst, item)
        if os.path.isdir(out):
            shutil.rmtree(out)
        if item == "data":
            # only the rollouts the manifest lists (local builds accumulate test rollouts)
            man = json.load(open(os.path.join(src, "manifest.json")))
            keep = {r["id"] for c in man["configs"] for r in c["rollouts"]}
            def _ignore(d, names):
                if os.path.basename(d) != "rollouts":
                    return []
                return [n for n in names if n.split(".")[0].replace("_policy", "") not in keep]
            shutil.copytree(src, out, ignore=_ignore)
        elif os.path.isdir(src):
            shutil.copytree(src, out)
        elif os.path.exists(src):
            shutil.copy(src, out)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dst) for f in fs)
    print(f"deployed {WEB} -> {dst} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
