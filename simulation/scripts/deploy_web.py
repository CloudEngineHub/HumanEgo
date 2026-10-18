# -*- coding: utf-8 -*-
"""
Copy the built web viewer (simulation/web) into a static site, e.g. the HumanEgo project page:

    python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim
    python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim --teaser   # + main-page teaser video

Copies index.html, css/, js/, assets/, data/ and media/ (not tools/). Only what the packed site
loads is copied: the rollouts the manifest lists (as .bin when packed, JSON otherwise), WebP
textures instead of their PNG/JPEG sources. Stale files in the target are removed so the site
mirrors the current build exactly. --teaser also copies media/teaser.mp4 / teaser.jpg to the
main page's static/videos/sim_teaser.mp4 and static/images/sim_teaser.jpg (next to <dst>).
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
WEB = os.path.join(SIM, "web")
sys.path.insert(0, SIM)


def plan(web=WEB):
    """(relative paths to copy, reasons for what is skipped) for a built web/ directory."""
    from humanego_sim.web_manifest import validate
    from humanego_sim.web_pack import deploy_skip
    man = json.load(open(os.path.join(web, "data", "manifest.json")))
    errs = validate(man, web)
    if errs:
        raise SystemExit("manifest invalid, not deploying:\n  " + "\n  ".join(errs))
    keep_rollouts = {r["id"]: r.get("fmt", "json") for c in man["configs"] for r in c["rollouts"]}
    skip = deploy_skip(web)
    files = []
    for item in ("index.html", "css", "js", "assets", "data", "media"):
        src = os.path.join(web, item)
        if os.path.isfile(src):
            files.append(item)
            continue
        for root, _, names in os.walk(src):
            for n in names:
                rel = os.path.relpath(os.path.join(root, n), web)
                if rel in skip or n.endswith((".packed", ".tmp.mp4")):
                    continue
                if os.path.basename(root) == "rollouts":
                    rid = n.split(".")[0].replace("_policy", "")
                    if rid not in keep_rollouts:
                        continue                      # local test builds accumulate rollouts
                    if n.endswith(".json") and keep_rollouts[rid] == "bin":
                        continue
                files.append(rel)
    return sorted(files)


def main(dst, teaser=False, dry_run=False):
    files = plan()
    total = sum(os.path.getsize(os.path.join(WEB, f)) for f in files)
    if dry_run:
        print(f"would deploy {len(files)} files ({total / 1e6:.1f} MB) -> {dst}")
        return files
    os.makedirs(dst, exist_ok=True)
    for item in ("index.html", "css", "js", "assets", "data", "media"):
        out = os.path.join(dst, item)
        if os.path.isdir(out):
            shutil.rmtree(out)
        elif os.path.isfile(out):
            os.remove(out)
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(dst, f)) or dst, exist_ok=True)
        shutil.copy2(os.path.join(WEB, f), os.path.join(dst, f))
    print(f"deployed {WEB} -> {dst} ({len(files)} files, {total / 1e6:.1f} MB)")
    if teaser:
        site = os.path.dirname(os.path.abspath(dst.rstrip("/")))
        for src, out in (("media/teaser.mp4", "static/videos/sim_teaser.mp4"),
                         ("media/teaser.jpg", "static/images/sim_teaser.jpg")):
            if not os.path.exists(os.path.join(WEB, src)):
                raise SystemExit(f"--teaser: {src} not built (scripts/rebuild_site.py records it)")
            os.makedirs(os.path.dirname(os.path.join(site, out)), exist_ok=True)
            shutil.copy2(os.path.join(WEB, src), os.path.join(site, out))
            print(f"teaser: {src} -> {os.path.join(site, out)}")
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dst")
    ap.add_argument("--teaser", action="store_true", help="also update the main page's sim teaser video + poster")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.dst, a.teaser, a.dry_run)
