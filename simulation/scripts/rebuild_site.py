# -*- coding: utf-8 -*-
"""
Rebuild the whole interactive-sim site in one command (it never deploys).

    python simulation/scripts/rebuild_site.py --workers 2            # build + pack + record + validate
    python simulation/scripts/rebuild_site.py --skip-build            # re-generate manifest, pack, videos
    python simulation/scripts/rebuild_site.py --dry-run               # show the plan
    python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim --teaser   # then, to publish

1. builds every site config with its own profile (SITE_CONFIGS below; robot:camera:mode:n:profile)
   -> models, Live-mode MJCF, rollouts (build_web.build_config);
2. writes the manifest: results table from those rollouts, held-out numbers and the tuning note from
   configs/tuned/<robot>.json and the autotune run's outputs (humanego_sim/web_manifest.py);
3. packs it for the web (humanego_sim/web_pack.py: binary rollouts, re-encoded policy videos,
   quantized robot GLBs, WebP textures);
4. records the teaser (media/teaser.mp4 + .jpg poster) and one video per robot (media/<robot>.mp4)
   with web/tools/record.py from a local server;
5. validates the manifest against the files and writes runs/site/report.json (sizes, numbers,
   timings, the deploy command).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
sys.path.append(HERE)

from humanego_sim.profiles import PROFILES, add_profile_arg  # noqa: E402

# what humanego-ai.github.io/sim shows: the real rig with every camera / both arm modes, and the other robots
SITE_CONFIGS = [
    "trossen_wxai:realsense_d405:bimanual:24:tuned",
    "trossen_wxai:realsense_d405:single:12:tuned",
    "trossen_wxai:realsense_d435:bimanual:10:tuned",
    "trossen_wxai:zed2i:bimanual:10:tuned",
    "franka_panda:realsense_d405:bimanual:10:tuned",
    "ur5e:realsense_d405:bimanual:10:tuned",
]
WEB = os.path.join(SIM, "web")
REPORT = os.path.join(SIM, "runs", "site", "report.json")
BUSY_PORTS = {8765, 8791, 8792, 8797, 8803}         # other apps / sessions on the shared box
RECORD_PY = os.path.expanduser("~/miniconda3/envs/v2d/bin/python")   # Playwright lives here


def free_port():
    for _ in range(50):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if p not in BUSY_PORTS:
            return p
    raise RuntimeError("no free port")


def featured(man, robot, camera="realsense_d405", mode="bimanual"):
    c = next((c for c in man["configs"] if (c["robot"], c["camera"], c["mode"]) == (robot, camera, mode)), None)
    if c is None or not c["rollouts"]:
        return None
    return (next((r for r in c["rollouts"] if r.get("featured")), None) or c["rollouts"][0])["id"]


def record_videos(man, py, robots, teaser=True, size="1280x720", dpr=1.5, max_seconds=45.0):
    """Teaser + one video per robot from the packed site, served locally like GitHub Pages."""
    port = free_port()
    srv = subprocess.Popen([sys.executable, os.path.join(WEB, "tools", "serve.py"), str(port), WEB, "--gzip"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    media = {}
    try:
        time.sleep(1.0)
        os.makedirs(os.path.join(WEB, "media"), exist_ok=True)
        jobs = []
        if teaser:
            rid = featured(man, "trossen_wxai")
            if rid:
                jobs.append(("teaser", "trossen_wxai", rid, dpr))
        for robot in robots:
            rid = featured(man, robot)
            if rid:
                jobs.append((robot, robot, rid, 1.0))
        url = f"http://127.0.0.1:{port}/index.html"
        for name, robot, rid, d in jobs:
            out = os.path.join(WEB, "media", f"{name}.mp4")
            cmd = [py, os.path.join(WEB, "tools", "record.py"), url, out, "--rollout", rid, "--robot", robot,
                   "--size", size, "--dpr", str(d), "--hide_controls", "--max_seconds", str(max_seconds)]
            print("[site] recording", name, rid, flush=True)
            t0 = time.time()
            subprocess.run(cmd, check=True)
            media[name] = f"media/{name}.mp4"
            print(f"[site]   {out} ({os.path.getsize(out) / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)
            if name == "teaser":                     # poster: a frame from the middle of the teaser
                jpg = os.path.join(WEB, "media", "teaser.jpg")
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                            "csv=p=0", out], capture_output=True, text=True).stdout or 10)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{0.55 * dur:.2f}", "-i", out, "-frames:v", "1",
                                "-q:v", "3", jpg], check=True)
                media["teaser_poster"] = "media/teaser.jpg"
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    return media


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild the interactive-sim site (build, pack, record, validate).")
    ap.add_argument("--configs", default=",".join(SITE_CONFIGS), help="robot:camera:mode[:n[:profile]], comma-separated")
    ap.add_argument("--workers", type=int, default=2, help="evaluation workers (1.2 GB GPU memory each)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-build", action="store_true", help="keep the built rollouts; manifest, pack, videos only")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--record-python", default=RECORD_PY if os.path.exists(RECORD_PY) else sys.executable,
                    help="python with Playwright + Chromium for web/tools/record.py")
    ap.add_argument("--video-size", default="1280x720")
    ap.add_argument("--video-seconds", type=float, default=45.0, help="cap per video (sim seconds)")
    ap.add_argument("--dry-run", action="store_true")
    add_profile_arg(ap, default=None)
    args = ap.parse_args(argv)
    import build_web
    specs = []
    for s in args.configs.split(","):
        robot, camera, mode, n, profile = build_web.parse_spec(s, 10, "tuned")
        specs.append((robot, camera, mode, n, args.profile or profile))
    print("[site] configs:")
    for s in specs:
        print("   ", ":".join(map(str, s)))
    if args.dry_run:
        print(f"[site] would build {sum(s[3] for s in specs)} rollouts with {args.workers} workers, pack, "
              f"record {'no' if args.no_videos else 1 + len({s[0] for s in specs})} videos, validate; never deploys")
        return
    t0 = time.time()
    timings = {}
    built = []
    if not args.skip_build:
        for robot, camera, mode, n, profile in specs:
            t1 = time.time()
            print(f"[site] building {robot} / {camera} / {mode}: {n} rollouts, profile {profile}", flush=True)
            built.append(build_web.build_config(robot, camera, mode, n, args.workers, args.seed, {}, profile))
            build_web.write_manifest(built)
            timings[f"build {robot}_{camera}_{mode}"] = round(time.time() - t1)
    man = build_web.write_manifest(built)
    from humanego_sim.web_pack import pack_web
    t1 = time.time()
    pack_rep = pack_web(WEB, log_root=build_web.LOG_ROOT)
    timings["pack"] = round(time.time() - t1)
    man = json.load(open(os.path.join(WEB, "data", "manifest.json")))
    if not args.no_videos:
        t1 = time.time()
        media = record_videos(man, args.record_python, sorted({s[0] for s in specs}, key=list(build_web.ROBOTS).index),
                              size=args.video_size, max_seconds=args.video_seconds)
        timings["videos"] = round(time.time() - t1)
        man = build_web.write_manifest([], media=media)
        pack_web(WEB, log_root=build_web.LOG_ROOT, verbose=False)     # re-applies fmt to the rewritten manifest
        man = json.load(open(os.path.join(WEB, "data", "manifest.json")))
    from humanego_sim.web_manifest import validate
    errs = validate(man, WEB)
    import deploy_web
    files = deploy_web.plan() if not errs else []
    size = sum(os.path.getsize(os.path.join(WEB, f)) for f in files)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "commit": build_web._git_rev(), "configs": [":".join(map(str, s)) for s in specs],
              "summary": man["summary"], "tuning_note": (man.get("tuning") or {}).get("note"), "media": man.get("media"),
              "pack": pack_rep, "deploy": {"files": len(files), "bytes": size}, "timings_s": timings,
              "errors": errs, "deploy_command": "python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim --teaser"}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    if errs:
        raise SystemExit("[site] manifest invalid:\n  " + "\n  ".join(errs))
    print(f"[site] done in {(time.time() - t0) / 60:.0f} min: {len(files)} files, {size / 1e6:.1f} MB staged in {WEB}")
    for r in man["summary"]:
        print(f"   {r['robot']:13s} {r['camera']:15s} {r['mode']:9s} {r['profile']:10s} n={r['n']:2d}  "
              f"R {r['right'] if r['right'] is None else round(r['right'], 2)}  L {r['left'] if r['left'] is None else round(r['left'], 2)}  "
              f"both {round(r['both'], 2)}")
    print(f"[site] report: {REPORT}\n[site] to publish (not done here): {report['deploy_command']}")


if __name__ == "__main__":
    main()
