# -*- coding: utf-8 -*-
"""Screenshot two builds of the viewer at fixed rollout times and diff them (before/after checks).

    python simulation/web/tools/compare_shots.py http://127.0.0.1:8811/index.html http://127.0.0.1:8812/index.html \\
        --out shots/ [--cases default]

Each case pins robot / camera / arms / rollout / camera view / time, pauses the replay and waits
for the policy-input video to finish seeking. Reports per case: PSNR, mean / max abs difference
and the share of pixels that differ by more than 8/255, and writes A | B | 4x diff strips.
"""
import argparse
import asyncio
import json
import os

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
            "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox"]
CASES = [   # name, query
    ("trossen_front_t-0.5", "robot=trossen_wxai&camera=realsense_d405&mode=bimanual&rollout=trossen_wxai_realsense_d405_bimanual_000&view=front&t=-0.5"),
    ("trossen_front_t6", "robot=trossen_wxai&camera=realsense_d405&mode=bimanual&rollout=trossen_wxai_realsense_d405_bimanual_000&view=front&t=6"),
    ("trossen_side_t11", "robot=trossen_wxai&camera=realsense_d405&mode=bimanual&rollout=trossen_wxai_realsense_d405_bimanual_000&view=side&t=11"),
    ("trossen_close_t9", "robot=trossen_wxai&camera=realsense_d405&mode=bimanual&rollout=trossen_wxai_realsense_d405_bimanual_000&cam=0.62,-0.42,0.22,0.46,-0.25,0.04&t=9"),
    ("trossen_policycam_t20", "robot=trossen_wxai&camera=realsense_d405&mode=bimanual&rollout=trossen_wxai_realsense_d405_bimanual_005&view=policy&t=20"),
    ("single_front_t8", "robot=trossen_wxai&camera=realsense_d405&mode=single&rollout=trossen_wxai_realsense_d405_single_002&view=front&t=8"),
    ("franka_front_t10", "robot=franka_panda&camera=realsense_d405&mode=bimanual&rollout=franka_panda_realsense_d405_bimanual_001&view=front&t=10"),
    ("franka_close_t7", "robot=franka_panda&camera=realsense_d405&mode=bimanual&rollout=franka_panda_realsense_d405_bimanual_001&cam=0.75,-0.55,0.35,0.45,-0.2,0.08&t=7"),
    ("ur5e_front_t10", "robot=ur5e&camera=realsense_d405&mode=bimanual&rollout=ur5e_realsense_d405_bimanual_002&view=front&t=10"),
    ("ur5e_side_t14", "robot=ur5e&camera=realsense_d405&mode=bimanual&rollout=ur5e_realsense_d405_bimanual_002&view=side&t=14"),
]


async def shoot(b, url, path, size, selector):
    pg = await b.new_page(viewport={"width": size[0], "height": size[1]})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    await pg.goto(url, wait_until="networkidle", timeout=180000)
    await pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=180000)
    await pg.evaluate("() => { const s = window.__sim; s.viewer.playing = false; s.viewer.setTime(s.viewer.time); }")
    await pg.evaluate("() => new Promise(r => { const v = document.querySelector('#pip-video');"
                      " if (!v || !v.seeking) return r(); v.addEventListener('seeked', () => r(), {once: true});"
                      " setTimeout(r, 3000); })")
    await pg.wait_for_timeout(2500)        # camera tween (1.1 s), GTAO, video frame
    await pg.locator(selector).screenshot(path=path)
    await pg.close()
    return errs


def diff(a, b):
    A = np.asarray(Image.open(a).convert("RGB"), np.float64)
    B = np.asarray(Image.open(b).convert("RGB"), np.float64)
    if A.shape != B.shape:
        return {"error": f"shape {A.shape} vs {B.shape}"}, None
    d = np.abs(A - B)
    mse = (d ** 2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    strip = np.hstack([A, B, np.clip(4 * d, 0, 255)]).astype(np.uint8)
    return {"psnr": round(psnr, 2), "mean_abs": round(float(d.mean()), 3), "max_abs": int(d.max()),
            "frac_gt8": round(float((d.max(2) > 8).mean()), 5)}, strip


async def main(a):
    os.makedirs(a.out, exist_ok=True)
    cases = [c for c in CASES if not a.only or any(k in c[0] for k in a.only.split(","))]
    rep = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU_ARGS)
        for name, q in cases:
            q += "&kiosk=1"
            pa, pb = os.path.join(a.out, f"{name}_A.png"), os.path.join(a.out, f"{name}_B.png")
            ea = await shoot(b, f"{a.url_a}?{q}", pa, (a.width, a.height), a.selector)
            eb = await shoot(b, f"{a.url_b}?{q}", pb, (a.width, a.height), a.selector)
            r, strip = diff(pa, pb)
            if strip is not None:
                Image.fromarray(strip).save(os.path.join(a.out, f"{name}_diff.png"))
            r["errors_a"], r["errors_b"] = ea[:5], eb[:5]
            rep[name] = r
            print(f"{name:26s} {json.dumps(r)}", flush=True)
        await b.close()
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url_a")
    ap.add_argument("url_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated substrings of case names")
    ap.add_argument("--selector", default="#viewport")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    asyncio.run(main(ap.parse_args()))
