# -*- coding: utf-8 -*-
"""Record a cinematic teaser video of a policy rollout from the web viewer.

    python simulation/web/tools/record.py http://127.0.0.1:8791/index.html teaser.mp4 \
        --rollout trossen_wxai_realsense_d405_bimanual_003 --fps 30 --speed 1.0 --size 1920x1080 --clean

The camera slowly orbits the workspace while the recorded rollout (the released policy
in simulation) plays; frames are rendered deterministically via viewer.setTime(t).
"""
import argparse
import asyncio
import math
import os
import shutil
import subprocess
import tempfile

from playwright.async_api import async_playwright


async def main(a):
    W, H = map(int, a.size.split("x"))
    tmp = tempfile.mkdtemp(prefix="rec_")
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
                                          "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox"])
        pg = await b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        q = f"?kiosk={'clean' if a.clean else '1'}&rollout={a.rollout}&t=-9"
        if a.robot:
            q += f"&robot={a.robot}&camera={a.camera}&mode={a.mode}"
        await pg.goto(a.url + q, wait_until="networkidle")
        await pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=120000)
        await pg.wait_for_timeout(1500)
        t0, t1 = await pg.evaluate("() => [window.__sim.viewer.t0, window.__sim.viewer.t1]")
        t1 = min(t1, t0 + a.max_seconds)
        n = int((t1 - t0) / a.speed * a.fps)
        print(f"recording {n} frames ({t0:.1f}..{t1:.1f} s sim)")
        for i in range(n):
            t = t0 + i * a.speed / a.fps
            # cinematic camera: slow orbit + gentle dolly around the workspace
            ang = math.radians(a.az0 + a.orbit_deg * i / max(n - 1, 1))
            r = a.radius * (1.0 - 0.10 * math.sin(math.pi * i / max(n - 1, 1)))
            cx, cy, cz = 0.42, 0.0, 0.06
            px, py, pz = cx + r * math.cos(ang), cy + r * math.sin(ang), a.height
            await pg.evaluate(f"() => {{ const v = window.__sim.viewer; v.setCustomView([{px},{py},{pz}],[{cx},{cy},{cz}]); v.setTime({t}); }}")
            await pg.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            await pg.locator("#viewport").screenshot(path=os.path.join(tmp, f"{i:05d}.png"))
        await b.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
                    a.out], check=True)
    shutil.rmtree(tmp)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--camera", default="realsense_d405")
    ap.add_argument("--mode", default="bimanual")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=60)
    ap.add_argument("--az0", type=float, default=-20)
    ap.add_argument("--orbit_deg", type=float, default=50)
    ap.add_argument("--radius", type=float, default=1.15)
    ap.add_argument("--height", type=float, default=0.62)
    asyncio.run(main(ap.parse_args()))
