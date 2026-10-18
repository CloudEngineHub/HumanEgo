# -*- coding: utf-8 -*-
"""First-load bytes and time of the web viewer in headless Chrome (cold cache).

    # local build, served like GitHub Pages (gzip), on a throttled 'Fast 4G' link:
    python simulation/web/tools/serve.py 8811 simulation/web --gzip &
    python simulation/web/tools/measure_load.py http://127.0.0.1:8811/index.html --net fast4g --out load.json
    # the deployed site, real network:
    python simulation/web/tools/measure_load.py https://humanego-ai.github.io/sim/
    # phone: viewport + touch + UA + 4x CPU slowdown; WebGPU removed (WASM fallback for Live mode)
    python simulation/web/tools/measure_load.py URL --mobile --no-webgpu --live

Reports wire bytes (encodedDataLength: compressed, incl. headers) per resource class, the
number of requests, time to the first rendered frame of the replay (`body[data-ready]`),
and with --live the time until Live mode is ready (MuJoCo WASM + policy downloaded/compiled),
plus every console error / page error.
"""
import argparse
import asyncio
import json
import time
from urllib.parse import urlparse

from playwright.async_api import async_playwright

NETS = {   # Chrome DevTools presets (download/upload bytes per s, latency ms)
    "none": None,
    "fast4g": (9e6 / 8, 1.5e6 / 8, 60),
    "slow4g": (1.6e6 / 8, 0.75e6 / 8, 150),
}
GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
            "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox"]
PHONE = dict(viewport={"width": 393, "height": 852}, device_scale_factor=3, is_mobile=True, has_touch=True,
             user_agent="Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/129.0.0.0 Mobile Safari/537.36")


def classify(url, origin):
    u = urlparse(url)
    if u.netloc != origin:
        return "cdn:" + u.netloc
    p = u.path
    for k in ("/data/rollouts/", "/data/models/", "/data/manifest", "/assets/robots/", "/assets/objects/",
              "/assets/textures/", "/assets/policy/", "/assets/live/", "/js/", "/css/", "/media/"):
        if k in p:
            return k.strip("/")
    return "page"


async def run(a):
    origin = urlparse(a.url).netloc
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU_ARGS)
        ctx = await b.new_context(**(PHONE if a.mobile else {"viewport": {"width": 1440, "height": 1000}}))
        if a.no_webgpu:
            await ctx.add_init_script("try { delete Navigator.prototype.gpu; } catch (e) {}")
        pg = await ctx.new_page()
        cdp = await ctx.new_cdp_session(pg)
        await cdp.send("Network.enable")
        await cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        net = NETS[a.net]
        if net:
            await cdp.send("Network.emulateNetworkConditions", {"offline": False, "downloadThroughput": net[0],
                                                                "uploadThroughput": net[1], "latency": net[2]})
        if a.mobile:
            await cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})
        reqs, done = {}, {}
        cdp.on("Network.requestWillBeSent", lambda e: reqs.__setitem__(e["requestId"], e["request"]["url"]))
        cdp.on("Network.loadingFinished", lambda e: done.__setitem__(e["requestId"], e["encodedDataLength"]))
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        t0 = time.time()
        await pg.goto(a.url, wait_until="commit", timeout=600000)
        out = {"url": a.url, "net": a.net, "mobile": a.mobile, "webgpu": not a.no_webgpu}
        try:
            await pg.wait_for_selector("body[data-ready='1']", state="attached", timeout=a.timeout * 1000)
            out["t_ready_s"] = round(time.time() - t0, 2)
        except Exception as e:
            out["t_ready_s"] = None
            logs.append(f"[measure] never ready: {e!r}"[:300])
        out["hud"] = await pg.evaluate("() => document.querySelector('#loading')?.classList.contains('hide')")
        await pg.wait_for_timeout(a.settle * 1000)       # trailing requests (policy video, fonts)
        first = dict(done)
        out["first_load"] = summarize(reqs, first, origin)
        if a.urls:
            out["first_load"]["urls"] = sorted(((reqs.get(k, ""), int(v)) for k, v in first.items()), key=lambda x: -x[1])
        if a.live:
            t1 = time.time()
            await pg.evaluate("() => [...document.querySelectorAll('#pills-run .pill')].find(b => b.textContent.includes('Live')).click()")
            try:
                await pg.wait_for_function("() => /Ready/.test(document.querySelector('#hud-title')?.textContent || '')"
                                           " || /failed/i.test(document.querySelector('#loading-msg')?.textContent || '')",
                                           timeout=a.timeout * 1000)
                out["live_status"] = await pg.evaluate("() => document.querySelector('#hud-title').textContent")
                out["t_live_ready_s"] = round(time.time() - t1, 2)
                if a.live_steps:
                    await pg.evaluate("() => document.querySelector('#btn-replay').click()")
                    t2 = time.time()
                    await pg.wait_for_function(f"() => {{ const m = (document.querySelector('#hud-title')?.textContent || '')"
                                               f".match(/re-plan (\\d+)/); return m && +m[1] >= {a.live_steps}; }}",
                                               timeout=a.timeout * 1000)
                    out["t_live_steps_s"] = round(time.time() - t2, 2)
                    out["live_status"] = await pg.evaluate("() => document.querySelector('#hud-title').textContent")
            except Exception as e:
                out["live_error"] = repr(e)[:300]
                out["live_status"] = await pg.evaluate("() => (document.querySelector('#hud-title')?.textContent || '') + ' / ' + (document.querySelector('#loading-msg')?.textContent || '')")
            live = {k: v for k, v in done.items() if k not in first}
            out["live_load"] = summarize(reqs, live, origin)
        if a.shot:
            await pg.screenshot(path=a.shot)
        out["console"] = [l for l in logs if "GPU stall" not in l][:40]
        await b.close()
    return out


def summarize(reqs, done, origin):
    by = {}
    for rid, n in done.items():
        k = classify(reqs.get(rid, ""), origin)
        e = by.setdefault(k, {"n": 0, "bytes": 0})
        e["n"] += 1
        e["bytes"] += int(n)
    tot = sum(e["bytes"] for e in by.values())
    return {"requests": sum(e["n"] for e in by.values()), "bytes": tot, "MB": round(tot / 1e6, 2),
            "by_class": dict(sorted(by.items(), key=lambda kv: -kv[1]["bytes"]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--net", default="none", choices=list(NETS))
    ap.add_argument("--mobile", action="store_true")
    ap.add_argument("--no-webgpu", dest="no_webgpu", action="store_true", help="hide navigator.gpu (WASM fallback)")
    ap.add_argument("--live", action="store_true", help="also switch to Live mode and wait until it is ready")
    ap.add_argument("--live_steps", type=int, default=0, help="... then run it for this many re-plans")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--shot", default=None, help="screenshot at the end")
    ap.add_argument("--urls", action="store_true", help="list every first-load request")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = asyncio.run(run(a))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
