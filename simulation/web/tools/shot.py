# -*- coding: utf-8 -*-
"""Headless screenshots of the web viewer (dev tool).

    python simulation/web/tools/shot.py http://127.0.0.1:8791/index.html out.png --view front --t 9
"""
import argparse
import asyncio

from playwright.async_api import async_playwright


async def main(a):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-angle=gl", "--ignore-gpu-blocklist", "--enable-gpu", "--no-sandbox"])
        pg = await b.new_page(viewport={"width": a.width, "height": a.height}, device_scale_factor=a.dpr)
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        url = a.url
        qs = []
        if a.view:
            qs.append(f"view={a.view}")
        if a.t is not None:
            qs.append(f"t={a.t}")
        if a.query:
            qs.append(a.query)
        if qs:
            url += ("&" if "?" in url else "?") + "&".join(qs)
        await pg.goto(url, wait_until="networkidle", timeout=120000)
        try:
            await pg.wait_for_selector("body[data-ready='1']", timeout=90000)
        except Exception:
            logs.append("[shot] page never became ready")
        await pg.wait_for_timeout(a.wait)
        if a.full:
            await pg.screenshot(path=a.out, full_page=True)
        elif a.selector:
            await pg.locator(a.selector).screenshot(path=a.out)
        else:
            await pg.screenshot(path=a.out)
        for l in logs:
            if "GPU stall" not in l and "deprecated" not in l:
                print(l)
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--view", default=None)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--query", default=None)
    ap.add_argument("--selector", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--wait", type=int, default=2500)
    asyncio.run(main(ap.parse_args()))
