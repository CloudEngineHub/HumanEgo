# -*- coding: utf-8 -*-
"""Decode every packed rollout (<id>.bin) in headless Chrome with web/js/rollout.js and compare it,
number for number, with its JSON source (both must be in the served build).

    python simulation/web/tools/serve.py 8812 simulation/web &
    python simulation/web/tools/check_rollouts.py http://127.0.0.1:8812/index.html
"""
import argparse
import asyncio
import json

from playwright.async_api import async_playwright

JS = """async (base) => {
  const { decodeRollout } = await import(new URL('js/rollout.js', base).href);
  const get = (p) => fetch(new URL(p, base).href);
  const man = await (await get('data/manifest.json')).json();
  const plain = (x) => (ArrayBuffer.isView(x) ? Array.from(x) : Array.isArray(x) ? x.map(plain)
                       : (x && typeof x === 'object') ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, plain(v)])) : x);
  const diff = (a, b, path) => {
    if (typeof a === 'number' && typeof b === 'number') return a === b ? null : `${path}: ${a} != ${b}`;
    if (Array.isArray(a) || Array.isArray(b)) {
      if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return `${path}: array shape`;
      for (let i = 0; i < a.length; i++) { const d = diff(a[i], b[i], `${path}[${i}]`); if (d) return d; }
      return null;
    }
    if (a && b && typeof a === 'object') {
      const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
      if (ka.join() !== kb.join()) return `${path}: keys ${ka} vs ${kb}`;
      for (const k of ka) { const d = diff(a[k], b[k], `${path}.${k}`); if (d) return d; }
      return null;
    }
    return a === b ? null : `${path}: ${a} != ${b}`;
  };
  const out = { n: 0, ok: 0, bad: [], decode_ms: 0, parse_ms: 0, bin_bytes: 0, json_bytes: 0 };
  for (const c of man.configs) for (const r of c.rollouts) {
    if (r.fmt !== 'bin') continue;
    const bin = await (await get(`data/rollouts/${r.id}.bin`)).arrayBuffer();
    const txt = await (await get(`data/rollouts/${r.id}.json`)).text();
    let t0 = performance.now(); const a = plain(decodeRollout(bin)); out.decode_ms += performance.now() - t0;
    t0 = performance.now(); const b = JSON.parse(txt); out.parse_ms += performance.now() - t0;
    out.n++; out.bin_bytes += bin.byteLength; out.json_bytes += txt.length;
    const d = diff(a, b, r.id);
    if (d) out.bad.push(d); else out.ok++;
  }
  return out;
}"""


async def main(url):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page()
        base = url.rsplit("/", 1)[0] + "/"
        await pg.goto(base + "css/style.css")          # a same-origin document without the 3D viewer
        res = await pg.evaluate(JS, base)
        await b.close()
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    r = asyncio.run(main(ap.parse_args().url))
    raise SystemExit(0 if r["n"] and not r["bad"] else 1)
