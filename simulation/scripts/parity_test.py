# -*- coding: utf-8 -*-
"""
Python <-> browser parity of the closed loop (Live mode = web/js/live/*).

One scene, fixed seeds, depth noise off. The Python stack (runner.py, controller.py, driver.py) runs the
episode and records a trace: the physics state at t = 0 and before every control tick, its perception,
every policy call (inputs and the network's output, with the flow-ODE initial noise) and every command.
The browser stack (runner.js, controller.js, driver.js on MuJoCo WebAssembly, in headless Chrome) then
starts from the same physics state and perception, with the same policy noise:

  sync : physics re-synced to Python's before every control tick, Python's policy outputs. Any command
         difference is a logic difference between the two ports; the state just before each re-sync
         measures how far WASM physics drifts from native in one 0.1 s tick.
  free : free-running WASM physics, Python's policy outputs replayed by (side, time).

Per control tick it compares the EE targets, gripper commands, grasp-skill state / events, release phase
and the bread poses, and reports the first divergence and the max errors.

    python simulation/scripts/parity_test.py                              # Trossen, tuned, dev seed 7777
    python simulation/scripts/parity_test.py --robot ur5e --modes sync free --scene 3
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
BROWSER_PY = os.path.expanduser("~/miniconda3/envs/v2d/bin/python")
CHROME_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
               "--ignore-gpu-blocklist", "--enable-gpu", "--enable-unsafe-webgpu", "--no-sandbox"]

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>parity</title>
<script type="importmap">{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/" } }</script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.all.min.js"></script>
</head><body><pre id="out">running</pre><div id="vp" style="width:640px;height:480px"></div>
<script type="module">
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.16.0/mujoco.js';
import { runParity } from './js/live/parity.js';
const q = new URLSearchParams(location.search), key = q.get('key'), mode = q.get('mode') || 'sync';
const out = document.getElementById('out');
try {
  const mj = await loadMujoco();
  const base = `assets/live/${key}/`;
  const files = await (await fetch(base + 'files.json')).json();
  const spec = await (await fetch(base + 'spec.json')).json();
  mj.FS.mkdir('/p'); mj.FS.mkdir('/p/meshes');
  for (const f of [files.xml, ...files.files]) mj.FS.writeFile('/p/' + f, new Uint8Array(await (await fetch(base + f)).arrayBuffer()));
  const model = mj.MjModel.from_xml_path('/p/scene.xml'), data = new mj.MjData(model);
  const trace = await (await fetch('trace.json')).json();
  const tMax = q.get('tmax') ? +q.get('tmax') : null;
  let cam = null;
  if (mode === 'full') {         // the page's own perception: the viewer's three.js scene from the policy camera
    const { SimViewer } = await import('./js/viewer.js');
    const { PolicyCamera } = await import('./js/live/perception.js');
    const viewer = new SimViewer(document.getElementById('vp'));
    await viewer.loadModel(await (await fetch(`data/models/${q.get('model')}.json`)).json());
    viewer.beginLive(spec);
    cam = new PolicyCamera(viewer, viewer.model.policy_camera);
  }
  const log = await runParity({ mj, model, data, spec, trace, mode, tMax, cam, onProgress: (t) => { out.textContent = 't = ' + t.toFixed(1); } });
  window.__result = JSON.stringify(log);
  out.textContent = 'done';
} catch (e) { window.__result = JSON.stringify({ error: String(e) + '\\n' + e.stack }); out.textContent = 'error ' + e; }
</script></body></html>
"""


# ---------------------------------------------------------------------------------------------- browser side
def browser_main(a):
    """Runs in an environment with Playwright: serve the web root, run the harness page, save its log."""
    import asyncio
    from playwright.async_api import async_playwright

    async def go():
        async with async_playwright() as p:
            b = await p.chromium.launch(args=CHROME_ARGS)
            pg = await b.new_page()
            pg.on("console", lambda m: print("[chrome]", m.text) if m.type in ("error", "warning") else None)
            await pg.goto(f"http://127.0.0.1:{a.port}/parity.html?key={a.key}&mode={a.mode}&model={a.model}"
                          + (f"&tmax={a.tmax}" if a.tmax else ""))
            await pg.wait_for_function("() => window.__result !== undefined", timeout=a.timeout * 1000, polling=500)
            res = await pg.evaluate("() => window.__result")
            await b.close()
        with open(a.out, "w") as f:
            f.write(res)

    asyncio.run(go())


# ---------------------------------------------------------------------------------------------- comparison
def _rot_err_deg(r1, r2):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    return float(np.degrees((R.from_rotvec(r1) * R.from_rotvec(r2).inv()).magnitude()))


def compare(py: dict, js: dict, sides, tol_mm: float = 1.0, tol_deg: float = 0.5) -> dict:
    import numpy as np
    rep = {"mode": js["mode"], "n_ticks_py": len(py["ticks"]), "n_ticks_js": len(js["ticks"]), "first": None}
    mx = {k: 0.0 for k in ("cmd_mm", "cmd_deg", "goal_mm", "ee_mm", "bread_mm", "drift_bread_mm", "drift_ee_mm",
                           "ict", "uv")}
    series = {s: [] for s in sides}
    first = None
    n = min(len(py["ticks"]), len(js["ticks"]))
    for i in range(n):
        tp, tj = py["ticks"][i], js["ticks"][i]
        if abs(tp["t"] - tj["t"]) > 1e-6:
            first = first or {"t": tp["t"], "what": f"tick time {tp['t']} vs {tj['t']}"}
            break
        for s in sides:
            a, b = tp["sides"][s], tj["sides"][s]
            row = {"t": tp["t"]}
            if a["cmd"] is not None and b["cmd"] is not None:
                row["cmd_mm"] = 1000 * float(np.linalg.norm(np.subtract(a["cmd"][:3], b["cmd"][:3])))
                row["cmd_deg"] = _rot_err_deg(a["cmd"][3:], b["cmd"][3:])
            elif (a["cmd"] is None) != (b["cmd"] is None):
                row["cmd_missing"] = True
            row["goal_mm"] = 1000 * float(np.linalg.norm(np.subtract(a["goal"], b["goal"])))
            row["ee_mm"] = 1000 * float(np.linalg.norm(np.subtract(a["ee"][:3], b["ee"][:3])))
            row["bread_mm"] = 1000 * float(np.linalg.norm(np.subtract(a["bread"], b["bread"])))
            for k in ("cmd_mm", "cmd_deg", "goal_mm", "ee_mm", "bread_mm"):
                if k in row:
                    mx[k] = max(mx[k], row[k])
            disc = [k for k in ("g_to", "skill", "tries", "wrap", "grasping", "n_events", "mode")
                    if (abs(a[k] - b[k]) > 1e-6 if isinstance(a[k], float) else a[k] != b[k])]
            if row.get("cmd_missing"):
                disc.append("cmd")
            row["mismatch"] = disc
            series[s].append(row)
            if first is None:
                why = None
                if disc:
                    why = ", ".join(f"{k}: py {a.get(k)} / js {b.get(k)}" for k in disc)
                elif row.get("cmd_mm", 0) > tol_mm or row.get("cmd_deg", 0) > tol_deg:
                    why = f"EE target {row.get('cmd_mm', 0):.2f} mm / {row.get('cmd_deg', 0):.2f} deg"
                elif row["bread_mm"] > 5.0:
                    why = f"bread {row['bread_mm']:.1f} mm"
                if why:
                    first = {"t": tp["t"], "side": s, "what": why}
    # one tick of physics drift (sync mode): JS state just before the re-sync vs Python's state at that tick
    if js["mode"] == "sync":
        for i in range(1, min(len(js["pre"]), len(py["ticks"]))):
            for s in sides:
                mx["drift_bread_mm"] = max(mx["drift_bread_mm"], 1000 * float(np.linalg.norm(
                    np.subtract(js["pre"][i]["sides"][s]["bread"], py["ticks"][i]["sides"][s]["bread"]))))
                mx["drift_ee_mm"] = max(mx["drift_ee_mm"], 1000 * float(np.linalg.norm(
                    np.subtract(js["pre"][i]["sides"][s]["ee"][:3], py["ticks"][i]["sides"][s]["ee"][:3]))))
    # policy inputs (ICT, anchor uv, grasp flag, latching) at matching re-plans
    pk = {(round(r["t"], 4), r["side"]): r for r in py["policy"]}
    pol_mis = []
    for r in js["policy"]:
        q = pk.get((round(r["t"], 4), r["side"]))
        if q is None:
            pol_mis.append({"t": r["t"], "side": r["side"], "what": "re-plan only in the browser"})
            continue
        d_ict = float(np.max(np.abs(np.subtract(q["ict"], r["ict"]))))
        mx["ict"] = max(mx["ict"], d_ict)
        if q["uv"] is not None and r["uv"] is not None:
            mx["uv"] = max(mx["uv"], float(np.max(np.abs(np.subtract(q["uv"], r["uv"])))))
        if q["gv"] != r["gv"] or q["latched"] != r["latched"] or q["lift"] != r["lift"] or d_ict > 1e-3:
            pol_mis.append({"t": r["t"], "side": r["side"], "gv": [q["gv"], r["gv"]], "latched": [q["latched"], r["latched"]],
                            "lift": [q["lift"], r["lift"]], "ict": d_ict})
    # re-perception before a re-grasp: the browser must ask at exactly Python's (side, time)s
    rp_mis = [r for r in js.get("reperceive", []) if r["py"] == "missing"]
    n_py = len(py.get("reperceive", []))
    if len(js.get("reperceive", [])) != n_py or rp_mis:
        pol_mis.insert(0, {"t": rp_mis[0]["t"] if rp_mis else None, "side": rp_mis[0]["side"] if rp_mis else None,
                           "what": f"re-perception calls py {n_py} / browser {len(js.get('reperceive', []))}"})
    rep["n_reperceive"] = [n_py, len(js.get("reperceive", []))]
    rep["policy_mismatch"] = pol_mis[:20]
    rep["n_policy"] = [len(py["policy"]), len(js["policy"])]
    rep["first"] = first
    if first is None and pol_mis:
        rep["first"] = {"t": pol_mis[0]["t"], "side": pol_mis[0]["side"], "what": f"policy input {pol_mis[0]}"}
    rep["max"] = {k: round(v, 6) for k, v in mx.items()}
    rep["geoms_py"], rep["geoms_js"] = py["geoms"], js.get("geoms")
    if js["mode"] == "full":
        rep.update(compare_full(py, js, sides))
    rep["result_js"] = js.get("result")
    rep["series"] = series
    return rep


def compare_full(py: dict, js: dict, sides) -> dict:
    """The browser's own perception, policy images and ONNX policy against Python's."""
    import numpy as np
    out = {"perception": {}, "images": [], "onnx": []}
    for s in sides:
        gp, gj = py["geoms"].get(s, {}).get("bread"), (js.get("geoms") or {}).get(s, {}).get("bread")
        if gp and gj:
            ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(gp["a"][:2], gj["a"][:2]))))))
            out["perception"][s] = {"bread_c_mm": [round(1000 * (b - a), 2) for a, b in zip(gp["c"], gj["c"])],
                                    "axis_deg": round(float(ang), 2),
                                    "place_z_mm": round(1000 * (js["geoms"][s]["placeZ"] - py["geoms"][s]["place_z"]), 2)}
        for k in ("obj1", "obj2"):
            Tp = np.asarray(py["perception"][s]["meta"][k]["T_static"])
            Tj = np.asarray(js["perception"][s]["meta"][k]["T_static"]).reshape(4, 4)
            out["perception"][s][f"{k}_pos_mm"] = round(1000 * float(np.linalg.norm(Tp[:3, 3] - Tj[:3, 3])), 2)
            out["perception"][s][f"{k}_n_kpts"] = [len(py["perception"][s]["meta"][k]["kpts_local"]),
                                                   len(js["perception"][s]["meta"][k]["kpts_local"])]
    pimg = [r for r in py["policy"] if r.get("img")]
    for im in js.get("images", []):
        q = next((r for r in pimg if r["side"] == im["side"] and abs(r["t"] - im["t"]) < 1e-6), None)
        if q is None:
            continue
        d = np.abs(np.asarray(q["img"], float) - np.asarray(im["img"], float))
        mse = float(np.mean(d ** 2))
        out["images"].append({"t": q["t"], "side": q["side"], "mean_abs": round(float(d.mean()), 2),
                              "psnr_db": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 1),
                              "frac_px_gt_32": round(float((d.reshape(3, -1).max(0) > 32).mean()), 4)})
    for o in js.get("onnx", []):
        q = next(r for r in pimg if r["side"] == o["side"] and abs(r["t"] - o["t"]) < 1e-6)
        out["onnx"].append({"t": q["t"], "side": q["side"],
                            "pos": float(np.max(np.abs(np.subtract(q["pos"], o["pos"])))),
                            "o6d": float(np.max(np.abs(np.subtract(q["o6d"], o["o6d"])))),
                            "grasp": float(np.max(np.abs(np.subtract(q["grasp"], o["grasp"])))),
                            "done": abs(q["done"] - o["done"])})
    return out


def save_images(py: dict, js: dict, out_dir: str):
    """First policy-input images, Python | browser | |difference| x4, as PNGs."""
    import cv2
    import numpy as np
    for im in js.get("images", []):
        q = next((r for r in py["policy"] if r.get("img") and r["side"] == im["side"] and abs(r["t"] - im["t"]) < 1e-6), None)
        if q is None:
            continue
        a = np.asarray(q["img"], np.uint8).reshape(3, 240, 320).transpose(1, 2, 0)
        b = np.clip(np.asarray(im["img"]), 0, 255).astype(np.uint8).reshape(3, 240, 320).transpose(1, 2, 0)
        d = np.clip(4 * np.abs(a.astype(int) - b.astype(int)), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"img_{im['side']}_{im['t']:.1f}.png"), cv2.cvtColor(np.hstack([a, b, d]), cv2.COLOR_RGB2BGR))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="trossen_wxai")
    ap.add_argument("--params", default=None, help="params json (default configs/tuned/<robot>.json; '-' = production)")
    ap.add_argument("--set", action="append", default=[], help="extra overrides, e.g. control.alpha=0.3")
    ap.add_argument("--seed", type=int, default=7777, help="scene seed (dev seeds 7777 / 8888)")
    ap.add_argument("--scene", type=int, default=0, help="index of the scene in that seed's sequence")
    ap.add_argument("--modes", nargs="+", default=["sync", "free"])
    ap.add_argument("--tmax", type=float, default=None, help="stop the browser run at this sim time")
    ap.add_argument("--name", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep_web", action="store_true")
    # browser-only entry point (runs under the Playwright env)
    ap.add_argument("--browser", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--key", default="", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="sync", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.browser:
        return browser_main(a)

    import numpy as np
    from autotune.params import params_to_overrides
    from evaluate import parse_sets
    from humanego_sim.env import deep_update, load_config
    from humanego_sim.live_export import export_live
    from humanego_sim.runner import EpisodeRunner
    from humanego_sim.sampling import sample_scene

    extra = parse_sets(a.set)
    base = load_config(None, deep_update({"robot": a.robot}, extra))
    ppath = a.params or os.path.join(SIM, "configs", "tuned", f"{a.robot}.json")
    params = {} if ppath == "-" else json.load(open(ppath))["params"]
    ov = deep_update({"robot": a.robot, "perception": {"depth_noise": False},
                      "control": {"grasp_assist": {"enabled": ppath != "-"}}},
                     params_to_overrides(params, base) if params else {})
    ov = deep_update(ov, extra)
    cfg = load_config(None, ov)
    rng = np.random.default_rng(a.seed)
    scenes = [sample_scene(rng, cfg.get("mode", "bimanual")) for _ in range(a.scene + 1)]
    scene, ep_seed = scenes[a.scene], a.seed * 1000 + a.scene        # as evaluate.py
    name = a.name or f"{a.robot}_s{a.seed}_{a.scene}"
    out_dir = os.path.join(SIM, "runs", "parity", name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Python run with the trace on
    runner = EpisodeRunner(cfg)
    runner.trace = {}
    t0 = time.time()
    res = runner.run(scene=scene, seed=ep_seed, verbose=False, intro=False)
    tr = runner.trace
    tr["result"] = res.per_side
    tr["events"] = runner._controller.events
    print(f"[parity] python: {json.dumps({s: v['success'] for s, v in res.per_side.items()})} "
          f"t={res.duration:.1f}s ticks={len(tr['ticks'])} replans={len(tr['policy'])} ({time.time() - t0:.0f}s)")

    # ---- web root: the live JS of this checkout + a fresh physics-only export of this very model
    web = os.path.join(out_dir, "web")
    if os.path.isdir(web):
        shutil.rmtree(web)
    os.makedirs(os.path.join(web, "assets"))
    os.symlink(os.path.join(SIM, "web", "js"), os.path.join(web, "js"))
    for d in ("policy", "objects", "robots", "textures"):     # policy + the viewer's visual assets (mode full)
        os.symlink(os.path.realpath(os.path.join(SIM, "web", "assets", d)), os.path.join(web, "assets", d))
    os.symlink(os.path.realpath(os.path.join(SIM, "web", "data")), os.path.join(web, "data"))
    key = "parity"
    export_live(runner.env, web, key)
    with open(os.path.join(web, "parity.html"), "w") as f:
        f.write(HTML)
    with open(os.path.join(web, "trace.json"), "w") as f:
        json.dump(tr, f)

    port = _free_port()
    srv = subprocess.Popen([sys.executable, os.path.join(SIM, "web", "tools", "serve.py"), str(port), web],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reports = {}
    try:
        time.sleep(1.0)
        for mode in a.modes:
            jsf = os.path.join(out_dir, f"js_{mode}.json")
            t1 = time.time()
            cmd = [BROWSER_PY, os.path.abspath(__file__), "--browser", "--port", str(port), "--key", key,
                   "--mode", mode, "--out", jsf, "--timeout", str(a.timeout),
                   "--model", f"{cfg['robot']}_{cfg['camera']}_{cfg['mode']}"]
            if a.tmax:
                cmd += ["--tmax", str(a.tmax)]
            subprocess.run(cmd, check=True)
            js = json.load(open(jsf))
            if "error" in js:
                print(f"[parity] {mode}: browser error\n{js['error']}")
                continue
            rep = compare(tr, js, runner.env.sides)
            if mode == "full":
                save_images(tr, js, out_dir)
            reports[mode] = rep
            mxr = rep["max"]
            print(f"[parity] {mode}: {time.time() - t1:.0f}s  browser result "
                  f"{json.dumps({s: v.get('success') for s, v in (rep['result_js'] or {}).items()})}")
            print(f"         first divergence: {rep['first']}")
            print(f"         max: EE target {mxr['cmd_mm']:.3f} mm / {mxr['cmd_deg']:.3f} deg, driver goal {mxr['goal_mm']:.3f} mm, "
                  f"EE {mxr['ee_mm']:.3f} mm, bread {mxr['bread_mm']:.2f} mm, ICT {mxr['ict']:.2e}, uv {mxr['uv']:.2e}"
                  + (f", 1-tick drift bread {mxr['drift_bread_mm']:.4f} mm / EE {mxr['drift_ee_mm']:.4f} mm"
                     if mode == "sync" else ""))
            if mode == "full":
                print(f"         device {js.get('device')}  perception {json.dumps(rep['perception'])}")
                print(f"         first images {json.dumps(rep['images'])}")
                print(f"         ONNX vs PyTorch on identical inputs (max abs): {json.dumps(rep['onnx'])}")
    finally:
        srv.terminate()
    summ = {"robot": a.robot, "seed": a.seed, "scene": a.scene, "scene_init": scene, "episode_seed": ep_seed,
            "python_result": res.per_side, "reports": reports}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(summ, f)
    if not a.keep_web:
        os.remove(os.path.join(web, "trace.json"))
    print(f"[parity] report: {os.path.relpath(os.path.join(out_dir, 'report.json'))}")


if __name__ == "__main__":
    main()
