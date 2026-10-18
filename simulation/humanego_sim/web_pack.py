# -*- coding: utf-8 -*-
"""
Pack the web viewer's build outputs into smaller, gzip-friendly files (after build_web.py).

  rollout .bin   data/rollouts/<id>.json -> <id>.bin: every number of the JSON as an integer at
                 exactly the precision the exporter rounded it to, columns delta-coded over time
                 (int16 where it fits). Decoding (web/js/rollout.js) gives back the same doubles
                 the JSON parses to, so replays are unchanged bit for bit.
  policy video   H.264 crf 30 with a 32-frame GOP: half the bytes of crf 28 / GOP 8 at SSIM 0.994
                 (vs 0.995); a 512x192 GOP still decodes instantly when the viewer seeks.
  robot GLB      KHR_mesh_quantization: int16 positions (uniform scale + offset on the node),
                 int8 normals, uint16 texcoords / indices, re-welded.
  bread GLB      textures -> 1024 px WebP (EXT_texture_webp), duplicate images shared, uint16
                 texcoords / indices; positions stay float32 because the viewer deforms them on
                 the CPU (web/js/softbread.js).
  textures       PNG / JPEG -> WebP, references in data/models/*.json rewritten.

    from humanego_sim.web_pack import pack_web
    pack_web("simulation/web")                  # idempotent; returns a size report
"""
from __future__ import annotations

import glob
import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

MAGIC = b"HEGR"
VERSION = 1
POLICY_VIDEO_ARGS = ["-c:v", "libx264", "-crf", "30", "-g", "32", "-preset", "veryslow", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart"]


# ============================================================================ rollouts
def _decimals(v: np.ndarray, max_d: int = 6) -> Optional[int]:
    """Smallest d with v * 10^d integral (to 1e-6): the precision the exporter rounded to."""
    v = np.asarray(v, np.float64)
    for d in range(max_d + 1):
        s = v * 10.0 ** d
        if np.all(np.abs(s - np.round(s)) < 1e-6):
            return d
    return None


def _ints(v, d: int) -> np.ndarray:
    return np.round(np.asarray(v, np.float64) * 10.0 ** d).astype(np.int64)


class _Payload:
    def __init__(self):
        self.buf = bytearray()

    def add(self, arr: np.ndarray) -> int:
        while len(self.buf) % 4:
            self.buf += b"\x00"
        off = len(self.buf)
        self.buf += np.ascontiguousarray(arr).tobytes()
        return off


def _width(a: np.ndarray) -> int:
    """Bytes per value that hold every entry of the integer array a: 1, 2 or 4."""
    if a.size == 0 or (int(a.min()) >= -128 and int(a.max()) <= 127):
        return 1
    return 2 if (int(a.min()) >= -32768 and int(a.max()) <= 32767) else 4


_DT = {1: "<i1", 2: "<i2", 4: "<i4"}


def _pack_frames(frames: List[list], pay: _Payload) -> Optional[dict]:
    F = np.asarray(frames, np.float64)
    if F.ndim != 2 or len(F) == 0:
        return None
    cols, blocks = [], {1: [], 2: [], 4: []}
    for j in range(F.shape[1]):
        d = _decimals(F[:, j])
        if d is None:
            return None
        q = _ints(F[:, j], d)
        dq = np.diff(q)                                  # column-major, delta over time
        w = _width(dq)
        blocks[w].append(dq)
        cols.append({"d": d, "base": int(q[0]), "w": w})
    out = {"n": len(F), "cols": cols}
    for w, name in ((1, "i8"), (2, "i16"), (4, "i32")):
        a = np.concatenate(blocks[w]).astype(_DT[w]) if blocks[w] else np.zeros(0, _DT[w])
        out[name] = [pay.add(a), int(a.size)]
    return out


def _pack_deform(deform: dict, pay: _Payload) -> Optional[dict]:
    out = {}
    for name, tr in deform.items():
        t, r = tr.get("t", []), tr.get("r", [])
        if set(tr) - {"t", "r"} or len(t) != len(r):
            return None
        dt = _decimals(t) if len(t) else 0
        if dt is None:
            return None
        mask = np.array([1 if (isinstance(x, list) and len(x)) else 0 for x in r], np.uint8)
        if any((not isinstance(x, list)) and x != 0 for x in r):
            return None
        rows = [x for x in r if isinstance(x, list) and len(x)]
        L = len(rows[0]) if rows else 0
        if any(len(x) != L for x in rows) or any(isinstance(v, float) and not float(v).is_integer()
                                                for x in rows for v in x):
            return None
        R = np.asarray(rows, np.int64).reshape(-1, L) if rows else np.zeros((0, 0), np.int64)
        D = np.diff(np.vstack([np.zeros((1, L), np.int64), R]), axis=0) if len(R) else R
        w = _width(D)
        data = D.astype(_DT[w]).ravel()
        out[name] = {"n": len(t), "dt": dt, "t": pay.add(_ints(t, dt).astype("<i4")), "mask": pay.add(mask),
                     "nd": len(rows), "len": L, "w": w, "data": pay.add(data)}
    return out


_SIDE_KEYS = {"traj", "traj_step", "grasp", "gmax", "done", "grasping"}


def _pack_policy(policy: List[dict], pay: _Payload) -> Optional[dict]:
    P = len(policy)
    if P == 0 or any(set(e) != {"t", "sides"} for e in policy):
        return None
    sides = sorted({s for e in policy for s in e["sides"]})
    t = [e["t"] for e in policy]
    dt = _decimals(t)
    if dt is None:
        return None
    out = {"n": P, "dt": dt, "t": pay.add(_ints(t, dt).astype("<i4")), "sides": {}}
    for s in sides:
        ents = [e["sides"].get(s) for e in policy]
        if any(x is not None and set(x) != _SIDE_KEYS for x in ents):
            return None
        present = np.array([x is not None for x in ents], np.uint8)
        E = [x for x in ents if x is not None]
        tl = np.array([len(x["traj"]) for x in E], np.int64)
        gl = np.array([len(x["grasp"]) for x in E], np.int64)
        if (tl.max(initial=0) > 255) or (gl.max(initial=0) > 255):
            return None
        pts = np.array([p for x in E for p in x["traj"]], np.float64).reshape(-1, 3)
        dpt = _decimals(pts) if len(pts) else 0
        g = np.array([v for x in E for v in x["grasp"]] + [x["gmax"] for x in E], np.float64)
        dg = _decimals(g) if len(g) else 0
        dn = _decimals([x["done"] for x in E])
        if None in (dpt, dg, dn) or dg > 2 or dn > 4:
            return None
        q = _ints(pts, dpt)
        # delta along each predicted trajectory (consecutive points are close)
        starts = np.r_[0, np.cumsum(tl)[:-1]] if len(tl) else np.zeros(0, np.int64)
        dq = q.copy()
        if len(q):
            dq[1:] = q[1:] - q[:-1]
            dq[starts[tl > 0]] = q[starts[tl > 0]]
        w = max(2, _width(dq))
        gq = _ints(g, dg)
        if gq.size and (gq.min() < 0 or gq.max() > 255):
            return None
        out["sides"][s] = {
            "present": pay.add(present), "traj_len": pay.add(tl.astype(np.uint8)), "traj_dec": dpt, "traj_w": w,
            "traj": pay.add(dq.astype(_DT[w]).ravel()), "npts": int(len(q)),
            "traj_step": pay.add(np.array([x["traj_step"] for x in E], np.uint8)),
            "grasp_len": pay.add(gl.astype(np.uint8)), "grasp_dec": dg, "grasp": pay.add(gq.astype(np.uint8)),
            "ng": int(gl.sum()), "done_dec": dn, "done": pay.add(_ints([x["done"] for x in E], dn).astype("<i4")),
            "grasping": pay.add(np.array([bool(x["grasping"]) for x in E], np.uint8)),
            "grasping_bool": all(isinstance(x["grasping"], bool) for x in E), "m": len(E),
        }
    return out


def pack_rollout(ro: dict) -> bytes:
    """Rollout dict (web_export JSON) -> HEGR container. Sections that do not fit the scheme stay JSON."""
    pay = _Payload()
    hdr = {k: v for k, v in ro.items() if k not in ("frames", "deform", "policy")}
    pack = {}
    for key, fn in (("frames", _pack_frames), ("deform", _pack_deform), ("policy", _pack_policy)):
        if key not in ro:
            continue
        r = fn(ro[key], pay) if ro[key] else None
        if r is None:
            hdr[key] = ro[key]
        else:
            pack[key] = r
    hdr["pack"] = pack
    jb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    return MAGIC + struct.pack("<II", VERSION, len(jb)) + jb + bytes(pay.buf)


def unpack_rollout(data: bytes) -> dict:
    """Reference decoder (same arithmetic as web/js/rollout.js): HEGR container -> rollout dict."""
    assert data[:4] == MAGIC, "not a HEGR rollout"
    ver, jl = struct.unpack_from("<II", data, 4)
    assert ver == VERSION
    hdr = json.loads(data[12:12 + jl].decode("utf-8"))
    P0 = 12 + jl
    view = lambda dt, off, n: np.frombuffer(data, dt, n, P0 + off)   # noqa: E731
    pack = hdr.pop("pack", {})
    ro = dict(hdr)
    if "frames" in pack:
        f = pack["frames"]
        T, C = f["n"], len(f["cols"])
        blk = {1: view("<i1", *f["i8"]), 2: view("<i2", *f["i16"]), 4: view("<i4", *f["i32"])}
        off = {1: 0, 2: 0, 4: 0}
        out = np.zeros((T, C), np.float64)
        for j, c in enumerate(f["cols"]):
            w = c["w"]
            d = blk[w][off[w]:off[w] + T - 1]
            off[w] += T - 1
            q = c["base"] + np.r_[0, np.cumsum(d.astype(np.int64))]
            out[:, j] = q / 10 ** c["d"]
        ro["frames"] = out.tolist()
    if "deform" in pack:
        ro["deform"] = {}
        for name, d in pack["deform"].items():
            n = d["n"]
            t = view("<i4", d["t"], n).astype(np.int64)
            mask = view(np.uint8, d["mask"], n)
            D = view(_DT[d["w"]], d["data"], d["nd"] * d["len"]).astype(np.int64)
            R = np.cumsum(D.reshape(d["nd"], d["len"]), axis=0) if d["nd"] else np.zeros((0, 0), np.int64)
            rows, k = [], 0
            for m in mask:
                if m:
                    rows.append(R[k].tolist())
                    k += 1
                else:
                    rows.append(0)
            ro["deform"][name] = {"t": (t / 10 ** d["dt"]).tolist(), "r": rows}
    if "policy" in pack:
        p = pack["policy"]
        n = p["n"]
        t = view("<i4", p["t"], n).astype(np.int64) / 10 ** p["dt"]
        pol = [{"t": float(t[i]), "sides": {}} for i in range(n)]
        for s, d in p["sides"].items():
            m = d["m"]
            present = view(np.uint8, d["present"], n)
            tl = view(np.uint8, d["traj_len"], m).astype(np.int64)
            dq = view(_DT[d["traj_w"]], d["traj"], 3 * d["npts"]).astype(np.int64).reshape(-1, 3)
            q = np.zeros_like(dq)
            o = 0
            for L in tl:
                if L:
                    q[o:o + L] = np.cumsum(dq[o:o + L], axis=0)
                o += L
            pts = q / 10 ** d["traj_dec"]
            step = view(np.uint8, d["traj_step"], m)
            gl = view(np.uint8, d["grasp_len"], m).astype(np.int64)
            gq = view(np.uint8, d["grasp"], d["ng"] + m).astype(np.int64)
            g = gq / 10 ** d["grasp_dec"]
            done = view("<i4", d["done"], m).astype(np.int64) / 10 ** d["done_dec"]
            gr = view(np.uint8, d["grasping"], m)
            k = op = og = 0
            for i in range(n):
                if not present[i]:
                    continue
                L, G = int(tl[k]), int(gl[k])
                pol[i]["sides"][s] = {"traj": pts[op:op + L].tolist(), "traj_step": int(step[k]),
                                      "grasp": g[og:og + G].tolist(), "gmax": float(g[d["ng"] + k]),
                                      "done": float(done[k]),
                                      "grasping": bool(gr[k]) if d["grasping_bool"] else int(gr[k])}
                op += L
                og += G
                k += 1
        ro["policy"] = pol
    return ro


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b and type(a) is type(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def pack_rollout_file(json_path: str, bin_path: Optional[str] = None, verify: bool = True) -> Tuple[str, int, int]:
    bin_path = bin_path or json_path[:-5] + ".bin"
    with open(json_path) as f:
        ro = json.load(f)
    data = pack_rollout(ro)
    if verify and not _same(ro, unpack_rollout(data)):
        raise ValueError(f"{json_path}: packed rollout does not decode to the JSON")
    with open(bin_path, "wb") as f:
        f.write(data)
    return bin_path, os.path.getsize(json_path), len(data)


# ============================================================================ policy videos
def encode_policy_video(frames_bgr: List[np.ndarray], out: str, fps: float):
    """Policy-input video (one tile per arm) at the packed settings."""
    tmp = tempfile.mkdtemp(prefix="polvid_")
    try:
        import cv2
        for i, im in enumerate(frames_bgr):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), im)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.6f}", "-i", os.path.join(tmp, "%05d.png"),
                        *POLICY_VIDEO_ARGS, out], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def policy_tiles(log_dir: str, sides: List[str], tile=(256, 192)) -> Tuple[List[np.ndarray], float]:
    """The policy-view tiles of a logged episode, as web_export.export_rollout composes them."""
    import cv2
    with open(os.path.join(log_dir, "episode.json")) as f:
        pol = json.load(f)["policy"]
    TW, TH = tile
    last = {s: np.zeros((TH, TW, 3), np.uint8) for s in sides}
    frames = []
    for st in pol:
        for s in sides:
            e = st["sides"].get(s)
            if e and e.get("image"):
                im = cv2.imread(os.path.join(log_dir, "policy_view", e["image"]))
                if im is not None:
                    last[s] = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
        frames.append(np.hstack([last[s] for s in sides]))
    P = pol[1]["t"] - pol[0]["t"] if len(pol) > 1 else 0.3
    return frames, 1.0 / P


def reencode_policy_video(mp4: str, log_dir: Optional[str] = None, sides: Optional[List[str]] = None,
                          n_frames: Optional[int] = None) -> int:
    """Re-encode one policy video at the packed settings: from the logged policy views when the
    episode log is there and matches (`n_frames` policy steps; no generation loss), else from the
    existing MP4. Returns the new size."""
    tag = mp4 + ".packed"
    if os.path.exists(tag) and os.path.getmtime(tag) >= os.path.getmtime(mp4):
        return os.path.getsize(mp4)
    tmp_out = mp4[:-4] + ".tmp.mp4"
    frames = None
    if log_dir and sides and os.path.isdir(os.path.join(log_dir, "policy_view")):
        frames, fps = policy_tiles(log_dir, sides)
        if n_frames is not None and len(frames) != n_frames:
            frames = None                               # a different episode: do not mix them up
    if frames:
        encode_policy_video(frames, tmp_out, fps)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, *POLICY_VIDEO_ARGS, tmp_out], check=True)
    if os.path.getsize(tmp_out) < os.path.getsize(mp4):
        os.replace(tmp_out, mp4)
    else:
        os.remove(tmp_out)
    open(tag, "w").close()
    return os.path.getsize(mp4)


# ============================================================================ GLB
COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_accessor(js, bin_, i) -> np.ndarray:
    a = js["accessors"][i]
    v = js["bufferViews"][a["bufferView"]]
    dt = np.dtype(COMP[a["componentType"]]).newbyteorder("<")
    nc = NCOMP[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", 0) or dt.itemsize * nc
    raw = np.frombuffer(bin_, np.uint8, a["count"] * stride, off).reshape(a["count"], stride)
    out = raw[:, :dt.itemsize * nc].copy().view(dt).reshape(a["count"], nc)
    if a.get("normalized"):
        info = np.iinfo(dt)
        out = np.maximum(out.astype(np.float64) / info.max, -1.0)
    return out if nc > 1 else out.ravel()


class _GlbWriter:
    """Rebuild buffer views / accessors of a glTF JSON around new binary data."""

    def __init__(self, js):
        self.js = js
        self.bin = bytearray()
        self.views, self.accs = [], []

    def view(self, data: bytes, stride: int = 0, target: Optional[int] = None) -> int:
        while len(self.bin) % 4:
            self.bin += b"\x00"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if stride:
            v["byteStride"] = stride
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, typ: str, normalized=False, minmax=False, pad_to=0,
                 target=34962) -> int:
        arr = np.ascontiguousarray(arr)
        n = arr.shape[0]
        nc = NCOMP[typ]
        isz = np.dtype(COMP[ctype]).itemsize
        row = isz * nc
        stride = row if not pad_to else max(pad_to, row)
        if stride != row:                                   # vertex attributes: 4-byte aligned elements
            buf = np.zeros((n, stride), np.uint8)
            buf[:, :row] = arr.reshape(n, nc).astype(COMP[ctype]).view(np.uint8).reshape(n, row)
            data = buf.tobytes()
        else:
            data = arr.astype(COMP[ctype]).tobytes()
        bv = self.view(data, stride if (target == 34962 and stride != row) else 0, target)
        a = {"bufferView": bv, "componentType": ctype, "count": int(n), "type": typ}
        if normalized:
            a["normalized"] = True
        if minmax:
            a2 = arr.reshape(n, nc)
            a["min"] = [v.item() for v in a2.min(0)]
            a["max"] = [v.item() for v in a2.max(0)]
        self.accs.append(a)
        return len(self.accs) - 1


def _read_glb(path):
    from .glb_tools import _read_glb as rd
    return rd(path)


def _write_glb(path, js, bin_):
    from .glb_tools import _write_glb as wr
    wr(path, js, bin_)


def _quantize_uv(uv: np.ndarray):
    if uv.size and uv.min() >= 0.0 and uv.max() <= 1.0:
        return np.round(uv * 65535).astype(np.uint16), 5123, True
    return uv.astype(np.float32), 5126, False


def _flat_normals(pos: np.ndarray, nrm: np.ndarray, idx: np.ndarray, tol_deg: float = 2.0) -> float:
    """Fraction of triangle corners whose normal is the triangle's own normal (flat shading)."""
    I = idx.reshape(-1, 3)
    fn = np.cross(pos[I[:, 1]] - pos[I[:, 0]], pos[I[:, 2]] - pos[I[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-14
    if not ok.any():
        return 0.0
    fn = fn[ok] / ln[ok, None]
    c = np.concatenate([(nrm[I[ok, k]] * fn).sum(1) for k in range(3)])
    return float((c > np.cos(np.radians(tol_deg))).mean())


def quantize_glb(path: str, out: Optional[str] = None, quantize_positions: bool = True,
                 flat_frac: float = 0.95) -> dict:
    """KHR_mesh_quantization for static meshes (robots). Images / materials / samplers are kept.
    Meshes whose normals are (>= flat_frac of corners) just the face normals lose them and are
    welded by position: the viewer rebuilds flat normals (Menagerie's Franka: 5x fewer vertices)."""
    js, bin_ = _read_glb(path)
    if "KHR_mesh_quantization" in js.get("extensionsUsed", []):
        return {"skipped": True}
    w = _GlbWriter(js)
    img_views = {}
    for im in js.get("images", []):                         # carry embedded images over unchanged
        if "bufferView" in im:
            v = js["bufferViews"][im["bufferView"]]
            o = v.get("byteOffset", 0)
            im["bufferView"] = img_views.setdefault(im["bufferView"], w.view(bytes(bin_[o:o + v["byteLength"]])))
    node_of_mesh = {n["mesh"]: n for n in js.get("nodes", []) if "mesh" in n}
    n_in = n_out = n_flat = 0
    for mi, mesh in enumerate(js["meshes"]):
        node = node_of_mesh.get(mi)
        prims = mesh["primitives"]
        allpos = np.vstack([_read_accessor(js, bin_, p["attributes"]["POSITION"]) for p in prims]).astype(np.float64)
        lo, hi = allpos.min(0), allpos.max(0)
        c = 0.5 * (lo + hi)
        s = float(max(np.abs(allpos - c).max(), 1e-9))
        can_move = quantize_positions and node is not None and not any(k in node for k in ("matrix", "rotation",
                                                                                             "scale", "translation"))
        for p in prims:
            A = p["attributes"]
            pos = _read_accessor(js, bin_, A["POSITION"]).astype(np.float64)
            idx = _read_accessor(js, bin_, p["indices"]).astype(np.int64) if "indices" in p else np.arange(len(pos))
            cols, n_in = [], n_in + len(pos)
            if can_move:
                qp = np.round((pos - c) / s * 32767).astype(np.int64)
            else:
                qp = np.round(pos * 1e6).astype(np.int64)    # weld only
            cols.append(qp)
            nrm = None
            if "NORMAL" in A:
                nrm = _read_accessor(js, bin_, A["NORMAL"]).astype(np.float64)
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                if _flat_normals(pos, nrm, idx) >= flat_frac:
                    nrm = None
                    del A["NORMAL"]
                    n_flat += 1
                else:
                    cols.append(np.round(nrm * 127).astype(np.int64))
            uvq = None
            if "TEXCOORD_0" in A:
                uv = _read_accessor(js, bin_, A["TEXCOORD_0"]).astype(np.float64)
                uvq, uvt, uvn = _quantize_uv(uv)
                cols.append(uvq.astype(np.int64) if uvt == 5123 else np.round(uv * 1e6).astype(np.int64))
            other = set(A) - {"POSITION", "NORMAL", "TEXCOORD_0"}
            if other:
                raise ValueError(f"{path}: unsupported attributes {other}")
            key = np.hstack(cols)
            uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inv = inv.ravel()
            n_out += len(uniq)
            idx2 = inv[idx]
            if can_move:
                A["POSITION"] = w.accessor(qp[first], 5122, "VEC3", normalized=True, minmax=True, pad_to=8)
            else:
                A["POSITION"] = w.accessor(pos[first].astype(np.float32), 5126, "VEC3", minmax=True)
            if nrm is not None:
                A["NORMAL"] = w.accessor(np.round(nrm[first] * 127), 5120, "VEC3", normalized=True, pad_to=4)
            if uvq is not None:
                A["TEXCOORD_0"] = (w.accessor(uvq[first], 5123, "VEC2", normalized=True) if uvt == 5123
                                   else w.accessor(uvq[first], 5126, "VEC2"))
            ict = 5123 if len(uniq) < 65536 else 5125
            p["indices"] = w.accessor(idx2.reshape(-1, 1), ict, "SCALAR", target=34963)
        if can_move:
            node["translation"] = [float(v) for v in c]
            node["scale"] = [s, s, s]
    js["bufferViews"], js["accessors"] = w.views, w.accs
    js["buffers"] = [{"byteLength": len(w.bin)}]
    ext = set(js.get("extensionsUsed", [])) | {"KHR_mesh_quantization"}
    js["extensionsUsed"] = sorted(ext)
    js["extensionsRequired"] = sorted(set(js.get("extensionsRequired", [])) | {"KHR_mesh_quantization"})
    before = os.path.getsize(path)
    _write_glb(out or path, js, bytes(w.bin))
    return {"verts_in": n_in, "verts_out": n_out, "flat_meshes": n_flat, "bytes_in": before,
            "bytes_out": os.path.getsize(out or path)}


def _webp(raw: bytes, max_size: int, quality: int, lossless: bool = False) -> Tuple[bytes, tuple]:
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") and "A" in im.getbands() else "RGB")
    if max(im.size) > max_size:
        s = max_size / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6, lossless=lossless)
    return buf.getvalue(), im.size


def webp_glb(path: str, out: Optional[str] = None, max_size: int = 1024, quality: int = 88,
             normal_quality: int = 92) -> dict:
    """Bread GLBs: embedded textures -> WebP at <= max_size px (EXT_texture_webp), identical images
    shared, texcoords / indices quantized; float32 positions (deformed on the CPU in the viewer)."""
    js, bin_ = _read_glb(path)
    if "EXT_texture_webp" in js.get("extensionsUsed", []):
        return {"skipped": True}
    before = os.path.getsize(path)
    normal_imgs = {js["textures"][m["normalTexture"]["index"]]["source"] for m in js.get("materials", [])
                   if "normalTexture" in m}
    w = _GlbWriter(js)
    # images: dedupe identical bytes, re-encode
    seen, new_src = {}, {}
    images = []
    for i, im in enumerate(js.get("images", [])):
        v = js["bufferViews"][im["bufferView"]]
        o = v.get("byteOffset", 0)
        raw = bytes(bin_[o:o + v["byteLength"]])
        if raw in seen:
            new_src[i] = seen[raw]
            continue
        data, size = _webp(raw, max_size, normal_quality if i in normal_imgs else quality)
        images.append({"bufferView": w.view(data), "mimeType": "image/webp"})
        seen[raw] = new_src[i] = len(images) - 1
    js["images"] = images
    for t in js.get("textures", []):
        src = new_src[t.pop("source")]
        t.setdefault("extensions", {})["EXT_texture_webp"] = {"source": src}
    for m in js["meshes"]:
        for p in m["primitives"]:
            A = p["attributes"]
            for k in list(A):
                arr = _read_accessor(js, bin_, A[k])
                if k == "TEXCOORD_0":
                    q, ct, nrm = _quantize_uv(arr.astype(np.float64))
                    A[k] = w.accessor(q, ct, "VEC2", normalized=nrm)
                elif k == "POSITION":
                    A[k] = w.accessor(arr.astype(np.float32), 5126, "VEC3", minmax=True)
                elif k == "NORMAL":
                    A[k] = w.accessor(arr.astype(np.float32), 5126, "VEC3")
                else:
                    raise ValueError(f"{path}: unsupported attribute {k}")
            if "indices" in p:
                idx = _read_accessor(js, bin_, p["indices"]).astype(np.int64)
                n = w.accs[A["POSITION"]]["count"]
                p["indices"] = w.accessor(idx.reshape(-1, 1), 5123 if n < 65536 else 5125, "SCALAR", target=34963)
    js["bufferViews"], js["accessors"] = w.views, w.accs
    js["buffers"] = [{"byteLength": len(w.bin)}]
    js["extensionsUsed"] = sorted(set(js.get("extensionsUsed", [])) | {"EXT_texture_webp"})
    js["extensionsRequired"] = sorted(set(js.get("extensionsRequired", [])) | {"EXT_texture_webp"})
    _write_glb(out or path, js, bytes(w.bin))
    return {"bytes_in": before, "bytes_out": os.path.getsize(out or path)}


# ============================================================================ site
def webp_texture(path: str, quality: int = 90) -> str:
    out = os.path.splitext(path)[0] + ".webp"
    if not (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path)):
        with open(path, "rb") as f:
            data, _ = _webp(f.read(), 4096, quality)
        with open(out, "wb") as f:
            f.write(data)
    return out


def pack_web(web: str, log_root: Optional[str] = None, rollouts: bool = True, videos: bool = True,
             glbs: bool = True, textures: bool = True, verbose: bool = True) -> Dict[str, dict]:
    """Pack a built web/ directory in place (idempotent). `log_root`: runs/web, where build_web
    keeps the episode logs (<key>/epNNN), used to re-encode policy videos without generation loss."""
    rep: Dict[str, dict] = {}
    data = os.path.join(web, "data")
    man_p = os.path.join(data, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"configs": []}
    if rollouts:
        n, a, b = 0, 0, 0
        for c in man["configs"]:
            for r in c["rollouts"]:
                jp = os.path.join(data, "rollouts", f"{r['id']}.json")
                bp = jp[:-5] + ".bin"
                if not os.path.exists(jp):
                    continue
                if not (os.path.exists(bp) and os.path.getmtime(bp) >= os.path.getmtime(jp)):
                    pack_rollout_file(jp, bp)
                r["fmt"] = "bin"
                n, a, b = n + 1, a + os.path.getsize(jp), b + os.path.getsize(bp)
        rep["rollouts"] = {"n": n, "json_bytes": a, "bin_bytes": b}
    if videos:
        n, a, b = 0, 0, 0
        for c in man["configs"]:
            sides = ["right"] if c["mode"] == "single" else ["right", "left"]
            for r in c["rollouts"]:
                mp4 = os.path.join(data, "rollouts", f"{r['id']}_policy.mp4")
                if not os.path.exists(mp4):
                    continue
                a += os.path.getsize(mp4)
                ld = os.path.join(log_root, c["model"], f"ep{int(r['id'].rsplit('_', 1)[1]):03d}") if log_root else None
                jp = os.path.join(data, "rollouts", f"{r['id']}.json")
                n_pol = len(json.load(open(jp))["policy"]) if os.path.exists(jp) else None
                b += reencode_policy_video(mp4, ld, sides, n_pol)
                n += 1
        rep["videos"] = {"n": n, "bytes_before": a, "bytes_after": b}
    if glbs:
        rep["glb"] = {}
        for p in sorted(glob.glob(os.path.join(web, "assets", "robots", "*.glb"))):
            rep["glb"][os.path.relpath(p, web)] = quantize_glb(p)
        for p in sorted(glob.glob(os.path.join(web, "assets", "objects", "*.glb"))):
            js, _ = _read_glb(p)
            if js.get("images"):
                rep["glb"][os.path.relpath(p, web)] = webp_glb(p)
    if textures:
        rep["textures"] = {}
        conv = {}
        for p in sorted(glob.glob(os.path.join(web, "assets", "textures", "*"))):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                out = webp_texture(p)
                conv[os.path.relpath(p, web)] = os.path.relpath(out, web)
                rep["textures"][os.path.relpath(p, web)] = {"bytes_in": os.path.getsize(p),
                                                            "bytes_out": os.path.getsize(out)}
        for mp in glob.glob(os.path.join(data, "models", "*.json")):
            model = json.load(open(mp))
            changed = False
            for spec in model.get("materials", {}).values():
                if spec.get("map") in conv:
                    spec["map"] = conv[spec["map"]]
                    changed = True
            if changed:
                with open(mp, "w") as f:
                    json.dump(model, f, separators=(",", ":"))
    man["packed"] = {"version": VERSION}
    if man["configs"]:
        with open(man_p, "w") as f:
            json.dump(man, f, separators=(",", ":"))
    if verbose:
        print("[pack]", json.dumps(rep, default=int))
    return rep


def deploy_skip(web: str) -> set:
    """Build outputs the deployed site does not need once packed (JSON rollouts with a .bin, PNG/JPEG
    textures with a WebP, pack markers), as paths relative to `web`."""
    skip = set()
    for jp in glob.glob(os.path.join(web, "data", "rollouts", "*.json")):
        if os.path.exists(jp[:-5] + ".bin"):
            skip.add(os.path.relpath(jp, web))
    for p in glob.glob(os.path.join(web, "data", "rollouts", "*.packed")):
        skip.add(os.path.relpath(p, web))
    for p in glob.glob(os.path.join(web, "assets", "textures", "*")):
        if p.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(os.path.splitext(p)[0] + ".webp"):
            skip.add(os.path.relpath(p, web))
    return skip
