# -*- coding: utf-8 -*-
"""Tiny GLB utilities: re-encode embedded textures as JPEG (smaller web downloads)."""
from __future__ import annotations

import io
import json
import struct

from PIL import Image


def _read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, ver, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF", path
    off = 12
    js, bin_ = None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            bin_ = chunk
        off += 8 + clen
    return js, bin_


def _write_glb(path, js, bin_):
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    bin_ += b"\x00" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(jb), b"JSON") + jb)
        f.write(struct.pack("<I4s", len(bin_), b"BIN\x00") + bin_)


def jpeg_textures(path, out=None, max_size=1024, quality=85):
    js, bin_ = _read_glb(path)
    views = js.get("bufferViews", [])
    replace = {}
    for img in js.get("images", []):
        bv = img.get("bufferView")
        if bv is None:
            continue
        v = views[bv]
        raw = bin_[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_size:
            s = max_size / max(im.size)
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        replace[bv] = buf.getvalue()
        img["mimeType"] = "image/jpeg"
    new_bin = bytearray()
    for i, v in enumerate(views):
        chunk = replace.get(i)
        if chunk is None:
            o = v.get("byteOffset", 0)
            chunk = bin_[o: o + v["byteLength"]]
        while len(new_bin) % 4:
            new_bin += b"\x00"
        v["byteOffset"] = len(new_bin)
        v["byteLength"] = len(chunk)
        v["buffer"] = 0
        new_bin += chunk
    js["buffers"] = [{"byteLength": len(new_bin)}]
    _write_glb(out or path, js, bytes(new_bin))


def set_pbr(path, metallic=None, roughness=None):
    """Override metallic / roughness factors of every material in a GLB."""
    js, bin_ = _read_glb(path)
    for m in js.get("materials", []):
        pbr = m.setdefault("pbrMetallicRoughness", {})
        if metallic is not None:
            pbr["metallicFactor"] = metallic
        if roughness is not None:
            pbr["roughnessFactor"] = roughness
    _write_glb(path, js, bin_)
