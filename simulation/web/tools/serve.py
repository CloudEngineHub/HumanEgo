# -*- coding: utf-8 -*-
"""Static file server with HTTP Range support (video seeking), like GitHub Pages.

    python simulation/web/tools/serve.py 8791 simulation/web
    python simulation/web/tools/serve.py 8791 simulation/web --gzip     # + GitHub Pages' gzip (load tests)

With --gzip every response that is not a Range request and not media (video / audio / raster
image / woff2) is gzip-encoded when the browser accepts it, as GitHub Pages does (it gzips
JSON, JS, GLB, ONNX, .bin ... but not MP4 / JPEG / PNG / WebP).
"""
import gzip
import http.server
import io
import os
import re
import sys

GZIP = False
_GZ_CACHE = {}


def _compressible(ctype: str) -> bool:
    if ctype.startswith(("video/", "audio/")) or ctype == "font/woff2":
        return False
    return not ctype.startswith("image/") or ctype == "image/svg+xml"


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".glb": "model/gltf-binary", ".wasm": "application/wasm", ".mjs": "text/javascript",
                      ".webp": "image/webp", ".bin": "application/octet-stream", ".onnx": "application/octet-stream"}

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        if not rng:
            ctype = self.guess_type(path)
            if GZIP and "gzip" in self.headers.get("Accept-Encoding", "") and _compressible(ctype):
                st = os.stat(path)
                key = (path, st.st_mtime_ns, st.st_size)
                if key not in _GZ_CACHE:
                    with open(path, "rb") as f:
                        _GZ_CACHE[key] = gzip.compress(f.read(), 6)
                body = _GZ_CACHE[key]
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return io.BytesIO(body)
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            buf = source.read(min(65536, left))
            if not buf:
                break
            outputfile.write(buf)
            left -= len(buf)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        if os.environ.get("SERVE_VERBOSE"):
            super().log_message(*a)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    GZIP = "--gzip" in sys.argv
    port = int(args[0]) if args else 8791
    os.chdir(args[1] if len(args) > 1 else ".")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
