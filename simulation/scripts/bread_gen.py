# -*- coding: utf-8 -*-
"""
Procedural models of the two breads used on the real HumanEgo rig.

* ``croissant``  the straight rolled croissant of the human demos and the robot
                 rollouts: ~7 chevron bands, glossy golden-orange egg wash, pale
                 yellow dough in the grooves, tips resting on the table.
* ``bun``        the laminated danish loaf of the bimanual rollouts: blunt oval
                 loaf, diagonal folds across the dome, white sugar crystals on top.

Each bread is a closed tube over (s, theta): s in [0, 1] runs along the length,
theta around the cross-section (theta = 0 under the bread, so the UV seam is
hidden). The same surface function gives

  * a dense textured visual mesh (UV = (s, theta / 2pi)) with base-colour,
    roughness and normal maps painted in the same (s, theta) space, and
  * a coarse closed surface for the MuJoCo flex (soft-body collision geometry).

Sizes are bounding boxes in metres (length x width x height), measured from the
released dataset (croissant) and from the real rollout video against the
24.5 cm plate (bun).
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

TAU = 2.0 * np.pi


# ----------------------------------------------------------------------------- noise
def _value_noise(shape, cells, rng, octaves=4, persistence=0.55):
    """Tileable-in-u fBm value noise on a (H, W) grid; cells = coarse cell count along W."""
    H, W = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cw = max(2, int(cells * 2 ** o))
        ch = max(2, int(round(cw * H / W)))
        g = rng.random((ch + 1, cw + 1))
        g[:, -1] = g[:, 0]          # wrap along u
        g[-1, :] = g[0, :]          # and along v (theta wraps)
        yy = np.linspace(0, ch, H, endpoint=False)
        xx = np.linspace(0, cw, W, endpoint=False)
        y0 = yy.astype(int); x0 = xx.astype(int)
        fy = (yy - y0)[:, None]; fx = (xx - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        tot += amp
        amp *= persistence
    return out / tot


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _srgb(c):
    return np.asarray(c, dtype=np.float64) / 255.0


# ----------------------------------------------------------------------------- spec
@dataclass
class BreadSpec:
    name: str
    size: Tuple[float, float, float]            # bounding box L x W x H (m)
    half_width: Callable                         # s -> half width (unitless, max 1)
    half_height: Callable                        # s -> top half height (unitless, max 1)
    bottom: float                                # bottom half-height as a fraction of the top
    n_top: float                                 # cross-section superellipse exponent (top)
    n_bot: float                                 # (bottom: large = flat)
    spine_y: Callable                            # s -> lateral bow (unitless, x W/2)
    relief: Callable                             # (s, theta) -> radial relief (fraction, <= 0 grooves)
    paint: Callable                              # (s, theta, rng) -> rgb, rough, height (texture space)


def _section(sp: BreadSpec, s, th, with_relief=True):
    """Surface points (N, 3) for parameter arrays s, th (unscaled: x in [-1, 1])."""
    s = np.asarray(s, float); th = np.asarray(th, float)
    hw = sp.half_width(s); hh = sp.half_height(s)
    c, sn = np.cos(th), np.sin(th)
    # theta = 0 is the bottom: y ~ sin, z ~ -cos
    y = np.sign(sn) * np.abs(sn) ** (2.0 / sp.n_top)
    zc = -c
    up = zc >= 0
    z = np.where(up, np.abs(zc) ** (2.0 / sp.n_top), -np.abs(zc) ** (2.0 / sp.n_bot))
    yb = np.sign(sn) * np.abs(sn) ** (2.0 / np.where(up, sp.n_top, sp.n_bot))
    y = np.where(up, y, yb)
    r = 1.0 + (sp.relief(s, th) if with_relief else 0.0)
    rz_bot = hh * sp.bottom
    zc_line = rz_bot                         # every section rests on the table (z = 0)
    X = 2.0 * s - 1.0
    Y = sp.spine_y(s) + hw * y * r
    Z = zc_line + np.where(up, hh, rz_bot) * z * np.where(up, r, 1.0)
    return np.stack([X, Y, Z], -1)


def _scale_to_box(V, size):
    lo, hi = V.min(0), V.max(0)
    V = (V - (lo + hi) / 2) / (hi - lo) * np.asarray(size)
    return V


def tube_mesh(sp: BreadSpec, ns: int, nt: int, with_relief=True, size=None):
    """Closed tube: (ns - 1) interior rings + 2 tip vertices. Returns V, F, UV, (s, th) per vertex."""
    s = np.linspace(0, 1, ns + 1)[1:-1]
    # cluster rings toward the tips so the rounded ends are resolved
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    th = np.linspace(0, TAU, nt, endpoint=False)
    S, T = np.meshgrid(s, th, indexing="ij")
    P = _section(sp, S.ravel(), T.ravel(), with_relief).reshape(len(s), nt, 3)
    # tips: centre of the last ring pushed slightly outward
    def tip(ring, ds):
        c = ring.mean(0)
        return c + np.array([ds, 0, 0])
    gap = (P[1, :, 0].mean() - P[0, :, 0].mean())
    t0 = tip(P[0], -0.6 * gap); t1 = tip(P[-1], 0.6 * gap)
    V = np.concatenate([P.reshape(-1, 3), t0[None], t1[None]])
    n_r = len(s)
    F = []
    idx = lambda i, j: i * nt + (j % nt)
    for i in range(n_r - 1):
        for j in range(nt):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            F += [[a, c, b], [b, c, d]]
    i0, i1 = n_r * nt, n_r * nt + 1
    for j in range(nt):
        F.append([i0, idx(0, j), idx(0, j + 1)])
        F.append([i1, idx(n_r - 1, j + 1), idx(n_r - 1, j)])
    F = np.asarray(F, np.int64)
    ST = np.concatenate([np.stack([S.ravel(), T.ravel()], -1), [[0.0, 0.0], [1.0, 0.0]]])
    if size is not None:
        V = _scale_to_box(V, size)
    # outward winding check (volume must be positive)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    if vol < 0:
        F = F[:, [0, 2, 1]]
    return V, F, ST


def uv_mesh(sp: BreadSpec, ns: int, nt: int, size):
    """Visual mesh with a UV seam column duplicated at theta = 2pi and per-ring tip fans."""
    s = np.linspace(0, 1, ns + 1)[1:-1]
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    th = np.linspace(0, TAU, nt + 1)                      # seam duplicated
    S, T = np.meshgrid(s, th, indexing="ij")
    P = _section(sp, S.ravel(), T.ravel()).reshape(len(s), nt + 1, 3)
    gap = (P[1, :, 0].mean() - P[0, :, 0].mean())
    tips = [P[0, :-1].mean(0) + [-0.6 * gap, 0, 0], P[-1, :-1].mean(0) + [0.6 * gap, 0, 0]]
    n_r = len(s); W = nt + 1
    V = [P.reshape(-1, 3)]; UV = [np.stack([S.ravel(), T.ravel() / TAU], -1)]
    F = []
    idx = lambda i, j: i * W + j
    for i in range(n_r - 1):
        for j in range(nt):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            F += [[a, c, b], [b, c, d]]
    base = n_r * W
    # one tip vertex per segment so UVs do not smear
    tv, tuv = [], []
    for k, (ring, su) in enumerate([(0, 0.0), (n_r - 1, 1.0)]):
        for j in range(nt):
            tv.append(tips[k]); tuv.append([su, (j + 0.5) / nt])
            t = base + k * nt + j
            if k == 0:
                F.append([t, idx(ring, j), idx(ring, j + 1)])
            else:
                F.append([t, idx(ring, j + 1), idx(ring, j)])
    V = np.concatenate(V + [np.asarray(tv)]); UV = np.concatenate(UV + [np.asarray(tuv)])
    F = np.asarray(F, np.int64)
    # scale with the SAME transform as the relief-free collision tube would use: use own bbox
    V = _scale_to_box(V, size)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    if vol < 0:
        F = F[:, [0, 2, 1]]
    return V, F, UV


def paint_textures(sp: BreadSpec, W=2048, H=1024, seed=0):
    """Base colour (sRGB uint8), roughness (uint8) and tangent-space normal map (uint8)."""
    rng = np.random.default_rng(seed)
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    S, TH = np.meshgrid(u, v * TAU)                     # (H, W)
    rgb, rough, height = sp.paint(S, TH, rng)
    # relief also shades the albedo slightly (cavity) and enters the height map
    rel = sp.relief(S, TH)
    height = height + 6.0 * rel
    # normal map from the height field (texel units -> tangent space, u along s, v around)
    gy, gx = np.gradient(height)
    strength = 2.2
    n = np.stack([-gx * strength * W / 512, -gy * strength * H / 256, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    col = (np.clip(rgb, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)
    rgh = (np.clip(rough, 0, 1) * 255).astype(np.uint8)
    return col, rgh, nrm


# ----------------------------------------------------------------------------- croissant
def _croissant() -> BreadSpec:
    N_HALF = 3.3          # bands from the centre to each tip (~7 in total)
    TILT = 0.28           # chevron: grooves lean toward the centre on top
    A = 0.16              # groove depth (fraction of the local radius)

    def band(s, th):
        d = np.abs(2 * s - 1)                        # 0 centre .. 1 tip
        top = np.clip(-np.cos(th), 0, 1)             # 1 on top, 0 at the sides / bottom
        b = N_HALF * d + TILT * top * (0.4 + 1.2 * d) + 0.5
        return b

    def lobe(s, th):
        b = band(s, th)
        f = b - np.floor(b)
        return np.sqrt(np.clip(np.sin(np.pi * f), 0, 1))   # rounded roll, pinched groove

    def groove(s, th):
        return np.clip(1 - lobe(s, th), 0, 1) ** 2.5

    def relief(s, th):
        d = np.abs(2 * s - 1)
        side = np.clip(-np.cos(th) + 0.45, 0, 1)    # grooves fade under the bread
        return -A * (1 - lobe(s, th)) * side * (1 - 0.4 * d ** 3)

    def half_width(s):
        x = 2 * s - 1
        return (1 - np.abs(x) ** 2.0) ** 0.68 * (1 - 0.12 * x ** 2)

    def half_height(s):
        x = 2 * s - 1
        return (1 - np.abs(x) ** 1.8) ** 0.85

    def spine_y(s):
        x = 2 * s - 1
        return 0.10 * (1 - x ** 2)                   # gentle bow

    def paint(S, TH, rng):
        H, W = S.shape
        g = groove(S, TH)
        d = np.abs(2 * S - 1)
        top = np.clip(-np.cos(TH), 0, 1)
        n1 = _value_noise((H, W), 10, rng, 5)
        n2 = _value_noise((H, W), 64, rng, 3)
        crest_c = _srgb((214, 120, 34))
        crest_hi = _srgb((236, 156, 52))
        groove_c = _srgb((246, 206, 120))
        under_c = _srgb((222, 160, 80))
        tip_c = _srgb((168, 82, 26))
        t = np.clip(0.55 * top + 0.45 * n1 + 0.15 * (n2 - 0.5), 0, 1)
        col = crest_hi[None, None] * (1 - t[..., None]) + crest_c[None, None] * t[..., None]
        col = col * (1 - g[..., None] * 0.9) + groove_c[None, None] * g[..., None] * 0.9
        under = _smoothstep(0.25, -0.35, -np.cos(TH))       # bottom of the bread
        col = col * (1 - under[..., None]) + under_c[None, None] * under[..., None]
        tipw = _smoothstep(0.78, 1.0, d) * (0.6 + 0.4 * n1)
        col = col * (1 - tipw[..., None]) + tip_c[None, None] * tipw[..., None]
        col *= (0.92 + 0.16 * n2)[..., None]
        rough = 0.32 + 0.25 * g + 0.15 * n2 + 0.2 * under
        flakes = _value_noise((H, W), 180, rng, 2)
        height = 1.5 * n2 + 2.5 * (flakes - 0.5) * (1 - g)
        return col, rough, height

    return BreadSpec("croissant", (0.138, 0.062, 0.050), half_width, half_height,
                     bottom=0.62, n_top=2.0, n_bot=3.2, spine_y=spine_y, relief=relief, paint=paint)


# ----------------------------------------------------------------------------- danish loaf
def _bun() -> BreadSpec:
    N_FOLD = 6.5          # diagonal laminated folds across the dome
    SLANT = 1.4
    A = 0.022

    def fold(s, th):
        top = -np.cos(th)
        b = N_FOLD * s + SLANT * np.sin(th) * np.clip(top + 0.3, 0, 1.3)
        f = b - np.floor(b)
        crest = 0.5 - 0.5 * np.cos(TAU * f)
        return crest, (1 - crest) ** 3

    def relief(s, th):
        _, g = fold(s, th)
        top = np.clip(-np.cos(th) + 0.2, 0, 1)
        x = 2 * s - 1
        return -A * g * top * (1 - x ** 8)

    def half_width(s):
        x = 2 * s - 1
        return (1 - np.abs(x) ** 2.8) ** (1 / 2.8)

    def half_height(s):
        x = 2 * s - 1
        return (1 - np.abs(x) ** 3.2) ** (1 / 2.6) * (1 - 0.05 * x ** 2)

    def spine_y(s):
        return 0.0 * s

    def paint(S, TH, rng):
        H, W = S.shape
        crest, g = fold(S, TH)
        top = np.clip(-np.cos(TH), 0, 1)
        x = 2 * S - 1
        n1 = _value_noise((H, W), 8, rng, 5)
        n2 = _value_noise((H, W), 70, rng, 3)
        side_c = _srgb((238, 160, 58))
        top_c = _srgb((200, 108, 30))
        layer_c = _srgb((250, 214, 130))
        under_c = _srgb((228, 170, 90))
        t = np.clip(0.75 * top ** 1.3 + 0.35 * (n1 - 0.5), 0, 1)
        col = side_c[None, None] * (1 - t[..., None]) + top_c[None, None] * t[..., None]
        col = col * (1 - 0.35 * g[..., None]) + layer_c[None, None] * 0.35 * g[..., None]
        under = _smoothstep(0.2, -0.4, -np.cos(TH))
        col = col * (1 - under[..., None]) + under_c[None, None] * under[..., None]
        col *= (0.93 + 0.14 * n2)[..., None]
        rough = 0.3 + 0.2 * g + 0.12 * n2 + 0.25 * under
        height = 1.2 * n2
        # sugar crystals: dense on the dome, on the fold crests, thinning down the sides
        dens = _smoothstep(0.25, 0.85, top) * (0.45 + 0.55 * crest) * (1 - _smoothstep(0.82, 1.0, np.abs(x)))
        n_cr = int(W * H * 0.022)
        cy = rng.integers(0, H, n_cr); cx = rng.integers(0, W, n_cr)
        keep = rng.random(n_cr) < dens[cy, cx]
        cy, cx = cy[keep], cx[keep]
        sugar = np.zeros((H, W))
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                rr = dx * dx + dy * dy
                w = 1.0 if rr <= 1 else (0.8 if rr <= 2 else (0.35 if rr <= 5 else 0.0))
                yy = np.clip(cy + dy, 0, H - 1); xx = (cx + dx) % W
                np.maximum.at(sugar, (yy, xx), w * (0.75 + 0.25 * rng.random(len(cy))))
        sug_c = _srgb((255, 255, 252))
        col = col * (1 - sugar[..., None]) + sug_c[None, None] * sugar[..., None]
        rough = rough * (1 - 0.5 * sugar) + 0.25 * sugar
        height = height + 7.0 * sugar
        return col, rough, height

    return BreadSpec("bun", (0.140, 0.074, 0.060), half_width, half_height,
                     bottom=0.30, n_top=2.7, n_bot=6.0, spine_y=spine_y, relief=relief, paint=paint)


SPECS = {"croissant": _croissant, "bun": _bun}


def build(name: str, vis_res=(150, 96), flex_res=(22, 14), tex=(2048, 1024), seed=0):
    """Returns dict(vis=(V, F, UV), flex=(V, F), tex=(rgb, rough, normal), size)."""
    sp = SPECS[name]()
    Vv, Fv, UV = uv_mesh(sp, vis_res[0], vis_res[1], sp.size)
    # the flex surface follows the bands' crests (relief-free tube) so the fingers touch
    # the visible crust; same bounding box as the visual mesh
    Vf, Ff, _ = tube_mesh(sp, flex_res[0], flex_res[1], with_relief=False)
    lo, hi = Vf.min(0), Vf.max(0)
    Vf = (Vf - (lo + hi) / 2) / (hi - lo) * np.asarray(sp.size)
    col, rgh, nrm = paint_textures(sp, tex[0], tex[1], seed)
    return dict(vis=(Vv, Fv, UV), flex=(Vf, Ff), tex=(col, rgh, nrm), size=np.asarray(sp.size))
