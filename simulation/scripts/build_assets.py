# -*- coding: utf-8 -*-
"""
Build the serve_bread object assets for MuJoCo (OBJ + PNG + convex collision
parts) and for the web viewer (GLB).

Sources
-------
* croissant      Poly Haven "Croissant" photoscan (CC0)         -> rescaled to the
                 real HumanEgo croissant measured from the released dataset
* bread roll     Poly Haven "Hamburger Buns" photoscan (CC0), one bun reshaped
                 into the oval roll used in the real bimanual rollouts
* plate          procedural lathe (light-blue ceramic dinner plate)
* tabletop       maple texture from TrossenRobotics/trossen_arm_mujoco (BSD-3)

Real sizes come from the 20 triangulated contour keypoints per object in the
61 released serve_bread recordings (camtriangulator_results.json):
    croissant  max extent 12.8 cm (inner-eroded contour; ~13.8 cm outer)
    plate      diameter   23.3 cm (inner-eroded contour; ~24.5 cm outer)

Run:  python simulation/scripts/build_assets.py
"""
import json
import os
import shutil

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
SRC = os.path.join(SIM, "assets_src")
OUT = os.path.join(SIM, "humanego_sim", "assets", "objects")
WEB = os.path.join(SIM, "web", "assets", "objects")

# Y-up (glTF) -> Z-up (MuJoCo): rotate +90 deg about X
YUP_TO_ZUP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def _single_mesh(scene, name=None):
    if isinstance(scene, trimesh.Trimesh):
        return scene
    geoms = scene.dump()  # world-transformed copies
    if name is not None:
        geoms = [g for g in geoms if name in g.metadata.get("name", "")] or geoms
    return geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)


def _fit_to_size(mesh, size_xyz):
    """Center on the bbox (xy) and table-contact (z=0 is the bottom), then scale
    non-uniformly to the target (L, W, H) in metres. Returns the mesh centred on
    its bbox centre (body frame at the object's centre)."""
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)
    ext = hi - lo
    S = np.diag(list(np.asarray(size_xyz) / ext) + [1.0])
    mesh.apply_transform(S)
    return mesh


def _texture_from_material(mesh, size=1024):
    mat = mesh.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        return None
    img = img.convert("RGB")
    if max(img.size) > size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def _export_mujoco(mesh, name, tex_img, n_hulls=6):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    # --- visual: OBJ with UVs (MuJoCo flips v internally for OBJ) ---
    vis = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                          visual=mesh.visual, process=False)
    vis.export(os.path.join(d, f"{name}.obj"), include_texture=False)
    if tex_img is not None:
        tex_img.save(os.path.join(d, f"{name}.png"))
    # --- collision: approximate convex decomposition (CoACD) ---
    parts = []
    try:
        import coacd
        cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        res = coacd.run_coacd(cm, threshold=0.05, max_convex_hull=n_hulls, preprocess_mode="auto",
                              resolution=2000, mcts_iterations=100)
        for v, f in res:
            parts.append(trimesh.Trimesh(v, f).convex_hull)
    except Exception as e:  # pragma: no cover
        print(f"[assets] CoACD unavailable ({e}); falling back to a single hull")
        parts = [mesh.convex_hull]
    for i, p in enumerate(parts):
        p.export(os.path.join(d, f"{name}_col{i}.obj"))
    return len(parts)


def _export_glb(mesh, name, tex_paths=None, color=None, roughness=0.8, metallic=0.0, tex_size=1024):
    os.makedirs(WEB, exist_ok=True)
    m = mesh.copy()
    if tex_paths:
        imgs = {}
        for k, p in tex_paths.items():
            im = Image.open(p).convert("RGB")
            if max(im.size) > tex_size:
                im = im.resize((tex_size, tex_size), Image.LANCZOS)
            imgs[k] = im
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=imgs.get("diff"),
            normalTexture=imgs.get("nor"),
            metallicRoughnessTexture=imgs.get("arm"),
            occlusionTexture=imgs.get("arm"),
            roughnessFactor=1.0, metallicFactor=1.0 if "arm" in imgs else 0.0)
        m.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mat)
    elif color is not None:
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=color, roughnessFactor=roughness,
                                                  metallicFactor=metallic)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
    # glTF is Y-up: rotate back so three.js can use a Z-up parent consistently
    m.apply_transform(np.linalg.inv(YUP_TO_ZUP))
    m.export(os.path.join(WEB, f"{name}.glb"))


# ---------------------------------------------------------------------------
def bake_crust(img: Image.Image) -> Image.Image:
    """Colour-grade the pale (unbaked-looking) bun scan into a golden-brown roll crust,
    keeping the scan's luminance detail."""
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    L = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    L = np.clip((L - L.min()) / (np.percentile(L, 99.5) - L.min() + 1e-6), 0, 1) ** 1.35
    dark = np.array([0.36, 0.17, 0.06])
    gold = np.array([0.93, 0.66, 0.33])
    out = dark[None, None] * (1 - L[..., None]) + gold[None, None] * L[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def build_photoscan(src_name, out_name, size_xyz, pick=None, tex_prefix=None, grade=None):
    sc = trimesh.load(os.path.join(SRC, src_name, f"{src_name}.gltf"))
    mesh = _single_mesh(sc, pick)
    mesh.apply_transform(YUP_TO_ZUP)
    mesh = _fit_to_size(mesh, size_xyz)
    tex = _texture_from_material(mesh)
    tp = tex_prefix or src_name
    texdir = os.path.join(SRC, src_name, "textures")
    diff_path = os.path.join(texdir, f"{tp}_diff_2k.jpg")
    if grade is not None:
        tex = grade(tex)
        graded = os.path.join(texdir, f"{tp}_diff_2k_graded.jpg")
        grade(Image.open(diff_path)).save(graded, quality=92)
        diff_path = graded
    n = _export_mujoco(mesh, out_name, tex)
    _export_glb(mesh, out_name, {
        "diff": diff_path,
        "nor": os.path.join(texdir, f"{tp}_nor_gl_2k.jpg"),
        "arm": os.path.join(texdir, f"{tp}_arm_2k.jpg"),
    })
    print(f"[assets] {out_name}: extents {np.round(mesh.extents, 4)}  hulls={n}  verts={len(mesh.vertices)}")
    return mesh


def plate_profile(R=0.1225, H=0.022, r_well=0.080, r_foot=0.072, t=0.0045, foot_h=0.004):
    """(r, z) polyline of a dinner plate cross-section, outer surface then inner."""
    # top surface from centre to rim
    top = [(0.0, foot_h + t), (r_well, foot_h + t)]
    for a in np.linspace(0, 1, 10)[1:]:
        r = r_well + (R - r_well) * a
        z = foot_h + t + (H - foot_h - t) * (a ** 1.6)
        top.append((r, z))
    # rim lip
    top.append((R + 0.002, H - 0.0015))
    # underside back to centre
    bot = []
    for a in np.linspace(1, 0, 10):
        r = r_well + (R - r_well) * a
        z = foot_h + (H - foot_h - t) * (a ** 1.6)
        bot.append((r - 0.001, z))
    bot += [(r_foot + 0.004, foot_h), (r_foot, 0.0), (r_foot - 0.004, 0.0), (r_foot - 0.008, foot_h), (0.0, foot_h)]
    return np.array(top + bot)


def build_plate(R=0.1225, H=0.022):
    prof = plate_profile(R=R, H=H)
    # trimesh.creation.revolve wants (r, z) with a closed polyline; build manually
    n_seg = 96
    rings = []
    for (r, z) in prof:
        ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        rings.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n_seg, z)], 1))
    V = np.concatenate(rings, 0)
    F = []
    n_r = len(prof)
    for i in range(n_r - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + j
            d = (i + 1) * n_seg + (j + 1) % n_seg
            F += [[a, c, b], [b, c, d]]
    mesh = trimesh.Trimesh(V, np.array(F), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    color = [0.67, 0.78, 0.90, 1.0]  # light periwinkle-blue ceramic (real rollouts)
    d = os.path.join(OUT, "plate")
    os.makedirs(d, exist_ok=True)
    mesh.export(os.path.join(d, "plate.obj"))
    _export_glb(mesh, "plate", color=color, roughness=0.18)
    # collision primitives (box ring + base disc) are generated in the MJCF builder
    meta = {"R": R, "H": H, "r_well": 0.080, "t": 0.0045, "foot_h": 0.004, "rgba": color}
    with open(os.path.join(d, "plate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[assets] plate: R={R} H={H} verts={len(mesh.vertices)}")
    return mesh


def copy_table_texture():
    src = os.path.join(SIM, "third_party", "trossen_arm_mujoco", "trossen_arm_mujoco", "assets",
                       "maple_tabletop.png")
    d = os.path.join(OUT, "table")
    os.makedirs(d, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img.save(os.path.join(d, "maple.png"))
    os.makedirs(os.path.join(SIM, "web", "assets", "textures"), exist_ok=True)
    img.resize((1024, int(1024 * img.size[1] / img.size[0])), Image.LANCZOS).save(
        os.path.join(SIM, "web", "assets", "textures", "maple.jpg"), quality=88)
    print(f"[assets] table texture {img.size}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fetch_polyhaven("croissant")
    fetch_polyhaven("hamburger_buns")
    build_photoscan("croissant", "croissant", size_xyz=(0.138, 0.060, 0.045))
    build_photoscan("hamburger_buns", "roll", size_xyz=(0.120, 0.070, 0.052), pick="mesh.023", grade=bake_crust)
    build_plate()
    copy_table_texture()
