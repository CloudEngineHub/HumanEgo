# -*- coding: utf-8 -*-
"""
Parallel batch evaluation: N episodes (randomized or given initial conditions)
spread over worker processes, each holding its own simulator + policy.

    from humanego_sim.evaluate import evaluate
    stats = evaluate(cfg_overrides, n_episodes=32, workers=8, seed=0)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

_W = {}


def _worker_init(cfg_overrides: dict, device: str, intro: bool = False):
    os.environ.setdefault("MUJOCO_GL", "egl")
    # one thread per worker: OpenCV / torch / BLAS thread pools oversubscribe the CPU otherwise
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")
    from .env import load_config
    from .runner import EpisodeRunner
    cfg = load_config(None, cfg_overrides)
    _W["runner"] = EpisodeRunner(cfg, device=device)
    _W["cfg"] = cfg
    _W["intro"] = intro


def _worker_run(job):
    idx, scene, seed, log_dir = job
    try:
        r = _W["runner"].run(scene=scene, seed=seed, log_dir=log_dir, save_policy_images=log_dir is not None,
                             verbose=False, intro=_W.get("intro", False))
        return {"idx": idx, "seed": seed, "scene": scene, "success": r.success, "per_side": r.per_side,
                "duration": r.duration, "wall": r.wall_time, "metrics": r.metrics}
    except Exception as e:  # keep the batch alive, report the failure
        return {"idx": idx, "seed": seed, "scene": scene, "success": False, "error": repr(e),
                "trace": traceback.format_exc()[-2000:]}


def evaluate(cfg_overrides: Optional[dict] = None, n_episodes: int = 16, workers: int = 6, seed: int = 0,
             scenes: Optional[List[dict]] = None, difficulty: float = 1.0, device: str = "cuda",
             log_root: Optional[str] = None, verbose: bool = True, intro: bool = False) -> Dict:
    from .env import load_config
    from .sampling import sample_scene
    cfg_overrides = cfg_overrides or {}
    cfg = load_config(None, cfg_overrides)
    rng = np.random.default_rng(seed)
    if scenes is None:
        scenes = [sample_scene(rng, cfg.get("mode", "bimanual"), difficulty) for _ in range(n_episodes)]
    jobs = []
    for i, sc in enumerate(scenes):
        ld = os.path.join(log_root, f"ep{i:03d}") if log_root else None
        jobs.append((i, sc, seed * 1000 + i, ld))
    t0 = time.time()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"           # inherited by the spawned workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(jobs)), initializer=_worker_init,
                  initargs=(cfg_overrides, device, intro)) as pool:
        results = list(pool.imap_unordered(_worker_run, jobs))
    results.sort(key=lambda r: r["idx"])
    stats = summarize(results)
    stats["wall"] = time.time() - t0
    if verbose:
        print(f"[eval] success {stats['success_rate']:.2f} ({stats['n_success']}/{stats['n']}) "
              f"side={stats['side_success']} fails={stats['failure_modes']} wall={stats['wall']:.0f}s", flush=True)
    if log_root:
        os.makedirs(log_root, exist_ok=True)
        with open(os.path.join(log_root, "results.json"), "w") as f:
            json.dump({"stats": stats, "results": results, "overrides": cfg_overrides}, f, indent=1)
    stats["results"] = results
    return stats


def classify_failure(side_res: dict) -> str:
    if side_res.get("success"):
        return "success"
    if not side_res.get("grasped"):
        return "never_closed"
    lifted = side_res.get("bread_max_z", 1.0) > 0.07
    d = side_res.get("dist_to_plate_center", 1.0)
    if not lifted:
        return "grasp_missed"            # gripper closed but the bread never left the table
    if d > 0.16:
        return "dropped_in_transit"
    if side_res.get("done_reason") == "timeout":
        return "timeout"
    return "rim_or_edge"


def summarize(results: List[dict]) -> Dict:
    n = len(results)
    ns = sum(1 for r in results if r.get("success"))
    side_succ, modes = {}, {}
    dists = []
    for r in results:
        if "error" in r:
            modes["error"] = modes.get("error", 0) + 1
            continue
        for s, v in r["per_side"].items():
            side_succ.setdefault(s, []).append(1.0 if v["success"] else 0.0)
            m = classify_failure(v)
            modes[m] = modes.get(m, 0) + 1
            dists.append(v.get("dist_to_plate_center", 1.0))
    return {
        "n": n, "n_success": ns, "success_rate": ns / max(n, 1),
        "side_success": {s: round(float(np.mean(v)), 3) for s, v in side_succ.items()},
        "per_side_success_rate": float(np.mean([x for v in side_succ.values() for x in v])) if side_succ else 0.0,
        "failure_modes": modes,
        "mean_dist_to_plate": float(np.mean(dists)) if dists else 1.0,
    }
