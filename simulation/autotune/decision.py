# -*- coding: utf-8 -*-
"""
Per-batch evidence and the four-way decision rule.

After every population batch the tuner summarises what happened and picks one of:

  continue_local     the best candidate improved: keep sampling around it
  retune_search      plateau, but some parameters correlate with success or sit
                     on a bound: re-centre / widen those dims, freeze inert ones
  revise_global      plateau and no usable signal: ask the agent (file inbox)
                     to revise the search space or propose candidates
  validate_and_stop  the best candidate clears the target: re-test it on fresh
                     scenes and stop if it holds
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def batch_evidence(names: List[str], us: List[np.ndarray], scores: List[float], details: List[dict]) -> Dict:
    U = np.stack(us)
    s = np.asarray(scores, float)
    corr = {n: round(spearman(U[:, i], s), 3) for i, n in enumerate(names)}
    best = int(np.argmax(s))
    on_bound = {n: bool(U[best, i] < 0.03 or U[best, i] > 0.97) for i, n in enumerate(names)}
    modes: Dict[str, int] = {}
    for d in details:
        for k, v in d.get("failure_modes", {}).items():
            modes[k] = modes.get(k, 0) + v
    return {
        "best_idx": best, "best_score": float(s[best]), "mean_score": float(s.mean()),
        "best_success": details[best].get("success_rate", 0.0),
        "spearman": corr, "best_on_bound": on_bound, "failure_modes": modes,
    }


def decide(history: List[Dict], ev: Dict, target: float = 0.9, min_gain: float = 0.02,
           patience: int = 2, signal: float = 0.35) -> Dict:
    best_so_far = max([h["evidence"]["best_score"] for h in history[:-1]], default=-1e9)
    gain = ev["best_score"] - best_so_far
    stalls = 0
    for h in reversed(history[:-1]):
        if h["decision"]["action"] in ("continue_local",):
            break
        stalls += 1
    if ev["best_success"] >= target:
        return {"action": "validate_and_stop", "reason": f"best success {ev['best_success']:.2f} >= {target}"}
    if gain >= min_gain:
        return {"action": "continue_local", "reason": f"improved by {gain:+.3f}"}
    strong = [n for n, c in ev["spearman"].items() if abs(c) >= signal]
    bounded = [n for n, b in ev["best_on_bound"].items() if b]
    if (strong or bounded) and stalls < patience:
        return {"action": "retune_search", "reason": f"plateau ({gain:+.3f}); signals {strong}, on-bound {bounded}",
                "focus": strong, "on_bound": bounded}
    return {"action": "revise_global", "reason": f"plateau ({gain:+.3f}) for {stalls + 1} batch(es); no usable signal"}
