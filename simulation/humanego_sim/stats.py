# -*- coding: utf-8 -*-
"""
Statistics for success rates measured on randomized scenes.

    wilson(k, n)                 95 % Wilson score interval of a binomial rate
    scene_bootstrap_ci(...)      per-arm rate CI resampling whole scenes (arms of one scene are correlated)
    paired(a, b)                 A vs B on the SAME scenes: exact McNemar on discordant arm outcomes
                                 + paired scene bootstrap of the difference

Per-arm success has two outcomes per bimanual scene. The Wilson interval treats them as
independent; the scene bootstrap does not, and is the one to quote when they disagree.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Z95 = 1.959963984540054


def wilson(k: float, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for k successes out of n (returns (lo, hi) in [0, 1])."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k: float, n: int, pct: bool = True) -> str:
    """'91 % [84, 95]' (Wilson 95 %)."""
    lo, hi = wilson(k, n)
    p = k / max(n, 1)
    if pct:
        return f"{100 * p:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}]"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: P(X <= min(b, c)) * 2 under Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def per_scene_outcomes(results: List[dict], sides: Optional[Sequence[str]] = None) -> Dict[int, Dict[str, float]]:
    """{scene idx: {side: 0/1}} from evaluate() results. Errored episodes count as failures."""
    out = {}
    for r in results:
        if "per_side" in r:
            out[r["idx"]] = {s: float(v["success"]) for s, v in r["per_side"].items()}
        else:
            out[r["idx"]] = {s: 0.0 for s in (sides or ("right", "left"))}
    return out


def scene_bootstrap_ci(outcomes: Dict[int, Dict[str, float]], n_boot: int = 4000, seed: int = 0,
                       alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of the per-arm rate, resampling scenes (keeps the two arms of a scene together)."""
    if not outcomes:
        return 0.0, 1.0
    M = np.array([[v for _, v in sorted(o.items())] for o in outcomes.values()], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), (n_boot, len(M)))
    rates = M[idx].mean(axis=(1, 2))
    return float(np.quantile(rates, alpha / 2)), float(np.quantile(rates, 1 - alpha / 2))


def summarize_ci(results: List[dict]) -> dict:
    """Wilson CIs for per-arm / per-side / both-served rates, plus the scene-bootstrap per-arm CI."""
    oc = per_scene_outcomes(results)
    arms = [v for o in oc.values() for v in o.values()]
    sides: Dict[str, List[float]] = {}
    for o in oc.values():
        for s, v in o.items():
            sides.setdefault(s, []).append(v)
    both = [float(all(v > 0.5 for v in o.values())) for o in oc.values()]
    k, n = sum(arms), len(arms)
    return {
        "per_arm": {"k": k, "n": n, "rate": k / max(n, 1), "wilson95": list(wilson(k, n)),
                    "scene_boot95": list(scene_bootstrap_ci(oc))},
        "sides": {s: {"k": sum(v), "n": len(v), "rate": sum(v) / max(len(v), 1), "wilson95": list(wilson(sum(v), len(v)))}
                  for s, v in sides.items()},
        "both": {"k": sum(both), "n": len(both), "rate": sum(both) / max(len(both), 1),
                 "wilson95": list(wilson(sum(both), len(both)))},
    }


def paired(res_a: List[dict], res_b: List[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    """A vs B evaluated on the same scenes (same idx). Positive diff = A better.

    mcnemar_p   exact two-sided test on discordant arm outcomes (A ok / B fail vs A fail / B ok)
    boot95      paired percentile CI of the per-arm difference, resampling scenes
    boot_p      two-sided bootstrap p-value (share of resamples on the other side of 0, x2)
    """
    oa, ob = per_scene_outcomes(res_a), per_scene_outcomes(res_b)
    common = sorted(set(oa) & set(ob))
    if not common:
        return {"n_scenes": 0}
    sides = sorted(set(oa[common[0]]) & set(ob[common[0]]))
    A = np.array([[oa[i][s] for s in sides] for i in common])
    B = np.array([[ob[i][s] for s in sides] for i in common])
    b = int(((A > 0.5) & (B < 0.5)).sum())       # A ok, B failed
    c = int(((A < 0.5) & (B > 0.5)).sum())       # A failed, B ok
    D = (A - B).mean(axis=1)                      # per-scene per-arm difference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), (n_boot, len(D)))
    boots = D[idx].mean(axis=1)
    diff = float(D.mean())
    if diff > 0:
        bp = 2.0 * float((boots <= 0).mean())
    elif diff < 0:
        bp = 2.0 * float((boots >= 0).mean())
    else:
        bp = 1.0
    return {"n_scenes": len(common), "n_arms": int(A.size), "rate_a": float(A.mean()), "rate_b": float(B.mean()),
            "diff": diff, "a_only": b, "b_only": c, "mcnemar_p": _binom_two_sided(b, c),
            "boot95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "boot_p": min(1.0, bp)}


def fmt_paired(p: dict, name_a: str = "A", name_b: str = "B") -> str:
    if not p.get("n_scenes"):
        return f"{name_a} vs {name_b}: no common scenes"
    lo, hi = p["boot95"]
    verdict = "significant" if p["mcnemar_p"] < 0.05 and (lo > 0 or hi < 0) else "not significant"
    return (f"{name_a} vs {name_b}: per-arm {100 * p['rate_a']:.0f} % vs {100 * p['rate_b']:.0f} %, "
            f"diff {100 * p['diff']:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] (paired scene bootstrap), "
            f"discordant arms {p['a_only']}/{p['b_only']}, McNemar p={p['mcnemar_p']:.3g} -> {verdict} "
            f"(n={p['n_scenes']} scenes)")


# ----------------------------------------------------------------------------- protocol
DEV_SEEDS = (7777, 8888)                 # scene sets for selecting / tuning parameters
RESERVED_SEEDS = range(9001, 9100)       # scene sets for final reports only (never for selection)


def check_seed(seeds: Sequence[int], final: bool) -> None:
    """Refuse reserved scene seeds unless the caller declares a final report (--final)."""
    bad = [s for s in seeds if s in RESERVED_SEEDS]
    if bad and not final:
        raise SystemExit(f"seeds {bad} are reserved for final reports (9001-9099); pass --final to use them, "
                         f"and never pick parameters on them. Development seeds: {list(DEV_SEEDS)}")
