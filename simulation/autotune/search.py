# -*- coding: utf-8 -*-
"""
Rank-weighted diagonal CEM in the unit hypercube.

Re-implementation (not a copy) of the population-search pattern used by the
recursive-tamp harness: sample a population around the current mean, evaluate,
move the mean to the rank-weighted elite average and shrink/grow the per-dim
spread, with a floor so the search never collapses before validation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .params import ParamSpec


class RankCEM:
    def __init__(self, space: List[ParamSpec], seed: int = 0, init_std: float = 0.18, std_floor: float = 0.03,
                 elite_frac: float = 0.35, mean: Optional[np.ndarray] = None):
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.mean = np.array([p.to_unit(p.init) for p in space]) if mean is None else np.asarray(mean, float)
        self.std = np.full(len(space), init_std)
        self.std_floor = std_floor
        self.elite_frac = elite_frac
        self.frozen = np.zeros(len(space), bool)

    def ask(self, n: int, include_mean: bool = True) -> List[np.ndarray]:
        out = []
        if include_mean:
            out.append(self.mean.copy())
        while len(out) < n:
            z = self.mean + self.std * self.rng.standard_normal(len(self.space))
            z[self.frozen] = self.mean[self.frozen]
            out.append(np.clip(z, 0.0, 1.0))
        return out

    def decode(self, u: np.ndarray) -> Dict[str, float]:
        return {p.name: p.from_unit(x) for p, x in zip(self.space, u)}

    def tell(self, us: List[np.ndarray], scores: List[float]):
        us = np.stack(us)
        scores = np.asarray(scores, float)
        order = np.argsort(-scores)
        k = max(2, int(round(self.elite_frac * len(us))))
        elite = us[order[:k]]
        w = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
        w /= w.sum()
        new_mean = (w[:, None] * elite).sum(0)
        new_std = np.sqrt((w[:, None] * (elite - new_mean) ** 2).sum(0))
        self.mean = np.where(self.frozen, self.mean, new_mean)
        self.std = np.where(self.frozen, self.std, np.maximum(0.7 * self.std + 0.3 * new_std, self.std_floor))

    def inflate(self, dims: Optional[np.ndarray] = None, factor: float = 1.8, cap: float = 0.35):
        dims = np.ones(len(self.space), bool) if dims is None else dims
        self.std = np.where(dims, np.minimum(self.std * factor, cap), self.std)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "frozen": self.frozen.tolist(),
                "names": [p.name for p in self.space]}
