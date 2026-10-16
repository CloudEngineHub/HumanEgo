# -*- coding: utf-8 -*-
"""
Self-improving parameter search for the HumanEgo simulator + low-level controller.

Pattern (after ColinYu1/recursive-tamp's population loop, re-implemented here):

    repeat:
        population <- CEM.ask()                     (candidate parameter vectors)
        evaluate every candidate on the SAME K scenes (common random numbers)
        evidence <- success, failure modes, Spearman(param, score), bound hits
        decision <- continue_local | retune_search | revise_global | validate_and_stop
        revise_global  -> write inbox/request_NNN.json, wait for response_NNN.json
                          from an agent (e.g. Claude Code); fall back to a heuristic
        validate_and_stop -> re-run the best on fresh scenes; accept or keep going

Everything is written under simulation/autotune_runs/<name>/ so a run can be
inspected or resumed, and the agent sees exactly what the tuner saw.

Agent response format (response_NNN.json), every key optional:
    {
      "note": "why",
      "set_bounds": {"control.alpha": [0.0, 0.4]},
      "freeze": {"physics.gripper_force": 12.0},
      "unfreeze": ["control.alpha_rot"],
      "candidates": [{"control.alpha": 0.2, "control.look_ahead_offset": 30}],
      "recenter": {"control.alpha": 0.2},
      "inflate": 1.5,
      "stop": false
    }
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from humanego_sim.env import load_config
from humanego_sim.evaluate import evaluate
from humanego_sim.sampling import sample_scene

from .decision import batch_evidence, decide
from .params import ParamSpec, params_to_overrides, space_from_names, spec_dict
from .search import RankCEM


def score_of(stats: Dict) -> float:
    """Per-side success rate, minus a little for near-misses that ended far from the plate."""
    return float(stats["per_side_success_rate"]) - 0.05 * float(min(stats["mean_dist_to_plate"], 0.3))


class AutoTuner:
    def __init__(self, name: str, base_overrides: Optional[dict] = None, space: Optional[List[ParamSpec]] = None,
                 pop: int = 8, scenes_per_candidate: int = 8, workers: int = 8, seed: int = 0, target: float = 0.9,
                 validate_n: int = 32, agent_timeout: float = 1800.0, max_batches: int = 12):
        self.root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autotune_runs", name)
        os.makedirs(os.path.join(self.root, "inbox"), exist_ok=True)
        self.base_overrides = base_overrides or {}
        self.base_cfg = load_config(None, self.base_overrides)
        self.space = space or space_from_names()
        self.cem = RankCEM(self.space, seed=seed)
        self.pop, self.K, self.workers = pop, scenes_per_candidate, workers
        self.seed, self.target, self.validate_n = seed, target, validate_n
        self.agent_timeout, self.max_batches = agent_timeout, max_batches
        self.history: List[Dict] = []
        self.extra_candidates: List[Dict[str, float]] = []
        self.best: Optional[Dict] = None

    # ------------------------------------------------------------------
    def _overrides(self, params: Dict[str, float]) -> dict:
        from humanego_sim.env import deep_update
        return deep_update(self.base_overrides, params_to_overrides(params, self.base_cfg))

    def _eval(self, params: Dict[str, float], scenes, tag: str) -> Dict:
        st = evaluate(self._overrides(params), scenes=scenes, workers=self.workers, seed=self.seed,
                      verbose=False)
        st.pop("results", None)
        st["score"] = score_of(st)
        print(f"  [{tag}] score={st['score']:.3f} side_success={st['side_success']} modes={st['failure_modes']}",
              flush=True)
        return st

    def _write(self, rel: str, obj):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f, indent=1, default=float)

    # ------------------------------------------------------------------
    def ask_agent(self, it: int, ev: Dict) -> Optional[Dict]:
        req = {
            "schema": "humanego-sim.autotune.request.v1",
            "iteration": it,
            "instructions": ("The population search plateaued. Read the evidence and propose a revision: new bounds, "
                             "frozen values, explicit candidates, or stop. Keep physics inside plausible ranges."),
            "space": spec_dict(self.space),
            "cem": self.cem.state(),
            "evidence": ev,
            "history": [{"iteration": h["iteration"], "best_score": h["evidence"]["best_score"],
                         "best_params": h["best_params"], "decision": h["decision"]} for h in self.history],
            "best": self.best,
        }
        rq = os.path.join(self.root, "inbox", f"request_{it:03d}.json")
        rs = os.path.join(self.root, "inbox", f"response_{it:03d}.json")
        with open(rq, "w") as f:
            json.dump(req, f, indent=1, default=float)
        print(f"AGENT_REQUEST {rq}", flush=True)
        t0 = time.time()
        while time.time() - t0 < self.agent_timeout:
            if os.path.exists(rs):
                time.sleep(0.5)
                with open(rs) as f:
                    return json.load(f)
            time.sleep(1.0)
        print("  [agent] no response; heuristic fallback (inflate search)", flush=True)
        return None

    def apply_agent(self, resp: Dict):
        names = [p.name for p in self.space]
        for n, (lo, hi) in resp.get("set_bounds", {}).items():
            i = names.index(n)
            p = self.space[i]
            v = p.from_unit(self.cem.mean[i])
            p.lo, p.hi = float(lo), float(hi)
            self.cem.mean[i] = p.to_unit(min(max(v, p.lo), p.hi))
        for n, v in resp.get("freeze", {}).items():
            i = names.index(n)
            self.cem.mean[i] = self.space[i].to_unit(v)
            self.cem.frozen[i] = True
        for n in resp.get("unfreeze", []):
            self.cem.frozen[names.index(n)] = False
        for n, v in resp.get("recenter", {}).items():
            i = names.index(n)
            self.cem.mean[i] = self.space[i].to_unit(v)
        if resp.get("inflate"):
            self.cem.inflate(factor=float(resp["inflate"]))
        self.extra_candidates += resp.get("candidates", [])

    # ------------------------------------------------------------------
    def run(self):
        rng = np.random.default_rng(self.seed + 7)
        mode = self.base_cfg.get("mode", "bimanual")
        for it in range(self.max_batches):
            scenes = [sample_scene(rng, mode) for _ in range(self.K)]       # shared by the whole batch
            us = self.cem.ask(self.pop)
            # agent-proposed candidates replace the tail of the population
            for j, c in enumerate(self.extra_candidates[: self.pop - 1]):
                base = self.cem.decode(self.cem.mean)
                base.update(c)
                us[-(j + 1)] = np.array([p.to_unit(base[p.name]) for p in self.space])
            self.extra_candidates = []
            print(f"[autotune] batch {it}: {len(us)} candidates x {self.K} scenes", flush=True)
            details, scores = [], []
            for j, u in enumerate(us):
                params = self.cem.decode(u)
                st = self._eval(params, scenes, f"b{it}c{j}")
                details.append(st)
                scores.append(st["score"])
            names = [p.name for p in self.space]
            ev = batch_evidence(names, us, scores, details)
            best_params = self.cem.decode(us[ev["best_idx"]])
            if self.best is None or ev["best_score"] > self.best["score"]:
                self.best = {"score": ev["best_score"], "params": best_params, "stats": details[ev["best_idx"]],
                             "iteration": it}
            self.cem.tell(us, scores)
            rec = {"iteration": it, "evidence": ev, "best_params": best_params,
                   "candidates": [{"params": self.cem.decode(u), "stats": d} for u, d in zip(us, details)]}
            self.history.append(rec)
            dec = decide(self.history, ev, target=self.target)
            rec["decision"] = dec
            print(f"[autotune] batch {it}: best={ev['best_score']:.3f} mean={ev['mean_score']:.3f} -> {dec['action']} "
                  f"({dec['reason']})", flush=True)
            self._write(f"iterations/iter_{it:03d}.json", rec)
            self._write("population_memory.json", {"cem": self.cem.state(), "best": self.best, "space": spec_dict(self.space)})

            if dec["action"] == "retune_search":
                names_i = {n: i for i, n in enumerate(names)}
                dims = np.zeros(len(names), bool)
                for n in dec.get("focus", []) + dec.get("on_bound", []):
                    dims[names_i[n]] = True
                self.cem.inflate(dims, factor=1.6)
                # widen bounds that the best candidate is pushing against (within hard caps)
                for n in dec.get("on_bound", []):
                    p = self.space[names_i[n]]
                    u = self.cem.mean[names_i[n]]
                    span = p.hi - p.lo
                    if u > 0.97:
                        p.hi = p.hi + 0.25 * span if p.kind != "log" else p.hi * 1.5
                    elif u < 0.03 and p.lo > 0:
                        p.lo = max(0.0 if p.kind != "log" else p.lo / 1.5, p.lo - 0.25 * span)
            elif dec["action"] == "revise_global":
                resp = self.ask_agent(it, ev)
                if resp is None:
                    self.cem.inflate(factor=1.5)
                else:
                    if resp.get("stop"):
                        break
                    self.apply_agent(resp)
            elif dec["action"] == "validate_and_stop":
                vs = [sample_scene(np.random.default_rng(10_000 + it), mode) for _ in range(self.validate_n)]
                st = self._eval(self.best["params"], vs, "validate")
                self.best["validation"] = st
                self._write("best.json", self.best)
                if st["per_side_success_rate"] >= self.target - 0.05:
                    print(f"[autotune] accepted: validation per-side success {st['per_side_success_rate']:.2f}", flush=True)
                    break
                print("[autotune] validation did not hold; continuing", flush=True)
        self._write("best.json", self.best)
        return self.best
