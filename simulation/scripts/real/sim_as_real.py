# -*- coding: utf-8 -*-
"""
Run one simulated episode and write it in the real-robot log format of AriaMimic's
inference/SimLogger.py ("humanego-real-log/1"), as if the simulator were the real rig.

It is the test fixture of replay_real.py (the scene and outcome are known exactly, under
meta.json "sim_truth") and an example of what a real log contains.

    python simulation/scripts/real/sim_as_real.py --seed 3 --out /tmp/fake_real_log
    python simulation/scripts/real/replay_real.py /tmp/fake_real_log

Differences from a real log: no joint velocities / efforts, the gripper position is the sim
joint, wrap-up event times follow the runner's schedule, and images are renders.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede `import mujoco`
import json
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "scripts"))

import humanego_sim.runner as R  # noqa: E402
from humanego_sim.controller import SimController  # noqa: E402
from humanego_sim.env import deep_update, load_config  # noqa: E402
from humanego_sim.perception import mirror_pose  # noqa: E402
from humanego_sim.sampling import sample_scene  # noqa: E402

FORMAT = "humanego-real-log/1"


def _r(a, n=5):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


def _depth_png(d):
    return np.clip(np.round(np.nan_to_num(d) * 1e4), 0, 65535).astype(np.uint16)


class Recorder:
    def __init__(self, out):
        self.out = out
        os.makedirs(os.path.join(out, "frames"), exist_ok=True)
        self.steps, self.commands, self.events = {}, [], []
        self.n = 0
        self.t_index = {}
        self.init_done = False


REC = None


class RecController(SimController):
    def tick(self):
        super().tick()
        for s, arm in self.arms.items():
            if self.muted[s] or self.last_cmd[s] is None:
                continue
            REC.commands.append({"t": round(self.t - 1.0 / self.control_freq, 4), "side": s, "cmd": _r(self.last_cmd[s]),
                                 "grip_closed": bool(arm.gripper_cmd_closed()),
                                 "goal_time": round(self.duration_smooth_param / self.control_freq, 3)})


class RecRunner(R.EpisodeRunner):
    def _run_side(self, ctx, clean_bg, t):
        env, arm = self.env, self.env.arms[ctx.side]
        if not REC.init_done:
            self._write_init(t)
        Tr = arm.get_T_ee_in_cam() @ self.T_align
        Tv = mirror_pose(Tr) if ctx.is_flip else Tr
        gv = self._grasp_val(arm)
        was_grasp, was_done = ctx.hand["is_grasping"], ctx.is_done
        raw, grasp, vis, pW, dp = super()._run_side(ctx, clean_bg, t)
        key = round(t, 4)
        if key not in REC.t_index:
            REC.t_index[key] = REC.n
            REC.steps[REC.n] = {"t": key, "n": REC.n, "segment": 0, "sides": {}}
            cv2.imwrite(os.path.join(REC.out, "frames", f"{REC.n:06d}_rgb.jpg"), env.render_rgb(), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(REC.out, "frames", f"{REC.n:06d}_depth.png"), _depth_png(env.render_depth()))
            REC.n += 1
        n = REC.t_index[key]
        if ctx.side == "left" and not any(e.get("kind") == "activate" for e in REC.events):
            REC.events.append({"t": key, "kind": "activate", "side": "left"})
        cv2.imwrite(os.path.join(REC.out, "frames", f"{n:06d}_policy_{ctx.side}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        q = list(arm.get_j()) + [float(env.d.qpos[arm.g_qadr])]
        REC.steps[n]["sides"][ctx.side] = {
            "active": True, "gq": round(arm.get_gripper_q(), 4), "gv": gv,
            "T_ee_in_cam": _r(arm.get_T_ee_in_cam(), 6), "T_virtual": _r(Tv, 6),
            "plan_cam": _r(np.asarray(raw).reshape(-1, 6) if len(raw) else []),
            "grasp_probs": _r(np.asarray(grasp).ravel(), 4), "done_prob": round(float(dp), 4),
            "is_grasping": bool(ctx.hand["is_grasping"]), "latched": ctx.hand["latched_obj"], "is_done": bool(ctx.is_done),
            "obj_poses": {k: _r(v) for k, v in ctx.objs.dynamic_obj_poses.items()}, "kb_muted": False,
            "forced_lift_counter": int(ctx.forced_lift_counter),
            "state": {"q": _r(q), "p_base": _r(arm.get_p())}}
        if ctx.hand["is_grasping"] and not was_grasp:
            REC.events.append({"t": key, "kind": "grasp_on", "side": ctx.side, "latched": ctx.hand["latched_obj"]})
        if was_grasp and not ctx.hand["is_grasping"]:
            REC.events.append({"t": key, "kind": "grasp_off", "side": ctx.side})
        if ctx.is_done and not was_done:
            REC.events += [{"t": key, "kind": "done", "side": ctx.side, "reason": ctx.done_reason},
                           {"t": key, "kind": "wrap_halt", "side": ctx.side},
                           {"t": round(key + 0.8, 4), "kind": "wrap_opened", "side": ctx.side},
                           {"t": round(key + 0.8 + 3.1, 4), "kind": "wrap_home", "side": ctx.side}]
        return raw, grasp, vis, pW, dp

    def _write_init(self, t):
        REC.init_done = True
        env = self.env
        d = os.path.join(REC.out, "init_0")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "rgb.png"), env.render_rgb())
        cv2.imwrite(os.path.join(d, "depth_0.png"), _depth_png(env.render_depth()))
        masks = self.perception.object_masks()
        for k, m in masks.items():
            cv2.imwrite(os.path.join(d, f"mask_{k}_full.png"), m)
        self._init_t = t


def main():
    global REC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene_seed", type=int, default=None, help="sample a random scene (sampling.py) with this seed")
    ap.add_argument("--mode", default="bimanual", choices=["bimanual", "single"])
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    from evaluate import parse_sets
    cfg = load_config(None, deep_update({"mode": args.mode}, parse_sets(args.set)))
    REC = Recorder(args.out)
    R.SimController = RecController
    runner = RecRunner(cfg)
    scene = sample_scene(np.random.default_rng(args.scene_seed), args.mode) if args.scene_seed is not None else None
    sim_dir = os.path.join(args.out, "sim_episode")
    res = runner.run(scene=scene, seed=args.seed, log_dir=sim_dir, record_hz=50.0, intro=False)
    env = runner.env
    ep = json.load(open(os.path.join(sim_dir, "episode.json")))
    meta_s = ep["meta"]
    # ---- init objects (perception at t = 0) ----
    objs = {}
    for s, per in meta_s["perception"].items():
        objs[s] = {}
        for k, v in per.items():
            T, kl = np.asarray(v["T_obj_in_cam_static"]), np.asarray(v["kpts_local"]).reshape(-1, 3)
            objs[s][k] = {"T_obj_in_cam_static": v["T_obj_in_cam_static"], "kpts_local": v["kpts_local"],
                          "kpts_3d": _r((T[:3, :3] @ kl.T).T + T[:3, 3]), "kpts_2d": []}
        objs[s]["_is_flip"] = s == "left"
    q0 = {s: ep["frames"][0]["qpos"][s] for s in env.sides}
    json.dump({"t": 0.0, "segment": 0, "objects": objs, "arms": {s: {"q": q0[s]} for s in env.sides},
               "note": "left-side poses are in the MIRRORED camera frame (x -> -x), as the policy sees them"},
              open(os.path.join(args.out, "init_0", "objects.json"), "w"), indent=1)
    # ---- state at 50 Hz from the recorded frames ----
    fr = [f for f in ep["frames"] if f["t"] >= 0]
    st = {"t": np.array([f["t"] for f in fr])}
    for s in env.sides:
        arm = env.arms[s]
        Q, P, G = [], [], []
        for f in fr:
            env.d.qpos[:] = f["qfull"]
            mujoco.mj_kinematics(env.m, env.d)
            Q.append(f["qpos"][s])
            P.append(arm.get_p())
            G.append(float(env.d.qpos[arm.g_qadr]))
        st[f"{s}_q"] = np.array(Q)
        st[f"{s}_p_base"] = np.array(P)
        st[f"{s}_g_pos"] = np.array(G)
        st[f"{s}_qd"] = np.full((len(fr), 7), np.nan)
        st[f"{s}_tau"] = np.full((len(fr), 7), np.nan)
        st[f"{s}_g_eff"] = np.full(len(fr), np.nan)
    np.savez_compressed(os.path.join(args.out, "state.npz"), **st)
    # ---- jsonl ----
    with open(os.path.join(args.out, "steps.jsonl"), "w") as f:
        for n in sorted(REC.steps):
            f.write(json.dumps(REC.steps[n]) + "\n")
    with open(os.path.join(args.out, "commands.jsonl"), "w") as f:
        for c in REC.commands:
            f.write(json.dumps(c) + "\n")
    REC.events.insert(0, {"t": 0.0, "kind": "start"})
    with open(os.path.join(args.out, "events.jsonl"), "w") as f:
        for e in sorted(REC.events, key=lambda e: e["t"]):
            f.write(json.dumps(e) + "\n")
    # ---- meta + outcome ----
    from grasp_audit import audit_episode
    per_side = {}
    for s in env.sides:
        try:
            per_side[s] = audit_episode(env, ep, s).get("outcome", "other")
        except Exception:  # noqa: BLE001
            per_side[s] = "served" if res.per_side[s]["success"] else "other"
    meta = {"format": FORMAT, "task": "serve_bread", "source": "sim_as_real.py", "t0_wall": 0.0,
            "camera": {"K": _r(env.K, 6), "D": [0, 0, 0, 0, 0], "w": env.W, "h": env.H, "w_raw": 1280, "h_raw": 720,
                       "fps": 30, "crop": [160, 0, 960, 720], "serial": "sim"},
            "arms": {s: {"T_base_in_cam": _r(env.arms[s].T_base_in_cam, 8),
                         "gripper_max": float(env.robot.gripper_joint_open)} for s in env.sides},
            "T_align": _r(R.T_ALIGN_ARIA, 6), "bimanual": len(env.sides) > 1, "bimanual_mode": "shared_context",
            "bimanual_start_delay": float(cfg["control"].get("bimanual_start_delay", 6.0)),
            "split_x_ratio": float(cfg["control"].get("split_x_ratio", 0.5)),
            "control_cfg": cfg["control"], "policy": {"ckpt": cfg["policy"]["ckpt"], "pred_horizon": 50},
            "state_hz": 50.0,
            "sim_truth": {"scene_init": meta_s["scene_init"], "result": meta_s["result"]}}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1, default=lambda o: None)
    kinds = {s: meta_s["objects"][f"bread_{s}"]["kind"] for s in env.sides}
    json.dump({"per_side": per_side, **{f"bread_{s}": ("croissant" if k == "croissant" else "danish") for s, k in kinds.items()},
               "notes": "simulated (sim_as_real.py)"}, open(os.path.join(args.out, "outcome.json"), "w"), indent=1)
    print(json.dumps({"out": args.out, "per_side": per_side, "scene": meta_s["scene_init"]}, indent=1))


if __name__ == "__main__":
    main()
