# -*- coding: utf-8 -*-
"""
Where does the grasp height come from? For every arm of every logged episode:

  kc_minus_center    perception: centroid of the lifted object keypoints above the bread centre
  intent_minus_*     policy: first predicted step with grasp > 0.6, relative to centre / keypoints
  exec_minus_*       execution: fingertip point when the gripper command switches to closed
  exec_low_minus_*   lowest fingertip point in the 1.5 s after that

All heights in cm. Usage (logs from evaluate.py --log <name> or run_episode.py):

    python simulation/scripts/grasp_height.py simulation/runs/<name>
"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
from humanego_sim.env import ServeBreadSim, load_config

root = sys.argv[1]
rows = []
env = None
for epd in sorted(glob.glob(os.path.join(root, 'ep*'))):
    ep = json.load(open(os.path.join(epd, 'episode.json')))
    meta = ep['meta']
    if env is None:
        cfg = load_config(None, dict(mode=meta['mode'], robot=meta['robot'], camera=meta['camera']))
        env = ServeBreadSim(cfg, render=False)
    Tcw = np.array(meta['T_cam_in_world'])
    for s in meta['sides']:
        arm = env.arms[s]
        kind = meta['objects'][f'bread_{s}']['kind']
        pm = meta['perception'][s]['obj1']
        Ts = np.array(pm['T_obj_in_cam_static']); kl = np.array(pm['kpts_local'])
        if s == 'left':
            S = np.diag([-1, 1, 1, 1.0]); Ts = S @ Ts @ S; kl = kl * [-1, 1, 1]
        kw = (Tcw @ Ts @ np.c_[kl, np.ones(len(kl))].T).T[:, :3]
        # bread at rest (first non-intro frame)
        fr0 = next(f for f in ep['frames'] if f['t'] >= 0)
        bpos0 = np.array(fr0['objects'][f'bread_{s}'][:3])
        ext = np.array(meta['objects'][f'bread_{s}']['extents'])
        top = bpos0[2] + ext[2] / 2
        # policy: first chunk with grasp > thr; intended grasp point = first index above thr
        intent = None
        for pe in ep['policy']:
            e = pe['sides'].get(s)
            if not e: continue
            g = np.array(e['grasp']); tw = np.array(e['traj_world'])
            if (g > 0.6).any():
                k = int(np.argmax(g > 0.6))
                intent = dict(t=pe['t'], p=tw[k], zmin=tw[:k + 1, 2].min() if k > 0 else tw[0, 2], k=k)
                break
        # executed: TCP at the frame the close command starts
        prev, ex = None, None
        for f in ep['frames']:
            gc = f['grip_cmd'][s]
            closed = abs(gc - arm.robot.gripper_closed) < abs(gc - arm.robot.gripper_open)
            if prev is not None and closed and not prev:
                env.d.qpos[:] = f['qfull']; mujoco.mj_forward(env.m, env.d)
                T = arm.T_base_in_world @ arm.get_T_ee_in_base()
                ex = dict(t=f['t'], p=T[:3, 3].copy(), b=np.array(f['objects'][f'bread_{s}'][:3]))
                break
            prev = closed
        # lowest TCP z in the 1.5 s after closing starts
        zlow = None
        if ex:
            zs = []
            for f in ep['frames']:
                if ex['t'] <= f['t'] <= ex['t'] + 1.5:
                    env.d.qpos[:] = f['qfull']; mujoco.mj_forward(env.m, env.d)
                    zs.append((arm.T_base_in_world @ arm.get_T_ee_in_base())[2, 3])
            zlow = min(zs)
        r = dict(ep=os.path.basename(epd), side=s, kind=kind,
                 kc_minus_center=100 * (kw[:, 2].mean() - bpos0[2]),
                 kpts_z_range=(100 * (kw[:, 2].min() - bpos0[2]), 100 * (kw[:, 2].max() - bpos0[2])),
                 succ=meta['result']['per_side'][s]['success'] if 'result' in meta else None)
        if intent:
            r['intent_minus_center'] = 100 * (intent['p'][2] - bpos0[2])
            r['intent_minus_kc'] = 100 * (intent['p'][2] - kw[:, 2].mean())
            r['intent_horiz_from_center'] = 100 * np.linalg.norm(intent['p'][:2] - bpos0[:2])
            r['intent_k'] = intent['k']
        if ex:
            r['exec_minus_center'] = 100 * (ex['p'][2] - ex['b'][2])
            r['exec_minus_intent'] = 100 * (ex['p'][2] - intent['p'][2]) if intent else None
            r['exec_low_minus_center'] = 100 * (zlow - ex['b'][2])
            r['dt_close_after_intent'] = ex['t'] - intent['t'] if intent else None
        r['bread_half_h'] = 100 * ext[2] / 2
        rows.append(r)


keys = ['kc_minus_center','intent_minus_center','intent_minus_kc','exec_minus_center','exec_minus_intent','exec_low_minus_center','intent_horiz_from_center','dt_close_after_intent','intent_k','bread_half_h']
print('ep    side  kind       succ ' + ' '.join(k[:14].rjust(14) for k in keys))
for r in rows:
    print(f"{r['ep']} {r['side']:5s} {r['kind']:10s} {str(r['succ'])[:5]:5s} " + ' '.join((f"{r[k]:14.2f}" if isinstance(r.get(k),(int,float)) and r.get(k) is not None else ' '*14) for k in keys))
for s in ('right','left'):
    print(s, {k: round(float(np.nanmean([r[k] for r in rows if r['side']==s and r.get(k) is not None])),2) for k in keys})
