# -*- coding: utf-8 -*-
"""5 s of the closed loop per robot (render -> perception -> released policy -> controller -> physics).
Skips without the checkpoint or an OpenGL context; uses the GPU when there is one."""
import os

import numpy as np
import pytest

from conftest import CKPT, robots_with_assets

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def policy():
    if not os.path.exists(CKPT):
        pytest.skip("checkpoint missing (huggingface-cli download Leo-TX/HumanEgo --include 'serve_bread/*')")
    torch = pytest.importorskip("torch")
    from humanego_sim.env import load_config
    from humanego_sim.runner import load_policy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return load_policy(load_config(), dev), dev


@pytest.mark.parametrize("profile", ["production", "tuned"])
@pytest.mark.parametrize("robot", robots_with_assets())
def test_five_second_episode(robot, profile, policy):
    from humanego_sim.env import ServeBreadSim, load_config
    from humanego_sim.runner import EpisodeRunner
    pol, dev = policy
    cfg = load_config(None, {"robot": robot, "profile": profile, "control": {"max_episode_time": 5.0}})
    try:
        env = ServeBreadSim(cfg)
    except Exception as e:                       # no EGL / OSMesa context on this machine
        pytest.skip(f"no OpenGL context: {e!r}")
    res = EpisodeRunner(cfg, policy=pol, device=dev, env=env).run(seed=0, verbose=False, intro=False)
    assert res.n_policy_steps >= 5                # the policy re-planned (every ~0.3 s once perception is done)
    assert res.duration == pytest.approx(5.0, abs=0.5)
    assert set(res.per_side) == set(env.sides)
    for side, r in res.per_side.items():
        assert np.isfinite(r["dist_to_plate_center"]) and "success" in r
    # the right arm left its start pose towards the bread
    arm = env.arms["right"]
    assert np.linalg.norm(arm.get_j() - np.asarray(env.initial_joints("right"))) > 0.05
