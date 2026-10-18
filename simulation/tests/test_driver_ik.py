# -*- coding: utf-8 -*-
"""Driver emulator: SE(3) helpers, frame conversions and the IK round trip on the real scene
(no rendering, no policy)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from conftest import robots_with_assets
from humanego_sim.driver import T_to_p, p_to_T
from humanego_sim.env import ServeBreadSim, load_config

_ENVS = {}


def env_for(robot):
    if robot not in _ENVS:
        env = ServeBreadSim(load_config(None, {"robot": robot}), render=False)
        env.reset(settle=0.0)
        _ENVS[robot] = env
    return _ENVS[robot]


def test_pose_vector_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.r_[rng.uniform(-1, 1, 3), R.random(random_state=rng).as_rotvec()]
        np.testing.assert_allclose(T_to_p(p_to_T(p)), p, atol=1e-9)
        T = p_to_T(p)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("robot", robots_with_assets())
def test_cam_base_conversions_are_inverse(robot):
    env = env_for(robot)
    for side, arm in env.arms.items():
        p = arm.get_p_in_cam()
        np.testing.assert_allclose(arm.p_in_base_to_p_in_cam(arm.p_in_cam_to_p_in_base(p)), p, atol=1e-9)
        np.testing.assert_allclose(arm.p_in_cam_to_p_in_base(p[:3]), arm.get_p()[:3], atol=1e-9)
        # the camera sees the EE where world kinematics put it
        T_ee_w = env.T_cam_in_world @ arm.get_T_ee_in_cam()
        T_ee_w2 = arm.T_base_in_world @ arm.get_T_ee_in_base()
        np.testing.assert_allclose(T_ee_w, T_ee_w2, atol=1e-9)


@pytest.mark.parametrize("robot", robots_with_assets())
def test_ik_round_trip(robot):
    """fk(q*) -> ik from the start pose -> the same tool pose (within servo tolerance)."""
    env = env_for(robot)
    rng = np.random.default_rng(1)
    for side, arm in env.arms.items():
        q0 = arm.get_j()
        n_ok = 0
        for _ in range(8):
            q_t = np.clip(q0 + rng.uniform(-0.25, 0.25, len(q0)), arm.qlo, arm.qhi)
            T = arm.fk_raw(q_t)
            q, err = arm.ik(T, q0, iters=300)
            T2 = arm.fk_raw(q)
            rot_err = (R.from_matrix(T2[:3, :3]) * R.from_matrix(T[:3, :3]).inv()).magnitude()
            assert err == pytest.approx(np.linalg.norm(T2[:3, 3] - T[:3, 3]), abs=1e-9)
            assert np.all(q >= arm.qlo - 1e-9) and np.all(q <= arm.qhi + 1e-9)
            n_ok += err < 1e-3 and rot_err < 1e-2
        assert n_ok >= 7, f"{robot}/{side}: IK converged on {n_ok}/8 reachable targets"


@pytest.mark.parametrize("robot", robots_with_assets())
def test_move_p_reaches_the_commanded_pose(robot):
    """Cartesian command in the camera frame -> physics -> measured pose (the servo path of the runner)."""
    env = ServeBreadSim(load_config(None, {"robot": robot}), render=False)
    env.reset(settle=0.1)
    arm = env.arms["right"]
    p0 = arm.get_p_in_cam()
    goal = p0.copy()
    # 4 cm down in the world (up is joint-limited on the Trossen: its start pose sits 2.6 deg from
    # wrist joint 4's lower limit, so a pure lift at a fixed orientation is not reachable)
    goal[:3] += env.T_world_in_cam[:3, :3] @ np.array([0.0, 0.0, -0.04])
    assert arm.move_p_in_cam(goal, duration=0.6)
    for _ in range(int(1.5 / env.dt)):
        env.step()
    p = arm.get_p_in_cam()
    assert np.linalg.norm(p[:3] - goal[:3]) < 0.002
    assert (R.from_rotvec(p[3:]) * R.from_rotvec(goal[3:]).inv()).magnitude() < 0.01
