# -*- coding: utf-8 -*-
"""Config loading: defaults, per-robot calibration, overrides and the named profiles."""
import glob
import json
import os

import pytest

from humanego_sim import profiles
from humanego_sim.env import deep_update, load_config
from humanego_sim.robots import ROBOTS

TUNED = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(profiles.TUNED_DIR, "*.json")))


def test_defaults_are_the_real_rig():
    cfg = load_config()
    assert cfg["robot"] == "trossen_wxai" and cfg["mode"] == "bimanual"
    assert cfg["profile"] == "production"
    c = cfg["control"]
    # cfg/inference/serve_bread/Control.yaml of the real rig
    assert (c["look_ahead_offset"], c["step_stride"], c["control_freq"]) == (45, 2, 10.0)
    assert (c["alpha"], c["alpha_rot"], c["duration_smooth_param"]) == (0.5, 0.0, 12.0)
    assert c["grasp_threshold"] == 0.6 and c["safe_z_min"] == -0.20
    assert not (c.get("grasp_assist") or {}).get("enabled", False)


def test_deep_update_merges_nested_and_copies():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 3}
    upd = {"a": {"b": 5}, "e": {"f": 1}}
    out = deep_update(base, upd)
    assert out == {"a": {"b": 5, "c": [1, 2]}, "d": 3, "e": {"f": 1}}
    out["a"]["c"].append(9)
    assert base["a"]["c"] == [1, 2]                     # no aliasing into the input


def test_per_robot_calibration_applies_to_that_robot_only():
    ur = load_config(None, {"robot": "ur5e"})
    fr = load_config(None, {"robot": "franka_panda"})
    tr = load_config(None, {"robot": "trossen_wxai"})
    assert ur["physics"]["soft"]["dim"] == 3
    assert ur["control"]["grasp_assist"]["gap_empty"] == 0.015
    assert fr["physics"]["soft"]["surface"] == "fine" and fr["physics"]["soft"]["radius"] == 0.002
    assert "dim" not in tr["physics"]["soft"] and "surface" not in tr["physics"]["soft"]
    # per_robot merges into the defaults instead of replacing the section
    assert ur["physics"]["soft"]["young"] == tr["physics"]["soft"]["young"]


def test_explicit_overrides_beat_per_robot():
    cfg = load_config(None, {"robot": "ur5e", "physics": {"soft": {"dim": 2}}, "control": {"alpha": 0.1}})
    assert cfg["physics"]["soft"]["dim"] == 2 and cfg["control"]["alpha"] == 0.1


def test_production_profile_is_the_default():
    for robot in ROBOTS:
        a = load_config(None, {"robot": robot})
        b = load_config(None, {"robot": robot, "profile": "production"})
        assert a == b


@pytest.mark.parametrize("robot", TUNED)
def test_tuned_profile_applies_tuned_params_and_skill(robot):
    t = profiles.load_tuned(robot)
    cfg = load_config(None, {"robot": robot, "profile": "tuned"})
    assert cfg["profile"] == "tuned"
    assert cfg["control"]["grasp_assist"]["enabled"] is True
    for key, val in t["params"].items():
        cur = cfg
        for p in key.split("."):
            cur = cur[p]
        assert cur == val, key


@pytest.mark.parametrize("robot", TUNED)
@pytest.mark.parametrize("mode", ["bimanual", "single"])
def test_tuned_profile_equals_the_old_set_route(robot, mode):
    """--profile tuned == --params configs/tuned/<robot>.json --set control.grasp_assist={"enabled": true}."""
    from autotune.params import params_to_overrides
    extra = {"robot": robot, "mode": mode, "control": {"grasp_assist": {"enabled": True}}}
    base = load_config(None, extra)
    old = load_config(None, deep_update(params_to_overrides(profiles.load_tuned(robot)["params"], base), extra))
    new = load_config(None, {"robot": robot, "mode": mode, "profile": "tuned"})
    old.pop("profile"), new.pop("profile")
    assert old == new


def test_overrides_apply_on_top_of_the_profile():
    cfg = load_config(None, {"profile": "tuned", "control": {"alpha": 0.42, "grasp_assist": {"regrasp_max": 0}}})
    assert cfg["control"]["alpha"] == 0.42
    assert cfg["control"]["grasp_assist"]["regrasp_max"] == 0
    assert cfg["control"]["grasp_assist"]["enabled"] is True        # rest of the profile survives


def test_profile_errors():
    with pytest.raises(ValueError):
        load_config(None, {"profile": "fastest"})
    with pytest.raises(FileNotFoundError):
        profiles.load_tuned("no_such_robot")


def test_dotted_to_nested_handles_list_elements():
    base = load_config()
    ov = profiles.dotted_to_nested({"physics.bread_friction.0": 1.5, "control.alpha": 0.2}, base)
    assert ov["physics"]["bread_friction"] == [1.5] + base["physics"]["bread_friction"][1:]
    assert ov["control"]["alpha"] == 0.2
    with pytest.raises(ValueError):
        profiles.dotted_to_nested({"physics.bread_friction.0": 1.5})


@pytest.mark.parametrize("robot", TUNED)
def test_tuned_file_schema(robot):
    """configs/tuned/<robot>.json carries what rebuild_site.py and the web manifest read."""
    t = json.load(open(profiles.tuned_path(robot)))
    assert t["robot"] == robot and robot in ROBOTS
    assert t["params"] and all(isinstance(k, str) and k.count(".") >= 1 for k in t["params"])
    assert t["requires"] == {"control.grasp_assist.enabled": True}
    for k in ("held_out", "held_out_2"):
        h = t[k]
        assert h["seed"] in (7777, 8888) and h["n_scenes"] > 0
        for v in [h["per_arm"], h["both"], *h["per_side"].values()]:
            assert 0.0 <= v <= 1.0
    info = profiles.profile_info("tuned", robot)
    assert info["source"] == f"configs/tuned/{robot}.json" and info["held_out"] == t["held_out"]


def test_cli_scripts_accept_profile():
    """Every entry point that builds a config takes --profile (default production)."""
    import subprocess
    import sys
    scripts = os.path.join(profiles.SIM_ROOT, "scripts")
    for s in ("run_episode.py", "evaluate.py", "compare_params.py", "autotune.py", "autotune_validate.py",
              "build_web.py", "rebuild_site.py"):
        out = subprocess.run([sys.executable, os.path.join(scripts, s), "--help"], capture_output=True, text=True,
                             timeout=120)
        assert out.returncode == 0, (s, out.stderr[-2000:])
        assert "--profile" in out.stdout, s
