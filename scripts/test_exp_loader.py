# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test experiment YAML loading and exp_loader overrides.

Examples:
    ../IsaacLab/isaaclab.sh -p scripts/test_exp_loader.py --headless
    ../IsaacLab/isaaclab.sh -p scripts/test_exp_loader.py --headless --exp_cfg /path/to/exp.yaml
    ../IsaacLab/isaaclab.sh -p scripts/test_exp_loader.py --headless --create_env --num_envs 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", os.path.abspath(os.path.join(_REPO_ROOT, "..", "IsaacLab")))
for _path in (
    os.path.join(_ISAACLAB_ROOT, "source", "isaaclab"),
    os.path.join(_ISAACLAB_ROOT, "source", "isaaclab_assets"),
    os.path.join(_ISAACLAB_ROOT, "source", "isaaclab_contrib"),
    os.path.join(_ISAACLAB_ROOT, "source", "isaaclab_rl"),
    os.path.join(_ISAACLAB_ROOT, "source", "isaaclab_tasks"),
    os.path.join(_REPO_ROOT, "source", "kirack"),
):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test kirack.utils.exp_loader with an Isaac Lab task config.")
parser.add_argument("--task", type=str, default="Kapex-Standing-v0", help="Gym task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Gym registry key for the agent config.",
)
parser.add_argument("--exp_cfg", type=str, default=None, help="Experiment YAML path. Defaults to an embedded sample.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to configure.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations when --create_env is used.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    default=False,
    help="Pass the YAML to exp_loader exactly. By default unsupported sections are skipped for this task cfg.",
)
parser.add_argument(
    "--create_env",
    action="store_true",
    default=False,
    help="After applying overrides, create the Gym env and run a few zero-action steps.",
)
parser.add_argument("--steps", type=int, default=3, help="Number of zero-action env steps for --create_env.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import kirack.tasks  # noqa: F401
from kirack.utils import exp_loader


DEFAULT_EXP = {
    "agent": {
        "max_iterations": 12,
        "save_interval": 3,
        "policy": {
            "init_noise_std": 0.42,
        },
        "algorithm": {
            "learning_rate": 7.0e-4,
            "schedule": "fixed",
            "entropy_coef": 0.02,
        },
    },
    "rewards": {
        "alive": {"weight": 3.0},
        "base_height": {"weight": -9.0, "params": {"target_height": 0.88}},
        "feet_slide": None,
    },
    "events": {
        "push_robot": None,
        "reset_robot_joints": {"params": {"position_range": [-0.02, 0.02]}},
    },
    "terminations": {
        "root_too_low": {"params": {"minimum_height": 0.44}},
    },
    "scene": {
        "env_spacing": 3.3,
    },
}


def _write_default_exp() -> str:
    """Write the embedded sample to a temporary YAML file so get_exp_cfg() is exercised."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="kapex_exp_loader_", delete=False) as f:
        yaml.safe_dump(DEFAULT_EXP, f, sort_keys=False)
        return f.name


def _same_value(actual: Any, expected: Any) -> bool:
    """Compare values while accepting tuple/list differences from YAML."""
    if isinstance(actual, tuple) and isinstance(expected, list):
        return list(actual) == expected
    if isinstance(actual, list) and isinstance(expected, tuple):
        return actual == list(expected)
    return actual == expected


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if not _same_value(actual, expected):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    print(f"[OK] {label}: {actual!r}")


def _has_velocity_command(env_cfg: Any) -> bool:
    commands = getattr(env_cfg, "commands", None)
    return commands is not None and getattr(commands, "base_velocity", None) is not None


def _has_terrain_generator(env_cfg: Any) -> bool:
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    return terrain is not None and getattr(terrain, "terrain_generator", None) is not None


def _filter_exp_for_cfg(env_cfg: Any, exp: dict[str, Any]) -> dict[str, Any]:
    """Skip sections that exp_loader assumes but this standing cfg may not define."""
    filtered = copy.deepcopy(exp)

    if not _has_velocity_command(env_cfg):
        for key in ("vel_command", "vel_limit"):
            if key in filtered:
                print(f"[SKIP] {key}: env_cfg.commands.base_velocity is not defined by {type(env_cfg).__name__}")
                filtered.pop(key)

    if "curriculum" in filtered and not hasattr(env_cfg, "curriculum"):
        print(f"[SKIP] curriculum: env_cfg.curriculum is not defined by {type(env_cfg).__name__}")
        filtered.pop("curriculum")

    if "terrain" in filtered and not _has_terrain_generator(env_cfg):
        print(f"[SKIP] terrain: env_cfg.scene.terrain.terrain_generator is not defined by {type(env_cfg).__name__}")
        filtered.pop("terrain")

    return filtered


def _verify_agent(agent_cfg: Any, exp: Mapping[str, Any]) -> None:
    agent_exp = exp.get("agent")
    if not isinstance(agent_exp, Mapping):
        return

    for key, expected in agent_exp.items():
        if not hasattr(agent_cfg, key):
            print(f"[SKIP] agent.{key}: field is not present")
            continue

        target = getattr(agent_cfg, key)
        if isinstance(expected, Mapping) and not isinstance(target, (int, float, str, bool, list, tuple, dict)):
            for sub_key, sub_expected in expected.items():
                if not hasattr(target, sub_key):
                    print(f"[SKIP] agent.{key}.{sub_key}: field is not present")
                    continue
                sub_target = getattr(target, sub_key)
                if isinstance(sub_expected, Mapping) and not isinstance(sub_target, dict):
                    print(f"[SKIP] agent.{key}.{sub_key}: exp_loader skips dict values for non-dict fields")
                    continue
                _assert_equal(f"agent.{key}.{sub_key}", sub_target, sub_expected)
        else:
            if isinstance(expected, Mapping) and not isinstance(target, dict):
                print(f"[SKIP] agent.{key}: exp_loader skips dict values for non-dict fields")
                continue
            _assert_equal(f"agent.{key}", target, expected)


def _verify_term_group(env_cfg: Any, group_name: str, exp: Mapping[str, Any]) -> None:
    overrides = exp.get(group_name)
    if not isinstance(overrides, Mapping):
        return

    group = getattr(env_cfg, group_name, None)
    if group is None:
        print(f"[SKIP] {group_name}: group is not present")
        return

    for term_name, override in overrides.items():
        term = getattr(group, term_name, None)
        if override is None:
            _assert_equal(f"{group_name}.{term_name}", term, None)
            continue
        if term is None:
            print(f"[SKIP] {group_name}.{term_name}: term is not present")
            continue
        if group_name == "rewards" and "weight" in override:
            _assert_equal(f"{group_name}.{term_name}.weight", term.weight, override["weight"])
        if "params" in override:
            for param_name, expected in override["params"].items():
                _assert_equal(f"{group_name}.{term_name}.params.{param_name}", term.params[param_name], expected)


def _verify_env(env_cfg: Any, exp: Mapping[str, Any]) -> None:
    _verify_term_group(env_cfg, "rewards", exp)
    _verify_term_group(env_cfg, "events", exp)
    _verify_term_group(env_cfg, "terminations", exp)
    _verify_term_group(env_cfg, "curriculum", exp)

    scene_exp = exp.get("scene")
    if isinstance(scene_exp, Mapping) and "env_spacing" in scene_exp:
        _assert_equal("scene.env_spacing", env_cfg.scene.env_spacing, scene_exp["env_spacing"])

    if "vel_command" in exp and _has_velocity_command(env_cfg):
        ranges = env_cfg.commands.base_velocity.ranges
        for name, expected in exp["vel_command"].items():
            _assert_equal(f"commands.base_velocity.ranges.{name}", getattr(ranges, name), expected)

    if "vel_limit" in exp and _has_velocity_command(env_cfg):
        ranges = env_cfg.commands.base_velocity.limit_ranges
        for name, expected in exp["vel_limit"].items():
            _assert_equal(f"commands.base_velocity.limit_ranges.{name}", getattr(ranges, name), expected)


def main() -> None:
    """Load configs, apply experiment YAML, verify selected overrides, and optionally create an env."""
    exp_path = args_cli.exp_cfg or os.environ.get("KIRACK_EXP_CFG") or _write_default_exp()
    os.environ["KIRACK_EXP_CFG"] = exp_path
    exp_loader._EXP_CFG = None

    print(f"[INFO] task: {args_cli.task}")
    print(f"[INFO] exp_cfg: {exp_path}")

    exp = exp_loader.get_exp_cfg()
    if exp is None:
        raise RuntimeError("get_exp_cfg() returned None")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    applied_exp = exp if args_cli.strict else _filter_exp_for_cfg(env_cfg, exp)

    exp_loader.apply_exp_cfg(env_cfg, applied_exp)
    exp_loader.apply_agent_exp_cfg(agent_cfg, applied_exp)

    _verify_env(env_cfg, applied_exp)
    _verify_agent(agent_cfg, applied_exp)

    if args_cli.create_env:
        print("[INFO] Creating env with overridden cfg.")
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
        env.close()
        print(f"[OK] Created env and ran {args_cli.steps} zero-action steps.")

    print("[OK] exp_loader smoke test completed.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        sys.stdout.flush()
