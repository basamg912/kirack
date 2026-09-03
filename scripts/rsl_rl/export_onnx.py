# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a trained RSL-RL policy checkpoint to ONNX (and JIT)."""

# python scripts/rsl_rl/export_onnx.py --task Kapex-Standing-Play-v0 --exp <exp_name> --checkpoint <path/to/model.pt>
# 또는 load_run 사용
# python scripts/rsl_rl/export_onnx.py --task <Task-Name> --exp <exp_name> --load_run <run_name>
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export a trained RSL-RL policy to ONNX.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--exp", type=str, default=None, help="Name of the experiment to load.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save the exported policy. Defaults to '<checkpoint_dir>/exported'.",
)
parser.add_argument("--onnx_filename", type=str, default="policy.onnx", help="Filename for the exported ONNX policy.")
parser.add_argument("--jit_filename", type=str, default="policy.pt", help="Filename for the exported JIT policy.")
parser.add_argument("--skip_jit", action="store_true", default=False, help="Skip exporting the JIT policy.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# force minimal env count for fast init
args_cli.num_envs = 1
# headless export — no rendering needed
args_cli.headless = True

if args_cli.exp is not None:
    import os

    os.environ["KIRACK_EXP_CFG"] = args_cli.exp

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os

import gymnasium as gym
import kirack.tasks  # noqa: F401
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Export a trained RSL-RL policy to ONNX."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # agent-related experiment config is applied here, matching train.py / play.py
    from kirack.utils.exp_loader import apply_agent_exp_cfg, get_exp_cfg

    _exp = get_exp_cfg()
    if _exp is not None:
        apply_agent_exp_cfg(agent_cfg, _exp)

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve the checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment (needed by runner for obs/action spec)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # decide where to write exported files
    export_model_dir = args_cli.output_dir or os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    if version.parse(installed_version) >= version.parse("4.0.0"):
        if not args_cli.skip_jit:
            runner.export_policy_to_jit(path=export_model_dir, filename=args_cli.jit_filename)
        runner.export_policy_to_onnx(path=export_model_dir, filename=args_cli.onnx_filename)
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        if not args_cli.skip_jit:
            export_policy_as_jit(
                policy_nn, normalizer=normalizer, path=export_model_dir, filename=args_cli.jit_filename
            )
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=args_cli.onnx_filename)

    onnx_path = os.path.join(export_model_dir, args_cli.onnx_filename)
    print(f"[INFO]: Exported ONNX policy to: {onnx_path}")
    if not args_cli.skip_jit:
        jit_path = os.path.join(export_model_dir, args_cli.jit_filename)
        print(f"[INFO]: Exported JIT policy to: {jit_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
