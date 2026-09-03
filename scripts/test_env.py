"""Diagnostic inspector for the Kapex / G1 standing envs.

Builds a small env (few parallel robots, headless by default), resets once, and
dumps the contents of ``robot.data.*`` plus the five attention observations.
Useful for verifying URDF parsing, joint ordering, body name resolution, and
observation pipeline output before launching real training.

Usage:
    python scripts/test_env.py                              # Kapex play env
    python scripts/test_env.py --task G1-Standing-Play-v0   # G1 instead
    python scripts/test_env.py --num_envs 2 --no_terrain    # smaller, no terrain
    python scripts/test_env.py --steps 5                    # run a few steps too
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inspect articulation data + attention obs.")
parser.add_argument(
    "--task",
    type=str,
    default="Kapex-Standing-Play-v0",
    help="Registered gym task id (e.g. Kapex-Standing-Play-v0 or G1-Standing-Play-v0).",
)
parser.add_argument("--num_envs", type=int, default=3, help="Number of parallel envs to spawn.")
parser.add_argument(
    "--no_terrain",
    action="store_true",
    help="Disable terrain + height_scanner for a minimal scene.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of zero-action steps to run after reset (default 0: just print and exit).",
)
parser.add_argument(
    "--max_rows",
    type=int,
    default=8,
    help="Truncate tensor previews to this many rows for readability.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always headless for a quick inspection script.
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Heavy imports happen AFTER AppLauncher initialises Isaac Sim.
# -----------------------------------------------------------------------------
import gymnasium as gym
import kirack.tasks  # noqa: F401  (registers gym tasks)
import torch

# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------
SEP = "=" * 78


def banner(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def tensor_preview(name: str, t, max_rows: int) -> None:
    """Print tensor shape, dtype, device, and a small numeric preview."""
    if t is None:
        print(f"  {name}: None")
        return
    if not isinstance(t, torch.Tensor):
        print(f"  {name}: {t!r}")
        return
    shape = tuple(t.shape)
    print(f"  {name:32s}  shape={str(shape):24s} dtype={t.dtype} device={t.device}")
    # Show a clipped preview
    with torch.no_grad():
        preview = t.detach()
        # Take the first env, then truncate by max_rows on the next axis if possible.
        if preview.ndim >= 1 and preview.shape[0] > 0:
            preview = preview[0]
        if preview.ndim >= 1 and preview.shape[0] > max_rows:
            preview = preview[:max_rows]
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
        print(f"    env0 preview:\n{preview.cpu().numpy()}\n")


# -----------------------------------------------------------------------------
# Build env
# -----------------------------------------------------------------------------
banner(f"Building task '{args_cli.task}' with {args_cli.num_envs} envs")

# Construct the env_cfg directly via hydra entry_point isn't trivial here;
# we rely on gym.make + cli-style cfg overrides via the env_cfg_entry_point mechanism.
# Easiest path: pull env_cfg_entry_point ourselves so we can mutate the cfg.
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)

if args_cli.no_terrain:
    print("[INFO] Disabling terrain + height_scanner for a minimal scene.")
    env_cfg.scene.terrain = None
    env_cfg.scene.height_scanner = None

env = gym.make(args_cli.task, cfg=env_cfg)
obs_dict, _ = env.reset()

unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]
data = robot.data

# -----------------------------------------------------------------------------
# 1) Names / indices
# -----------------------------------------------------------------------------
banner("Joint names")
print(f"  num_joints = {len(data.joint_names)}")
for i, name in enumerate(data.joint_names):
    print(f"    [{i:>2d}] {name}")

banner("Body names")
print(f"  num_bodies = {len(data.body_names)}")
for i, name in enumerate(data.body_names):
    print(f"    [{i:>2d}] {name}")

# -----------------------------------------------------------------------------
# 2) Static articulation data
# -----------------------------------------------------------------------------
banner("Joint static fields (default / limits)")
tensor_preview("default_joint_pos", data.default_joint_pos, args_cli.max_rows)
tensor_preview("default_joint_vel", data.default_joint_vel, args_cli.max_rows)
tensor_preview("default_joint_pos_limits", data.default_joint_pos_limits, args_cli.max_rows)
tensor_preview("joint_pos_limits", data.joint_pos_limits, args_cli.max_rows)
tensor_preview("joint_effort_limits", data.joint_effort_limits, args_cli.max_rows)
tensor_preview("joint_vel_limits", data.joint_vel_limits, args_cli.max_rows)
tensor_preview("soft_joint_pos_limits", data.soft_joint_pos_limits, args_cli.max_rows)
tensor_preview("default_joint_stiffness", data.default_joint_stiffness, args_cli.max_rows)
tensor_preview("default_joint_damping", data.default_joint_damping, args_cli.max_rows)
tensor_preview("default_joint_armature", data.default_joint_armature, args_cli.max_rows)
tensor_preview("default_joint_friction_coeff", data.default_joint_friction_coeff, args_cli.max_rows)

# -----------------------------------------------------------------------------
# 3) Dynamic state (after reset, before any step)
# -----------------------------------------------------------------------------
banner("Joint dynamic state (post-reset)")
tensor_preview("joint_pos", data.joint_pos, args_cli.max_rows)
tensor_preview("joint_vel", data.joint_vel, args_cli.max_rows)

banner("Root / body kinematics")
tensor_preview("root_pos_w", data.root_pos_w, args_cli.max_rows)
tensor_preview("root_quat_w", data.root_quat_w, args_cli.max_rows)
tensor_preview("root_lin_vel_b", data.root_lin_vel_b, args_cli.max_rows)
tensor_preview("root_ang_vel_b", data.root_ang_vel_b, args_cli.max_rows)
tensor_preview("projected_gravity_b", data.projected_gravity_b, args_cli.max_rows)

# All body positions for env 0
banner("body_pos_w / body_quat_w (env 0, all bodies)")
tensor_preview("body_pos_w", data.body_pos_w, args_cli.max_rows)
tensor_preview("body_quat_w", data.body_quat_w, args_cli.max_rows)

# -----------------------------------------------------------------------------
# 4) Attention observations (5 groups)
# -----------------------------------------------------------------------------
banner("Attention obs groups (env.observation_manager.compute)")
obs = unwrapped.observation_manager.compute()
# obs is a dict-of-dict or dict-of-tensor depending on group structure.
for group_name, group_value in obs.items():
    if isinstance(group_value, dict):
        for term_name, t in group_value.items():
            tensor_preview(f"{group_name}/{term_name}", t, args_cli.max_rows)
    else:
        tensor_preview(group_name, group_value, args_cli.max_rows)

# Cache attributes set by description events
banner("Description caches on env")
for attr in ("_kapex_joint_desc", "_kapex_feet_desc"):
    if hasattr(unwrapped, attr):
        tensor_preview(attr, getattr(unwrapped, attr), args_cli.max_rows)
    else:
        print(f"  {attr}: <not set>")

# -----------------------------------------------------------------------------
# 5) Action manager info
# -----------------------------------------------------------------------------
banner("Action manager")
am = unwrapped.action_manager
print(f"  total_action_dim = {am.total_action_dim}")
print(f"  active terms     = {list(am.active_terms)}")
for term_name in am.active_terms:
    term = am.get_term(term_name)
    print(f"  term '{term_name}' action_dim = {term.action_dim}")
    if hasattr(term, "_joint_names") and term._joint_names is not None:
        print(f"             joints = {term._joint_names}")

# -----------------------------------------------------------------------------
# 6) Optional: run a few zero-action steps
# -----------------------------------------------------------------------------
if args_cli.steps > 0:
    banner(f"Stepping {args_cli.steps} times with zero actions")
    zero_action = torch.zeros(unwrapped.num_envs, am.total_action_dim, device=unwrapped.device)
    for step in range(args_cli.steps):
        obs_dict, rew, terminated, truncated, _ = env.step(zero_action)
        print(
            f"  step {step:>3d}: "
            f"reward.mean={rew.mean().item():+.4f}  "
            f"terminated.sum={int(terminated.sum().item())}  "
            f"truncated.sum={int(truncated.sum().item())}"
        )

    banner("Joint state after stepping")
    tensor_preview("joint_pos", data.joint_pos, args_cli.max_rows)
    tensor_preview("joint_vel", data.joint_vel, args_cli.max_rows)
    tensor_preview("root_pos_w", data.root_pos_w, args_cli.max_rows)

# -----------------------------------------------------------------------------
# Teardown
# -----------------------------------------------------------------------------
banner("Done")
env.close()
simulation_app.close()
