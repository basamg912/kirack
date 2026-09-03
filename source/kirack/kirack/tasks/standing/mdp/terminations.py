# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for the Kapex standing task.

Built-in terminations from ``isaaclab.envs.mdp`` (e.g. ``time_out``,
``bad_orientation``, ``root_height_below_minimum``) are re-exported via
``mdp/__init__.py`` and cover most needs. Add custom terms here as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_body_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any of the specified bodies experiences contact above ``threshold``."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    max_force = forces.norm(dim=-1).max(dim=1)[0]
    return torch.any(max_force > threshold, dim=-1)


def root_height_below_minimum_relative(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), minimum_height: float = 0.3
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # env.scene.env_origins: 각 환경의 원점 (num_envs, 3)
    terrain_height = env.scene.env_origins[:, 2]
    relative_height = asset.data.root_pos_w[:, 2] - terrain_height
    return relative_height < minimum_height
