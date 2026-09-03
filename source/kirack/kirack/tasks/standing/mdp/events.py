# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event terms for the Kapex standing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_position_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    noise_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add uniform noise to the current joint positions of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    pos = asset.data.joint_pos[env_ids].clone()
    noise = torch.empty_like(pos).uniform_(*noise_range)
    asset.write_joint_position_to_sim(pos + noise, env_ids=env_ids)
