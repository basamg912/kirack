# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the Kapex standing task.

Built-in observations from ``isaaclab.envs.mdp`` (e.g. ``base_lin_vel``,
``base_ang_vel``, ``projected_gravity``, ``joint_pos_rel``, ``joint_vel_rel``,
``last_action``) are re-exported via ``mdp/__init__.py`` and cover the standard
proprioceptive observations used here.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def torso_com(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="WL3")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    body_ids = asset_cfg.body_ids

    coms = asset.root_physx_view.get_coms().clone()

    # default_com = torch.tensor([-0.066071861, -0.19773669, 0.0024721903], device=env.device)
    default_com = torch.tensor([-0.066071861, -0.19773669, 0.0024721903], device=env.device)
    # 기본값을 뺀 delta com 반환 (num_envs, 3)
    return coms[:, body_ids, :3].squeeze(1).to(env.device) - default_com


def obs_feet_air_time(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"])
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return air_time
