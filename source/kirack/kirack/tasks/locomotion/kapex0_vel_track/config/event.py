from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

VELOCITY_RANGE = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
}

def print_com(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene["robot"]
    wl3_idx = robot.find_bodies("WL3")[0][0]
    coms = robot.root_physx_view.get_coms().clone()
    print(f"[COM] env0 WL3: {coms[0, wl3_idx, :3]}")
    
def joint_position_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    noise_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Applies random noise to the robot's joint positions.

    This function adds noise to the joint positions to simulate sensing errors or perturbations.
    It samples the noise for each joint from the given range dictionary and applies it to the simulation.

    The function takes a dictionary of noise ranges, where the keys are joint indices (as strings)
    and the values are tuples of the form ``(min, max)``.
    If a joint is not specified in the dictionary, no noise is added to that joint.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor): The environment IDs to apply the noise to.
        noise_range (dict[str, tuple[float, float]]): Noise range dictionary for each joint.
        asset_cfg (SceneEntityCfg): Configuration for the asset (robot).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[env_ids].clone()

    joint_pos += math_utils.sample_uniform(*noise_range, joint_pos.shape, joint_pos.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_state_to_sim(joint_pos, asset.data.joint_vel[env_ids], env_ids=env_ids)
