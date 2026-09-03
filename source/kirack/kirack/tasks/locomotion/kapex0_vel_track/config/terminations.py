from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
import torch

def root_height_below_minimum_relative(
    env:ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.3
) -> torch.Tensor :
    asset : Articulation = env.scene[asset_cfg.name]
    # env.scene.env_origins: 각 환경의 원점 (num_envs, 3)
    terrain_height = env.scene.env_origins[:, 2]
    relative_height = asset.data.root_pos_w[:, 2] - terrain_height
    return relative_height < minimum_height