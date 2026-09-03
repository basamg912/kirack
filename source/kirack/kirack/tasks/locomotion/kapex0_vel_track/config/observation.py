from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
import torch

def torso_com(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="WL3")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    body_ids = asset_cfg.body_ids

    coms = asset.root_physx_view.get_coms().clone()

    # default_com = torch.tensor([-0.066071861, -0.19773669, 0.0024721903], device=env.device)
    default_com = torch.tensor([-0.066071861, -0.19773669, 0.0024721903], device=env.device)
    # 기본값을 뺀 delta com 반환 (num_envs, 3)
    return coms[:, body_ids, :3].squeeze(1).to(env.device) - default_com
def obs_feet_air_time(env : ManagerBasedEnv, sensor_cfg : SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["LL[67]", "RL[67]"])) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return air_time
