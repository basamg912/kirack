from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from kirack.assets.robot import KAPEX0_CFG
from kirack.tasks.locomotion.kapex0_vel_track.kapex0_vel_track_env_cfg import Kapex0VelTrackEnvCfg


from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from kirack.utils.exp_loader import get_exp_cfg, apply_exp_cfg


@configclass
class KirackEnvCfg(Kapex0VelTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = KAPEX0_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.observations.policy.joint_pos_rel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["(L|R)LJ[1-7]"])
        # self.observations.policy.joint_vel_rel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["(L|R)LJ[1-7]"])

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (5.0, -5.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


        # load experiment config if --exp is provided
        exp = get_exp_cfg()
        if exp is not None:
            apply_exp_cfg(self, exp)

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import (
    HfRandomUniformTerrainCfg,  # 랜덤 ruggedly terrain
    HfWaveTerrainCfg,  # 파도모양 terrain
    MeshPlaneTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)

from kirack.tasks.locomotion.kapex0_vel_track.kapex0_vel_track_env_cfg import PLAY_TERRAIN_CFG
@configclass
class KirackPlayEnvCfg(KirackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.viewer.origin_type = "world"
        self.viewer.eye = (-10.0, 0.0, 15.0)
        self.viewer.lookat = (0.0, 0.0, 2.5)

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_generator=PLAY_TERRAIN_CFG,
            max_init_terrain_level=0,  # 초기 학습 안정성 위해 lowest terrain 에 스폰
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # self.commands.base_velocity.ranges = UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x = (0.5, 0.5),
        #     lin_vel_y = (0.0, 0.0),
        #     ang_vel_z = (0, 0),
        # )

        self.curriculum = None
