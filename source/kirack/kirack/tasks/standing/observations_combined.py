"""Combined Kapex + G1 observation builders.

Concatenates per-joint / per-foot tensors from both robots along the element
axis so a single URMA-style attention policy can consume them. Order is
deterministic: ``[Kapex, G1]`` for joints, feet, and global vector.

Schemas (single-robot → combined):
- ``joint_desc``: (E, 21, 16) + (E, 23, 16)  → (E, 44, 16)
- ``joint_obs`` : (E, 21, 3)  + (E, 23, 3)   → (E, 44, 3)
- ``feet_desc`` : (E, 2, 3)   + (E, 2, 3)    → (E, 4, 3)
- ``feet_obs``  : (E, 2, 2)   + (E, 2, 2)    → (E, 4, 2)
- ``global_obs``: (E, 15)     + (E, 15)      → (E, 30)
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from kirack.tasks.environment.g1.observations.observation import (
    g1_cache_feet_description,
    g1_cache_joint_description,
    g1_get_feet_description,
    g1_get_feet_observation,
    g1_get_global_observation,
    g1_get_joint_description,
    g1_get_joint_observation,
)
from kirack.tasks.environment.kapex.observations.observation import (
    cache_feet_description,
    cache_joint_description,
    get_feet_description,
    get_feet_observation,
    get_global_observation,
    get_joint_description,
    get_joint_observation,
)

# Asset names. Must match scene config keys.
KAPEX_ASSET = "robot"
G1_ASSET = "robot_g1"

# Body / morphology constants.
KAPEX_ROOT_BODY = "WL3"
KAPEX_FOOT_BODIES = ("LL7", "RL7")
G1_ROOT_BODY = "pelvis"
G1_TORSO_BODY = "torso_link"
G1_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


# ---------------------------------------------------------------------------
# Event-mode cache wrappers (one per robot; called on reset)
# ---------------------------------------------------------------------------

def cache_joint_desc_combined(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    cache_joint_description(env, env_ids, asset_cfg=SceneEntityCfg(KAPEX_ASSET), root_body=KAPEX_ROOT_BODY)
    g1_cache_joint_description(env, env_ids, asset_cfg=SceneEntityCfg(G1_ASSET), root_body=G1_ROOT_BODY)


def cache_feet_desc_combined(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    cache_feet_description(
        env, env_ids, asset_cfg=SceneEntityCfg(KAPEX_ASSET),
        root_body=KAPEX_ROOT_BODY, foot_bodies=KAPEX_FOOT_BODIES,
    )
    g1_cache_feet_description(
        env, env_ids, asset_cfg=SceneEntityCfg(G1_ASSET),
        root_body=G1_ROOT_BODY, foot_bodies=G1_FOOT_BODIES,
    )


# ---------------------------------------------------------------------------
# ObsTerm functions (one per of the 5 URMA keys)
# ---------------------------------------------------------------------------

def combined_joint_description(env: ManagerBasedRLEnv) -> torch.Tensor:
    kapex = get_joint_description(env, asset_cfg=SceneEntityCfg(KAPEX_ASSET))
    g1 = g1_get_joint_description(env, asset_cfg=SceneEntityCfg(G1_ASSET))
    return torch.cat([kapex, g1], dim=1)  # (E, 44, 16)


def combined_joint_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    kapex = get_joint_observation(env, asset_cfg=SceneEntityCfg(KAPEX_ASSET), action_name="kapex_joint_pos")
    g1 = g1_get_joint_observation(env, asset_cfg=SceneEntityCfg(G1_ASSET), action_name="g1_joint_pos")
    return torch.cat([kapex, g1], dim=1)  # (E, 44, 3)


def combined_feet_description(env: ManagerBasedRLEnv) -> torch.Tensor:
    kapex = get_feet_description(env, asset_cfg=SceneEntityCfg(KAPEX_ASSET))
    g1 = g1_get_feet_description(env, asset_cfg=SceneEntityCfg(G1_ASSET))
    return torch.cat([kapex, g1], dim=1)  # (E, 4, 3)


def combined_feet_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    kapex = get_feet_observation(
        env, asset_cfg=SceneEntityCfg(KAPEX_ASSET),
        root_body=KAPEX_ROOT_BODY, foot_bodies=KAPEX_FOOT_BODIES,
    )
    g1 = g1_get_feet_observation(
        env, asset_cfg=SceneEntityCfg(G1_ASSET),
        root_body=G1_ROOT_BODY, foot_bodies=G1_FOOT_BODIES,
    )
    return torch.cat([kapex, g1], dim=1)  # (E, 4, 2)


def combined_global_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    kapex = get_global_observation(env, asset_cfg=SceneEntityCfg(KAPEX_ASSET))
    g1 = g1_get_global_observation(
        env, asset_cfg=SceneEntityCfg(G1_ASSET), torso_body=G1_TORSO_BODY,
    )
    return torch.cat([kapex, g1], dim=-1)  # (E, 30)
