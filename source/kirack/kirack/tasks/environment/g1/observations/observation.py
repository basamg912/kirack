"""Per-joint / per-foot / global observation builders for the G1 23DOF robot.

Mirrors the Kapex obs structure in
``tasks/environment/kapex/observations/observation.py`` so the same URMA-style
attention policy can ingest G1's morphology. Differences from Kapex:

- URDF path → ``g1_23dof.urdf`` (23 revolute joints).
- Root body → ``pelvis`` (URDF base).
- Foot bodies → ``("left_ankle_roll_link", "right_ankle_roll_link")``.
- Cache attributes are namespaced (``_g1_*``) so both robots can coexist on the
  same ``env`` instance for the future combined-spawn config.

The output schemas (``joint_desc`` 16-dim, ``joint_obs`` 3-dim, ``feet_desc``
3-dim, ``feet_obs`` 2-dim, ``global_obs`` 15-dim) match Kapex's so the same
``AttentionActorModel`` / ``AttentionCriticModel`` can be reused.
"""

import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from kirack import kirack_EXT_DIR

_URDF_PATH = os.path.join(
    kirack_EXT_DIR,
    "tasks",
    "environment",
    "g1",
    "robots",
    "g1_description",
    "g1_23dof.urdf",
)
_DEFAULT_ROOT_BODY = "pelvis"
_DEFAULT_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_DEFAULT_TORSO_BODY = "torso_link"


# ---------------------------------------------------------------------------
# URDF parsing (per-URDF lru_cache; keyed by path)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _parse_urdf(urdf_path: str = _URDF_PATH) -> dict[str, dict]:
    """Return ``joint_name -> {axis, parent, child, nr_child_joints}`` from URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    info: dict[str, dict] = {}
    link_to_children: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        name = j.get("name")
        axis_el = j.find("axis")
        axis = (
            tuple(float(x) for x in axis_el.get("xyz").split())
            if axis_el is not None
            else (1.0, 0.0, 0.0)
        )
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        info[name] = {"axis": axis, "parent": parent, "child": child}
        link_to_children.setdefault(parent, []).append(child)
    for d in info.values():
        d["nr_child_joints"] = len(link_to_children.get(d["child"], []))
    return info


# ---------------------------------------------------------------------------
# Description tensors
# ---------------------------------------------------------------------------

JOINT_DESC_FEATURES = (
    "rel_pos_x", "rel_pos_y", "rel_pos_z",        # joint anchor in root frame (3)
    "axis_x", "axis_y", "axis_z",                  # joint axis in root frame (3)
    "nr_child_joints",                              # tree fanout (1)
    "nominal_pos",                                  # default joint position (1)
    "effort_limit",                                 # (1)
    "velocity_limit",                               # (1)
    "stiffness",                                    # (1)
    "damping",                                      # (1)
    "armature",                                     # (1)
    "friction_coeff",                               # (1)
    "pos_lower", "pos_upper",                       # joint pos limits (2)
)
JOINT_DESC_SIZE = len(JOINT_DESC_FEATURES)  # 16


def _compute_joint_description(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body: str = _DEFAULT_ROOT_BODY,
) -> torch.Tensor:
    """Build the static per-joint description tensor for G1.

    Returns ``(num_envs, n_joints, JOINT_DESC_SIZE)`` in raw physical units.
    Must be invoked at the default pose (first call after reset).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    device = data.default_joint_pos.device
    num_envs, n_joints = data.default_joint_pos.shape

    urdf_info = _parse_urdf()
    joint_names = list(data.joint_names)
    body_names = list(data.body_names)

    if root_body not in body_names:
        raise RuntimeError(f"Root body '{root_body}' not in body_names: {body_names}")
    root_idx = body_names.index(root_body)

    missing = [n for n in joint_names if n not in urdf_info]
    if missing:
        raise RuntimeError(
            f"Articulation joints not found in URDF '{_URDF_PATH}': {missing}"
        )

    child_body_idx = torch.empty(n_joints, dtype=torch.long, device=device)
    parent_body_idx = torch.empty(n_joints, dtype=torch.long, device=device)
    axes_local = torch.empty((n_joints, 3), device=device)
    nr_children = torch.empty(n_joints, device=device)
    for i, name in enumerate(joint_names):
        info = urdf_info[name]
        child_body_idx[i] = body_names.index(info["child"])
        parent_body_idx[i] = body_names.index(info["parent"])
        axes_local[i] = torch.tensor(info["axis"], device=device)
        nr_children[i] = info["nr_child_joints"]

    body_pos_w = data.body_pos_w
    body_quat_w = data.body_quat_w

    root_pos_w = body_pos_w[:, root_idx, :]
    root_quat_w = body_quat_w[:, root_idx, :]
    joint_pos_w = body_pos_w[:, child_body_idx, :]
    rel_pos_w = joint_pos_w - root_pos_w.unsqueeze(1)

    root_q_flat = root_quat_w.unsqueeze(1).expand(-1, n_joints, -1).reshape(-1, 4)
    rel_pos_root = quat_apply_inverse(root_q_flat, rel_pos_w.reshape(-1, 3)).view(
        num_envs, n_joints, 3
    )

    parent_quat_w = body_quat_w[:, parent_body_idx, :].reshape(-1, 4)
    axes_local_exp = axes_local.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3)
    axes_world = quat_apply(parent_quat_w, axes_local_exp)
    axes_root = quat_apply_inverse(root_q_flat, axes_world).view(num_envs, n_joints, 3)
    axes_root = axes_root / axes_root.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    nr_children_exp = nr_children.view(1, -1, 1).expand(num_envs, -1, -1)

    desc = torch.cat(
        [
            rel_pos_root,                                       # 3
            axes_root,                                          # 3
            nr_children_exp,                                    # 1
            data.default_joint_pos.unsqueeze(-1),              # 1
            data.joint_effort_limits.unsqueeze(-1),            # 1
            data.joint_vel_limits.unsqueeze(-1),               # 1
            data.default_joint_stiffness.unsqueeze(-1),        # 1
            data.default_joint_damping.unsqueeze(-1),          # 1
            data.default_joint_armature.unsqueeze(-1),         # 1
            data.default_joint_friction_coeff.unsqueeze(-1),   # 1
            data.default_joint_pos_limits,                      # 2
        ],
        dim=-1,
    )
    assert desc.shape == (num_envs, n_joints, JOINT_DESC_SIZE), desc.shape
    return desc


# ---------------------------------------------------------------------------
# Public ObsTerm functions (G1-namespaced)
# ---------------------------------------------------------------------------

_JOINT_DESC_CACHE_ATTR = "_g1_joint_desc"


def g1_cache_joint_description(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body: str = _DEFAULT_ROOT_BODY,
) -> None:
    """EventTerm: (re)compute the G1 joint description and cache it on env."""
    desc = _compute_joint_description(env, asset_cfg, root_body=root_body)
    setattr(env, _JOINT_DESC_CACHE_ATTR, desc)


def g1_get_joint_description(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the cached G1 joint description; populate lazily on first call."""
    desc = getattr(env, _JOINT_DESC_CACHE_ATTR, None)
    if desc is None:
        desc = _compute_joint_description(env, asset_cfg)
        setattr(env, _JOINT_DESC_CACHE_ATTR, desc)
    return desc


def g1_get_joint_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str | None = None,
) -> torch.Tensor:
    """Per-joint dynamic observation: ``(num_envs, n_joints, 3)`` -- [pos_rel, vel_rel, last_action]."""
    robot = env.scene[asset_cfg.name]
    data = robot.data
    pos_rel = data.joint_pos - data.default_joint_pos
    vel_rel = data.joint_vel - data.default_joint_vel
    if action_name is None:
        last_action = env.action_manager.action
    else:
        last_action = env.action_manager.get_term(action_name).raw_actions
    return torch.stack([pos_rel, vel_rel, last_action], dim=-1)


# ---------------------------------------------------------------------------
# Feet description / observation
# ---------------------------------------------------------------------------

FEET_DESC_FEATURES = ("rel_pos_x", "rel_pos_y", "rel_pos_z")
FEET_DESC_SIZE = len(FEET_DESC_FEATURES)  # 3

_FEET_DESC_CACHE_ATTR = "_g1_feet_desc"


def _compute_feet_description(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body: str = _DEFAULT_ROOT_BODY,
    foot_bodies: tuple[str, ...] = _DEFAULT_FOOT_BODIES,
) -> torch.Tensor:
    """Per-foot static description: ``(num_envs, n_feet, 3)`` -- foot position in root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    device = data.body_pos_w.device
    num_envs = data.body_pos_w.shape[0]
    body_names = list(data.body_names)

    if root_body not in body_names:
        raise RuntimeError(f"Root body '{root_body}' not in body_names: {body_names}")
    missing = [b for b in foot_bodies if b not in body_names]
    if missing:
        raise RuntimeError(f"Foot bodies {missing} not in body_names: {body_names}")

    root_idx = body_names.index(root_body)
    foot_idx = torch.tensor(
        [body_names.index(b) for b in foot_bodies], dtype=torch.long, device=device
    )
    n_feet = len(foot_bodies)

    body_pos_w = data.body_pos_w
    body_quat_w = data.body_quat_w
    root_pos_w = body_pos_w[:, root_idx, :]
    root_quat_w = body_quat_w[:, root_idx, :]
    foot_pos_w = body_pos_w[:, foot_idx, :]
    rel_pos_w = foot_pos_w - root_pos_w.unsqueeze(1)

    root_q_flat = root_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_pos_root = quat_apply_inverse(root_q_flat, rel_pos_w.reshape(-1, 3)).view(
        num_envs, n_feet, 3
    )
    assert rel_pos_root.shape == (num_envs, n_feet, FEET_DESC_SIZE), rel_pos_root.shape
    return rel_pos_root


def g1_cache_feet_description(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body: str = _DEFAULT_ROOT_BODY,
    foot_bodies: tuple[str, ...] = _DEFAULT_FOOT_BODIES,
) -> None:
    """EventTerm: (re)compute the G1 feet description and cache it on env."""
    desc = _compute_feet_description(env, asset_cfg, root_body=root_body, foot_bodies=foot_bodies)
    setattr(env, _FEET_DESC_CACHE_ATTR, desc)


def g1_get_feet_description(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the cached G1 feet description; populate lazily on first call."""
    desc = getattr(env, _FEET_DESC_CACHE_ATTR, None)
    if desc is None:
        desc = _compute_feet_description(env, asset_cfg)
        setattr(env, _FEET_DESC_CACHE_ATTR, desc)
    return desc


FEET_OBS_FEATURES = ("foot_z_in_root", "foot_vz_in_root")
FEET_OBS_SIZE = len(FEET_OBS_FEATURES)  # 2


def g1_get_feet_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body: str = _DEFAULT_ROOT_BODY,
    foot_bodies: tuple[str, ...] = _DEFAULT_FOOT_BODIES,
) -> torch.Tensor:
    """Per-foot dynamic observation: ``(num_envs, n_feet, 2)`` -- [foot_z_in_root, foot_vz_in_root]."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    device = data.body_pos_w.device
    num_envs = data.body_pos_w.shape[0]
    body_names = list(data.body_names)

    if root_body not in body_names:
        raise RuntimeError(f"Root body '{root_body}' not in body_names: {body_names}")
    missing = [b for b in foot_bodies if b not in body_names]
    if missing:
        raise RuntimeError(f"Foot bodies {missing} not in body_names: {body_names}")

    root_idx = body_names.index(root_body)
    foot_idx = torch.tensor(
        [body_names.index(b) for b in foot_bodies], dtype=torch.long, device=device
    )
    n_feet = len(foot_bodies)

    root_pos_w = data.body_pos_w[:, root_idx, :]
    root_quat_w = data.body_quat_w[:, root_idx, :]
    root_lin_vel_w = data.body_lin_vel_w[:, root_idx, :]
    root_ang_vel_w = data.body_ang_vel_w[:, root_idx, :]

    foot_pos_w = data.body_pos_w[:, foot_idx, :]
    foot_lin_vel_w = data.body_lin_vel_w[:, foot_idx, :]

    rel_pos_w = foot_pos_w - root_pos_w.unsqueeze(1)
    rel_vel_w = (
        foot_lin_vel_w
        - root_lin_vel_w.unsqueeze(1)
        - torch.cross(root_ang_vel_w.unsqueeze(1).expand(-1, n_feet, -1), rel_pos_w, dim=-1)
    )

    root_q_flat = root_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_pos_root = quat_apply_inverse(root_q_flat, rel_pos_w.reshape(-1, 3)).view(
        num_envs, n_feet, 3
    )
    rel_vel_root = quat_apply_inverse(root_q_flat, rel_vel_w.reshape(-1, 3)).view(
        num_envs, n_feet, 3
    )

    out = torch.stack([rel_pos_root[..., 2], rel_vel_root[..., 2]], dim=-1)
    assert out.shape == (num_envs, n_feet, FEET_OBS_SIZE), out.shape
    return out


# ---------------------------------------------------------------------------
# Global observation (robot-level context)
# ---------------------------------------------------------------------------

GLOBAL_OBS_FEATURES = (
    "base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z",       # 3
    "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",       # 3
    "proj_gravity_x", "proj_gravity_y", "proj_gravity_z",       # 3
    "vel_cmd_x", "vel_cmd_y", "vel_cmd_yaw",                    # 3
    "dif_torso_com_x", "dif_torso_com_y", "dif_torso_com_z",    # 3
)
GLOBAL_OBS_SIZE = len(GLOBAL_OBS_FEATURES)  # 15

# G1 torso CoM offset baseline. Without a measured value, zeros let the policy
# observe the absolute torso CoM and learn from there; empirical normalization
# (enabled in the URMA actor/critic) absorbs any constant bias.
_DEFAULT_TORSO_COM = (0.0, 0.0, 0.0)


def g1_get_global_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    torso_body: str = _DEFAULT_TORSO_BODY,
    default_com: tuple[float, float, float] = _DEFAULT_TORSO_COM,
) -> torch.Tensor:
    """Robot-level context vector: ``(num_envs, 15)`` -- same schema as Kapex."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data

    base_lin_vel = data.root_lin_vel_b
    base_ang_vel = data.root_ang_vel_b
    proj_gravity = data.projected_gravity_b
    vel_cmd = env.command_manager.get_command(command_name)

    body_names = list(data.body_names)
    if torso_body not in body_names:
        raise RuntimeError(f"Torso body '{torso_body}' not in body_names: {body_names}")
    torso_idx = body_names.index(torso_body)
    coms = asset.root_physx_view.get_coms()
    default_com_t = torch.as_tensor(default_com, device=env.device, dtype=coms.dtype)
    dif_com = coms[:, torso_idx, :3].to(env.device) - default_com_t

    out = torch.cat([base_lin_vel, base_ang_vel, proj_gravity, vel_cmd, dif_com], dim=-1)
    assert out.shape[-1] == GLOBAL_OBS_SIZE, out.shape
    return out
