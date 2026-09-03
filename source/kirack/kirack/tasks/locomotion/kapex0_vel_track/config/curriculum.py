from __future__ import annotations
from isaaclab.envs import ManagerBasedRLEnv
import torch
from typing import Sequence
from isaaclab.managers import SceneEntityCfg
def torso_com_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "torso_com",
    termination_term_name: str = "time_out",
    threshold: float = 0.7,
    delta: float = 0.01,
    limit_range: dict = {
        "x": (-0.3, 0.3), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
    },
) -> torch.Tensor:
    event_term = env.event_manager.get_term_cfg(event_term_name)
    current_range = event_term.params["com_range"]  # dict with x, y, z

    # termination ratio 계산 (최근 episode 기준)
    term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
    termination_ratio = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()

    # 에피소드 경계에서만 업데이트
    if env.common_step_counter % env.max_episode_length == 0:
        # termination_ratio 가 threshold 이하면 난이도 상승 (range 확장)
        if termination_ratio > threshold:
            for key in current_range:
                lo, hi = current_range[key]
                lim_lo, lim_hi = limit_range[key]
                current_range[key] = (
                    max(lo - delta, lim_lo),
                    min(hi + delta, lim_hi),
                )

    return torch.tensor(current_range["x"][1], device=env.device)



def perturb_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "push_robot",
    termination_term_name: str = "time_out",
    survival_high = 0.6,
    survival_low = 0.3,
    orient_threshold = 0.10,
    height_threshold = 0.02,
    delta: float = 0.05,
    limit_range: dict = {
        "x": (-1.0, 1.0), "y": (-1.0, 1.0),
    },
) -> torch.Tensor:
    event_term = env.event_manager.get_term_cfg(event_term_name)
    current_range = event_term.params["velocity_range"]

    term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
    survival = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()

    rm = env.reward_manager
    ep_len = env.episode_length_buf.float()

    def _avg(name, default=0.0):
        if name not in rm._episode_sums:
            return default
        w = getattr(rm.cfg, name).weight
        if w == 0.0:
            return default
        return (rm._episode_sums[name] / ep_len).mean().item()

    orient_score = abs(_avg("flat_orientation_l2"))
    height_score = abs(_avg("height_l2"))

    if env.common_step_counter % env.max_episode_length == 0:
        well_recovered = (
            survival > survival_high
            and orient_score < orient_threshold
            and height_score < height_threshold
        )
        too_hard = survival < survival_low

        if well_recovered:
            for key in current_range:
                lo, hi = current_range[key]
                lim_lo, lim_hi = limit_range[key]
                current_range[key] = (
                    max(lo - delta, lim_lo),
                    min(hi + delta, lim_hi),
                )
        elif too_hard:
            for key in current_range:
                lo, hi = current_range[key]
                lim_lo, lim_hi = limit_range[key]
                current_range[key] = (
                    min(lo + delta, 0.0),
                    max(hi - delta, 0.0)
                )

    return torch.tensor(current_range["x"][1], device=env.device)

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    termination_term_name: str = "time_out",
    threshold: float = 0.7,
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
    termination_ratio = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()

    if env.common_step_counter % env.max_episode_length == 0:
        if termination_ratio > threshold:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

# ------------------- Randomize curriculum
def actuator_armature_range_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "scale_all_joint_armature",
    termination_term_name: str = "time_out",
    threshold: float = 0.7,
    delta: float = 0.02,
    limit_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    event_term = env.event_manager.get_term_cfg(event_term_name)
    params = event_term.params

    term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
    termination_ratio = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()

    if env.common_step_counter % env.max_episode_length == 0:
        if termination_ratio > threshold:
            armature = params["armature_distribution_params"]

            params["armature_distribution_params"] = (
                max(armature[0] - delta, limit_range[0]),
                min(armature[1] + delta, limit_range[1]),
            )

    return torch.tensor(params["armature_distribution_params"][0], device=env.device)

def actuator_gain_range_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "joint_stiffness_and_damping",
    termination_term_name: str = "time_out",
    threshold: float = 0.7,
    delta: float = 0.02,
    limit_range: tuple[float, float] = (0.7, 1.3),
) -> torch.Tensor:
    event_term = env.event_manager.get_term_cfg(event_term_name)
    params = event_term.params

    term_idx = env.termination_manager._term_name_to_term_idx[termination_term_name]
    termination_ratio = env.termination_manager._last_episode_dones[:, term_idx].float().mean().item()

    if env.common_step_counter % env.max_episode_length == 0:
        if termination_ratio > threshold:
            stiff = params["stiffness_distribution_params"]
            damp = params["damping_distribution_params"]

            params["stiffness_distribution_params"] = (
                max(stiff[0] - delta, limit_range[0]),
                min(stiff[1] + delta, limit_range[1]),
            )
            params["damping_distribution_params"] = (
                max(damp[0] - delta, limit_range[0]),
                min(damp[1] + delta, limit_range[1]),
            )

    return torch.tensor(params["stiffness_distribution_params"][0], device=env.device)


def terrain_levels_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_ratio: float = 0.7,
    move_down_ratio: float = 0.3,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ep_len = env.episode_length_buf[env_ids].float()
    terrain: TerrainImporter = env.scene.terrain
    max_len = float(env.max_episode_length)
    move_up = ep_len > max_len * move_up_ratio
    move_down = ep_len < max_len * move_down_ratio
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
