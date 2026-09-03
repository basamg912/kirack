"""Load experiment YAML and apply overrides to env config."""

import os
import yaml
from kirack.tasks.locomotion.kapex0_vel_track.kapex0_vel_track_env_cfg import Kapex0VelTrackEnvCfg
_EXP_CFG = None


def get_exp_cfg() -> dict | None:
    """Load experiment YAML from the path set by set_exp_path()."""
    global _EXP_CFG
    if _EXP_CFG is not None:
        return _EXP_CFG

    exp_path = os.environ.get("KIRACK_EXP_CFG")
    if exp_path is None:
        return None

    with open(exp_path, "r") as f:
        _EXP_CFG = yaml.safe_load(f)
    return _EXP_CFG

def apply_agent_exp_cfg(agent_cfg, exp: dict):
    """Apply experiment overrides to the RSL-RL agent (PPO) config.

    YAML structure:
        agent:
          max_iterations: 30000          # top-level RunnerCfg attrs
          policy:
            init_noise_std: 0.8          # PolicyCfg attrs
          algorithm:
            learning_rate: 5.0e-4        # AlgorithmCfg attrs
            schedule: fixed
            entropy_coef: 0.01
    """
    if "agent" not in exp or exp["agent"] is None:
        return

    for key, value in exp["agent"].items():
        if not hasattr(agent_cfg, key):
            continue
        target = getattr(agent_cfg, key)
        if isinstance(value, dict) and target is not None and not isinstance(target, (int, float, str, bool, list, tuple)):
            for sub_key, sub_value in value.items():
                if not hasattr(target, sub_key):
                    continue
                sub_target = getattr(target, sub_key)
                if isinstance(sub_value, dict) and not isinstance(sub_target, dict):
                    print(f"[exp_loader] WARNING: skipping agent.{key}.{sub_key} — got dict for non-dict field (did you leave a placeholder?)")
                    continue
                setattr(target, sub_key, sub_value)
        else:
            if isinstance(value, dict) and not isinstance(target, dict):
                print(f"[exp_loader] WARNING: skipping agent.{key} — got dict for non-dict field")
                continue
            setattr(agent_cfg, key, value)


# ! kirack_env.py 에서 self, exp 식으로 전달
def apply_exp_cfg(env_cfg:Kapex0VelTrackEnvCfg, exp: dict):
    """Apply experiment overrides to the env config.

    Supports:
        - vel_command / vel_limit: velocity command ranges
        - rewards: {term_name: null} to disable, {term_name: {weight: N, params: {...}}} to override
        - events: {term_name: null} to disable
        - terminations: {term_name: null} to disable, {term_name: {params: {...}}} to override
        - curriculum: {term_name: null} to disable
        - terrain: "flat" or "rough"
    """
    from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils

    # --- velocity command ---
    if "vel_command" in exp:
        vc = exp["vel_command"]
        env_cfg.commands.base_velocity.ranges = UniformVelocityCommandCfg.Ranges(
            lin_vel_x=tuple(vc.get("lin_vel_x", [0, 0])),
            lin_vel_y=tuple(vc.get("lin_vel_y", [0, 0])),
            ang_vel_z=tuple(vc.get("ang_vel_z", [0, 0])),
        )

    if "vel_limit" in exp:
        vl = exp["vel_limit"]
        env_cfg.commands.base_velocity.limit_ranges = UniformVelocityCommandCfg.Ranges(
            lin_vel_x=tuple(vl.get("lin_vel_x", [0, 0])),
            lin_vel_y=tuple(vl.get("lin_vel_y", [0, 0])),
            ang_vel_z=tuple(vl.get("ang_vel_z", [0, 0])),
        )

    # --- rewards ---
    if "rewards" in exp and exp["rewards"] is not None:
        for term_name, override in exp["rewards"].items():
            # ! yaml 에서 null 설정할 경우 None
            # ! {weight: -10} dict 통째가 override
            if override is None:
                setattr(env_cfg.rewards, term_name, None)
            else:
                term = getattr(env_cfg.rewards, term_name, None)
                if term is None:
                    continue
                if "weight" in override:
                    term.weight = override["weight"]
                if "params" in override:
                    term.params.update(override["params"])

    # --- observations ---
    # YAML structure:
    #   observations:
    #     policy:                       # or "critic"
    #       base_lin_vel: null          # disable obs term
    #       joint_vel_rel:
    #         scale: 0.1
    #         clip: [-1.0, 1.0]
    #         params: {...}
    #       enable_corruption: false    # group-level attr
    if "observations" in exp and exp["observations"] is not None:
        _GROUP_ATTRS = {"history_length", "enable_corruption", "concatenate_terms"}
        _ALIASES = {"actor": "policy"}

        for group_name, group_override in exp["observations"].items():
            if group_override is None:
                continue
            group_name = _ALIASES.get(group_name, group_name)
            group = getattr(env_cfg.observations, group_name, None)
            if group is None:
                print(f"[exp_loader] WARNING: unknown observation group '{group_name}' "
                      f"(expected 'policy' or 'critic')")
                continue

            for term_name, term_override in group_override.items():
                # group-level attribute (history_length, enable_corruption, ...)
                if term_name in _GROUP_ATTRS:
                    setattr(group, term_name, term_override)
                    continue
                # disable obs term
                if term_override is None:
                    setattr(group, term_name, None)
                    continue
                # modify obs term fields
                if isinstance(term_override, dict):
                    term = getattr(group, term_name, None)
                    if term is None:
                        print(f"[exp_loader] WARNING: observations.{group_name}.{term_name} not found")
                        continue
                    if "scale" in term_override:
                        term.scale = term_override["scale"]
                    if "clip" in term_override:
                        term.clip = tuple(term_override["clip"])
                    if "params" in term_override:
                        if term.params is None:
                            term.params = {}
                        term.params.update(term_override["params"])

    # --- events ---
    if "events" in exp and exp["events"] is not None:
        for term_name, override in exp["events"].items():
            if override is None:
                setattr(env_cfg.events, term_name, None)
            elif isinstance(override, dict):
                term = getattr(env_cfg.events, term_name, None)
                if term is None:
                    continue
                if "mode" in override:
                    term.mode = override["mode"]
                if "params" in override:
                    term.params.update(override["params"])

    # --- terminations ---
    if "terminations" in exp and exp["terminations"] is not None:
        for term_name, override in exp["terminations"].items():
            if override is None:
                setattr(env_cfg.terminations, term_name, None)
            elif isinstance(override, dict):
                term = getattr(env_cfg.terminations, term_name, None)
                if term is None:
                    continue
                if "params" in override:
                    term.params.update(override["params"])

    # --- curriculum ---
    if "curriculum" in exp and exp["curriculum"] is not None:
        for term_name, override in exp["curriculum"].items():
            if override is None:
                setattr(env_cfg.curriculum, term_name, None)
            elif isinstance(override, dict):
                term = getattr(env_cfg.curriculum, term_name, None)
                if term is None:
                    continue
                if "params" in override:
                    term.params.update(override["params"])

    # --- terrain ---
    if "terrain" in exp and exp["terrain"] is not None:
        terrain_cfg = exp["terrain"]
        terrain_type = terrain_cfg.get("type", "rough")

        if (terrain_type == 'flat'):
            env_cfg.scene.terrain = AssetBaseCfg(
                prim_path="/World/ground",
                collision_group=-1,
                spawn=sim_utils.GroundPlaneCfg(),
            )
            if hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None

        elif (terrain_type == "rough"):
            rough_ratio = terrain_cfg.get("rough_ratio", 0.5)
            if rough_ratio is not None:
                sub_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
                sub_terrains["rough"].proportion = rough_ratio
                if "flat" in sub_terrains:
                    sub_terrains["flat"].proportion = 1.0 - rough_ratio

    # --- scene ---
    if "scene" in exp and exp["scene"] is not None:
        scene_cfg = exp["scene"]
        if "env_spacing" in scene_cfg:
            env_cfg.scene.env_spacing = scene_cfg["env_spacing"]
