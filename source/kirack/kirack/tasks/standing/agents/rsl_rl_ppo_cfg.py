# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .attention_model import RslRlAttentionActorModelCfg, RslRlAttentionCriticModelCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 999999
    save_interval = 100
    experiment_name = "kirack"

    # New (>=4.0.0) model API. The deprecation handler reads `policy` only when
    # `actor`/`critic` are MISSING, so leaving `policy` at its parent MISSING
    # default is fine.
    actor = RslRlAttentionActorModelCfg()
    critic = RslRlAttentionCriticModelCfg()

    # Map env observation groups to algorithm-side observation sets.
    # The env exposes five flat top-level groups; the attention model reads each
    # by name. Actor and critic share the same view here.
    obs_groups = {
        "actor": ["joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs"],
        "critic": ["joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs"],
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
