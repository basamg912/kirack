"""Gaussian policy distribution with a SINGLE shared standard deviation.

The stock ``rsl_rl.modules.distribution.GaussianDistribution`` allocates a
``std_param`` of shape ``(output_dim,)`` -- i.e. one learnable std per action
dimension. For multi-morphology distributed training (KAPEX 21-joint actions,
G1 23-joint actions), this shape differs across ranks and NCCL ``all_reduce``
on the parameter list fails.

This variant keeps the std as a single learnable scalar shared across all
action dimensions, so every rank holds an identically-shaped ``std_param`` and
gradient synchronization works. Exploration becomes uniform across joints,
which is fine for our setup (per-joint actions are already on a comparable
scale via ``JointPositionActionCfg(scale=0.25)``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.distribution import Distribution, GaussianDistribution


class SharedStdGaussianDistribution(GaussianDistribution):
    """Gaussian distribution with a single learnable std broadcast across dims."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        # Skip GaussianDistribution.__init__ so we don't allocate the per-dim std_param.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type

        # Single shared scalar -- shape (1,) instead of (output_dim,).
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(1), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(
                torch.log(init_std * torch.ones(1)), requires_grad=learn_std
            )
        else:
            raise ValueError(f"Unknown std_type: {std_type}")

        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # update() / sample() / log_prob() inherit from GaussianDistribution.
    # Broadcasting handles the (1,)-vs-(B, J) shape difference automatically.
