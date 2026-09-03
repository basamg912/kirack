"""Attention-pooling Actor / Critic for the KAPEX standing task.

Mirrors one_policy_to_run_them_all/policy.py & critic.py in PyTorch and adapts
the architecture to rsl_rl 5.3.0's :class:`MLPModel`-style interface.

The forward pass expects observations as a TensorDict with one nested group
(default ``"policy"``) containing five term-level tensors produced by the
helpers in ``kirack/tasks/environment/kapex/observations/observation.py``:

    joint_desc:   (E, J, D_jd)  -- static morphology per joint
    joint_obs:    (E, J, D_js)  -- dynamic state per joint
    feet_desc:    (E, F, D_fd)  -- static morphology per foot
    feet_obs:     (E, F, D_fs)  -- dynamic state per foot
    global_obs:   (E, D_g)      -- robot-level context

The actor outputs ``(E, J)`` action means (one per joint); the critic outputs
``(E, 1)`` state values.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

# ---------------------------------------------------------------------------
# Shared attention-pooling trunk
# ---------------------------------------------------------------------------


# Trunk = Backbone
class AttentionTrunk(nn.Module):
    """Joint + foot attention pooling encoder, producing a fixed-size latent.

    The "attention" is not transformer self-attention: it's a learned per-element
    softmax over the *feature* axis, used to gate each element's latent state
    before summing across elements. Architecture matches the JAX original.
    """

    def __init__(
        self,
        # input : 5개
        joint_desc_dim: int,
        joint_state_dim: int,
        foot_desc_dim: int,
        foot_state_dim: int,
        global_dim: int,
        joint_mask_dim: int = 64,  # mask : 64개 슬롯에 대한 가중치 의미 -> softmax
        foot_mask_dim: int = 32,
        state_latent_dim: int = 32,
        trunk_hidden: tuple[int, ...] = (512, 256, 128),
        init_temperature: float = 1.0,
        min_temperature: float = 0.015,
        stability_eps: float = 1e-7,
    ) -> None:
        super().__init__()
        assert len(trunk_hidden) == 3, "Trunk uses a 3-layer MLP (512, 256, 128 by default)."
        self.joint_mask_dim = joint_mask_dim
        self.foot_mask_dim = foot_mask_dim
        self.state_latent_dim = state_latent_dim
        self.min_temperature = min_temperature
        self.stability_eps = stability_eps

        # Per-element branches: description -> mask logits; state -> latent.
        self.joint_mask_mlp_1 = nn.Linear(joint_desc_dim, 64)
        self.joint_mask_ln = nn.LayerNorm(64)
        self.joint_mask_mlp_2 = nn.Linear(64, joint_mask_dim)
        self.joint_state_enc = nn.Linear(joint_state_dim, state_latent_dim)

        self.foot_mask_mlp_1 = nn.Linear(foot_desc_dim, 32)
        self.foot_mask_ln = nn.LayerNorm(32)
        self.foot_mask_mlp_2 = nn.Linear(32, foot_mask_dim)
        self.foot_state_enc = nn.Linear(foot_state_dim, state_latent_dim)

        # Learnable softmax temperatures (stored as log(T - T_min) so positivity is automatic).
        init_log = math.log(max(init_temperature - min_temperature, 1e-6))
        self.joint_log_temp = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))
        self.foot_log_temp = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))

        # Trunk MLP: concat(joint_latent, foot_latent, global) -> trunk_hidden -> output latent.
        trunk_in = joint_mask_dim * state_latent_dim + foot_mask_dim * state_latent_dim + global_dim
        h1, h2, h3 = trunk_hidden
        self.trunk_fc1 = nn.Linear(trunk_in, h1)
        self.trunk_ln1 = nn.LayerNorm(h1)
        self.trunk_fc2 = nn.Linear(h1, h2)
        self.trunk_fc3 = nn.Linear(h2, h3)
        self.trunk_out_dim = h3

        # Orthogonal init for the trunk -- matches the JAX original (sqrt(2) gain).
        for layer in (self.trunk_fc1, self.trunk_fc2, self.trunk_fc3):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)

    def _attention_pool(
        self,
        mlp_1: nn.Linear,
        ln: nn.LayerNorm,
        mlp_2: nn.Linear,
        state_enc: nn.Linear,
        log_temp: torch.Tensor,
        desc: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one attention-pooled latent from per-element (description, state)."""
        # Mask logits per element: (B, N, M)
        m = mlp_1(desc)
        m = ln(m)
        m = F.elu(m)
        m = mlp_2(m)
        m = torch.tanh(m).clamp(-1.0 + self.stability_eps, 1.0 - self.stability_eps)
        # Per-element state latent: (B, N, S)
        state_latent = F.elu(state_enc(state))
        # Softmax over the feature axis (not the joint axis -- matches the original).
        temperature = torch.exp(log_temp) + self.min_temperature
        e_x = torch.exp(m / temperature)
        # softmax 적용
        weights = e_x / (e_x.sum(dim=-1, keepdim=True) + self.stability_eps)  # (B, N, M)
        # Outer product per element: (B, N, M, 1) * (B, N, 1, S) -> (B, N, M, S)
        gated = weights.unsqueeze(-1) * state_latent.unsqueeze(-2)
        # Flatten (M, S) and aggregate across elements: (B, M*S)

        # * 는 tuple 풀어서 인자로 넣어주는것 **kwargs dict 를 풀어주는것과 같은 이치
        gated = gated.reshape(*gated.shape[:-2], -1)  # [B, Joint, 64 * 32] 로 reshape
        pooled = gated.sum(dim=-2)  # Joint 축으로 summation -> 각 joint 들의 기여도를 합치기
        return pooled, state_latent

    def forward(
        self,
        joint_desc: torch.Tensor,
        joint_state: torch.Tensor,
        foot_desc: torch.Tensor,
        foot_state: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_latent, joint_state_latent = self._attention_pool(
            self.joint_mask_mlp_1,
            self.joint_mask_ln,
            self.joint_mask_mlp_2,
            self.joint_state_enc,
            self.joint_log_temp,
            joint_desc,
            joint_state,
        )
        foot_latent, _ = self._attention_pool(
            self.foot_mask_mlp_1,
            self.foot_mask_ln,
            self.foot_mask_mlp_2,
            self.foot_state_enc,
            self.foot_log_temp,
            foot_desc,
            foot_state,
        )
        x = torch.cat([joint_latent, foot_latent, global_state], dim=-1)
        x = F.elu(self.trunk_ln1(self.trunk_fc1(x)))
        x = F.elu(self.trunk_fc2(x))
        x = self.trunk_fc3(x)  # No activation here -- matches the actor trunk in the original.
        return x, joint_state_latent


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


# Decoder part
class ActionMeanHead(nn.Module):
    """Per-joint action-mean head.

    Takes the trunk latent (E, T), per-joint description (E, J, D_jd), and the
    stop-gradient joint-state latent (E, J, S); outputs (E, J) means.
    """

    def __init__(
        self,
        joint_desc_dim: int,
        trunk_dim: int,
        state_latent_dim: int,
        desc_hidden: int = 128,
        mean_hidden: int = 128,
        mean_abs_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.mean_abs_clip = mean_abs_clip

        # Per-joint description encoder (separate from the trunk's mask MLP).
        self.desc_fc1 = nn.Linear(joint_desc_dim, desc_hidden)
        self.desc_ln = nn.LayerNorm(desc_hidden)
        self.desc_fc2 = nn.Linear(desc_hidden, desc_hidden)

        # Mean MLP: [trunk_tile, sg(state_latent), desc_encoded] -> mean per joint.
        head_in = trunk_dim + state_latent_dim + desc_hidden
        self.mean_fc1 = nn.Linear(head_in, mean_hidden)
        self.mean_ln = nn.LayerNorm(mean_hidden)
        self.mean_fc2 = nn.Linear(mean_hidden, 1)

        nn.init.orthogonal_(self.mean_fc1.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.mean_fc1.bias)
        nn.init.orthogonal_(self.mean_fc2.weight, gain=0.01)  # tiny final-layer gain
        nn.init.zeros_(self.mean_fc2.bias)

    def description_latent(self, joint_desc: torch.Tensor) -> torch.Tensor:
        d = F.elu(self.desc_ln(self.desc_fc1(joint_desc)))
        d = self.desc_fc2(d)
        return d

    def forward(
        self,
        trunk_latent: torch.Tensor,
        joint_desc: torch.Tensor,
        joint_state_latent: torch.Tensor,
    ) -> torch.Tensor:
        n_joints = joint_desc.shape[-2]
        desc = self.description_latent(joint_desc)  # (B, J, D)
        trunk_tile = trunk_latent.unsqueeze(-2).expand(-1, n_joints, -1)  # (B, J, T)
        state_sg = joint_state_latent.detach()  # (B, J, S)
        x = torch.cat([trunk_tile, state_sg, desc], dim=-1)  # (B, J, T+S+D)
        m = F.elu(self.mean_ln(self.mean_fc1(x)))
        m = self.mean_fc2(m).squeeze(-1)  # (B, J)
        return m.clamp(-self.mean_abs_clip, self.mean_abs_clip)


class ValueHead(nn.Module):
    """Scalar value head: trunk latent -> ELU -> Dense(1)."""

    def __init__(self, trunk_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(trunk_dim, 1)
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

    def forward(self, trunk_latent: torch.Tensor) -> torch.Tensor:
        return self.fc(F.elu(trunk_latent))  # (B, 1)


# ---------------------------------------------------------------------------
# Model interface (matches rsl_rl MLPModel)
# ---------------------------------------------------------------------------


_EXPECTED_OBS_KEYS = ("joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs")


def _read_dims(obs: TensorDict) -> dict[str, int]:
    """Pull tensor dimensions from a *flat* top-level observation TensorDict.

    Each per-element term must be a 3D tensor ``(E, N, D)``; global is 2D.
    """
    return {
        "joint_desc_dim": obs["joint_desc"].shape[-1],
        "n_joints": obs["joint_desc"].shape[-2],
        "joint_state_dim": obs["joint_obs"].shape[-1],
        "foot_desc_dim": obs["feet_desc"].shape[-1],
        "n_feet": obs["feet_desc"].shape[-2],
        "foot_state_dim": obs["feet_obs"].shape[-1],
        "global_dim": obs["global_obs"].shape[-1],
    }


class _AttentionModelBase(nn.Module):
    """Common scaffolding for the attention actor / critic models."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        *,
        joint_mask_dim: int = 64,
        foot_mask_dim: int = 32,
        state_latent_dim: int = 32,
        trunk_hidden: tuple[int, ...] = (512, 256, 128),
        init_temperature: float = 1.0,
        min_temperature: float = 0.015,
        stability_eps: float = 1e-7,
        obs_normalization: bool = True,
    ) -> None:
        super().__init__()
        groups = list(obs_groups[obs_set])
        missing = [k for k in _EXPECTED_OBS_KEYS if k not in groups]
        if missing:
            raise ValueError(
                f"AttentionModel needs obs groups {_EXPECTED_OBS_KEYS} for set '{obs_set}'; "
                f"missing: {missing}. Got: {groups}."
            )
        dims = _read_dims(obs)

        self.trunk = AttentionTrunk(
            joint_desc_dim=dims["joint_desc_dim"],
            joint_state_dim=dims["joint_state_dim"],
            foot_desc_dim=dims["foot_desc_dim"],
            foot_state_dim=dims["foot_state_dim"],
            global_dim=dims["global_dim"],
            joint_mask_dim=joint_mask_dim,
            foot_mask_dim=foot_mask_dim,
            state_latent_dim=state_latent_dim,
            trunk_hidden=trunk_hidden,
            init_temperature=init_temperature,
            min_temperature=min_temperature,
            stability_eps=stability_eps,
        )
        self._dims = dims

        # Per-group empirical normalization. Description tensors (joint_desc in
        # particular) mix features with effort/stiffness/damping/armature scales
        # that span 5+ orders of magnitude; without normalization the attention
        # mask MLP receives inputs dominated by the largest-scale columns.
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.joint_desc_normalizer = EmpiricalNormalization(dims["joint_desc_dim"])
            self.joint_obs_normalizer = EmpiricalNormalization(dims["joint_state_dim"])
            self.feet_desc_normalizer = EmpiricalNormalization(dims["foot_desc_dim"])
            self.feet_obs_normalizer = EmpiricalNormalization(dims["foot_state_dim"])
            self.global_obs_normalizer = EmpiricalNormalization(dims["global_dim"])
        else:
            self.joint_desc_normalizer = nn.Identity()
            self.joint_obs_normalizer = nn.Identity()
            self.feet_desc_normalizer = nn.Identity()
            self.feet_obs_normalizer = nn.Identity()
            self.global_obs_normalizer = nn.Identity()

    @staticmethod
    def _apply_norm_per_element(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.Identity) or x.ndim == 2:
            return norm(x)
        flat = x.reshape(-1, x.shape[-1])
        return norm(flat).reshape_as(x)

    def _extract(self, obs: TensorDict) -> tuple[torch.Tensor, ...]:
        return (
            self._apply_norm_per_element(self.joint_desc_normalizer, obs["joint_desc"]),
            self._apply_norm_per_element(self.joint_obs_normalizer, obs["joint_obs"]),
            self._apply_norm_per_element(self.feet_desc_normalizer, obs["feet_desc"]),
            self._apply_norm_per_element(self.feet_obs_normalizer, obs["feet_obs"]),
            self.global_obs_normalizer(obs["global_obs"]),
        )

    # --- recurrent no-ops (required by rsl_rl Model interface) ---
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    # --- empirical-normalization update (called per rollout by PPO) ---
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return

        def _update(norm: nn.Module, x: torch.Tensor) -> None:
            if isinstance(norm, nn.Identity):
                return
            flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
            norm.update(flat)  # type: ignore[attr-defined]

        _update(self.joint_desc_normalizer, obs["joint_desc"])
        _update(self.joint_obs_normalizer, obs["joint_obs"])
        _update(self.feet_desc_normalizer, obs["feet_desc"])
        _update(self.feet_obs_normalizer, obs["feet_obs"])
        _update(self.global_obs_normalizer, obs["global_obs"])


class _OnnxAttentionActorModel(nn.Module):
    """ONNX-export wrapper: deterministic forward with 5 named inputs."""

    is_recurrent: bool = False

    def __init__(self, model: "AttentionActorModel") -> None:
        super().__init__()
        self.trunk = copy.deepcopy(model.trunk)
        self.head = copy.deepcopy(model.head)
        self._dims = dict(model._dims)
        self.joint_desc_normalizer = copy.deepcopy(model.joint_desc_normalizer)
        self.joint_obs_normalizer = copy.deepcopy(model.joint_obs_normalizer)
        self.feet_desc_normalizer = copy.deepcopy(model.feet_desc_normalizer)
        self.feet_obs_normalizer = copy.deepcopy(model.feet_obs_normalizer)
        self.global_obs_normalizer = copy.deepcopy(model.global_obs_normalizer)

    @staticmethod
    def _norm_seq(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.Identity):
            return x
        return norm(x.reshape(-1, x.shape[-1])).reshape_as(x)

    def forward(self, joint_desc, joint_obs, feet_desc, feet_obs, global_obs):
        joint_desc = self._norm_seq(self.joint_desc_normalizer, joint_desc)
        joint_obs = self._norm_seq(self.joint_obs_normalizer, joint_obs)
        feet_desc = self._norm_seq(self.feet_desc_normalizer, feet_desc)
        feet_obs = self._norm_seq(self.feet_obs_normalizer, feet_obs)
        global_obs = self.global_obs_normalizer(global_obs)
        trunk_latent, joint_state_latent = self.trunk(joint_desc, joint_obs, feet_desc, feet_obs, global_obs)
        return self.head(trunk_latent, joint_desc, joint_state_latent)

    def get_dummy_inputs(self):
        J, F = self._dims["n_joints"], self._dims["n_feet"]
        return (
            torch.zeros(1, J, self._dims["joint_desc_dim"]),
            torch.zeros(1, J, self._dims["joint_state_dim"]),
            torch.zeros(1, F, self._dims["foot_desc_dim"]),
            torch.zeros(1, F, self._dims["foot_state_dim"]),
            torch.zeros(1, self._dims["global_dim"]),
        )

    @property
    def input_names(self):
        return ["joint_desc", "joint_obs", "feet_desc", "feet_obs", "global_obs"]

    @property
    def output_names(self):
        return ["actions"]

    @property
    def dynamic_axes(self):
        return {
            "joint_desc": {0: "batch", 1: "n_joints"},
            "joint_obs": {0: "batch", 1: "n_joints"},
            "feed_desc": {0: "batch", 1: "n_feet"},
            "feet_obs": {0: "batch", 1: "n_feet"},
            "global_obs": {0: "batch"},
            "actions": {0: "batch", 1: "n_joints"},
        }


class _TorchAttentionActorModel(nn.Module):
    """TorchScript-friendly inference wrapper for :class:`AttentionActorModel`.

    Mirrors the math of :class:`AttentionTrunk` + :class:`ActionMeanHead` but
    with the attention pool **inlined**, so the module is amenable to
    ``torch.jit.script``. ``AttentionTrunk._attention_pool`` takes ``nn.Module``
    instances as arguments, which TorchScript cannot handle directly; lifting
    every leaf submodule onto ``self`` and inlining the forward sidesteps that.
    """

    is_recurrent: bool = False

    def __init__(self, model: "AttentionActorModel") -> None:
        super().__init__()

        trunk, head = model.trunk, model.head

        # Observation normalizers (Identity if disabled).
        self.joint_desc_normalizer = copy.deepcopy(model.joint_desc_normalizer)
        self.joint_obs_normalizer = copy.deepcopy(model.joint_obs_normalizer)
        self.feet_desc_normalizer = copy.deepcopy(model.feet_desc_normalizer)
        self.feet_obs_normalizer = copy.deepcopy(model.feet_obs_normalizer)
        self.global_obs_normalizer = copy.deepcopy(model.global_obs_normalizer)

        # Lift trunk leaves onto self as plain attributes.
        self.joint_mask_mlp_1 = copy.deepcopy(trunk.joint_mask_mlp_1)
        self.joint_mask_ln = copy.deepcopy(trunk.joint_mask_ln)
        self.joint_mask_mlp_2 = copy.deepcopy(trunk.joint_mask_mlp_2)
        self.joint_state_enc = copy.deepcopy(trunk.joint_state_enc)
        self.foot_mask_mlp_1 = copy.deepcopy(trunk.foot_mask_mlp_1)
        self.foot_mask_ln = copy.deepcopy(trunk.foot_mask_ln)
        self.foot_mask_mlp_2 = copy.deepcopy(trunk.foot_mask_mlp_2)
        self.foot_state_enc = copy.deepcopy(trunk.foot_state_enc)
        self.trunk_fc1 = copy.deepcopy(trunk.trunk_fc1)
        self.trunk_ln1 = copy.deepcopy(trunk.trunk_ln1)
        self.trunk_fc2 = copy.deepcopy(trunk.trunk_fc2)
        self.trunk_fc3 = copy.deepcopy(trunk.trunk_fc3)

        # Learned softmax temperatures (re-parameterized as plain parameters).
        self.joint_log_temp = nn.Parameter(trunk.joint_log_temp.detach().clone())
        self.foot_log_temp = nn.Parameter(trunk.foot_log_temp.detach().clone())

        # Action head leaves.
        self.desc_fc1 = copy.deepcopy(head.desc_fc1)
        self.desc_ln = copy.deepcopy(head.desc_ln)
        self.desc_fc2 = copy.deepcopy(head.desc_fc2)
        self.mean_fc1 = copy.deepcopy(head.mean_fc1)
        self.mean_ln = copy.deepcopy(head.mean_ln)
        self.mean_fc2 = copy.deepcopy(head.mean_fc2)

        # Scalar constants (must be plain floats for TorchScript).
        self.min_temperature = float(trunk.min_temperature)
        self.stability_eps = float(trunk.stability_eps)
        self.mean_abs_clip = float(head.mean_abs_clip)

    def forward(
        self,
        joint_desc: torch.Tensor,
        joint_obs: torch.Tensor,
        feet_desc: torch.Tensor,
        feet_obs: torch.Tensor,
        global_obs: torch.Tensor,
    ) -> torch.Tensor:
        # ---- Empirical normalization (Identity if disabled) ----
        jd_flat = joint_desc.reshape(-1, joint_desc.shape[-1])
        joint_desc = self.joint_desc_normalizer(jd_flat).reshape_as(joint_desc)
        jo_flat = joint_obs.reshape(-1, joint_obs.shape[-1])
        joint_obs = self.joint_obs_normalizer(jo_flat).reshape_as(joint_obs)
        fd_flat = feet_desc.reshape(-1, feet_desc.shape[-1])
        feet_desc = self.feet_desc_normalizer(fd_flat).reshape_as(feet_desc)
        fo_flat = feet_obs.reshape(-1, feet_obs.shape[-1])
        feet_obs = self.feet_obs_normalizer(fo_flat).reshape_as(feet_obs)
        global_obs = self.global_obs_normalizer(global_obs)

        # ---- Joint attention pool (inlined) ----
        m = self.joint_mask_mlp_2(F.elu(self.joint_mask_ln(self.joint_mask_mlp_1(joint_desc))))
        m = torch.tanh(m).clamp(-1.0 + self.stability_eps, 1.0 - self.stability_eps)
        joint_state_latent = F.elu(self.joint_state_enc(joint_obs))
        t_j = torch.exp(self.joint_log_temp) + self.min_temperature
        e_x = torch.exp(m / t_j)
        w = e_x / (e_x.sum(dim=-1, keepdim=True) + self.stability_eps)
        gated = w.unsqueeze(-1) * joint_state_latent.unsqueeze(-2)
        gated = gated.reshape(gated.shape[0], gated.shape[1], -1)
        joint_pool = gated.sum(dim=-2)

        # ---- Foot attention pool (inlined) ----
        m = self.foot_mask_mlp_2(F.elu(self.foot_mask_ln(self.foot_mask_mlp_1(feet_desc))))
        m = torch.tanh(m).clamp(-1.0 + self.stability_eps, 1.0 - self.stability_eps)
        foot_state_latent = F.elu(self.foot_state_enc(feet_obs))
        t_f = torch.exp(self.foot_log_temp) + self.min_temperature
        e_x = torch.exp(m / t_f)
        # softmax 구하는 수식
        w = e_x / (e_x.sum(dim=-1, keepdim=True) + self.stability_eps)
        gated = w.unsqueeze(-1) * foot_state_latent.unsqueeze(-2)
        gated = gated.reshape(gated.shape[0], gated.shape[1], -1)
        foot_pool = gated.sum(dim=-2)

        # ---- Trunk MLP ----
        x = torch.cat([joint_pool, foot_pool, global_obs], dim=-1)
        x = F.elu(self.trunk_ln1(self.trunk_fc1(x)))
        x = F.elu(self.trunk_fc2(x))
        trunk_latent = self.trunk_fc3(x)

        # ---- Action mean head ----
        desc = F.elu(self.desc_ln(self.desc_fc1(joint_desc)))
        desc = self.desc_fc2(desc)
        n_joints = joint_desc.shape[1]
        trunk_tile = trunk_latent.unsqueeze(-2).expand(-1, n_joints, -1)
        h = torch.cat([trunk_tile, joint_state_latent, desc], dim=-1)
        h = F.elu(self.mean_ln(self.mean_fc1(h)))
        mean = self.mean_fc2(h).squeeze(-1)
        return mean.clamp(-self.mean_abs_clip, self.mean_abs_clip)

    @torch.jit.export
    def reset(self) -> None:
        pass


class AttentionActorModel(_AttentionModelBase):
    """Attention-based policy network producing per-joint action means."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        joint_mask_dim: int = 64,
        foot_mask_dim: int = 32,
        state_latent_dim: int = 32,
        trunk_hidden: tuple[int, ...] = (512, 256, 128),
        desc_hidden: int = 128,
        mean_hidden: int = 128,
        init_temperature: float = 1.0,
        min_temperature: float = 0.015,
        stability_eps: float = 1e-7,
        mean_abs_clip: float = 10.0,
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        # Ignored fields inherited from RslRlMLPModelCfg (MLP-specific or deprecated).
        hidden_dims: list[int] | None = None,
        activation: str | None = None,
        **_unused: object,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            joint_mask_dim=joint_mask_dim,
            foot_mask_dim=foot_mask_dim,
            state_latent_dim=state_latent_dim,
            trunk_hidden=trunk_hidden,
            init_temperature=init_temperature,
            min_temperature=min_temperature,
            stability_eps=stability_eps,
            obs_normalization=obs_normalization,
        )
        del hidden_dims, activation, _unused  # silenced
        if output_dim != self._dims["n_joints"]:
            raise ValueError(
                f"AttentionActorModel expects output_dim == n_joints "
                f"({output_dim} != {self._dims['n_joints']}). The actor produces "
                "one action per joint; ensure ActionsCfg covers all joints in the same order."
            )

        self.head = ActionMeanHead(
            joint_desc_dim=self._dims["joint_desc_dim"],
            trunk_dim=self.trunk.trunk_out_dim,
            state_latent_dim=state_latent_dim,
            desc_hidden=desc_hidden,
            mean_hidden=mean_hidden,
            mean_abs_clip=mean_abs_clip,
        )

        # Output distribution (Gaussian with state-independent per-joint std works well
        # and matches kapex's existing init_noise_std setting; switch to
        # HeteroscedasticGaussianDistribution later for state-dependent std).
        if distribution_cfg is None:
            self.distribution: Distribution | None = None
        else:
            cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **cfg)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        joint_desc, joint_state, foot_desc, foot_state, global_state = self._extract(obs)
        trunk_latent, _ = self.trunk(joint_desc, joint_state, foot_desc, foot_state, global_state)
        return trunk_latent

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        joint_desc, joint_state, foot_desc, foot_state, global_state = self._extract(obs)
        trunk_latent, joint_state_latent = self.trunk(joint_desc, joint_state, foot_desc, foot_state, global_state)
        mean = self.head(trunk_latent, joint_desc, joint_state_latent)  # (B, J)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    # --- distribution accessors ---
    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchAttentionActorModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxAttentionActorModel(self)


class AttentionCriticModel(_AttentionModelBase):
    """Attention-based value network producing a scalar state value per env."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        joint_mask_dim: int = 64,
        foot_mask_dim: int = 32,
        state_latent_dim: int = 32,
        trunk_hidden: tuple[int, ...] = (512, 256, 128),
        init_temperature: float = 1.0,
        min_temperature: float = 0.015,
        stability_eps: float = 1e-7,
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        # Ignored fields inherited from RslRlMLPModelCfg (MLP-specific or deprecated).
        hidden_dims: list[int] | None = None,
        activation: str | None = None,
        **_unused: object,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            joint_mask_dim=joint_mask_dim,
            foot_mask_dim=foot_mask_dim,
            state_latent_dim=state_latent_dim,
            trunk_hidden=trunk_hidden,
            init_temperature=init_temperature,
            min_temperature=min_temperature,
            stability_eps=stability_eps,
            obs_normalization=obs_normalization,
        )
        del hidden_dims, activation, _unused
        if output_dim != 1:
            raise ValueError(f"AttentionCriticModel expects output_dim=1, got {output_dim}.")
        if distribution_cfg is not None:
            raise ValueError("Critic does not use a distribution; pass distribution_cfg=None.")
        self.distribution = None
        self.value_head = ValueHead(trunk_dim=self.trunk.trunk_out_dim)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        joint_desc, joint_state, foot_desc, foot_state, global_state = self._extract(obs)
        trunk_latent, _ = self.trunk(joint_desc, joint_state, foot_desc, foot_state, global_state)
        return trunk_latent

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        joint_desc, joint_state, foot_desc, foot_state, global_state = self._extract(obs)
        trunk_latent, _ = self.trunk(joint_desc, joint_state, foot_desc, foot_state, global_state)
        return self.value_head(trunk_latent)  # (B, 1)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("JIT export for AttentionCriticModel is not implemented yet.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("ONNX export for AttentionCriticModel is not implemented yet.")


# ---------------------------------------------------------------------------
# Config classes (rsl_rl PPO uses dict(**cfg[name]) -> Model init)
# ---------------------------------------------------------------------------

_ACTOR_QUALNAME = "kirack.tasks.standing.agents.attention_model:AttentionActorModel"
_CRITIC_QUALNAME = "kirack.tasks.standing.agents.attention_model:AttentionCriticModel"


@configclass
class RslRlAttentionActorModelCfg(RslRlMLPModelCfg):
    """Configuration for :class:`AttentionActorModel`.

    Inherits from :class:`isaaclab_rl.rsl_rl.RslRlMLPModelCfg` so the legacy
    deprecation handler in ``isaaclab_rl.rsl_rl.utils`` finds the fields it
    expects (``stochastic``, ``init_noise_std``, etc.). The model's
    ``__init__`` ignores MLP-specific kwargs (``hidden_dims``, ``activation``)
    that come along for the ride.
    """

    class_name: str = _ACTOR_QUALNAME

    # MLP fields are unused but must be set (parent declares them MISSING).
    hidden_dims: list[int] = []
    activation: str = "elu"

    # Attention-specific architecture.
    joint_mask_dim: int = 64
    foot_mask_dim: int = 32
    state_latent_dim: int = 32
    trunk_hidden: list[int] = [512, 256, 128]
    desc_hidden: int = 128
    mean_hidden: int = 128
    init_temperature: float = 1.0
    min_temperature: float = 0.015
    stability_eps: float = 1e-7
    mean_abs_clip: float = 10.0

    obs_normalization: bool = True
    """Per-group EmpiricalNormalization on joint_desc/joint_obs/feet_desc/feet_obs/global_obs."""

    distribution_cfg: RslRlMLPModelCfg.GaussianDistributionCfg | None = RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="kirack.tasks.standing.agents.shared_std_gaussian:SharedStdGaussianDistribution",
        init_std=1.0,
        std_type="scalar",
    )
    """Output distribution: shared scalar std (morphology-invariant for distributed training)."""

    # Legacy fields kept so the deprecation handler does not crash; the
    # handler deletes them once ``distribution_cfg`` is resolved.
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: Literal["scalar", "log"] = "scalar"
    state_dependent_std: bool = False


@configclass
class RslRlAttentionCriticModelCfg(RslRlMLPModelCfg):
    """Configuration for :class:`AttentionCriticModel`."""

    class_name: str = _CRITIC_QUALNAME

    hidden_dims: list[int] = []
    activation: str = "elu"

    joint_mask_dim: int = 64
    foot_mask_dim: int = 32
    state_latent_dim: int = 32
    trunk_hidden: list[int] = [512, 256, 128]
    init_temperature: float = 1.0
    min_temperature: float = 0.015
    stability_eps: float = 1e-7

    obs_normalization: bool = True
    distribution_cfg: None = None
    """Critic has no output distribution -- value head produces (E, 1) directly."""

    # Legacy fields with safe defaults so the deprecation handler is a no-op.
    stochastic: bool = False
    init_noise_std: float = 0.0
    noise_std_type: Literal["scalar", "log"] = "scalar"
    state_dependent_std: bool = False
