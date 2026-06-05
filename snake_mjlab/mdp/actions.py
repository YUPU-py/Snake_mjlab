"""Residual Serpenoid Joint Position Action for MuJoCo.

Applies joint-position residuals on top of a fixed serpenoid gait pattern.
The action space controls deviations from the nominal sinusoidal gait.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from snake_mjlab import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _build_reversed_joint_phase_offsets(action_dim: int, beta: float, device: torch.device) -> torch.Tensor:
    """Return MuJoCo-aligned reversed spatial phase offsets.

    Yaw joints are ordered from head (yaw1) to tail (yaw7).
    Reversed order gives the tail the largest phase offset and head the smallest.
    """

    return torch.arange(action_dim - 1, -1, -1, device=device, dtype=torch.float32) * beta


class ResidualSerpenoidJointPositionAction(ActionTerm):
    """Apply joint-position residuals on top of a fixed serpenoid gait."""

    cfg: "ResidualSerpenoidJointPositionActionCfg"

    def __init__(self, cfg: "ResidualSerpenoidJointPositionActionCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self._physics_dt = env.physics_dt
        self._amplitude = float(cfg.amplitude)
        self._omega = float(cfg.omega)
        self._beta = float(cfg.beta)
        self._bias = float(cfg.theta_bias)

        # Resolve joints from config (parent class sets self._entity)
        joint_names = cfg.joint_names
        if isinstance(joint_names, tuple) and len(joint_names) == 1:
            joint_names = (joint_names[0],)
        self._joint_ids, self._joint_names = self._entity.find_joints(
            list(joint_names), preserve_order=cfg.preserve_order
        )
        self._joint_ids = list(self._joint_ids)
        self._num_joints = len(self._joint_ids)

        if self._num_joints == 0:
            raise RuntimeError(
                "No joints were resolved for ResidualSerpenoidJointPositionAction. "
                f"Check `joint_names`={joint_names} and the asset joint naming. "
                f"Available joints: {self._entity.joint_names}"
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._second_last_raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._phase_offsets = torch.zeros(self.num_envs, device=self.device)
        self._gait_time = torch.zeros(self.num_envs, device=self.device)
        self._current_phase = torch.zeros(self.num_envs, device=self.device)
        self._joint_phase_offsets = _build_reversed_joint_phase_offsets(self.action_dim, self._beta, self.device)
        self._nominal_joint_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._final_joint_targets = torch.zeros_like(self._nominal_joint_targets)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._scale = self._resolve_affine_param(cfg.scale, default_value=1.0)
        self._offset = (
            self._resolve_affine_param(cfg.offset, default_value=0.0)
            if not cfg.use_default_offset
            else torch.zeros(self.num_envs, self.action_dim, device=self.device)
        )
        self._clip = self._resolve_clip(cfg.clip)
        self.reset()

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    # Alias: base class ActionTerm expects raw_action (singular)
    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._final_joint_targets

    @property
    def current_phase(self) -> torch.Tensor:
        return self._current_phase

    @property
    def nominal_joint_targets(self) -> torch.Tensor:
        return self._nominal_joint_targets

    @property
    def last_raw_actions(self) -> torch.Tensor:
        return self._prev_raw_actions

    @property
    def second_last_raw_actions(self) -> torch.Tensor:
        return self._second_last_raw_actions

    def process_actions(self, actions: torch.Tensor):
        actions = torch.clamp(actions, min=-6.28, max=6.28)
        self._second_last_raw_actions = self._prev_raw_actions.clone()
        self._prev_raw_actions = self._raw_actions.clone()
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self._update_nominal_targets(slice(None))
        self._final_joint_targets = self._nominal_joint_targets + self._processed_actions
        self._entity.set_joint_position_target(self._final_joint_targets, joint_ids=self._joint_ids)
        self._gait_time += self._physics_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, torch.Tensor) and not isinstance(env_ids, slice):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._raw_actions[env_ids] = 0.0
        self._prev_raw_actions[env_ids] = 0.0
        self._second_last_raw_actions[env_ids] = 0.0
        self._phase_offsets[env_ids] = torch.empty_like(self._phase_offsets[env_ids]).uniform_(0.0, 2.0 * math.pi)
        self._gait_time[env_ids] = 0.0
        self._update_nominal_targets(env_ids)
        self._final_joint_targets[env_ids] = self._nominal_joint_targets[env_ids]
        self._processed_actions[env_ids] = 0.0

    def _update_nominal_targets(self, env_ids: Sequence[int] | slice) -> None:
        phase = self._phase_offsets[env_ids] + self._omega * self._gait_time[env_ids]
        phase = torch.remainder(phase, 2.0 * math.pi)
        self._current_phase[env_ids] = phase
        joint_angles = self._amplitude * torch.sin(phase.unsqueeze(-1) + self._joint_phase_offsets.unsqueeze(0)) + self._bias
        self._nominal_joint_targets[env_ids] = joint_angles

    def _resolve_affine_param(self, param, default_value: float) -> torch.Tensor:
        if isinstance(param, (float, int)):
            return torch.full((self.num_envs, self.action_dim), float(param), device=self.device)
        if isinstance(param, dict):
            value = torch.full((self.num_envs, self.action_dim), default_value, device=self.device)
            index_list, _, value_list = resolve_matching_names_values(param, self._joint_names)
            value[:, index_list] = torch.tensor(value_list, device=self.device)
            return value
        raise ValueError(f"Unsupported affine parameter type: {type(param)}. Supported types are float and dict.")

    def _resolve_clip(self, clip_cfg):
        if clip_cfg is None:
            return None
        if isinstance(clip_cfg, dict):
            clip = torch.tensor([[-float('inf'), float('inf')]], device=self.device).repeat(self.num_envs, self.action_dim, 1)
            index_list, _, value_list = resolve_matching_names_values(clip_cfg, self._joint_names)
            clip[:, index_list] = torch.tensor(value_list, device=self.device)
            return clip
        raise ValueError(f"Unsupported clip type: {type(clip_cfg)}. Supported types are dict.")


from dataclasses import dataclass


@dataclass(kw_only=True)
class JointPositionActionCfg(ActionTermCfg):
    """Configuration for joint position control."""

    actuator_names: tuple = ("joint.*",)
    scale: float = 1.0
    offset: float = 0.0
    use_default_offset: bool = True
    preserve_order: bool = True
    clip: Optional[dict[str, tuple[float, float]]] = None

    def build(self, env: "ManagerBasedRlEnv") -> "JointPositionAction":
        return JointPositionAction(self, env)


class JointPositionAction(ActionTerm):
    """Joint position controller that sets joint position targets from policy actions."""

    cfg: JointPositionActionCfg

    def __init__(self, cfg: JointPositionActionCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        joint_names = cfg.actuator_names
        if isinstance(joint_names, tuple) and len(joint_names) == 1:
            joint_names = (joint_names[0],)
        self._joint_ids, self._joint_names = self._entity.find_joints(
            list(joint_names), preserve_order=cfg.preserve_order
        )
        self._joint_ids = list(self._joint_ids)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise RuntimeError(
                "No joints resolved for JointPositionAction. "
                f"Check `actuator_names`={joint_names}. "
                f"Available joints: {self._entity.joint_names}"
            )
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._scale = float(cfg.scale)
        self._offset = float(cfg.offset) if not cfg.use_default_offset else 0.0
        self._clip = self._resolve_clip(cfg.clip)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    def _resolve_clip(self, clip_cfg):
        if clip_cfg is None:
            return None
        if isinstance(clip_cfg, dict):
            clip = torch.tensor([[-float('inf'), float('inf')]], device=self.device).repeat(self.num_envs, self.action_dim, 1)
            index_list, _, value_list = resolve_matching_names_values(clip_cfg, self._joint_names)
            clip[:, index_list] = torch.tensor(value_list, device=self.device)
            return clip
        return None

    def process_actions(self, actions: torch.Tensor) -> None:
        self._processed_actions = actions * self._scale + self._offset
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._processed_actions


@dataclass(kw_only=True)
class ResidualSerpenoidJointPositionActionCfg(ActionTermCfg):
    """Configuration for residual joint-position control on a fixed serpenoid gait."""

    # Backwards compatible: older configs may still pass `actuator_names`.
    joint_names: tuple[str, ...] = ("yaw.*",)
    actuator_names: Optional[tuple[str, ...]] = None
    preserve_order: bool = True
    amplitude: float = math.radians(40.0)
    omega: float = 3.0
    beta: float = math.pi * 2.2 / 7.0
    theta_bias: float = 0.0
    scale: float = 1.0
    offset: float = 0.0
    use_default_offset: bool = False
    clip: Optional[dict[str, tuple[float, float]]] = None

    def build(self, env: "ManagerBasedRlEnv") -> ResidualSerpenoidJointPositionAction:
        return ResidualSerpenoidJointPositionAction(self, env)
