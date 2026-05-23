"""Snake velocity reward terms — virtual chassis tracking and motion incentives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .virtual_chassis import compute_virtual_chassis_command_terms

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_amplitude(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward sustained motion by averaging absolute joint velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def motion_coordination(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize all joints bending or moving in the same direction."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pos_sign_mean = torch.abs(torch.mean(torch.sign(joint_pos), dim=1))
    vel_sign_mean = torch.abs(torch.mean(torch.sign(joint_vel), dim=1))
    return (pos_sign_mean + vel_sign_mean) / 2.0


def phase_propagation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reward alternating velocity directions between adjacent controlled joints."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if joint_vel.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    vel_product = joint_vel[:, :-1] * joint_vel[:, 1:]
    vel_mag = torch.abs(joint_vel[:, :-1]) * torch.abs(joint_vel[:, 1:]) + 1e-6
    normalized_product = vel_product / vel_mag
    return -torch.mean(normalized_product, dim=1)


class RawActionRatePenalty:
    """L2 penalty on the first-order raw action-rate."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._action_term_name = cfg.params.get("action_term_name", "joint_pos")
        self._rate_clip = float(cfg.params.get("rate_clip", 10.0))
        self._prev_raw_action = None

    def __call__(self, env: ManagerBasedRlEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
        action_term = env.action_manager.get_term(action_term_name)
        raw_action = action_term.raw_actions
        if self._prev_raw_action is None:
            self._prev_raw_action = torch.zeros_like(raw_action)
        delta = raw_action - self._prev_raw_action
        if self._rate_clip > 0.0:
            delta = torch.clamp(delta, min=-self._rate_clip, max=self._rate_clip)
        self._prev_raw_action.copy_(raw_action)
        return torch.sum(torch.square(delta), dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._prev_raw_action is None:
            return
        if env_ids is None:
            self._prev_raw_action.zero_()
        else:
            self._prev_raw_action[env_ids] = 0.0


class VirtualChassisTrackLinVelXYExp:
    """Reward planar command tracking in the virtual chassis frame."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Entity = env.scene[self._asset_cfg.name]
        self._prev_axes_w = torch.zeros(env.num_envs, 3, 3, device=env.device)
        self._has_prev_axes = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        body_pos_w = self._asset.data.body_link_pos_w[:, self._asset_cfg.body_ids, :]
        body_lin_vel_w = self._asset.data.body_link_lin_vel_w[:, self._asset_cfg.body_ids, :]
        body_ang_vel_w = self._asset.data.body_link_ang_vel_w[:, self._asset_cfg.body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return torch.zeros(env.num_envs, device=env.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self._prev_axes_w,
            has_prev=self._has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(env.num_envs, device=env.device)

        self._prev_axes_w.copy_(axes_w)
        self._has_prev_axes[:] = True

        command = env.command_manager.get_command(command_name)
        lin_vel_error = torch.sum(
            torch.square(command[:, :2] - actual_lin_vel_vc[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / std**2)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_axes_w.zero_()
            self._has_prev_axes.zero_()
        else:
            self._prev_axes_w[env_ids] = 0.0
            self._has_prev_axes[env_ids] = False


class VirtualChassisTrackAngVelZExp:
    """Reward yaw-rate command tracking around the virtual chassis z-axis."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Entity = env.scene[self._asset_cfg.name]
        self._prev_axes_w = torch.zeros(env.num_envs, 3, 3, device=env.device)
        self._has_prev_axes = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        body_pos_w = self._asset.data.body_link_pos_w[:, self._asset_cfg.body_ids, :]
        body_lin_vel_w = self._asset.data.body_link_lin_vel_w[:, self._asset_cfg.body_ids, :]
        body_ang_vel_w = self._asset.data.body_link_ang_vel_w[:, self._asset_cfg.body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return torch.zeros(env.num_envs, device=env.device)

        _, axes_w, _, actual_ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self._prev_axes_w,
            has_prev=self._has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_ang_vel_z_vc).all():
            return torch.zeros(env.num_envs, device=env.device)

        self._prev_axes_w.copy_(axes_w)
        self._has_prev_axes[:] = True

        command = env.command_manager.get_command(command_name)
        ang_vel_error = torch.square(command[:, 2] - actual_ang_vel_z_vc)
        return torch.exp(-ang_vel_error / std**2)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_axes_w.zero_()
            self._has_prev_axes.zero_()
        else:
            self._prev_axes_w[env_ids] = 0.0
            self._has_prev_axes[env_ids] = False


def contact_penalty(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "contact_sensor",
    threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize contact forces between the robot body and the ground.
    
    Args:
        env: The environment instance.
        sensor_name: Name of the contact sensor.
        threshold: Minimum force magnitude to consider as contact (0 = any contact).
    
    Returns:
        Tensor of shape [B] with 1.0 if any body is in contact, 0.0 otherwise.
    """
    contact_sensor = env.scene.sensors[sensor_name]
    contact_data = contact_sensor.data
    
    # Use 'found' field: [B, N] where 0 = no contact, >0 = contact count
    # If threshold > 0, also check force magnitude
    in_contact = torch.any(contact_data.found > 0, dim=1)
    
    if threshold > 0 and contact_data.force is not None:
        # Additional check on force magnitude
        force_magnitude = torch.norm(contact_data.force, dim=-1)  # [B, N]
        in_contact = in_contact & torch.any(force_magnitude > threshold, dim=1)
    
    return in_contact.float()
