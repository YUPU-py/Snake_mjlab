"""Snake velocity observations — mirrors Isaac Lab with NaN handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import observations as builtin_obs
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def last_raw_actions(env: "ManagerBasedRlEnv", action_name: str = "joint_pos") -> torch.Tensor:
    """Get raw ( unclipped ) actions from the action manager, with NaN sanitization."""
    raw = env.action_manager.get_term(action_name)._raw_actions
    return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def base_ang_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Base angular velocity in world frame, with NaN sanitization."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.base_ang_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def projected_gravity(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Gravity direction projected to asset base frame, with NaN sanitization."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.projected_gravity(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_pos_rel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Joint positions relative to default pose, with NaN sanitization."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_pos_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_vel_rel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Joint velocities relative to zero, with NaN sanitization."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_vel_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def generated_commands(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """Current command values, with NaN sanitization."""
    return torch.nan_to_num(builtin_obs.generated_commands(env, command_name), nan=0.0, posinf=0.0, neginf=0.0)
