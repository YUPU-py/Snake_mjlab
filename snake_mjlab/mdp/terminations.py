"""Snake velocity termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Episode time limit termination."""
    return env.episode_length_buf >= env.max_episode_length


def invalid_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_root_lin_vel: float = 2.0,
    max_root_ang_vel: float = 5.0,
    min_root_height: float = -0.2,
    max_root_height: float = 0.5,
) -> torch.Tensor:
    """Terminate if the robot enters an invalid state (flying, spinning too fast, etc.)."""
    asset: Entity = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_link_pos_w
    root_lin_vel_w = asset.data.root_link_lin_vel_w
    root_ang_vel_w = asset.data.root_link_ang_vel_w

    # Check height bounds
    height_ok = (root_pos_w[:, 2] > min_root_height) & (root_pos_w[:, 2] < max_root_height)

    # Check linear velocity magnitude
    lin_vel_ok = torch.norm(root_lin_vel_w, dim=1) < max_root_lin_vel

    # Check angular velocity magnitude
    ang_vel_ok = torch.norm(root_ang_vel_w, dim=1) < max_root_ang_vel

    # All conditions must be met
    return ~(height_ok & lin_vel_ok & ang_vel_ok)


def is_terminated(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Return 1.0 if the episode is terminated for any reason, 0.0 otherwise.

    This is used as a reward penalty for early termination.
    """
    # Returns tensor of zeros (no termination signal from this term)
    # The actual termination detection is handled by other termination terms
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
