"""Snake velocity event terms — reset and domain randomization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_snake_state(
    env: ManagerBasedRlEnv,
    env_ids,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_position_range: tuple[float, float] = (-0.05, 0.05),
    pose_range: dict | None = None,
    velocity_range: dict | None = None,
) -> dict:
    """Reset the snake robot to its initial state with randomization.

    Mirrors the Isaac Lab implementation adapted for mjlab's entity API.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset (None = all).
        asset_cfg: Robot asset configuration.
        joint_position_range: Range for joint position randomization around default.
        pose_range: Position range for root pose (x, y, yaw).
        velocity_range: Velocity range for root velocity.

    Returns:
        Empty dictionary (no extras).
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    asset = env.scene[asset_cfg.name]

    # Default ranges if not specified
    pose_range = pose_range or {
        "x": (-0.2, 0.2),
        "y": (0.0, 0.0),
        "yaw": (-3.14159, 3.14159),
    }
    velocity_range = velocity_range or {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }

    num_resets = env_ids.numel()

    # Get default root state and apply perturbations
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    # Sample root position
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))

    root_state[:, 0] += torch.empty(num_resets, device=env.device).uniform_(*x_range)
    root_state[:, 1] += torch.empty(num_resets, device=env.device).uniform_(*y_range)
    yaw = torch.empty(num_resets, device=env.device).uniform_(*yaw_range)

    # Build quaternion from yaw (around z-axis) - format: [w, x, y, z]
    # For rotation around z-axis: w = cos(yaw/2), x = 0, y = 0, z = sin(yaw/2)
    cos_yaw_half = torch.cos(yaw * 0.5)
    sin_yaw_half = torch.sin(yaw * 0.5)
    root_state[:, 3] = cos_yaw_half  # w
    root_state[:, 4] = torch.zeros(num_resets, device=env.device)  # x = 0
    root_state[:, 5] = torch.zeros(num_resets, device=env.device)  # y = 0
    root_state[:, 6] = sin_yaw_half  # z

    # Sample root velocity
    root_state[:, 7] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["x"])
    root_state[:, 8] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["y"])
    root_state[:, 9] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["z"])
    root_state[:, 10] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["roll"])
    root_state[:, 11] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["pitch"])
    root_state[:, 12] = torch.empty(num_resets, device=env.device).uniform_(*velocity_range["yaw"])

    # Get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])

    # Apply joint position randomization if range is non-zero
    if joint_position_range[0] != joint_position_range[1]:
        joint_range = joint_position_range[1] - joint_position_range[0]
        num_joints = joint_pos.shape[1]
        random_offset = (
            torch.empty(num_resets, num_joints, device=env.device).uniform_(0.0, joint_range)
            + joint_position_range[0]
        )
        joint_pos += random_offset

    # Write to simulation
    # root_state shape: [N, 13] = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    return {}
