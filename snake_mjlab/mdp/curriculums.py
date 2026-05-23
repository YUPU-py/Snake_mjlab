"""Snake velocity curriculum terms — EMA-based velocity command curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def command_velocity_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids,
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    max_curriculum: float = 0.4,
    min_curriculum: float = 0.1,
    step_size: float = 0.05,
    threshold_ratio: float = 0.8,
    ema_decay: float = 0.05,
    min_env_count: int = 10,
) -> dict:
    """Symmetrically expand/shrink the x/y command ranges using EMA-smoothed reward.

    Uses exponential moving average over per-episode tracking rewards. Only updates
    the range when at least ``min_env_count`` environments have finished, and the
    EMA crosses the expand / shrink thresholds.

    Adapted from Isaac Lab implementation for mjlab.

    Args:
        env: The environment instance.
        env_ids: Environment IDs that finished episodes.
        command_name: Name of the command term to modify.
        reward_term_name: Name of the reward term to track.
        max_curriculum: Maximum velocity magnitude for curriculum.
        min_curriculum: Minimum velocity magnitude for curriculum.
        step_size: Step size for expanding/shrinking ranges.
        threshold_ratio: Ratio of max reward to use as expand threshold.
        ema_decay: EMA decay factor for smoothing.
        min_env_count: Minimum number of finished envs to trigger update.

    Returns:
        Dictionary with logging info (lin_vel ranges, EMA reward).
    """
    command_term = env.command_manager.get_term(command_name)

    # Get current command ranges
    x_min, x_max = command_term.current_lin_vel_x_range
    y_min, y_max = command_term.current_lin_vel_y_range

    if env_ids is None or len(env_ids) == 0:
        return {
            "lin_vel_x_min": x_min,
            "lin_vel_x_max": x_max,
            "lin_vel_y_min": y_min,
            "lin_vel_y_max": y_max,
        }

    # Compute mean tracking reward from episode sums
    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    mean_tracking_reward = float(torch.mean(episode_sums) / env.max_episode_length_s)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = threshold_ratio * reward_cfg.weight

    # Initialize or update EMA
    if not hasattr(command_term, "_tracking_reward_ema"):
        command_term._tracking_reward_ema = mean_tracking_reward
    else:
        command_term._tracking_reward_ema = (
            (1.0 - ema_decay) * command_term._tracking_reward_ema
            + ema_decay * mean_tracking_reward
        )
    ema = command_term._tracking_reward_ema

    # Update command ranges based on EMA
    if len(env_ids) >= min_env_count:
        if ema > threshold:
            # Expand: increase max curriculum
            x_min, x_max = command_term.expand_lin_vel_x(step_size=step_size, max_curriculum=max_curriculum)
            y_min, y_max = command_term.expand_lin_vel_y(step_size=step_size, max_curriculum=max_curriculum)
        elif ema < 0.6 * threshold:
            # Shrink: decrease to min curriculum
            x_min, x_max = command_term.shrink_lin_vel_x(step_size=step_size, min_curriculum=min_curriculum)
            y_min, y_max = command_term.shrink_lin_vel_y(step_size=step_size, min_curriculum=min_curriculum)

    return {
        "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
        "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        "mean_tracking_reward": ema,
    }
