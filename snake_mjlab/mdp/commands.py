"""Snake velocity command — virtual chassis velocity tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .virtual_chassis import (
    arrow_quat_from_virtual_velocity,
    compute_virtual_chassis_command_terms,
    quat_from_axes_w,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Virtual chassis body names (matching 14DOF-DW.xml robot structure).
VIRTUAL_CHASSIS_BODY_NAMES = ("base_link",) + tuple(f"link{i}" for i in range(1, 15))


class SnakeVirtualChassisCommand(CommandTerm):
    """Uniform velocity command interpreted in the virtual chassis frame."""

    cfg: "SnakeVirtualChassisCommandCfg"

    def __init__(self, cfg: "SnakeVirtualChassisCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self._body_ids, _ = self.robot.find_bodies(list(cfg.body_names), preserve_order=True)
        self._body_ids = torch.tensor(self._body_ids, device=self.device, dtype=torch.long)
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

        self.current_lin_vel_x_range = list(cfg.ranges.lin_vel_x)
        self.current_lin_vel_y_range = list(cfg.ranges.lin_vel_y)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # --- Debug metrics for virtual chassis analysis ---
        # World frame velocity of base_link
        self.metrics["debug_world_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["debug_world_lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["debug_world_lin_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        # Virtual chassis heading angle (angle between VC x-axis and world x-axis)
        self.metrics["debug_vc_heading_angle"] = torch.zeros(self.num_envs, device=self.device)
        # Virtual chassis frame velocity
        self.metrics["debug_vc_lin_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["debug_vc_lin_vel_y"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _compute_virtual_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos_w = self.robot.data.body_link_pos_w[:, self._body_ids, :]
        body_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self._body_ids, :]
        body_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self._body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return (
                torch.zeros_like(body_pos_w.mean(dim=1)),
                torch.zeros_like(self.prev_axes_w),
                torch.zeros(self.num_envs, 3, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

        origin_w, axes_w, lin_vel_vc, ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(lin_vel_vc).all():
            return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc

    def _update_metrics(self) -> None:
        origin_w, axes_w, lin_vel_vc, ang_vel_z_vc = self._compute_virtual_state()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.norm(
            self.vel_command_b[:, :2] - lin_vel_vc[:, :2], dim=-1
        ) / max_command_step
        self.metrics["error_vel_yaw"] += torch.abs(
            self.vel_command_b[:, 2] - ang_vel_z_vc
        ) / max_command_step

        # --- Debug metrics for virtual chassis analysis ---
        # World frame velocity of base_link
        base_link_vel_w = self.robot.data.body_link_lin_vel_w[:, self._body_ids[0], :]
        self.metrics["debug_world_lin_vel_x"] += base_link_vel_w[:, 0]
        self.metrics["debug_world_lin_vel_y"] += base_link_vel_w[:, 1]
        self.metrics["debug_world_lin_vel_z"] += base_link_vel_w[:, 2]

        # Virtual chassis heading angle (angle between VC x-axis and world x-axis)
        vc_x_axis = axes_w[:, :, 0]  # [N, 3]
        heading_angle = torch.atan2(vc_x_axis[:, 1], vc_x_axis[:, 0])
        self.metrics["debug_vc_heading_angle"] += heading_angle

        # Virtual chassis frame velocity
        self.metrics["debug_vc_lin_vel_x"] += lin_vel_vc[:, 0]
        self.metrics["debug_vc_lin_vel_y"] += lin_vel_vc[:, 1]

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.current_lin_vel_x_range)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.current_lin_vel_y_range)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            _, axes_w, _, _ = self._compute_virtual_state()
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                vc_x_axis_w = axes_w[env_ids, :, 0]
                heading_w = torch.atan2(vc_x_axis_w[:, 1], vc_x_axis_w[:, 0])
                heading_error = wrap_to_pi(self.heading_target[env_ids] - heading_w)
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        small_planar = torch.norm(self.vel_command_b[:, :2], dim=1) <= self.cfg.planar_zero_threshold
        self.vel_command_b[small_planar, :2] = 0.0

    def expand_lin_vel_x(self, step_size: float, max_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_x_range[0] = max(-max_curriculum, self.current_lin_vel_x_range[0] - step_size)
        self.current_lin_vel_x_range[1] = min(max_curriculum, self.current_lin_vel_x_range[1] + step_size)
        return tuple(self.current_lin_vel_x_range)

    def expand_lin_vel_y(self, step_size: float, max_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_y_range[0] = max(-max_curriculum, self.current_lin_vel_y_range[0] - step_size)
        self.current_lin_vel_y_range[1] = min(max_curriculum, self.current_lin_vel_y_range[1] + step_size)
        return tuple(self.current_lin_vel_y_range)

    def shrink_lin_vel_x(self, step_size: float, min_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_x_range[0] = min(-min_curriculum, self.current_lin_vel_x_range[0] + step_size)
        self.current_lin_vel_x_range[1] = max(min_curriculum, self.current_lin_vel_x_range[1] - step_size)
        return tuple(self.current_lin_vel_x_range)

    def shrink_lin_vel_y(self, step_size: float, min_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_y_range[0] = min(-min_curriculum, self.current_lin_vel_y_range[0] + step_size)
        self.current_lin_vel_y_range[1] = max(min_curriculum, self.current_lin_vel_y_range[1] - step_size)
        return tuple(self.current_lin_vel_y_range)

    def reset(self, env_ids=None) -> dict:
        extras = super().reset(env_ids=env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return extras


from dataclasses import dataclass, field


@dataclass(kw_only=True)
class SnakeVirtualChassisCommandCfg(CommandTermCfg):
    entity_name: str
    body_names: tuple[str, ...] = VIRTUAL_CHASSIS_BODY_NAMES
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    planar_zero_threshold: float = 0.2

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float]
        lin_vel_y: tuple[float, float]
        ang_vel_z: tuple[float, float]
        heading: tuple[float, float] | None = None

    ranges: Ranges = field(default_factory=lambda: SnakeVirtualChassisCommandCfg.Ranges(
        lin_vel_x=(-0.1, 0.1),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(0.0, 0.0),
    ))

    def build(self, env: ManagerBasedRlEnv) -> SnakeVirtualChassisCommand:
        return SnakeVirtualChassisCommand(self, env)
