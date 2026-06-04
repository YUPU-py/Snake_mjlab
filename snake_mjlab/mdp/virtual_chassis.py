"""Virtual chassis utilities (torch).

This file is a local, mjlab-native implementation of the Virtual Chassis (VC)
frame computation used throughout this repository.

It mirrors the logic used in the IsaacLab-based implementation under:
`source/Snake_Residual/tasks/manager_based/velocity_tracking/mdp/virtual_chassis.py`

Exposed functions are used by:
- `snake_mjlab/mdp/commands.py` (for command metrics + debug)
- (optionally) other modules that want VC axes/quaternions

Important design points:
- Principal axes are computed with SVD on centered body positions.
- Axis sign ambiguity is resolved via:
  1) continuity w.r.t. previous axes (dot-product sign alignment),
  2) initial head-to-tail heuristic for x-axis,
  3) world-up heuristic for z-axis,
  4) right-handedness enforced by determinant check.
"""

from __future__ import annotations

import torch


def compute_virtual_chassis_frame(
    body_pos_w: torch.Tensor,
    prev_axes_w: torch.Tensor | None = None,
    has_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return geometric center and principal axes of the virtual chassis.

    Args:
        body_pos_w: Tensor [B, N, 3] of body positions in world frame.
        prev_axes_w: Optional tensor [B, 3, 3] of previous axes.
        has_prev: Optional bool tensor [B] indicating which envs have valid prev axes.

    Returns:
        origin_w: Tensor [B, 3]
        axes_w: Tensor [B, 3, 3] with columns as x/y/z axes in world frame.
    """

    origin_w = body_pos_w.mean(dim=1)
    centered_body_pos_w = body_pos_w - origin_w.unsqueeze(1)

    # Shape [B, 3, N]. Left singular vectors correspond to principal axes in world frame.
    data_matrix = centered_body_pos_w.transpose(1, 2)
    axes_w, _, _ = torch.linalg.svd(data_matrix, full_matrices=False)
    axes_w = axes_w.clone()

    if prev_axes_w is not None and has_prev is not None:
        prev_mask = has_prev.to(dtype=torch.bool)
    else:
        prev_mask = torch.zeros(body_pos_w.shape[0], dtype=torch.bool, device=body_pos_w.device)

    # Continuity: align sign with previous axes per column.
    if prev_mask.any():
        dots = torch.sum(axes_w[prev_mask] * prev_axes_w[prev_mask], dim=1)
        signs = torch.where(dots >= 0.0, 1.0, -1.0)
        axes_w[prev_mask] = axes_w[prev_mask] * signs.unsqueeze(1)

    # Initialization: choose consistent sign using head-to-tail and world-up heuristics.
    init_mask = ~prev_mask
    if init_mask.any():
        head_to_tail_w = body_pos_w[init_mask, -1] - body_pos_w[init_mask, 0]
        x_dots = torch.sum(axes_w[init_mask, :, 0] * head_to_tail_w, dim=1)
        x_flip = torch.where(x_dots >= 0.0, 1.0, -1.0)
        axes_w[init_mask, :, 0] = axes_w[init_mask, :, 0] * x_flip.unsqueeze(1)

        world_up = torch.zeros(int(init_mask.sum().item()), 3, device=body_pos_w.device, dtype=body_pos_w.dtype)
        world_up[:, 2] = 1.0
        z_dots = torch.sum(axes_w[init_mask, :, 2] * world_up, dim=1)
        z_flip = torch.where(z_dots >= 0.0, 1.0, -1.0)
        axes_w[init_mask, :, 2] = axes_w[init_mask, :, 2] * z_flip.unsqueeze(1)

    # Enforce right-handedness.
    det_mask = torch.det(axes_w) < 0.0
    if det_mask.any():
        axes_w[det_mask, :, 1] = -axes_w[det_mask, :, 1]

    return origin_w, axes_w


def project_world_vector_to_virtual_frame(vector_w: torch.Tensor, axes_w: torch.Tensor) -> torch.Tensor:
    """Project a world-frame vector into the virtual chassis frame.

    Args:
        vector_w: Tensor [B, 3]
        axes_w: Tensor [B, 3, 3]

    Returns:
        vector_vc: Tensor [B, 3]
    """

    return torch.einsum("bij,bj->bi", axes_w.transpose(1, 2), vector_w)


def compute_virtual_chassis_command_terms(
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    prev_axes_w: torch.Tensor | None = None,
    has_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VC frame and (mean) linear/angular velocity expressed in that frame.

    Returns:
        origin_w: [B, 3]
        axes_w: [B, 3, 3]
        lin_vel_vc: [B, 3]
        ang_vel_z_vc: [B]
    """

    origin_w, axes_w = compute_virtual_chassis_frame(body_pos_w, prev_axes_w, has_prev)

    # Keep a conservative definition here (mean across bodies) for command-side metrics.
    # Reward-side tracking may choose a different velocity proxy (e.g., base_link world velocity).
    vc_lin_vel_w = body_lin_vel_w.mean(dim=1)
    vc_ang_vel_w = body_ang_vel_w.mean(dim=1)

    actual_lin_vel_vc = project_world_vector_to_virtual_frame(vc_lin_vel_w, axes_w)
    actual_ang_vel_vc = project_world_vector_to_virtual_frame(vc_ang_vel_w, axes_w)
    return origin_w, axes_w, actual_lin_vel_vc, actual_ang_vel_vc[:, 2]


def quat_from_axes_w(axes_w: torch.Tensor) -> torch.Tensor:
    """Convert batched orthonormal axes matrices to quaternions in (w, x, y, z).

    This implementation avoids external dependencies by using a standard
    matrix-to-quaternion conversion.

    Args:
        axes_w: [B, 3, 3]

    Returns:
        quat_wxyz: [B, 4]
    """

    m = axes_w
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    qw = torch.empty(m.shape[0], device=m.device, dtype=m.dtype)
    qx = torch.empty_like(qw)
    qy = torch.empty_like(qw)
    qz = torch.empty_like(qw)

    cond = tr > 0.0
    if cond.any():
        S = torch.sqrt(tr[cond] + 1.0) * 2.0
        qw[cond] = 0.25 * S
        qx[cond] = (m[cond, 2, 1] - m[cond, 1, 2]) / S
        qy[cond] = (m[cond, 0, 2] - m[cond, 2, 0]) / S
        qz[cond] = (m[cond, 1, 0] - m[cond, 0, 1]) / S

    cond = ~cond
    if cond.any():
        m00 = m[cond, 0, 0]
        m11 = m[cond, 1, 1]
        m22 = m[cond, 2, 2]

        cond0 = (m00 > m11) & (m00 > m22)
        cond1 = (~cond0) & (m11 > m22)
        cond2 = (~cond0) & (~cond1)

        idx0 = cond.nonzero(as_tuple=False).flatten()[cond0]
        idx1 = cond.nonzero(as_tuple=False).flatten()[cond1]
        idx2 = cond.nonzero(as_tuple=False).flatten()[cond2]

        if idx0.numel() > 0:
            S = torch.sqrt(1.0 + m[idx0, 0, 0] - m[idx0, 1, 1] - m[idx0, 2, 2]) * 2.0
            qw[idx0] = (m[idx0, 2, 1] - m[idx0, 1, 2]) / S
            qx[idx0] = 0.25 * S
            qy[idx0] = (m[idx0, 0, 1] + m[idx0, 1, 0]) / S
            qz[idx0] = (m[idx0, 0, 2] + m[idx0, 2, 0]) / S

        if idx1.numel() > 0:
            S = torch.sqrt(1.0 + m[idx1, 1, 1] - m[idx1, 0, 0] - m[idx1, 2, 2]) * 2.0
            qw[idx1] = (m[idx1, 0, 2] - m[idx1, 2, 0]) / S
            qx[idx1] = (m[idx1, 0, 1] + m[idx1, 1, 0]) / S
            qy[idx1] = 0.25 * S
            qz[idx1] = (m[idx1, 1, 2] + m[idx1, 2, 1]) / S

        if idx2.numel() > 0:
            S = torch.sqrt(1.0 + m[idx2, 2, 2] - m[idx2, 0, 0] - m[idx2, 1, 1]) * 2.0
            qw[idx2] = (m[idx2, 1, 0] - m[idx2, 0, 1]) / S
            qx[idx2] = (m[idx2, 0, 2] + m[idx2, 2, 0]) / S
            qy[idx2] = (m[idx2, 1, 2] + m[idx2, 2, 1]) / S
            qz[idx2] = 0.25 * S

    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    quat = quat / (torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-12))
    return quat


def arrow_quat_from_virtual_velocity(axes_w: torch.Tensor, velocity_vc: torch.Tensor) -> torch.Tensor:
    """Build world-frame arrow quaternions for planar velocities expressed in VC frame."""

    yaw = torch.atan2(velocity_vc[:, 1], velocity_vc[:, 0])
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    local_rot = torch.zeros(velocity_vc.shape[0], 3, 3, device=velocity_vc.device, dtype=axes_w.dtype)
    local_rot[:, 0, 0] = cos_yaw
    local_rot[:, 0, 1] = -sin_yaw
    local_rot[:, 1, 0] = sin_yaw
    local_rot[:, 1, 1] = cos_yaw
    local_rot[:, 2, 2] = 1.0
    arrow_axes_w = torch.bmm(axes_w, local_rot)
    return quat_from_axes_w(arrow_axes_w)
