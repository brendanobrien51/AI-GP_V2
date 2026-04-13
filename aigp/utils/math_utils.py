"""Quaternion operations and coordinate frame transforms for drone racing."""

from __future__ import annotations

import torch


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz convention).

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) order.
        v: Vectors of shape (..., 3).

    Returns:
        Rotated vectors of shape (..., 3).
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q (wxyz convention)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v - q_w * t + torch.cross(q_vec, t, dim=-1)


@torch.jit.script
def quat_to_gravity_body(q: torch.Tensor) -> torch.Tensor:
    """Compute gravity vector in body frame from quaternion (wxyz).

    Returns:
        Gravity direction in body frame, shape (..., 3). Normalized.
    """
    gravity_world = torch.zeros_like(q[..., :3])
    gravity_world[..., 2] = -1.0  # NED: gravity points down (-Z in world)
    return quat_rotate_inverse(q, gravity_world)


@torch.jit.script
def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion (wxyz convention).

    Returns:
        Yaw angle in radians, shape (..., 1).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw.unsqueeze(-1)


@torch.jit.script
def world_to_body(pos_world: torch.Tensor, ref_pos: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
    """Transform a world-frame position into the body frame of the reference.

    Args:
        pos_world: Target positions (..., 3) in world frame.
        ref_pos: Reference positions (..., 3) in world frame.
        ref_quat: Reference orientations (..., 4) wxyz.

    Returns:
        Relative position in body frame (..., 3).
    """
    delta = pos_world - ref_pos
    return quat_rotate_inverse(ref_quat, delta)


@torch.jit.script
def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))
