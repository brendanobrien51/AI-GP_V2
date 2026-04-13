"""Episode termination and truncation conditions for drone racing.

Terminated (failure):
    - Collision with ground or gate frame
    - Out of bounds (geofence violation)

Truncated (timeout):
    - Episode step limit exceeded (500 steps default)
"""

from __future__ import annotations

import torch


@torch.jit.script
def check_collision(
    contact_forces: torch.Tensor,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Detect collisions from contact force sensor data.

    Args:
        contact_forces: Contact force magnitudes (N, num_bodies).
        force_threshold: Minimum force to count as collision (N).

    Returns:
        Boolean tensor (N,) — True where collision detected.
    """
    max_force = contact_forces.max(dim=-1).values
    return max_force > force_threshold


@torch.jit.script
def check_out_of_bounds(
    position: torch.Tensor,
    geofence_radius: float = 100.0,
    min_altitude: float = 0.3,
    max_altitude: float = 15.0,
) -> torch.Tensor:
    """Check if drone has left the allowed flight volume.

    The flight volume is a cylinder centred at the origin:
    horizontal radius <= geofence_radius, altitude in [min, max].

    Args:
        position: Drone position (N, 3) in world frame.
        geofence_radius: Maximum horizontal distance from origin (m).
        min_altitude: Minimum altitude AGL (m).
        max_altitude: Maximum altitude AGL (m).

    Returns:
        Boolean tensor (N,) — True where out of bounds.
    """
    horizontal_dist = torch.norm(position[:, :2], dim=-1)
    altitude = position[:, 2]

    too_far = horizontal_dist > geofence_radius
    too_low = altitude < min_altitude
    too_high = altitude > max_altitude

    return too_far | too_low | too_high


@torch.jit.script
def check_timeout(
    step_count: torch.Tensor,
    max_steps: int = 500,
) -> torch.Tensor:
    """Check if episode has exceeded the step limit.

    Args:
        step_count: Per-env step counter (N,).
        max_steps: Maximum steps per episode.

    Returns:
        Boolean tensor (N,) — True where timed out.
    """
    return step_count >= max_steps


@torch.jit.script
def compute_terminations(
    contact_forces: torch.Tensor,
    position: torch.Tensor,
    step_count: torch.Tensor,
    force_threshold: float = 1.0,
    geofence_radius: float = 100.0,
    min_altitude: float = 0.3,
    max_altitude: float = 15.0,
    max_steps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute both terminated and truncated flags.

    Args:
        contact_forces: Contact force magnitudes (N, num_bodies).
        position: Drone position (N, 3).
        step_count: Per-env step counter (N,).
        force_threshold: Collision force threshold (N).
        geofence_radius: Horizontal geofence radius (m).
        min_altitude: Minimum altitude (m).
        max_altitude: Maximum altitude (m).
        max_steps: Episode length limit.

    Returns:
        terminated: (N,) True for failure terminations (collision, OOB).
        truncated: (N,) True for timeout truncations.
    """
    collided = check_collision(contact_forces, force_threshold)
    oob = check_out_of_bounds(position, geofence_radius, min_altitude, max_altitude)

    terminated = collided | oob
    truncated = check_timeout(step_count, max_steps) & ~terminated

    return terminated, truncated
