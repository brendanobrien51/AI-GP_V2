"""Observation computation for the drone-racing environment.

Three observation groups are produced every step:

1. **policy** (actor) -- 13-D vector:
       angular_velocity_body (3)
     + gravity_body_frame    (3)
     + relative_gate_centroid_body (3)
     + previous_action       (4)

2. **critic** (privileged) -- actor obs + 18-D extras:
       world_position         (3)
     + world_linear_velocity  (3)
     + all_gate_positions     (up to 4 gates x 3 = 12)

3. **image** -- 80x80x3 uint8 from the TiledCamera (kept separate for
   the visual encoder branch).

All vector math is JIT-compiled; the orchestration function is called from
``RacingEnv._get_observations()``.
"""

from __future__ import annotations

import torch

from aigp.utils.math_utils import quat_rotate_inverse, quat_to_gravity_body, world_to_body

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTOR_OBS_DIM: int = 13
CRITIC_EXTRA_DIM: int = 18   # 3 + 3 + 12
NUM_PRIVILEGED_GATES: int = 4  # number of gate positions packed into critic obs


@torch.jit.script
def compute_actor_obs(
    ang_vel_world: torch.Tensor,
    quat_wxyz: torch.Tensor,
    drone_pos: torch.Tensor,
    next_gate_pos: torch.Tensor,
    prev_action: torch.Tensor,
) -> torch.Tensor:
    """Build the 13-D actor observation vector.

    All inputs have batch dimension N (number of parallel envs).

    Args:
        ang_vel_world: (N, 3) angular velocity in world frame.
        quat_wxyz:     (N, 4) drone orientation quaternion (w, x, y, z).
        drone_pos:     (N, 3) drone world position.
        next_gate_pos: (N, 3) next gate centroid in world frame.
        prev_action:   (N, 4) previous CTBR action (raw, [-1, 1]).

    Returns:
        (N, 13) observation tensor.
    """
    # Angular velocity in body frame
    ang_vel_body = quat_rotate_inverse(quat_wxyz, ang_vel_world)  # (N, 3)

    # Gravity direction in body frame (unit vector)
    gravity_body = quat_to_gravity_body(quat_wxyz)                # (N, 3)

    # Relative gate position in body frame
    rel_gate_body = world_to_body(next_gate_pos, drone_pos, quat_wxyz)  # (N, 3)

    return torch.cat([ang_vel_body, gravity_body, rel_gate_body, prev_action], dim=-1)


@torch.jit.script
def compute_critic_obs(
    actor_obs: torch.Tensor,
    drone_pos: torch.Tensor,
    lin_vel_world: torch.Tensor,
    all_gate_positions: torch.Tensor,
    num_privileged_gates: int = 4,
) -> torch.Tensor:
    """Build the privileged critic observation vector.

    Args:
        actor_obs:          (N, 13) actor observation.
        drone_pos:          (N, 3) world position.
        lin_vel_world:      (N, 3) world linear velocity.
        all_gate_positions: (N, G, 3) world positions of up to G gates.
        num_privileged_gates: Number of gate positions to include.

    Returns:
        (N, 13 + 18) = (N, 31) critic observation tensor.
    """
    N = actor_obs.shape[0]
    device = actor_obs.device

    # Flatten gate positions — pad/truncate to exactly num_privileged_gates
    G = all_gate_positions.shape[1]
    if G >= num_privileged_gates:
        gate_flat = all_gate_positions[:, :num_privileged_gates, :].reshape(N, num_privileged_gates * 3)
    else:
        # Pad with zeros for missing gates
        pad_size = num_privileged_gates - G
        padding = torch.zeros(N, pad_size, 3, device=device, dtype=all_gate_positions.dtype)
        padded = torch.cat([all_gate_positions, padding], dim=1)
        gate_flat = padded.reshape(N, num_privileged_gates * 3)

    return torch.cat([actor_obs, drone_pos, lin_vel_world, gate_flat], dim=-1)


def compute_observations(
    ang_vel_world: torch.Tensor,
    quat_wxyz: torch.Tensor,
    drone_pos: torch.Tensor,
    lin_vel_world: torch.Tensor,
    next_gate_pos: torch.Tensor,
    prev_action: torch.Tensor,
    all_gate_positions: torch.Tensor,
    camera_images: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute full observation dictionary for the racing environment.

    Args:
        ang_vel_world:      (N, 3) angular velocity in world frame.
        quat_wxyz:          (N, 4) orientation quaternion (w,x,y,z).
        drone_pos:          (N, 3) world position.
        lin_vel_world:      (N, 3) world linear velocity.
        next_gate_pos:      (N, 3) next target gate centroid (world).
        prev_action:        (N, 4) previous action.
        all_gate_positions: (N, G, 3) world gate positions.
        camera_images:      (N, H, W, 3) optional RGB uint8 images.

    Returns:
        Dictionary with keys ``"policy"`` and ``"critic"``, and optionally
        ``"image"`` if camera data is supplied.
    """
    actor_obs = compute_actor_obs(
        ang_vel_world, quat_wxyz, drone_pos, next_gate_pos, prev_action,
    )
    critic_obs = compute_critic_obs(
        actor_obs, drone_pos, lin_vel_world, all_gate_positions,
    )

    obs_dict: dict[str, torch.Tensor] = {
        "policy": actor_obs,
        "critic": critic_obs,
    }

    if camera_images is not None:
        # Normalize to [0, 1] float32 for the visual encoder
        obs_dict["image"] = camera_images.float() / 255.0

    return obs_dict
