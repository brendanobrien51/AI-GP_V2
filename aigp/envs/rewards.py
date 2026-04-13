"""JIT-compiled reward functions for drone racing.

All functions operate on batched tensors (N environments) and are
torch.jit.script compiled for maximum throughput on GPU.

Reward budget (per step):
    progress     : +1.0 * distance_reduction   (continuous)
    gate_pass    : +10.0                        (sparse)
    completion   : +50.0                        (sparse, all gates passed)
    time_penalty : -0.002                       (continuous)
    smoothness   : -0.01 * ||action_delta||     (continuous)
    collision    : -5.0                          (terminal)
"""

from __future__ import annotations

import torch


@torch.jit.script
def progress_reward(
    dist_to_gate: torch.Tensor,
    prev_dist_to_gate: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for reducing distance to the next gate centroid.

    Args:
        dist_to_gate: Current distance to next gate (N,).
        prev_dist_to_gate: Previous distance to next gate (N,).
        scale: Reward scaling factor.

    Returns:
        Reward tensor (N,). Positive when drone moves closer.
    """
    return scale * (prev_dist_to_gate - dist_to_gate)


@torch.jit.script
def gate_pass_bonus(
    passed_gate: torch.Tensor,
    bonus: float = 10.0,
) -> torch.Tensor:
    """Sparse bonus when drone passes through a gate plane.

    Args:
        passed_gate: Boolean mask (N,) — True if gate was passed this step.
        bonus: Reward value for passing a gate.

    Returns:
        Reward tensor (N,).
    """
    return bonus * passed_gate.float()


@torch.jit.script
def course_completion_bonus(
    all_gates_passed: torch.Tensor,
    bonus: float = 50.0,
) -> torch.Tensor:
    """Bonus when all gates in the course have been passed.

    Args:
        all_gates_passed: Boolean mask (N,) — True if entire course completed.
        bonus: Reward value.

    Returns:
        Reward tensor (N,).
    """
    return bonus * all_gates_passed.float()


@torch.jit.script
def time_penalty(
    num_envs: int,
    penalty: float = -0.002,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    """Constant per-step time penalty to encourage speed.

    Args:
        num_envs: Number of environments.
        penalty: Negative reward per step.
        device: Torch device.

    Returns:
        Reward tensor (N,) filled with penalty value.
    """
    return torch.full((num_envs,), penalty, device=device)


@torch.jit.script
def smoothness_penalty(
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    scale: float = -0.01,
) -> torch.Tensor:
    """Penalize jerk in CTBR commands to reduce motor wear and battery sag.

    Args:
        current_action: Current action (N, 4).
        previous_action: Previous action (N, 4).
        scale: Negative scaling factor.

    Returns:
        Reward tensor (N,). Always <= 0.
    """
    action_delta = current_action - previous_action
    jerk = torch.norm(action_delta, dim=-1)
    return scale * jerk


@torch.jit.script
def collision_penalty(
    collided: torch.Tensor,
    penalty: float = -5.0,
) -> torch.Tensor:
    """Terminal penalty for collisions with ground, gates, or obstacles.

    Args:
        collided: Boolean mask (N,) — True if drone collided.
        penalty: Negative reward value.

    Returns:
        Reward tensor (N,).
    """
    return penalty * collided.float()


@torch.jit.script
def compute_total_reward(
    dist_to_gate: torch.Tensor,
    prev_dist_to_gate: torch.Tensor,
    passed_gate: torch.Tensor,
    all_gates_passed: torch.Tensor,
    collided: torch.Tensor,
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    progress_scale: float = 1.0,
    gate_bonus: float = 10.0,
    completion_bonus: float = 50.0,
    time_pen: float = -0.002,
    smooth_scale: float = -0.01,
    collision_pen: float = -5.0,
) -> torch.Tensor:
    """Compute total reward as sum of all components.

    Args:
        dist_to_gate: Current distance to next gate (N,).
        prev_dist_to_gate: Previous distance to next gate (N,).
        passed_gate: Boolean (N,) — gate passed this step.
        all_gates_passed: Boolean (N,) — course completed.
        collided: Boolean (N,) — collision occurred.
        current_action: Current CTBR action (N, 4).
        previous_action: Previous CTBR action (N, 4).
        progress_scale: Scale for progress reward.
        gate_bonus: Gate pass reward.
        completion_bonus: Course completion reward.
        time_pen: Per-step time penalty.
        smooth_scale: Smoothness penalty scale.
        collision_pen: Collision penalty.

    Returns:
        Total reward tensor (N,).
    """
    n = dist_to_gate.shape[0]

    r = progress_scale * (prev_dist_to_gate - dist_to_gate)
    r = r + gate_bonus * passed_gate.float()
    r = r + completion_bonus * all_gates_passed.float()
    r = r + time_pen
    r = r + smooth_scale * torch.norm(current_action - previous_action, dim=-1)
    r = r + collision_pen * collided.float()

    return r
