"""Asymmetric actor-critic configuration for SKRL PPO.

The actor observes only what is available on real hardware:
    - Angular velocity (3)
    - Gravity in body frame (3)
    - Relative gate position (3)
    - Previous action (4)
    + 80x80 RGB image

The critic additionally observes privileged simulation state:
    - Exact position (3)
    - Exact velocity (3)
    - All upcoming gate positions (up to 4*3=12)

This asymmetry allows the value function to learn faster while the
policy remains deployable on real hardware.
"""

from __future__ import annotations

from aigp.agents.models import RacingPolicy, RacingValue


def create_asymmetric_models(
    observation_space,
    action_space,
    device,
    actor_vector_dim: int = 13,
    critic_vector_dim: int = 31,
    cnn_output_dim: int = 256,
) -> tuple[RacingPolicy, RacingValue]:
    """Create asymmetric actor-critic model pair.

    Args:
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
        device: Torch device.
        actor_vector_dim: Actor vector observation dimension (13).
        critic_vector_dim: Critic privileged observation dimension (31).
        cnn_output_dim: CNN feature extractor output dimension.

    Returns:
        Tuple of (policy, value) models.
    """
    policy = RacingPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        vector_dim=actor_vector_dim,
        cnn_output_dim=cnn_output_dim,
    )

    value = RacingValue(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        state_dim=critic_vector_dim,
        cnn_output_dim=cnn_output_dim,
    )

    return policy, value
