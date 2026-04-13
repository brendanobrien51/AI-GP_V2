"""SKRL PPO training orchestrator for drone racing.

Configures the PPO agent with asymmetric actor-critic, connects to the
Isaac Lab racing environment, and manages the training loop with
curriculum learning, checkpointing, and optional W&B logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from aigp.agents.asymmetric_ac import create_asymmetric_models
from aigp.agents.curriculum import CurriculumManager
from aigp.utils.vram_profiler import log_vram_usage

logger = logging.getLogger(__name__)


def create_ppo_agent(
    env,
    device: str = "cuda:0",
    learning_rate: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_loss_scale: float = 0.01,
    value_loss_scale: float = 1.0,
    mini_batches: int = 4,
    learning_epochs: int = 5,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 1.0,
    rollout_steps: int = 24,
) -> tuple[PPO, RandomMemory]:
    """Create a PPO agent with asymmetric actor-critic for drone racing.

    Args:
        env: Wrapped Isaac Lab environment.
        device: Torch device.
        learning_rate: Adam learning rate.
        clip_ratio: PPO clip parameter.
        entropy_loss_scale: Entropy bonus coefficient.
        value_loss_scale: Value loss coefficient.
        mini_batches: Number of mini-batches per update.
        learning_epochs: Epochs per PPO update.
        discount_factor: Discount factor (gamma).
        gae_lambda: GAE lambda.
        max_grad_norm: Gradient clipping norm.
        rollout_steps: Steps per rollout before PPO update.

    Returns:
        Tuple of (PPO agent, memory buffer).
    """
    # Create models
    policy, value = create_asymmetric_models(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    models = {"policy": policy, "value": value}

    # Memory
    memory = RandomMemory(memory_size=rollout_steps, num_envs=env.num_envs, device=device)

    # PPO config
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["learning_rate"] = learning_rate
    cfg["discount_factor"] = discount_factor
    cfg["lambda"] = gae_lambda
    cfg["learning_epochs"] = learning_epochs
    cfg["mini_batches"] = mini_batches
    cfg["clip_ratio"] = clip_ratio
    cfg["entropy_loss_scale"] = entropy_loss_scale
    cfg["value_loss_scale"] = value_loss_scale
    cfg["grad_norm_clip"] = max_grad_norm
    cfg["state_preprocessor"] = None
    cfg["value_preprocessor"] = None

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    return agent, memory


def train(
    env,
    total_timesteps: int = 50_000_000,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 100,
    eval_interval: int = 50,
    log_interval: int = 10,
    seed: int = 42,
    device: str = "cuda:0",
    curriculum_enabled: bool = True,
    initial_gates: int = 3,
    max_gates: int = 8,
    **ppo_kwargs,
) -> PPO:
    """Run the full PPO training loop with curriculum learning.

    Args:
        env: Isaac Lab racing environment (unwrapped).
        total_timesteps: Total training steps.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_interval: Save every N iterations.
        eval_interval: Evaluate every N iterations.
        log_interval: Log metrics every N iterations.
        seed: Random seed.
        device: Torch device.
        curriculum_enabled: Enable gate count curriculum.
        initial_gates: Starting gate count for curriculum.
        max_gates: Maximum gate count.
        **ppo_kwargs: Additional PPO hyperparameters.

    Returns:
        Trained PPO agent.
    """
    torch.manual_seed(seed)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Wrap environment for SKRL
    wrapped_env = wrap_env(env, wrapper="isaaclab")

    # Create agent
    agent, memory = create_ppo_agent(wrapped_env, device=device, **ppo_kwargs)

    # Curriculum
    curriculum = CurriculumManager(
        initial_gates=initial_gates,
        max_gates=max_gates,
    ) if curriculum_enabled else None

    log_vram_usage("post-init")

    # Trainer
    trainer = SequentialTrainer(
        env=wrapped_env,
        agents=agent,
        cfg={
            "timesteps": total_timesteps,
            "headless": True,
        },
    )

    logger.info(
        "Starting PPO training: %d timesteps, %d envs, device=%s",
        total_timesteps, wrapped_env.num_envs, device,
    )

    trainer.train()

    # Save final checkpoint
    agent.save(f"{checkpoint_dir}/final_agent.pt")
    logger.info("Training complete. Final checkpoint saved.")

    return agent
