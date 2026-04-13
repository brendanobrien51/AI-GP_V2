"""Teacher-student policy distillation (RSL-RL inspired).

Pipeline:
    1. Train a privileged teacher policy with full state access
    2. Collect teacher rollouts
    3. Train a student policy (vision-only) via behavior cloning
    4. Fine-tune the student with RL (PPO)

This bridges the gap between the privileged training environment
and the sensor-limited real-world deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def collect_teacher_rollouts(
    env,
    teacher_agent,
    num_steps: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect observation-action pairs from a trained teacher.

    Args:
        env: Isaac Lab environment.
        teacher_agent: Trained SKRL agent with privileged access.
        num_steps: Number of steps to collect.

    Returns:
        Tuple of (actor_obs, images, teacher_actions), each (N, ...).
    """
    actor_obs_list = []
    image_list = []
    action_list = []

    obs, info = env.reset()
    for _ in range(num_steps):
        with torch.no_grad():
            action = teacher_agent.act(obs, timestep=0, timesteps=0)[0]

        actor_obs_list.append(obs["policy"].clone())
        if "image" in obs:
            image_list.append(obs["image"].clone())
        action_list.append(action.clone())

        obs, _, terminated, truncated, info = env.step(action)
        if (terminated | truncated).any():
            obs, info = env.reset()

    actor_obs = torch.cat(actor_obs_list, dim=0)
    actions = torch.cat(action_list, dim=0)

    if image_list:
        images = torch.cat(image_list, dim=0)
    else:
        images = torch.empty(0)

    return actor_obs, images, actions


def distill_student(
    student_model: nn.Module,
    actor_obs: torch.Tensor,
    images: torch.Tensor,
    teacher_actions: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    save_path: str = "checkpoints/student_bc.pt",
) -> float:
    """Behavior cloning: train student to mimic teacher actions.

    Args:
        student_model: Student policy model (vision-only).
        actor_obs: Actor vector observations (N, 13).
        images: Image observations (N, 3, 80, 80).
        teacher_actions: Teacher action labels (N, 4).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        save_path: Path to save best student checkpoint.

    Returns:
        Final training loss.
    """
    device = next(student_model.parameters()).device

    if images.numel() > 0:
        # Concatenate vector + flattened image
        image_flat = images.reshape(images.shape[0], -1)
        full_obs = torch.cat([actor_obs, image_flat], dim=-1)
    else:
        full_obs = actor_obs

    dataset = TensorDataset(full_obs.to(device), teacher_actions.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            inputs = {"states": obs_batch}
            pred_mean, _, _ = student_model.compute(inputs)
            loss = loss_fn(pred_mean, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(student_model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            logger.info("Distillation epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss)

    logger.info("Distillation complete — best loss: %.6f, saved to %s", best_loss, save_path)
    return best_loss
