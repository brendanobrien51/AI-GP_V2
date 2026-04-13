"""Training entry point for the drone racing RL agent.

Usage:
    python scripts/train.py --headless --num-envs 512
    python scripts/train.py --config config/train_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Isaac Sim pip install workarounds (must run before any omni/isaacsim imports)
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    import osqp  # noqa: F401 — pre-import to avoid DLL conflict with Omniverse kernel
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drone racing PPO agent")
    parser.add_argument("--config", default="config/train_config.yaml", help="Training config YAML")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=50_000_000, help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    from aigp.utils.config_loader import load_config
    config = load_config(args.config)

    # Override config with CLI args
    train_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})
    env_cfg = config.get("env", {})
    curriculum_cfg = config.get("curriculum", {})

    logger.info("Starting training: %d envs, %d timesteps, device=%s",
                args.num_envs, args.timesteps, args.device)

    # Initialize Isaac Lab
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    from aigp.envs.racing_env import RacingEnv
    from aigp.envs.racing_env_cfg import RacingEnvCfg
    from aigp.agents.ppo_trainer import train

    # Build env config
    env_config = RacingEnvCfg()
    env_config.scene.num_envs = args.num_envs
    env_config.num_gates = curriculum_cfg.get("initial_gates", 3)

    # Create environment
    env = RacingEnv(cfg=env_config)

    # Train
    agent = train(
        env=env,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        initial_gates=curriculum_cfg.get("initial_gates", 3),
        max_gates=curriculum_cfg.get("max_gates", 8),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        clip_ratio=ppo_cfg.get("clip_ratio", 0.2),
        entropy_loss_scale=ppo_cfg.get("entropy_loss_scale", 0.01),
        discount_factor=ppo_cfg.get("discount_factor", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
    )

    logger.info("Training complete.")
    simulation_app.close()


if __name__ == "__main__":
    main()
