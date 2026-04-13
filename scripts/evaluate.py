"""Evaluate a trained policy on racing tracks.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final_agent.pt
    python scripts/evaluate.py --checkpoint checkpoints/final_agent.pt --track circular --num-gates 6
"""

from __future__ import annotations

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained racing policy")
    parser.add_argument("--checkpoint", required=True, help="Agent checkpoint path")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of eval environments")
    parser.add_argument("--num-episodes", type=int, default=100, help="Episodes to evaluate")
    parser.add_argument("--track", default="zigzag", help="Track type: zigzag, split_s, circular")
    parser.add_argument("--num-gates", type=int, default=5, help="Number of gates")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import wrap_env

    from aigp.envs.racing_env import RacingEnv
    from aigp.envs.racing_env_cfg import RacingEnvCfg

    env_config = RacingEnvCfg()
    env_config.scene.num_envs = args.num_envs
    env_config.track_type = args.track
    env_config.num_gates = args.num_gates

    env = RacingEnv(cfg=env_config)
    wrapped_env = wrap_env(env, wrapper="isaaclab")

    # Load agent
    from aigp.agents.asymmetric_ac import create_asymmetric_models
    policy, value = create_asymmetric_models(
        wrapped_env.observation_space, wrapped_env.action_space, args.device,
    )

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=None,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=args.device,
    )
    agent.load(args.checkpoint)

    # Evaluate
    gates_passed_total = 0
    episodes_completed = 0
    total_reward = 0.0

    obs, info = wrapped_env.reset()
    episode_count = 0

    while episode_count < args.num_episodes:
        with torch.no_grad():
            action = agent.act(obs, timestep=0, timesteps=0)[0]

        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward.sum().item()

        done = terminated | truncated
        episode_count += done.sum().item()

    avg_reward = total_reward / max(episode_count, 1)
    logger.info(
        "Evaluation: %d episodes, avg reward: %.2f, track: %s (%d gates)",
        episode_count, avg_reward, args.track, args.num_gates,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
