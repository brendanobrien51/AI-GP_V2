"""Export a trained agent's policy to ONNX for deployment.

Usage:
    python scripts/export.py --checkpoint checkpoints/final_agent.pt --output checkpoints/policy.onnx
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export policy to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Agent checkpoint path")
    parser.add_argument("--output", default="checkpoints/policy.onnx", help="Output ONNX path")
    parser.add_argument("--no-image", action="store_true", help="Export vector-only policy")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from aigp.agents.export_onnx import export_policy_onnx

    # Reconstruct agent to load checkpoint
    import gymnasium as gym
    from skrl.agents.torch.ppo import PPO
    from aigp.agents.asymmetric_ac import create_asymmetric_models

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(13 + 3*80*80,))
    act_space = gym.spaces.Box(low=-1, high=1, shape=(4,))

    policy, value = create_asymmetric_models(obs_space, act_space, args.device)
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=None,
        observation_space=obs_space,
        action_space=act_space,
        device=args.device,
    )
    agent.load(args.checkpoint)

    path = export_policy_onnx(
        agent,
        output_path=args.output,
        include_image=not args.no_image,
    )
    logger.info("Exported to %s", path)


if __name__ == "__main__":
    main()
