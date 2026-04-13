"""ONNX export utility for trained SKRL policies.

Exports the policy network to ONNX format for deployment on the
onboard compute module via ONNX Runtime or TensorRT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_policy_onnx(
    agent,
    output_path: str = "checkpoints/policy.onnx",
    vector_dim: int = 13,
    image_size: tuple[int, int] = (80, 80),
    include_image: bool = True,
    opset_version: int = 17,
) -> Path:
    """Export a trained SKRL agent's policy to ONNX.

    Args:
        agent: Trained SKRL agent with a policy model.
        output_path: Output .onnx file path.
        vector_dim: Vector observation dimension.
        image_size: (H, W) of input image.
        include_image: Whether the model expects image input.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    policy = agent.policy
    policy.set_mode("eval")
    device = next(policy.parameters()).device

    # Build example input
    if include_image:
        h, w = image_size
        obs_dim = vector_dim + 3 * h * w
    else:
        obs_dim = vector_dim

    example_obs = torch.zeros(1, obs_dim, device=device)

    # Try SKRL built-in export first
    try:
        agent.export(str(path), format="onnx")
        logger.info("Exported policy via SKRL to %s", path)
        return path
    except (AttributeError, TypeError):
        logger.info("SKRL export unavailable, using torch.onnx.export")

    # Manual export via wrapper module
    class _PolicyForward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            inputs = {"states": obs}
            mean, _, _ = self.model.compute(inputs)
            return mean

    wrapper = _PolicyForward(policy)
    wrapper.train(False)

    torch.onnx.export(
        wrapper,
        example_obs,
        str(path),
        opset_version=opset_version,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    logger.info("Exported policy to %s (opset %d)", path, opset_version)
    return path
