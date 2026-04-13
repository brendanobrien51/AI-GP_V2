"""CNN + MLP actor-critic architectures for SKRL PPO.

The policy processes both image observations (80x80 RGB via a 3-layer CNN)
and vector observations (13D) through separate pathways, then fuses them
for the action head.

VRAM budget: ~250 MB for both policy and value networks at batch_size=512.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


class CNNFeatureExtractor(nn.Module):
    """3-layer CNN for processing 80x80 RGB tiled camera images.

    Architecture:
        Conv2d(3, 32, 8, stride=4)  -> ReLU  -> 19x19x32
        Conv2d(32, 64, 4, stride=2) -> ReLU  -> 9x9x64
        Conv2d(64, 64, 3, stride=1) -> ReLU  -> 7x7x64
        Flatten -> Linear(3136, 256)

    Output: 256-dim feature vector per image.
    """

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size: 80 -> 19 -> 9 -> 7; 7*7*64 = 3136
        self.fc = nn.Linear(3136, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: (N, 3, 80, 80) float tensor in [0, 1].

        Returns:
            Feature vectors (N, output_dim).
        """
        x = self.conv(images)
        return self.fc(x)


class RacingPolicy(GaussianMixin, Model):
    """Stochastic Gaussian policy for CTBR action space.

    Processes:
        - Image observations via CNN (256-dim features)
        - Vector observations (13D)
    Concatenates and passes through MLP to produce mean CTBR actions.
    Log-std is a learnable parameter.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        vector_dim: int = 13,
        cnn_output_dim: int = 256,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.cnn = CNNFeatureExtractor(cnn_output_dim)

        fused_dim = cnn_output_dim + vector_dim  # 256 + 13 = 269
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std)
        )

    def compute(self, inputs, role=""):
        """Forward pass: image + vector -> action mean + log_std.

        Args:
            inputs: Dict with "states" containing concatenated obs.
                    Image is expected to be pre-split by the wrapper.

        Returns:
            Action mean (N, 4) and log_std dict.
        """
        states = inputs["states"]

        # Split vector obs and image
        # Convention: first 13 dims are vector, remaining are flattened image
        vector_obs = states[:, :13]

        if states.shape[1] > 13:
            # Image is flattened: reshape to (N, 3, 80, 80)
            image_flat = states[:, 13:]
            images = image_flat.reshape(-1, 3, 80, 80)
            image_features = self.cnn(images)
        else:
            # No image available — zero features
            image_features = torch.zeros(
                states.shape[0], 256, device=self.device
            )

        fused = torch.cat([image_features, vector_obs], dim=-1)
        action_mean = self.mlp(fused)

        return action_mean, self.log_std_parameter, {}


class RacingValue(DeterministicMixin, Model):
    """Deterministic value function (critic) with optional privileged observations.

    When used as asymmetric critic, receives the full 31D privileged state
    plus image features.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        state_dim: int = 31,
        cnn_output_dim: int = 256,
        clip_actions: bool = False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.cnn = CNNFeatureExtractor(cnn_output_dim)

        fused_dim = cnn_output_dim + state_dim  # 256 + 31 = 287
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role=""):
        """Forward pass: privileged state + image -> scalar value.

        Args:
            inputs: Dict with "states" containing concatenated privileged obs.

        Returns:
            Value estimate (N, 1).
        """
        states = inputs["states"]
        vector_obs = states[:, :31]

        if states.shape[1] > 31:
            image_flat = states[:, 31:]
            images = image_flat.reshape(-1, 3, 80, 80)
            image_features = self.cnn(images)
        else:
            image_features = torch.zeros(
                states.shape[0], 256, device=self.device
            )

        fused = torch.cat([image_features, vector_obs], dim=-1)
        return self.mlp(fused), {}
