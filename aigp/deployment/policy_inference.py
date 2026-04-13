"""ONNX Runtime policy inference engine for onboard deployment.

Loads a trained policy (.onnx) and runs inference at >100 Hz using
ONNX Runtime with CUDA or TensorRT execution providers.

Designed for ~100 TOPS onboard compute modules.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PolicyInferenceEngine:
    """High-frequency ONNX policy inference for real-time control.

    Args:
        model_path: Path to the exported .onnx policy file.
        execution_provider: ONNX Runtime EP — "CUDAExecutionProvider",
            "TensorrtExecutionProvider", or "CPUExecutionProvider".
        warmup_iterations: Number of warmup forward passes.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/policy.onnx",
        execution_provider: str = "CUDAExecutionProvider",
        warmup_iterations: int = 10,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy model not found: {path}")

        # Select execution provider
        available_eps = ort.get_available_providers()
        if execution_provider not in available_eps:
            logger.warning(
                "%s not available (have: %s), falling back to CPU",
                execution_provider, available_eps,
            )
            execution_provider = "CPUExecutionProvider"

        self._session = ort.InferenceSession(
            str(path),
            providers=[execution_provider],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape

        # Warmup
        dummy = np.zeros((1, self._input_shape[1]), dtype=np.float32)
        for _ in range(warmup_iterations):
            self._session.run([self._output_name], {self._input_name: dummy})

        logger.info(
            "PolicyInferenceEngine ready: %s on %s (input shape: %s)",
            path.name, execution_provider, self._input_shape,
        )

        # Timing
        self._last_inference_ms = 0.0

    @property
    def last_inference_ms(self) -> float:
        """Latency of the most recent inference in milliseconds."""
        return self._last_inference_ms

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run policy inference on a single observation.

        Args:
            observation: (obs_dim,) or (1, obs_dim) float32 array.

        Returns:
            Action array (4,) — [thrust, roll_rate, pitch_rate, yaw_rate]
            in raw [-1, 1] space.
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        obs = observation.astype(np.float32)

        t0 = time.perf_counter()
        result = self._session.run([self._output_name], {self._input_name: obs})
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        return result[0].squeeze(0)

    def infer_batch(self, observations: np.ndarray) -> np.ndarray:
        """Run policy inference on a batch of observations.

        Args:
            observations: (N, obs_dim) float32 array.

        Returns:
            Actions (N, 4) in raw [-1, 1] space.
        """
        obs = observations.astype(np.float32)

        t0 = time.perf_counter()
        result = self._session.run([self._output_name], {self._input_name: obs})
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        return result[0]

    def benchmark(self, num_iterations: int = 1000) -> dict[str, float]:
        """Benchmark inference latency.

        Args:
            num_iterations: Number of iterations to time.

        Returns:
            Dict with mean_ms, std_ms, min_ms, max_ms, hz.
        """
        dummy = np.zeros((1, self._input_shape[1]), dtype=np.float32)
        times = []

        for _ in range(num_iterations):
            t0 = time.perf_counter()
            self._session.run([self._output_name], {self._input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000.0)

        times_arr = np.array(times)
        mean_ms = float(np.mean(times_arr))

        return {
            "mean_ms": mean_ms,
            "std_ms": float(np.std(times_arr)),
            "min_ms": float(np.min(times_arr)),
            "max_ms": float(np.max(times_arr)),
            "hz": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        }
