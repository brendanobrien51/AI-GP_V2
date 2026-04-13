"""Benchmark ONNX policy inference latency.

Usage:
    python scripts/benchmark_inference.py --model checkpoints/policy.onnx
    python scripts/benchmark_inference.py --model checkpoints/policy.onnx --iterations 5000
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark policy inference")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--provider", default="CUDAExecutionProvider", help="ONNX EP")
    args = parser.parse_args()

    from aigp.deployment.policy_inference import PolicyInferenceEngine

    engine = PolicyInferenceEngine(
        model_path=args.model,
        execution_provider=args.provider,
    )

    results = engine.benchmark(num_iterations=args.iterations)

    print(f"\nInference Benchmark ({args.iterations} iterations)")
    print(f"{'='*45}")
    print(f"  Mean:  {results['mean_ms']:.3f} ms")
    print(f"  Std:   {results['std_ms']:.3f} ms")
    print(f"  Min:   {results['min_ms']:.3f} ms")
    print(f"  Max:   {results['max_ms']:.3f} ms")
    print(f"  Rate:  {results['hz']:.0f} Hz")
    print(f"{'='*45}")

    target_hz = 100
    if results["hz"] >= target_hz:
        print(f"  PASS: {results['hz']:.0f} Hz >= {target_hz} Hz target")
    else:
        print(f"  FAIL: {results['hz']:.0f} Hz < {target_hz} Hz target")


if __name__ == "__main__":
    main()
