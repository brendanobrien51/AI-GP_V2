"""Launch the full deployment stack on real hardware.

Usage:
    python scripts/deploy.py --model checkpoints/policy.onnx
    python scripts/deploy.py --model checkpoints/policy.onnx --config config/deploy_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    _running = False
    logger.info("Shutdown signal received")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy racing policy on hardware")
    parser.add_argument("--model", required=True, help="ONNX policy path")
    parser.add_argument("--config", default="config/deploy_config.yaml", help="Deploy config")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending commands")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)

    from aigp.utils.config_loader import load_config
    from aigp.deployment.policy_inference import PolicyInferenceEngine
    from aigp.deployment.safety_monitor import SafetyMonitor
    from aigp.deployment.px4_bridge import PX4Bridge
    from aigp.perception.gate_detector import GateDetector
    from aigp.perception.pnp_localizer import PnPLocalizer
    from aigp.perception.vio_ekf import VIOStateEstimator
    from aigp.perception.image_preprocessor import ImagePreprocessor

    config = load_config(args.config)
    inference_cfg = config.get("inference", {})
    safety_cfg = config.get("safety", {})
    px4_cfg = config.get("px4", {})

    # Initialize components
    policy = PolicyInferenceEngine(
        model_path=args.model,
        execution_provider=inference_cfg.get("execution_provider", "CUDAExecutionProvider"),
    )
    safety = SafetyMonitor(**safety_cfg)
    bridge = PX4Bridge(
        setpoint_rate_hz=px4_cfg.get("setpoint_rate_hz", 100),
    )
    detector = GateDetector()
    localizer = PnPLocalizer()
    estimator = VIOStateEstimator()
    preprocessor = ImagePreprocessor()

    logger.info("Deployment stack initialized. Waiting for arm...")

    if not args.dry_run:
        bridge.send_offboard_mode()
        bridge.arm()

    control_dt = 1.0 / inference_cfg.get("inference_frequency_hz", 100)
    prev_action = np.zeros(4, dtype=np.float32)

    while _running:
        t_start = time.perf_counter()

        # Get state
        bridge.spin_once()
        odom = bridge.get_odometry()

        if odom is not None:
            pos = odom["position"]
            quat = odom["quaternion"]
        else:
            pos = None
            quat = None

        # Safety check
        if not safety.check_state(pos, quat):
            logger.critical("Safety triggered: %s — disarming", safety.trigger_reason)
            if not args.dry_run:
                bridge.disarm()
            break

        # Build observation
        state = estimator.get_state()
        obs = np.concatenate([
            state.velocity,
            np.zeros(3),  # gravity body placeholder
            np.zeros(3),  # relative gate placeholder
            prev_action,
        ]).astype(np.float32)

        # Run policy
        action = policy.infer(obs)

        # Safety check action
        if not safety.check_action(action):
            if not args.dry_run:
                bridge.disarm()
            break

        # Send command
        if not args.dry_run:
            thrust = float(np.clip((action[0] + 1) / 2, 0, 1))
            bridge.send_ctbr(
                thrust=thrust,
                roll_rate=float(action[1] * 6.0),
                pitch_rate=float(action[2] * 6.0),
                yaw_rate=float(action[3] * 6.0),
            )
            bridge.send_offboard_mode()

        prev_action = action.copy()

        # Rate control
        elapsed = time.perf_counter() - t_start
        sleep_time = control_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Deployment loop ended. Cleaning up...")
    bridge.shutdown()


if __name__ == "__main__":
    main()
