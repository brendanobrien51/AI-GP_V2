"""ROS2 <-> PX4 bridge via Micro XRCE-DDS.

Translates CTBR policy outputs into PX4 VehicleAttitudeSetpoint messages
and subscribes to VehicleOdometry for state feedback.

Designed for the Neros Archer-class hardware running PX4 with
Micro XRCE-DDS agent.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttitudeSetpoint:
    """PX4-compatible attitude setpoint in NED body frame.

    Maps to px4_msgs/msg/VehicleAttitudeSetpoint.
    """

    q_d: np.ndarray          # (4,) desired quaternion wxyz
    thrust_body: np.ndarray  # (3,) thrust in body frame [0, 0, -thrust]
    timestamp_us: int


class PX4Bridge:
    """ROS2 bridge to PX4 flight controller via Micro XRCE-DDS.

    Publishes VehicleAttitudeSetpoint at the configured rate and
    subscribes to VehicleOdometry for state feedback.

    Args:
        setpoint_rate_hz: Publishing rate for attitude setpoints.
        offboard_mode: "attitude" (CTBR) or "velocity".
    """

    def __init__(
        self,
        setpoint_rate_hz: float = 100.0,
        offboard_mode: str = "attitude",
    ) -> None:
        self._rate_hz = setpoint_rate_hz
        self._mode = offboard_mode
        self._node = None
        self._pub = None
        self._sub = None
        self._latest_odom = None
        self._armed = False

        self._init_ros2()

    def _init_ros2(self) -> None:
        """Initialize ROS2 node, publishers, and subscribers."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node("aigp_px4_bridge")

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            # Import PX4 message types
            from px4_msgs.msg import (
                VehicleAttitudeSetpoint,
                VehicleOdometry,
                OffboardControlMode,
                VehicleCommand,
            )

            self._pub = self._node.create_publisher(
                VehicleAttitudeSetpoint,
                "/fmu/in/vehicle_attitude_setpoint",
                qos,
            )

            self._offboard_pub = self._node.create_publisher(
                OffboardControlMode,
                "/fmu/in/offboard_control_mode",
                qos,
            )

            self._cmd_pub = self._node.create_publisher(
                VehicleCommand,
                "/fmu/in/vehicle_command",
                qos,
            )

            self._sub = self._node.create_subscription(
                VehicleOdometry,
                "/fmu/out/vehicle_odometry",
                self._odom_callback,
                qos,
            )

            logger.info("PX4Bridge initialized — rate: %d Hz, mode: %s",
                       int(self._rate_hz), self._mode)

        except ImportError:
            logger.warning(
                "ROS2 (rclpy/px4_msgs) not available. "
                "PX4Bridge will operate in dry-run mode."
            )

    def _odom_callback(self, msg) -> None:
        """Store latest odometry for state estimation."""
        self._latest_odom = msg

    def send_ctbr(
        self,
        thrust: float,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
    ) -> None:
        """Publish a CTBR command as a VehicleAttitudeSetpoint.

        Args:
            thrust: Normalized thrust [0, 1].
            roll_rate: Roll rate (rad/s).
            pitch_rate: Pitch rate (rad/s).
            yaw_rate: Yaw rate (rad/s).
        """
        if self._pub is None:
            return

        from px4_msgs.msg import VehicleAttitudeSetpoint

        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(time.time() * 1e6)

        # Body rates
        msg.roll_body = 0.0   # not used in rate mode
        msg.pitch_body = 0.0
        msg.yaw_body = 0.0

        # Quaternion setpoint (identity — PX4 uses body rates when available)
        msg.q_d = [1.0, 0.0, 0.0, 0.0]

        # Thrust in body Z (NED: negative Z is up)
        msg.thrust_body = [0.0, 0.0, -thrust]

        self._pub.publish(msg)

    def send_offboard_mode(self) -> None:
        """Publish offboard control mode heartbeat."""
        if self._offboard_pub is None:
            return

        from px4_msgs.msg import OffboardControlMode

        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = True

        self._offboard_pub.publish(msg)

    def arm(self) -> None:
        """Send arm command to PX4."""
        self._send_vehicle_command(176, param1=1.0)  # MAV_CMD_COMPONENT_ARM_DISARM
        self._armed = True

    def disarm(self) -> None:
        """Send disarm command to PX4."""
        self._send_vehicle_command(176, param1=0.0)
        self._armed = False

    def _send_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0
    ) -> None:
        """Send a VehicleCommand to PX4."""
        if self._cmd_pub is None:
            return

        from px4_msgs.msg import VehicleCommand

        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self._cmd_pub.publish(msg)

    def get_odometry(self) -> dict | None:
        """Get latest odometry from PX4.

        Returns:
            Dict with position, velocity, quaternion, or None if no data.
        """
        if self._latest_odom is None:
            return None

        odom = self._latest_odom
        return {
            "position": np.array(odom.position, dtype=np.float64),
            "velocity": np.array(odom.velocity, dtype=np.float64),
            "quaternion": np.array(odom.q, dtype=np.float64),
            "timestamp_us": odom.timestamp,
        }

    def spin_once(self, timeout_sec: float = 0.001) -> None:
        """Process pending ROS2 callbacks."""
        if self._node is not None:
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        """Clean up ROS2 resources."""
        if self._armed:
            self.disarm()
        if self._node is not None:
            self._node.destroy_node()
            import rclpy
            rclpy.shutdown()
