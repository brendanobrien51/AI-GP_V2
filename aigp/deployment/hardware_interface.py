"""Hardware abstraction layer for drone control and sensing.

Provides an abstract DroneInterface (ABC) with concrete implementations
for simulation (Isaac Lab) and real hardware (PX4/ROS2).

Evolved from V1's hardware.py — adds CTBR-native interface and
structured state return types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DroneState:
    """Complete drone state snapshot.

    All quantities in NED frame unless noted otherwise.
    """

    position: np.ndarray       # (3,) metres
    velocity: np.ndarray       # (3,) m/s
    quaternion: np.ndarray     # (4,) wxyz
    angular_velocity: np.ndarray  # (3,) rad/s in body frame
    timestamp_s: float


@dataclass(slots=True)
class CTBRCommand:
    """Collective Thrust + Body Rate command.

    Attributes:
        thrust: Normalized thrust [0, 1].
        roll_rate: Roll rate in rad/s.
        pitch_rate: Pitch rate in rad/s.
        yaw_rate: Yaw rate in rad/s.
    """

    thrust: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float

    def to_array(self) -> np.ndarray:
        return np.array([self.thrust, self.roll_rate, self.pitch_rate, self.yaw_rate])


class DroneInterface(ABC):
    """Abstract interface for drone control and sensing."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Return current drone state."""
        ...

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Return latest camera frame as (H, W, 3) BGR uint8."""
        ...

    @abstractmethod
    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Return raw IMU data as (accel_xyz, gyro_xyz)."""
        ...

    @abstractmethod
    def send_ctbr(self, cmd: CTBRCommand) -> None:
        """Send a CTBR command to the flight controller."""
        ...

    @abstractmethod
    def arm(self) -> bool:
        """Arm the drone. Returns True on success."""
        ...

    @abstractmethod
    def disarm(self) -> None:
        """Disarm the drone immediately."""
        ...

    @abstractmethod
    def is_armed(self) -> bool:
        """Check if the drone is currently armed."""
        ...


class SimInterface(DroneInterface):
    """Isaac Lab simulation interface stub.

    In practice, the Isaac Lab environment handles control directly.
    This stub exists for API parity with the hardware interface.
    """

    def __init__(self):
        self._armed = False

    def get_state(self) -> DroneState:
        return DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            timestamp_s=0.0,
        )

    def get_image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    def send_ctbr(self, cmd: CTBRCommand) -> None:
        pass

    def arm(self) -> bool:
        self._armed = True
        return True

    def disarm(self) -> None:
        self._armed = False

    def is_armed(self) -> bool:
        return self._armed
