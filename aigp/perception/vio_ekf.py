"""Combined VIO + EKF state estimator for GPS-denied drone racing.

Merges V1's ``vio_estimator.py`` and ``kalman_fusion.py`` into a single
module. The EKF predicts at IMU rate (~250 Hz) using accelerometer and
gyroscope integration, and corrects at vision rate (~30 Hz) using PnP
position measurements.

Architecture (Swift-style):
    IMU integration → EKF predict (high frequency)
    Gate detection → PnP solve → EKF update (low frequency)
    Aerodynamic residual model (HDVIO-style drag correction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class StateEstimate:
    """Current state estimate from the VIO + EKF fusion.

    All quantities are in NED (North-East-Down) frame.
    """

    position: np.ndarray         # (3,) metres
    velocity: np.ndarray         # (3,) m/s
    quaternion: np.ndarray       # (4,) wxyz
    position_covariance: np.ndarray  # (3, 3)
    timestamp_s: float = 0.0

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def euler_deg(self) -> np.ndarray:
        """Euler angles [roll, pitch, yaw] in degrees."""
        r = Rotation.from_quat([
            self.quaternion[1], self.quaternion[2],
            self.quaternion[3], self.quaternion[0],
        ])  # scipy uses xyzw
        return r.as_euler("xyz", degrees=True)


class AeroResidualModel:
    """HDVIO-style aerodynamic drag correction.

    Models drag deceleration as: drag = k * speed^2 in the velocity direction.
    Default k=0 (no correction) until calibrated from flight data.

    Args:
        drag_coeff: Quadratic drag coefficient (m/s^2 per (m/s)^2).
    """

    def __init__(self, drag_coeff: float = 0.0) -> None:
        self.k = drag_coeff

    def predict(self, velocity: np.ndarray) -> np.ndarray:
        """Estimate drag deceleration vector in world frame.

        Args:
            velocity: [vx, vy, vz] in world frame (m/s).

        Returns:
            Drag correction [dx, dy, dz] in m/s^2 (subtract from accel).
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.01 or self.k == 0.0:
            return np.zeros(3)
        direction = velocity / speed
        return self.k * speed**2 * direction

    def load(self, path: str) -> None:
        """Load calibrated drag coefficient from file."""
        try:
            self.k = float(np.load(path))
            logger.info("AeroResidual loaded k=%.5f from %s", self.k, path)
        except Exception:
            logger.warning("Could not load aero residual from %s", path)


class VIOStateEstimator:
    """Visual-Inertial Odometry + Extended Kalman Filter state estimator.

    Combines high-frequency IMU integration with low-frequency PnP
    position corrections in a single 6-state EKF.

    State vector: [x, y, z, vx, vy, vz] (NED)
    Attitude tracked separately via quaternion integration.

    Args:
        accel_noise: Accelerometer noise std (m/s^2/sqrt(Hz)).
        gyro_noise: Gyroscope noise std (rad/s/sqrt(Hz)).
        accel_bias_drift: Accelerometer bias random walk.
        gyro_bias_drift: Gyroscope bias random walk.
        pos_process_noise: Position process noise for EKF.
        vel_process_noise: Velocity process noise for EKF.
    """

    def __init__(
        self,
        accel_noise: float = 0.05,
        gyro_noise: float = 0.01,
        accel_bias_drift: float = 0.002,
        gyro_bias_drift: float = 0.0005,
        pos_process_noise: float = 0.01,
        vel_process_noise: float = 0.1,
    ) -> None:
        # IMU noise parameters
        self._accel_noise = accel_noise
        self._gyro_noise = gyro_noise
        self._accel_bias_drift = accel_bias_drift
        self._gyro_bias_drift = gyro_bias_drift

        # IMU bias state (drifts over time)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

        # Attitude (quaternion wxyz)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])

        # EKF (6-state: position + velocity)
        self._kf = KalmanFilter(dim_x=6, dim_z=3)
        self._kf.F = np.eye(6)  # updated per predict step
        self._kf.H = np.zeros((3, 6))
        self._kf.H[0, 0] = self._kf.H[1, 1] = self._kf.H[2, 2] = 1.0
        self._kf.R = np.eye(3) * 0.1
        self._kf.P = np.eye(6) * 1.0
        self._pos_q = pos_process_noise
        self._vel_q = vel_process_noise

        # Aerodynamic model
        self.aero = AeroResidualModel()

        # Time tracking
        self._time_s = 0.0
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """Seed the estimator with an initial state.

        Args:
            position: [x, y, z] in NED metres.
            velocity: [vx, vy, vz] in m/s.
            quaternion: [w, x, y, z] orientation.
        """
        self._kf.x = np.concatenate([
            np.asarray(position, dtype=np.float64),
            np.asarray(velocity, dtype=np.float64),
        ])
        self._kf.P = np.eye(6) * 0.1
        self._quat = np.asarray(quaternion, dtype=np.float64).copy()
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._time_s = 0.0
        self._initialized = True

    def predict(
        self,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        dt: float,
    ) -> None:
        """IMU-driven prediction step (call at IMU rate, ~250 Hz).

        Integrates accelerometer for velocity/position and gyroscope for
        attitude. Applies bias correction, noise injection, and aero drag.

        Args:
            accel_raw: Raw accelerometer reading [ax, ay, az] in m/s^2 (body frame).
            gyro_raw: Raw gyroscope reading [gx, gy, gz] in rad/s (body frame).
            dt: Time step in seconds.
        """
        if not self._initialized:
            return

        # Bias drift (random walk)
        self._accel_bias += np.random.randn(3) * self._accel_bias_drift * np.sqrt(dt)
        self._gyro_bias += np.random.randn(3) * self._gyro_bias_drift * np.sqrt(dt)

        # Correct readings
        accel = accel_raw - self._accel_bias + np.random.randn(3) * self._accel_noise * np.sqrt(dt)
        gyro = gyro_raw - self._gyro_bias + np.random.randn(3) * self._gyro_noise * np.sqrt(dt)

        # Integrate attitude (quaternion kinematics)
        omega = gyro  # body-frame angular velocity
        q = self._quat
        # Quaternion derivative: dq/dt = 0.5 * q * [0, omega]
        dq = 0.5 * np.array([
            -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2],
             q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1],
             q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0],
             q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0],
        ])
        self._quat = q + dq * dt
        self._quat /= np.linalg.norm(self._quat)  # normalize

        # Rotate accel to world frame
        r = Rotation.from_quat([
            self._quat[1], self._quat[2], self._quat[3], self._quat[0]
        ])  # scipy xyzw
        accel_world = r.apply(accel)

        # Remove gravity (NED: gravity = [0, 0, 9.81])
        accel_world[2] += 9.81

        # Aerodynamic drag correction
        vel = self._kf.x[3:6].copy()
        drag = self.aero.predict(vel)
        accel_world -= drag

        # EKF predict
        self._kf.F = np.eye(6)
        self._kf.F[0, 3] = dt
        self._kf.F[1, 4] = dt
        self._kf.F[2, 5] = dt

        # Process noise
        q_mat = np.zeros((6, 6))
        q_mat[0, 0] = q_mat[1, 1] = q_mat[2, 2] = self._pos_q * dt**2
        q_mat[3, 3] = q_mat[4, 4] = q_mat[5, 5] = self._vel_q * dt
        self._kf.Q = q_mat

        # Control input (acceleration)
        B = np.zeros((6, 3))
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt
        B[0, 0] = 0.5 * dt**2
        B[1, 1] = 0.5 * dt**2
        B[2, 2] = 0.5 * dt**2

        self._kf.predict(u=accel_world, B=B)
        self._time_s += dt

    def update_pnp(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """PnP-based position correction (call when gate detected, ~30 Hz).

        Args:
            position: Measured drone position [x, y, z] from PnP solve.
            covariance: 3x3 measurement covariance from PnP localizer.
        """
        if not self._initialized:
            return

        self._kf.R = np.asarray(covariance, dtype=np.float64)
        self._kf.update(np.asarray(position, dtype=np.float64))

    def get_state(self) -> StateEstimate:
        """Get current state estimate.

        Returns:
            StateEstimate with position, velocity, orientation, and covariance.
        """
        return StateEstimate(
            position=self._kf.x[:3].copy(),
            velocity=self._kf.x[3:6].copy(),
            quaternion=self._quat.copy(),
            position_covariance=self._kf.P[:3, :3].copy(),
            timestamp_s=self._time_s,
        )
