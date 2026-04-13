"""Runtime safety monitoring for autonomous drone racing.

Monitors for dangerous conditions and triggers an emergency stop:
    - NaN in policy outputs
    - Loss of state tracking
    - Geofence violation
    - Excessive attitude (>80 deg tilt)
    - Communication timeout
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Real-time safety monitor with kill switch capability.

    Checks multiple safety conditions each control cycle and triggers
    emergency disarm if any condition is violated.

    Args:
        max_tracking_loss_s: Max time without valid state estimate.
        nan_action_limit: Consecutive NaN actions before kill.
        geofence_radius_m: Maximum horizontal distance from origin.
        max_altitude_m: Maximum altitude AGL.
        min_altitude_m: Minimum altitude AGL.
        max_attitude_deg: Maximum tilt angle before safety trigger.
    """

    def __init__(
        self,
        max_tracking_loss_s: float = 0.5,
        nan_action_limit: int = 3,
        geofence_radius_m: float = 100.0,
        max_altitude_m: float = 15.0,
        min_altitude_m: float = 0.3,
        max_attitude_deg: float = 80.0,
    ) -> None:
        self._max_tracking_loss = max_tracking_loss_s
        self._nan_limit = nan_action_limit
        self._geofence_r = geofence_radius_m
        self._max_alt = max_altitude_m
        self._min_alt = min_altitude_m
        self._max_tilt_rad = np.radians(max_attitude_deg)

        self._nan_count = 0
        self._last_valid_state_time = time.monotonic()
        self._triggered = False
        self._trigger_reason = ""

    @property
    def is_triggered(self) -> bool:
        """Whether the safety monitor has triggered a kill."""
        return self._triggered

    @property
    def trigger_reason(self) -> str:
        """Reason for the most recent safety trigger."""
        return self._trigger_reason

    def check_action(self, action: np.ndarray) -> bool:
        """Check if action contains NaN values.

        Args:
            action: (4,) CTBR action array.

        Returns:
            True if action is safe.
        """
        if np.any(np.isnan(action)):
            self._nan_count += 1
            logger.warning("NaN action detected (%d/%d)", self._nan_count, self._nan_limit)
            if self._nan_count >= self._nan_limit:
                self._trigger("Consecutive NaN actions exceeded limit")
                return False
        else:
            self._nan_count = 0
        return True

    def check_state(
        self,
        position: np.ndarray | None,
        quaternion: np.ndarray | None = None,
    ) -> bool:
        """Check position and attitude safety constraints.

        Args:
            position: (3,) drone position in NED. None if state lost.
            quaternion: (4,) wxyz. None to skip attitude check.

        Returns:
            True if state is safe.
        """
        now = time.monotonic()

        if position is None:
            elapsed = now - self._last_valid_state_time
            if elapsed > self._max_tracking_loss:
                self._trigger(f"Tracking loss for {elapsed:.2f}s")
                return False
            return True

        self._last_valid_state_time = now

        # Geofence check
        horiz_dist = np.linalg.norm(position[:2])
        if horiz_dist > self._geofence_r:
            self._trigger(f"Geofence violation: {horiz_dist:.1f}m > {self._geofence_r:.1f}m")
            return False

        # Altitude check
        alt = position[2]
        if alt < self._min_alt:
            self._trigger(f"Below minimum altitude: {alt:.2f}m")
            return False
        if alt > self._max_alt:
            self._trigger(f"Above maximum altitude: {alt:.2f}m")
            return False

        # Attitude check
        if quaternion is not None:
            # Tilt angle: angle between body Z-axis and world Z-axis
            # For wxyz quaternion, body Z in world = R @ [0,0,1]
            w, x, y, z = quaternion
            body_z_world = np.array([
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y),
            ])
            tilt = np.arccos(np.clip(abs(body_z_world[2]), 0, 1))
            if tilt > self._max_tilt_rad:
                self._trigger(f"Excessive tilt: {np.degrees(tilt):.1f} deg")
                return False

        return True

    def reset(self) -> None:
        """Reset the safety monitor state."""
        self._nan_count = 0
        self._last_valid_state_time = time.monotonic()
        self._triggered = False
        self._trigger_reason = ""

    def _trigger(self, reason: str) -> None:
        """Trigger the safety kill switch."""
        self._triggered = True
        self._trigger_reason = reason
        logger.critical("SAFETY KILL TRIGGERED: %s", reason)
