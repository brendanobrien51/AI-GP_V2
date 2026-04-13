"""Dataclass definitions for gate poses and track layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GatePose:
    """A single gate's 6-DOF pose (position + yaw) in the world frame.

    Attributes:
        position: [x, y, z] coordinates in meters (world frame).
        rotation_deg: Yaw rotation of the gate face in degrees.
            0 deg = gate faces +X; 90 deg = gate faces +Y.
    """

    position: list[float]  # [x, y, z] in world frame (meters)
    rotation_deg: float  # Yaw rotation of gate (degrees)

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError(
                f"GatePose.position must have exactly 3 elements [x, y, z], got {len(self.position)}"
            )

    @property
    def x(self) -> float:
        """X position in meters."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Y position in meters."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Z position in meters."""
        return self.position[2]

    @property
    def rotation_rad(self) -> float:
        """Yaw rotation in radians."""
        return math.radians(self.rotation_deg)

    def distance_to(self, other: GatePose) -> float:
        """Euclidean distance to another gate.

        Args:
            other: The other gate pose.

        Returns:
            Distance in meters.
        """
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass
class TrackLayout:
    """A complete track defined by an ordered sequence of gates.

    Attributes:
        name: Human-readable name for this track layout.
        gates: Ordered list of gate poses defining the race course.
    """

    name: str
    gates: list[GatePose] = field(default_factory=list)

    @property
    def num_gates(self) -> int:
        """Number of gates in the track."""
        return len(self.gates)

    def min_gate_separation(self) -> float:
        """Return the minimum pairwise distance between consecutive gates.

        Returns:
            Minimum distance in meters, or inf if fewer than 2 gates.
        """
        if len(self.gates) < 2:
            return float("inf")
        return min(
            self.gates[i].distance_to(self.gates[i + 1])
            for i in range(len(self.gates) - 1)
        )

    def total_path_length(self) -> float:
        """Sum of consecutive gate-to-gate distances.

        Returns:
            Total path length in meters.
        """
        if len(self.gates) < 2:
            return 0.0
        return sum(
            self.gates[i].distance_to(self.gates[i + 1])
            for i in range(len(self.gates) - 1)
        )
