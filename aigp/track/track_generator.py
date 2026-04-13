"""Procedural track generators for autonomous drone racing.

Three track layouts are supported:
    - **zigzag**: Gates alternate left-right with altitude variation.
    - **split_s**: Descending spiral pattern for inverted flight.
    - **circular**: Gates arranged in an elliptical loop.

All generators are deterministic given a seed, use PyTorch for tensor
operations, and return a :class:`TrackLayout` that can be used standalone
(no Isaac Sim dependency).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from aigp.track.track_types import GatePose, TrackLayout

# Hard safety floor -- no two gates may be closer than this.
MIN_GATE_SEPARATION_M: float = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_generator(seed: int | None) -> torch.Generator:
    """Create a seeded PyTorch generator for reproducible randomness.

    Args:
        seed: Random seed. If None a non-deterministic seed is used.

    Returns:
        A seeded torch.Generator on CPU.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
    return gen


def _angle_between_points(x0: float, y0: float, x1: float, y1: float) -> float:
    """Heading (degrees) from point 0 toward point 1 in the XY plane.

    Returns:
        Angle in degrees, measured counter-clockwise from +X axis.
    """
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _validate_separation(gates: list[GatePose]) -> None:
    """Raise ValueError if any consecutive gates are closer than the minimum.

    Args:
        gates: Ordered gate list.

    Raises:
        ValueError: If any consecutive pair violates the separation constraint.
    """
    for i in range(len(gates) - 1):
        dist = gates[i].distance_to(gates[i + 1])
        if dist < MIN_GATE_SEPARATION_M:
            raise ValueError(
                f"Gates {i} and {i + 1} are only {dist:.2f}m apart "
                f"(minimum is {MIN_GATE_SEPARATION_M}m)"
            )


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_zigzag(
    num_gates: int,
    *,
    corridor_width: float = 10.0,
    gate_separation: float = 12.0,
    altitude_variation: float = 2.0,
    jitter: float = 0.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> TrackLayout:
    """Generate a zigzag track with gates alternating left-right.

    Gates march along the +X axis, alternating Y-offset. Each gate faces
    roughly toward the next gate.

    Args:
        num_gates: Number of gates (>= 2).
        corridor_width: Total left-right swing in meters (5-15m).
        gate_separation: Forward (X) spacing between consecutive gates (8-15m).
        altitude_variation: Peak-to-peak Z variation in meters (1-4m).
        jitter: Random positional jitter radius in meters (0 = no jitter).
        seed: Random seed for reproducibility.
        device: Torch device (used for random tensor ops).

    Returns:
        A TrackLayout with the generated gate poses.

    Raises:
        ValueError: If parameters are out of bounds or separation violated.
    """
    if num_gates < 2:
        raise ValueError(f"num_gates must be >= 2, got {num_gates}")
    corridor_width = _clamp(corridor_width, 5.0, 15.0)
    gate_separation = _clamp(gate_separation, 8.0, 15.0)
    altitude_variation = _clamp(altitude_variation, 1.0, 4.0)

    gen = _seed_generator(seed)
    half_width = corridor_width / 2.0
    base_altitude = 3.0  # metres above ground

    gates: list[GatePose] = []
    for i in range(num_gates):
        x = i * gate_separation
        y = half_width * (1 if i % 2 == 0 else -1)
        z = base_altitude + altitude_variation * math.sin(2 * math.pi * i / max(num_gates - 1, 1))

        # Apply jitter
        if jitter > 0:
            dx = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dy = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dz = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            x += dx
            y += dy
            z = max(0.5, z + dz)  # keep above ground

        # Rotation placeholder -- will be set after all positions known
        gates.append(GatePose(position=[x, y, z], rotation_deg=0.0))

    # Orient each gate to face the next gate; last gate mirrors the second-to-last
    for i in range(num_gates):
        if i < num_gates - 1:
            gates[i].rotation_deg = _angle_between_points(
                gates[i].x, gates[i].y, gates[i + 1].x, gates[i + 1].y
            )
        else:
            gates[i].rotation_deg = gates[i - 1].rotation_deg

    _validate_separation(gates)
    return TrackLayout(name="zigzag", gates=gates)


def generate_split_s(
    num_gates: int,
    *,
    radius: float = 12.0,
    descent_per_gate: float = 2.0,
    start_altitude: float = 10.0,
    jitter: float = 0.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> TrackLayout:
    """Generate a descending spiral (Split-S) track.

    Gates are placed along a helix that descends with each step. Each gate
    faces tangent to the spiral in the direction of travel.

    Args:
        num_gates: Number of gates (>= 2).
        radius: Spiral radius in meters (8-15m).
        descent_per_gate: Altitude drop per gate in meters (1-3m).
        start_altitude: Altitude of the first gate in meters (8-12m).
        jitter: Random positional jitter radius in meters.
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        A TrackLayout with the generated gate poses.

    Raises:
        ValueError: If parameters are out of bounds or separation violated.
    """
    if num_gates < 2:
        raise ValueError(f"num_gates must be >= 2, got {num_gates}")
    radius = _clamp(radius, 8.0, 15.0)
    descent_per_gate = _clamp(descent_per_gate, 1.0, 3.0)
    start_altitude = _clamp(start_altitude, 8.0, 12.0)

    gen = _seed_generator(seed)

    # Angular separation: spread gates evenly across a full turn + extra
    # to avoid bunching.  Two full turns for a proper spiral feel.
    total_angle = 2 * math.pi * max(1.0, num_gates / 6.0)
    angle_step = total_angle / num_gates

    # Ensure the arc-length between consecutive gates >= MIN_GATE_SEPARATION_M.
    # arc = radius * angle_step, so we may need to bump the angle step.
    if radius * angle_step < MIN_GATE_SEPARATION_M:
        angle_step = MIN_GATE_SEPARATION_M / radius * 1.05  # 5% margin

    gates: list[GatePose] = []
    for i in range(num_gates):
        theta = i * angle_step
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = max(0.5, start_altitude - i * descent_per_gate)

        if jitter > 0:
            dx = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dy = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dz = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            x += dx
            y += dy
            z = max(0.5, z + dz)

        # Tangent direction: derivative of (cos(theta), sin(theta)) is
        # (-sin(theta), cos(theta)).  We want the gate to face the travel
        # direction, which is the tangent.
        tangent_deg = math.degrees(math.atan2(math.cos(theta), -math.sin(theta)))
        gates.append(GatePose(position=[x, y, z], rotation_deg=tangent_deg))

    _validate_separation(gates)
    return TrackLayout(name="split_s", gates=gates)


def generate_circular(
    num_gates: int,
    *,
    radius: float = 15.0,
    altitude: float = 3.0,
    eccentricity: float = 0.0,
    jitter: float = 0.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> TrackLayout:
    """Generate an elliptical (circular when eccentricity=0) track.

    Gates are spaced evenly around an ellipse and face inward along the
    path (toward the center of curvature).

    Args:
        num_gates: Number of gates (>= 2).
        radius: Semi-major axis in meters (10-20m).
        altitude: Constant gate altitude in meters (2-5m).
        eccentricity: Ellipse eccentricity 0-0.5 (0 = perfect circle).
        jitter: Random positional jitter radius in meters.
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        A TrackLayout with the generated gate poses.

    Raises:
        ValueError: If parameters are out of bounds or separation violated.
    """
    if num_gates < 2:
        raise ValueError(f"num_gates must be >= 2, got {num_gates}")
    radius = _clamp(radius, 10.0, 20.0)
    altitude = _clamp(altitude, 2.0, 5.0)
    eccentricity = _clamp(eccentricity, 0.0, 0.5)

    gen = _seed_generator(seed)

    a = radius  # semi-major axis
    b = a * math.sqrt(1.0 - eccentricity**2)  # semi-minor axis

    gates: list[GatePose] = []
    for i in range(num_gates):
        theta = 2 * math.pi * i / num_gates
        x = a * math.cos(theta)
        y = b * math.sin(theta)
        z = altitude

        if jitter > 0:
            dx = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dy = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            dz = (torch.rand(1, generator=gen).item() - 0.5) * 2 * jitter
            x += dx
            y += dy
            z = max(0.5, z + dz)

        # Gate faces inward along the path.  The tangent of the ellipse at
        # angle theta is (-a*sin(theta), b*cos(theta)).  We want the gate
        # to face the direction of travel along the path, which is the
        # tangent direction.
        tangent_deg = math.degrees(math.atan2(b * math.cos(theta), -a * math.sin(theta)))
        gates.append(GatePose(position=[x, y, z], rotation_deg=tangent_deg))

    _validate_separation(gates)
    return TrackLayout(name="circular", gates=gates)
