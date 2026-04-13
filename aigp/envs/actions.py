"""CTBR (Collective Thrust + Body Rates) action processing.

The policy outputs a 4-D action vector in [-1, 1]:
    a[0]   -> collective thrust   mapped to [0, max_thrust] m/s**2
    a[1:4] -> body rates (p,q,r)  mapped to [-max_rate, max_rate] rad/s

All heavy-lifting is JIT-compiled for throughput across 512+ envs.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Constants (match config/race_config.yaml)
# ---------------------------------------------------------------------------
GRAVITY: float = 9.81
MAX_COLLECTIVE_THRUST: float = 2.0 * GRAVITY   # 19.62 m/s^2
MAX_BODY_RATE: float = 6.0                      # rad/s
ACTION_DIM: int = 4


@torch.jit.script
def scale_ctbr_actions(
    raw_actions: torch.Tensor,
    max_thrust: float = 19.62,
    max_rate: float = 6.0,
) -> torch.Tensor:
    """Map raw [-1,1] policy output to physical CTBR commands.

    Args:
        raw_actions: Tensor of shape (N, 4) in [-1, 1].
        max_thrust:  Maximum collective thrust in m/s**2.
        max_rate:    Maximum body rate in rad/s.

    Returns:
        Tensor of shape (N, 4):
            col 0  — collective thrust [0, max_thrust]
            col 1-3 — body rates       [-max_rate, max_rate]
    """
    scaled = torch.empty_like(raw_actions)
    # Thrust: [-1,1] -> [0,1] -> [0, max_thrust]
    scaled[:, 0] = (raw_actions[:, 0] * 0.5 + 0.5) * max_thrust
    # Body rates: [-1,1] -> [-max_rate, max_rate]
    scaled[:, 1] = raw_actions[:, 1] * max_rate
    scaled[:, 2] = raw_actions[:, 2] * max_rate
    scaled[:, 3] = raw_actions[:, 3] * max_rate
    return scaled


@torch.jit.script
def ctbr_to_motor_forces(
    scaled_actions: torch.Tensor,
    mass: float = 2.5,
    arm_length: float = 0.15,
    moment_coeff: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CTBR commands to per-motor thrust forces and body torques.

    Uses a standard quadrotor X-configuration mixer:

        Motor layout (top view, NED body frame):
            M0: front-right (+x, -y)  CW
            M1: rear-left   (-x, +y)  CW
            M2: front-left  (+x, +y)  CCW
            M3: rear-right  (-x, -y)  CCW

    Args:
        scaled_actions: (N, 4) — [thrust_accel, p, q, r] in physical units.
        mass:           Drone mass in kg.
        arm_length:     Motor arm length in m (centre-to-motor).
        moment_coeff:   Torque-to-thrust ratio (k_m / k_f).

    Returns:
        forces:  (N, 4) per-motor thrust in Newtons (clamped >= 0).
        torques: (N, 3) desired body torques [tau_x, tau_y, tau_z] in N-m.
    """
    thrust_accel = scaled_actions[:, 0]   # m/s^2
    p = scaled_actions[:, 1]              # roll rate  (rad/s)
    q = scaled_actions[:, 2]              # pitch rate (rad/s)
    r = scaled_actions[:, 3]              # yaw rate   (rad/s)

    # Total thrust force (N) = mass * commanded acceleration
    total_thrust = mass * thrust_accel    # (N,)

    # Desired torques via a simple proportional rate controller
    # In a full stack, the outer loop would be the RL policy; here we
    # translate desired rates to torques using approximate inertia values.
    # Approximate inertia (from cuboid model)
    Ixx: float = 0.005625
    Iyy: float = 0.005625
    Izz: float = 0.0375

    # PD rate control — feed-forward torque for the desired rate
    # Since we run at high frequency, proportional gain suffices.
    kp_pq: float = 0.25   # roll / pitch proportional gain
    kp_r: float = 0.15    # yaw proportional gain

    tau_x = kp_pq * Ixx * p   # roll  torque
    tau_y = kp_pq * Iyy * q   # pitch torque
    tau_z = kp_r * Izz * r     # yaw   torque

    torques = torch.stack([tau_x, tau_y, tau_z], dim=-1)  # (N, 3)

    # Motor mixing matrix (X-config)
    #   F_total = f0 + f1 + f2 + f3
    #   tau_x   = L * (-f0 + f1 + f2 - f3)   (roll)
    #   tau_y   = L * ( f0 - f1 + f2 - f3)   (pitch)
    #   tau_z   = c * ( f0 + f1 - f2 - f3)   (yaw, c = moment_coeff)
    #
    # Solving for individual motor forces:
    quarter_F = total_thrust * 0.25
    roll_term = tau_x / (4.0 * arm_length)
    pitch_term = tau_y / (4.0 * arm_length)
    yaw_term = tau_z / (4.0 * moment_coeff) if moment_coeff > 0 else tau_z * 0.0

    f0 = quarter_F - roll_term + pitch_term + yaw_term
    f1 = quarter_F + roll_term - pitch_term + yaw_term
    f2 = quarter_F + roll_term + pitch_term - yaw_term
    f3 = quarter_F - roll_term - pitch_term - yaw_term

    forces = torch.stack([f0, f1, f2, f3], dim=-1)       # (N, 4)
    forces = torch.clamp(forces, min=0.0, max=12.0)      # per-motor limit

    return forces, torques


@torch.jit.script
def clamp_actions(raw_actions: torch.Tensor) -> torch.Tensor:
    """Clamp raw policy output to [-1, 1].

    Args:
        raw_actions: (N, 4) unclamped actions from the network.

    Returns:
        Clamped actions (N, 4).
    """
    return torch.clamp(raw_actions, min=-1.0, max=1.0)
