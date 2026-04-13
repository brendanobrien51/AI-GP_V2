"""Custom domain randomization event functions for Isaac Lab EventManager.

Each function follows the Isaac Lab event signature::

    def event_fn(env, env_ids: torch.Tensor, **params) -> None

Events modify simulation parameters in-place for the specified env_ids.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv


def randomize_mass(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    mass_range: tuple[float, float] = (2.3, 2.7),
) -> None:
    """Randomize drone mass within the specified range.

    Simulates battery/payload variation between flights.
    """
    num = len(env_ids)
    lo, hi = mass_range
    masses = torch.rand(num, device=env.device) * (hi - lo) + lo

    drone = env.scene["drone"]
    for i, eid in enumerate(env_ids.tolist()):
        drone.root_physx_view.set_masses(
            masses[i].unsqueeze(0).unsqueeze(0),
            indices=torch.tensor([eid], device=env.device, dtype=torch.long),
        )


def randomize_motor_thrust(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    thrust_range: tuple[float, float] = (0.85, 1.15),
) -> None:
    """Randomize motor thrust multiplier per environment.

    Simulates motor-to-motor manufacturing variation. Stored on the
    environment as ``_motor_thrust_scale`` for use in action processing.
    """
    num = len(env_ids)
    lo, hi = thrust_range
    scale = torch.rand(num, device=env.device) * (hi - lo) + lo

    if not hasattr(env, "_motor_thrust_scale"):
        env._motor_thrust_scale = torch.ones(env.num_envs, device=env.device)
    env._motor_thrust_scale[env_ids] = scale


def randomize_com_offset(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    offset_range: tuple[float, float] = (-0.01, 0.01),
) -> None:
    """Randomize centre-of-mass offset to simulate assembly tolerance.

    Stored as ``_com_offset`` for use in torque computation.
    """
    num = len(env_ids)
    lo, hi = offset_range
    offset = torch.rand(num, 3, device=env.device) * (hi - lo) + lo

    if not hasattr(env, "_com_offset"):
        env._com_offset = torch.zeros(env.num_envs, 3, device=env.device)
    env._com_offset[env_ids] = offset


def randomize_drag(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    drag_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize aerodynamic drag coefficient multiplier.

    Simulates frame/propeller variation affecting drag profile.
    """
    num = len(env_ids)
    lo, hi = drag_range
    scale = torch.rand(num, device=env.device) * (hi - lo) + lo

    if not hasattr(env, "_drag_scale"):
        env._drag_scale = torch.ones(env.num_envs, device=env.device)
    env._drag_scale[env_ids] = scale


def randomize_imu_noise(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    accel_noise_range: tuple[float, float] = (0.03, 0.08),
    gyro_noise_range: tuple[float, float] = (0.005, 0.02),
) -> None:
    """Randomize IMU noise levels per environment.

    Stored as ``_imu_accel_noise`` and ``_imu_gyro_noise`` for
    observation noise injection.
    """
    num = len(env_ids)

    a_lo, a_hi = accel_noise_range
    g_lo, g_hi = gyro_noise_range

    if not hasattr(env, "_imu_accel_noise"):
        env._imu_accel_noise = torch.full((env.num_envs,), 0.05, device=env.device)
    if not hasattr(env, "_imu_gyro_noise"):
        env._imu_gyro_noise = torch.full((env.num_envs,), 0.01, device=env.device)

    env._imu_accel_noise[env_ids] = torch.rand(num, device=env.device) * (a_hi - a_lo) + a_lo
    env._imu_gyro_noise[env_ids] = torch.rand(num, device=env.device) * (g_hi - g_lo) + g_lo


def randomize_camera_latency(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    latency_range: tuple[int, int] = (0, 30),
) -> None:
    """Randomize camera processing latency in milliseconds.

    Stored as ``_camera_latency_ms`` for frame delay simulation.
    """
    num = len(env_ids)
    lo, hi = latency_range

    if not hasattr(env, "_camera_latency_ms"):
        env._camera_latency_ms = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._camera_latency_ms[env_ids] = torch.randint(lo, hi + 1, (num,), device=env.device)


def randomize_action_delay(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    delay_range: tuple[int, int] = (0, 2),
) -> None:
    """Randomize action execution delay in simulation steps.

    Simulates communication latency between compute module and ESCs.
    Stored as ``_action_delay_steps``.
    """
    num = len(env_ids)
    lo, hi = delay_range

    if not hasattr(env, "_action_delay_steps"):
        env._action_delay_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._action_delay_steps[env_ids] = torch.randint(lo, hi + 1, (num,), device=env.device)


def randomize_image_brightness(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    brightness_range: tuple[float, float] = (0.7, 1.3),
) -> None:
    """Randomize image brightness multiplier to simulate lighting changes.

    Stored as ``_brightness_scale`` for camera image augmentation.
    """
    num = len(env_ids)
    lo, hi = brightness_range

    if not hasattr(env, "_brightness_scale"):
        env._brightness_scale = torch.ones(env.num_envs, device=env.device)

    env._brightness_scale[env_ids] = torch.rand(num, device=env.device) * (hi - lo) + lo


def apply_wind_gust(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    gust_range: tuple[float, float] = (0.0, 3.0),
) -> None:
    """Apply a random wind gust as an external force on the drone body.

    Wind direction is randomized in XY plane. Magnitude from gust_range.
    """
    num = len(env_ids)
    lo, hi = gust_range

    # Random wind direction in XY plane
    angle = torch.rand(num, device=env.device) * 2.0 * 3.14159
    magnitude = torch.rand(num, device=env.device) * (hi - lo) + lo

    wind_force = torch.zeros(num, 1, 3, device=env.device)
    wind_force[:, 0, 0] = magnitude * torch.cos(angle)
    wind_force[:, 0, 1] = magnitude * torch.sin(angle)

    # Store for sustained application until next gust event
    if not hasattr(env, "_wind_force"):
        env._wind_force = torch.zeros(env.num_envs, 1, 3, device=env.device)

    env._wind_force[env_ids] = wind_force

    # Apply to drone
    drone = env.scene["drone"]
    drone.set_external_force_and_torque(
        forces=env._wind_force,
        torques=torch.zeros_like(env._wind_force),
        body_ids=[0],
    )
