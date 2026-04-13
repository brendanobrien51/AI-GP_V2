"""Domain randomization event configuration for sim-to-real transfer.

Defines ``EventTermCfg`` entries for the Isaac Lab ``EventManager``.
Each entry references a callable in :mod:`aigp.domain_rand.custom_events`.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

import aigp.domain_rand.custom_events as events


@configclass
class DomainRandEventsCfg:
    """Collection of domain randomization events applied during training.

    Per-reset events fire once at the start of each episode.
    Per-step events fire every simulation step.
    Interval events fire at fixed time intervals.
    """

    # -- Per-reset events (applied at episode start) -----------------------

    randomize_mass = EventTermCfg(
        func=events.randomize_mass,
        mode="reset",
        params={"mass_range": (2.3, 2.7)},
    )

    randomize_motor_thrust = EventTermCfg(
        func=events.randomize_motor_thrust,
        mode="reset",
        params={"thrust_range": (0.85, 1.15)},
    )

    randomize_com_offset = EventTermCfg(
        func=events.randomize_com_offset,
        mode="reset",
        params={"offset_range": (-0.01, 0.01)},
    )

    randomize_drag = EventTermCfg(
        func=events.randomize_drag,
        mode="reset",
        params={"drag_range": (0.8, 1.2)},
    )

    randomize_imu_noise = EventTermCfg(
        func=events.randomize_imu_noise,
        mode="reset",
        params={
            "accel_noise_range": (0.03, 0.08),
            "gyro_noise_range": (0.005, 0.02),
        },
    )

    randomize_camera_latency = EventTermCfg(
        func=events.randomize_camera_latency,
        mode="reset",
        params={"latency_range": (0, 30)},
    )

    randomize_action_delay = EventTermCfg(
        func=events.randomize_action_delay,
        mode="reset",
        params={"delay_range": (0, 2)},
    )

    # -- Per-step events ---------------------------------------------------

    randomize_image_brightness = EventTermCfg(
        func=events.randomize_image_brightness,
        mode="interval",
        interval_range_s=(0.04, 0.04),  # every sim step
        params={"brightness_range": (0.7, 1.3)},
    )

    # -- Interval events ---------------------------------------------------

    apply_wind_gust = EventTermCfg(
        func=events.apply_wind_gust,
        mode="interval",
        interval_range_s=(4.0, 6.0),  # every ~5 seconds
        params={"gust_range": (0.0, 3.0)},
    )
