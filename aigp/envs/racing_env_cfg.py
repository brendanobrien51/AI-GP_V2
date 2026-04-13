"""Configuration dataclass for the drone racing DirectRLEnv."""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from aigp.envs.scene_cfg import RacingSceneCfg


@configclass
class RacingEnvCfg(DirectRLEnvCfg):
    """Full configuration for :class:`RacingEnv`.

    Pulls together scene, simulation, observation, action, and reward
    parameters into a single config consumed by ``DirectRLEnv.__init__``.
    """

    # -- Scene -----------------------------------------------------------------
    scene: RacingSceneCfg = RacingSceneCfg(num_envs=512, env_spacing=20.0)

    # -- Simulation ------------------------------------------------------------
    sim_dt: float = 0.02           # 50 Hz physics
    decimation: int = 2            # policy at 25 Hz (every 2 sim steps)
    episode_length_s: float = 10.0 # 500 steps at 25 Hz

    # -- Observations ----------------------------------------------------------
    # Actor: 13D vector + 80x80x3 image
    observation_space: int = 13    # vector obs dimension (image handled separately)
    state_space: int = 31          # critic: 13 + 3 (pos) + 3 (vel) + 12 (4 gates)

    # -- Actions ---------------------------------------------------------------
    action_space: int = 4          # [thrust, roll_rate, pitch_rate, yaw_rate]

    # -- Reward parameters -----------------------------------------------------
    progress_scale: float = 1.0
    gate_pass_bonus: float = 10.0
    course_completion_bonus: float = 50.0
    time_penalty: float = -0.002
    smoothness_scale: float = -0.01
    collision_penalty: float = -5.0

    # -- Termination parameters ------------------------------------------------
    geofence_radius: float = 100.0
    min_altitude: float = 0.3
    max_altitude: float = 15.0
    collision_force_threshold: float = 1.0

    # -- Track -----------------------------------------------------------------
    track_type: str = "zigzag"
    num_gates: int = 5
    gate_jitter: float = 0.3      # position randomization per reset

    # -- Curriculum (managed externally by CurriculumManager) ------------------
    initial_num_gates: int = 3
    max_num_gates: int = 8
