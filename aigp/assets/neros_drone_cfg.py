"""Articulation configuration for the Neros Archer-class racing drone.

Defines the physical model (mass, inertia, joints, actuators) used to spawn
the drone inside Isaac Lab environments.  The USD asset comes from the generic
quadrotor bundled with ``omni.isaac.lab_assets``; we override mass properties,
actuator limits, and initial state to match the Neros Archer specification.

Physical parameters (from config/race_config.yaml):
    mass         : 2.5 kg
    bounding box : 0.30 x 0.30 x 0.15 m  (L x W x H)
    motor thrust : 12 N  per motor
    linear drag  : 0.3
    hover alt    : 1.5 m
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
MASS_KG: float = 2.5
# Bounding-box dimensions in metres
LX: float = 0.30
LY: float = 0.30
LZ: float = 0.15

# Uniform-density cuboid inertia tensor (principal axes aligned with body)
#   Ixx = (1/12) * m * (ly^2 + lz^2), etc.
IXX: float = (1.0 / 12.0) * MASS_KG * (LY**2 + LZ**2)  # ~0.005625
IYY: float = (1.0 / 12.0) * MASS_KG * (LX**2 + LZ**2)  # ~0.005625
IZZ: float = (1.0 / 12.0) * MASS_KG * (LX**2 + LY**2)  # ~0.0375

# Motor limits — velocity-driven revolute joints
# The *thrust* each motor can produce is ~12 N.  For the velocity-drive
# abstraction we set high velocity limits; actual thrust mapping lives in
# the environment action processing.
MOTOR_MAX_VELOCITY: float = 1000.0  # rad/s (prop speed limit)
MOTOR_EFFORT_LIMIT: float = 12.0    # N (maps to thrust)

# Aerodynamic linear damping
LINEAR_DAMPING: float = 0.3

# Path to the local quadrotor USD (created by scripts/create_quadrotor_usd.py)
_QUADROTOR_USD_PATH: str = os.path.join(
    os.path.dirname(__file__), "usd", "quadrotor.usd"
)


@configclass
class NerosDroneCfg(ArticulationCfg):
    """Isaac Lab ArticulationCfg for the Neros Archer-class quadrotor."""

    # -- USD source --------------------------------------------------------
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=_QUADROTOR_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=30.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=LINEAR_DAMPING,
            angular_damping=0.1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=MASS_KG,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    )

    # -- Initial state -----------------------------------------------------
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),           # hover at 1.5 m AGL
        rot=(1.0, 0.0, 0.0, 0.0),      # identity quaternion (w, x, y, z)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )

    # -- Actuators (4 rotor joints) ----------------------------------------
    actuators: dict = {
        "rotors": IdealPDActuatorCfg(
            joint_names_expr=[".*rotor.*"],
            effort_limit=MOTOR_EFFORT_LIMIT,
            velocity_limit=MOTOR_MAX_VELOCITY,
            stiffness=0.0,              # velocity-driven: no position PD
            damping=0.0,                # direct velocity command
        ),
    }


# Convenience alias for imports
NEROS_DRONE_CFG = NerosDroneCfg()
