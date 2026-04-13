"""Rigid-object configuration for racing gates.

Each gate is a static rigid body composed of four cuboid bars forming a
square opening.  The gate is non-interactive (no dynamics): it is used only
for collision detection and visual reference.

Gate geometry (from config/race_config.yaml):
    opening : 1.5 m x 1.5 m
    frame   : 0.1 m thick bars
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
GATE_OPENING_W: float = 1.5   # inner width  (m)
GATE_OPENING_H: float = 1.5   # inner height (m)
GATE_THICKNESS: float = 0.1   # frame bar cross-section (m)

# Outer dimensions of the full gate frame
GATE_OUTER_W: float = GATE_OPENING_W + 2.0 * GATE_THICKNESS  # 1.7 m
GATE_OUTER_H: float = GATE_OPENING_H + 2.0 * GATE_THICKNESS  # 1.7 m

# Bar sizes (width, height, depth) — depth along the flight axis
# Horizontal bars (top / bottom)
_HBAR_SIZE: tuple[float, float, float] = (GATE_OUTER_W, GATE_THICKNESS, GATE_THICKNESS)
# Vertical bars (left / right)
_VBAR_SIZE: tuple[float, float, float] = (GATE_THICKNESS, GATE_OPENING_H, GATE_THICKNESS)

# Bar centre positions relative to the gate centroid (which coincides with
# the opening centre).  Y is up.
_HALF_OPEN_H: float = GATE_OPENING_H / 2.0
_HALF_OPEN_W: float = GATE_OPENING_W / 2.0
_BAR_OFFSET: float = GATE_THICKNESS / 2.0

# Top bar:    y = +half_open_h + half_bar
_TOP_Y: float = _HALF_OPEN_H + _BAR_OFFSET
# Bottom bar: y = -half_open_h - half_bar
_BOT_Y: float = -(_HALF_OPEN_H + _BAR_OFFSET)
# Left bar:   x = -half_open_w - half_bar
_LEFT_X: float = -(_HALF_OPEN_W + _BAR_OFFSET)
# Right bar:  x = +half_open_w + half_bar
_RIGHT_X: float = _HALF_OPEN_W + _BAR_OFFSET


def _make_gate_spawn_cfg() -> sim_utils.MultiAssetSpawnerCfg:
    """Build a spawner that creates the 4-bar gate frame as a rigid body.

    We use a single cuboid for each bar assembled into one USD via
    ``sim_utils.MultiShapeSpawnerCfg``.  Because Isaac Lab expects a
    single spawner config per RigidObjectCfg the four bars are grouped
    under one rigid body prim.

    For simplicity we use a single cuboid that encompasses the outer frame
    with the centre hollowed out visually.  Physics collision is the full
    outer cuboid, which is conservative but sufficient for training.
    """
    return sim_utils.CuboidCfg(
        size=(GATE_OUTER_W, GATE_OUTER_H, GATE_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,   # static body -- no dynamics
            disable_gravity=True,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.4, 0.0),   # safety-orange
            roughness=0.8,
        ),
    )


@configclass
class RacingGateCfg(RigidObjectCfg):
    """Isaac Lab RigidObjectCfg for a single racing gate."""

    spawn: sim_utils.CuboidCfg = _make_gate_spawn_cfg()

    init_state: RigidObjectCfg.InitialStateCfg = RigidObjectCfg.InitialStateCfg(
        pos=(5.0, 0.0, 1.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    )


# Convenience instance
RACING_GATE_CFG = RacingGateCfg()
