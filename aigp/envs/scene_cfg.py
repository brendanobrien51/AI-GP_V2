"""Interactive scene configuration for the drone-racing environment.

Composes the Neros drone articulation, up to 8 racing gates, a downward-
looking tiled camera sensor, and a ground plane into one
``InteractiveSceneCfg`` that Isaac Lab's ``InteractiveScene`` can
instantiate.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from aigp.assets.gate_cfg import RACING_GATE_CFG
from aigp.assets.neros_drone_cfg import NEROS_DRONE_CFG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GATES: int = 8             # max gates spawned per environment
CAMERA_WIDTH: int = 80
CAMERA_HEIGHT: int = 80
CAMERA_HFOV_DEG: float = 120.0


def _gate_cfg(index: int) -> RigidObjectCfg:
    """Return a gate config placed at a default offset for gate *index*.

    During training, ``_reset_idx`` in the environment class repositions
    gates according to the track layout; these defaults just space them out
    so they do not overlap at spawn time.
    """
    cfg = RACING_GATE_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Gate_" + str(index)
    cfg.init_state = RigidObjectCfg.InitialStateCfg(
        pos=(5.0 + index * 8.0, 0.0, 1.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    return cfg


@configclass
class RacingSceneCfg(InteractiveSceneCfg):
    """Scene with one drone, up to 8 gates, a camera, and ground plane."""

    # -- Terrain (flat ground plane) ----------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -- Drone (articulation) ----------------------------------------------
    drone: ArticulationCfg = NEROS_DRONE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Drone",
    )

    # -- Racing gates (rigid bodies) ----------------------------------------
    gate_0: RigidObjectCfg = _gate_cfg(0)
    gate_1: RigidObjectCfg = _gate_cfg(1)
    gate_2: RigidObjectCfg = _gate_cfg(2)
    gate_3: RigidObjectCfg = _gate_cfg(3)
    gate_4: RigidObjectCfg = _gate_cfg(4)
    gate_5: RigidObjectCfg = _gate_cfg(5)
    gate_6: RigidObjectCfg = _gate_cfg(6)
    gate_7: RigidObjectCfg = _gate_cfg(7)

    # -- Forward-facing FPV camera (TiledCamera) ---------------------------
    # NOTE: Disabled by default because RTX renderer is broken on Windows 11
    # Build 26200. Re-enable when OS or Isaac Sim update fixes RTX support.
    # camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Drone/body/FPVCamera",
    #     update_period=0.04,
    #     height=CAMERA_HEIGHT,
    #     width=CAMERA_WIDTH,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=1.93,
    #         horizontal_aperture=3.86,
    #     ),
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.05, 0.0, 0.02),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         convention="world",
    #     ),
    # )
