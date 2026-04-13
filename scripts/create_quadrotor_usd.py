"""Create a quadrotor USD articulation for Isaac Lab / Isaac Sim 4.5.

This script builds a minimal quadrotor asset consisting of:
  - A root rigid body "body" (box 0.3 x 0.3 x 0.15 m, mass 2.5 kg)
  - Four revolute joints ("rotor_0" .. "rotor_3"), each connecting a
    small cylinder to the body at the four corners.

The resulting USD is saved as an Articulation root so Isaac Lab's
ArticulationCfg can load it directly.

Run with:
    C:\\isaacsim\\python.bat scripts\\create_quadrotor_usd.py
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------- #
# Isaac Sim 4.5 Windows workarounds (MUST come before any omni imports)
# --------------------------------------------------------------------------- #
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    import osqp  # noqa: F401  pre-load to avoid DLL conflict
except ImportError:
    pass

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "create_new_stage": False})

# --------------------------------------------------------------------------- #
# Now safe to import pxr / omni
# --------------------------------------------------------------------------- #
import omni.usd  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    PhysxSchema,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OUTPUT_PATH = (
    r"C:\Users\brend\OneDrive\Desktop\AI-GP_V2\aigp\assets\usd\quadrotor.usd"
)

BODY_HALF_EXTENTS = Gf.Vec3f(0.15, 0.15, 0.075)  # half of 0.3 x 0.3 x 0.15
BODY_MASS = 2.5  # kg

ROTOR_RADIUS = 0.04  # m
ROTOR_HEIGHT = 0.02  # m
ROTOR_MASS = 0.02  # kg  (small, mostly symbolic)

# Rotor positions relative to body centre (on top of body)
ROTOR_OFFSETS = [
    Gf.Vec3d(0.12, 0.12, 0.075),   # rotor_0  front-left
    Gf.Vec3d(-0.12, 0.12, 0.075),  # rotor_1  rear-left
    Gf.Vec3d(-0.12, -0.12, 0.075), # rotor_2  rear-right
    Gf.Vec3d(0.12, -0.12, 0.075),  # rotor_3  front-right
]

ROTOR_JOINT_AXIS = "Z"  # spin axis


def _ensure_dir(path: str) -> None:
    """Create parent directories if they do not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _add_rigid_body(prim: Usd.Prim) -> None:
    """Apply RigidBodyAPI and CollisionAPI to *prim*."""
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)


def _set_mass(prim: Usd.Prim, mass: float) -> None:
    """Apply MassAPI and set the mass attribute."""
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)


def create_quadrotor_usd() -> str:
    """Build the quadrotor USD stage and save it to *OUTPUT_PATH*.

    Returns:
        The absolute path of the saved file.
    """
    _ensure_dir(OUTPUT_PATH)

    # ------------------------------------------------------------------ #
    # 1.  Create a fresh stage
    # ------------------------------------------------------------------ #
    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # ------------------------------------------------------------------ #
    # 2.  Root Xform  /quadrotor
    # ------------------------------------------------------------------ #
    root_path = Sdf.Path("/quadrotor")
    root_xform = UsdGeom.Xform.Define(stage, root_path)

    # Mark as an Articulation root so Isaac Lab recognises it
    UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root_xform.GetPrim())
    # Enable self-collision disable (common for drones)
    physx_art = PhysxSchema.PhysxArticulationAPI(root_xform.GetPrim())
    physx_art.CreateEnabledSelfCollisionsAttr(False)

    # Set as default prim so Isaac Lab can reference it by just the USD path
    stage.SetDefaultPrim(root_xform.GetPrim())

    # ------------------------------------------------------------------ #
    # 3.  Body rigid body  /quadrotor/body
    # ------------------------------------------------------------------ #
    body_path = root_path.AppendChild("body")
    body_xform = UsdGeom.Xform.Define(stage, body_path)
    _add_rigid_body(body_xform.GetPrim())
    _set_mass(body_xform.GetPrim(), BODY_MASS)

    # Visual / collision box under body
    body_geom_path = body_path.AppendChild("body_geom")
    body_cube = UsdGeom.Cube.Define(stage, body_geom_path)
    # Cube in USD has size=2 by default, so we scale to half-extents
    body_cube.AddScaleOp().Set(
        Gf.Vec3f(
            BODY_HALF_EXTENTS[0],
            BODY_HALF_EXTENTS[1],
            BODY_HALF_EXTENTS[2],
        )
    )
    UsdPhysics.CollisionAPI.Apply(body_cube.GetPrim())

    # ------------------------------------------------------------------ #
    # 4.  Rotors  /quadrotor/rotor_N  (siblings of body, not children)
    # ------------------------------------------------------------------ #
    for idx, offset in enumerate(ROTOR_OFFSETS):
        rotor_name = f"rotor_{idx}"
        rotor_path = root_path.AppendChild(rotor_name)
        rotor_xform = UsdGeom.Xform.Define(stage, rotor_path)

        # Position relative to articulation root
        rotor_xform.AddTranslateOp().Set(offset)

        # Each link in the articulation needs its own RigidBodyAPI
        _add_rigid_body(rotor_xform.GetPrim())
        _set_mass(rotor_xform.GetPrim(), ROTOR_MASS)

        # Visual / collision cylinder
        cyl_path = rotor_path.AppendChild("rotor_geom")
        cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
        cyl.CreateRadiusAttr(ROTOR_RADIUS)
        cyl.CreateHeightAttr(ROTOR_HEIGHT)
        cyl.CreateAxisAttr(ROTOR_JOINT_AXIS)
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

        # ----- Revolute joint -----
        joint_path = rotor_path.AppendChild(f"{rotor_name}_joint")
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

        # body0 = parent (body), body1 = child (rotor)
        joint.CreateBody0Rel().SetTargets([body_path])
        joint.CreateBody1Rel().SetTargets([rotor_path])

        joint.CreateAxisAttr(ROTOR_JOINT_AXIS)

        # Local pose: joint frame at rotor offset in parent, origin in child
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        # No joint limits -- free spinning
        joint.CreateLowerLimitAttr(float("-inf"))
        joint.CreateUpperLimitAttr(float("inf"))

        # Drive: allow effort (torque) control
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(1e4)
        drive.CreateDampingAttr(0.0)
        drive.CreateStiffnessAttr(0.0)

    # ------------------------------------------------------------------ #
    # 5.  Physics scene (needed for standalone sim, harmless for Lab)
    # ------------------------------------------------------------------ #
    physics_scene_path = Sdf.Path("/physicsScene")
    physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # ------------------------------------------------------------------ #
    # 6.  Save
    # ------------------------------------------------------------------ #
    stage.GetRootLayer().Save()
    print(f"[create_quadrotor_usd] Saved USD to: {OUTPUT_PATH}", flush=True)
    return OUTPUT_PATH


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        path = create_quadrotor_usd()
        print(f"[create_quadrotor_usd] SUCCESS  ->  {path}", flush=True)
    except Exception as exc:
        print(f"[create_quadrotor_usd] FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        app.close()
