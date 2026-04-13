"""Validate the quadrotor USD articulation structure.

Run with:
    C:\\isaacsim\\python.bat scripts\\validate_quadrotor_usd.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    import osqp  # noqa: F401
except ImportError:
    pass

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "create_new_stage": False})

from pxr import Usd, UsdGeom, UsdPhysics, Sdf  # noqa: E402

USD_PATH = (
    r"C:\Users\brend\OneDrive\Desktop\AI-GP_V2\aigp\assets\usd\quadrotor.usd"
)
REPORT_PATH = (
    r"C:\Users\brend\OneDrive\Desktop\AI-GP_V2\scripts\validation_report.txt"
)


def validate() -> bool:
    lines: list[str] = []

    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        lines.append("FAIL: Could not open stage")
        with open(REPORT_PATH, "w") as f:
            f.write("\n".join(lines))
        return False

    errors: list[str] = []
    passes: list[str] = []

    # 1. Default prim
    default_prim = stage.GetDefaultPrim()
    if not default_prim or default_prim.GetPath() != Sdf.Path("/quadrotor"):
        errors.append(f"Default prim is {default_prim.GetPath()}, expected /quadrotor")
    else:
        passes.append(f"Default prim: {default_prim.GetPath()}")

    # 2. Articulation root
    root = stage.GetPrimAtPath("/quadrotor")
    if root and root.HasAPI(UsdPhysics.ArticulationRootAPI):
        passes.append("ArticulationRootAPI on /quadrotor")
    else:
        errors.append("Missing ArticulationRootAPI on /quadrotor")

    # 3. Body rigid body
    body = stage.GetPrimAtPath("/quadrotor/body")
    if body and body.HasAPI(UsdPhysics.RigidBodyAPI):
        passes.append("RigidBodyAPI on /quadrotor/body")
    else:
        errors.append("Missing RigidBodyAPI on /quadrotor/body")

    if body and body.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(body)
        mass = mass_api.GetMassAttr().Get()
        passes.append(f"Body mass: {mass} kg")
    else:
        errors.append("Missing MassAPI on body")

    # 4. Body geometry
    body_geom = stage.GetPrimAtPath("/quadrotor/body/body_geom")
    if body_geom and body_geom.IsA(UsdGeom.Cube):
        passes.append("Body geometry is Cube")
    else:
        errors.append("Missing or wrong body geometry")

    # 5. Rotors and joints
    for i in range(4):
        rotor_path = f"/quadrotor/body/rotor_{i}"
        joint_path = f"{rotor_path}/rotor_{i}_joint"

        rotor = stage.GetPrimAtPath(rotor_path)
        if rotor and rotor.HasAPI(UsdPhysics.RigidBodyAPI):
            passes.append(f"RigidBodyAPI on {rotor_path}")
        else:
            errors.append(f"Missing RigidBodyAPI on {rotor_path}")

        joint = stage.GetPrimAtPath(joint_path)
        if joint and joint.IsA(UsdPhysics.RevoluteJoint):
            passes.append(f"RevoluteJoint at {joint_path}")
        else:
            errors.append(f"Missing RevoluteJoint at {joint_path}")

        if joint and joint.HasAPI(UsdPhysics.DriveAPI):
            passes.append(f"DriveAPI on {joint_path}")
        else:
            errors.append(f"Missing DriveAPI on {joint_path}")

        cyl_path = f"{rotor_path}/rotor_geom"
        cyl = stage.GetPrimAtPath(cyl_path)
        if cyl and cyl.IsA(UsdGeom.Cylinder):
            passes.append(f"Cylinder at {cyl_path}")
        else:
            errors.append(f"Missing Cylinder at {cyl_path}")

    # 6. Up axis
    up = UsdGeom.GetStageUpAxis(stage)
    if up == UsdGeom.Tokens.z:
        passes.append("Up axis: Z")
    else:
        errors.append(f"Up axis is {up}, expected Z")

    # 7. Meters per unit
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    passes.append(f"Meters per unit: {mpu}")

    # Summary
    lines.append("=" * 60)
    lines.append(f"  Quadrotor USD Validation: {USD_PATH}")
    lines.append("=" * 60)
    for p in passes:
        lines.append(f"  [PASS] {p}")
    for e in errors:
        lines.append(f"  [FAIL] {e}")
    lines.append("=" * 60)
    lines.append(f"  {len(passes)} passed, {len(errors)} failed")
    lines.append("=" * 60)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    return len(errors) == 0


if __name__ == "__main__":
    try:
        ok = validate()
        sys.exit(0 if ok else 1)
    finally:
        app.close()
