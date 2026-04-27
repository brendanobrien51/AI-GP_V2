"""Create a racing gate USD rigid body for Isaac Lab / Isaac Sim 4.5.

This script builds a gate frame consisting of four cuboid bars forming a
square opening that the drone can fly through.  Collision geometry is on
each bar individually, so the centre is passable.

The resulting USD is saved as a RigidBody root so Isaac Lab's
RigidObjectCfg can load it directly.

Run with:
    C:\\isaacsim\\python.bat scripts\\create_gate_usd.py
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
from pxr import (  # noqa: E402
    Gf,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

# --------------------------------------------------------------------------- #
# Configuration — matches aigp/assets/gate_cfg.py constants
# --------------------------------------------------------------------------- #
OUTPUT_PATH = (
    r"C:\Users\brend\OneDrive\Desktop\AI-GP_V2\aigp\assets\usd\racing_gate.usd"
)

GATE_OPENING_W = 1.5   # inner width (m)
GATE_OPENING_H = 1.5   # inner height (m)
GATE_THICKNESS = 0.1    # bar cross-section (m)

GATE_OUTER_W = GATE_OPENING_W + 2.0 * GATE_THICKNESS  # 1.7 m
GATE_OUTER_H = GATE_OPENING_H + 2.0 * GATE_THICKNESS  # 1.7 m

# Bar half-extents (USD Cube default size = 2, so scale = half-extent)
# Horizontal bars (top / bottom): span full outer width
HBAR_HALF = Gf.Vec3f(GATE_OUTER_W / 2.0, GATE_THICKNESS / 2.0, GATE_THICKNESS / 2.0)
# Vertical bars (left / right): span inner opening height
VBAR_HALF = Gf.Vec3f(GATE_THICKNESS / 2.0, GATE_OPENING_H / 2.0, GATE_THICKNESS / 2.0)

# Bar centre offsets from gate centroid (Y is up)
HALF_OPEN_H = GATE_OPENING_H / 2.0
HALF_OPEN_W = GATE_OPENING_W / 2.0
BAR_OFFSET = GATE_THICKNESS / 2.0

BARS = [
    ("bar_top",    Gf.Vec3d(0.0, HALF_OPEN_H + BAR_OFFSET, 0.0),  HBAR_HALF),
    ("bar_bottom", Gf.Vec3d(0.0, -(HALF_OPEN_H + BAR_OFFSET), 0.0), HBAR_HALF),
    ("bar_left",   Gf.Vec3d(-(HALF_OPEN_W + BAR_OFFSET), 0.0, 0.0), VBAR_HALF),
    ("bar_right",  Gf.Vec3d(HALF_OPEN_W + BAR_OFFSET, 0.0, 0.0),  VBAR_HALF),
]

# Safety-orange colour
GATE_COLOR = Gf.Vec3f(1.0, 0.4, 0.0)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_gate_usd() -> str:
    """Build the gate frame USD and save it to OUTPUT_PATH."""
    _ensure_dir(OUTPUT_PATH)

    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root Xform — /gate
    root_path = Sdf.Path("/gate")
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root_xform.GetPrim())

    # Mark as a rigid body so Isaac Lab's RigidObjectCfg recognises it
    UsdPhysics.RigidBodyAPI.Apply(root_xform.GetPrim())

    # Create each bar as a child Cube with collision
    for name, offset, half_ext in BARS:
        bar_path = root_path.AppendChild(name)
        bar_xform = UsdGeom.Xform.Define(stage, bar_path)
        bar_xform.AddTranslateOp().Set(offset)

        geom_path = bar_path.AppendChild("geom")
        cube = UsdGeom.Cube.Define(stage, geom_path)
        cube.AddScaleOp().Set(half_ext)

        # Collision on each bar individually
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # Visual colour
        mat_path = bar_path.AppendChild("material")
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(GATE_COLOR)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    # Physics scene
    physics_scene_path = Sdf.Path("/physicsScene")
    physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    stage.GetRootLayer().Save()
    print(f"[create_gate_usd] Saved USD to: {OUTPUT_PATH}", flush=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    try:
        path = create_gate_usd()
        print(f"[create_gate_usd] SUCCESS  ->  {path}", flush=True)
    except Exception as exc:
        print(f"[create_gate_usd] FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        app.close()
