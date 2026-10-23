#!/usr/bin/env python3
"""Stage B (Part 2) — Bridge 2: MJCF / CAD -> URDF & NVIDIA Omniverse OpenUSD (`.usda`) Compiler.

Parses the authoritative Microduck robot specification (`robot_groundcontact.xml`)
and Stage A CAD terrain manifest (`01_freecad_cad/exports/cad_export_manifest.json`),
audits all 14 Dynamixel XL330 servo joints (joint limits, effort = 0.963 Nm,
velocity = 9.95 rad/s, BAM M6 stiffness/damping), and outputs:
  1. `02_blender_omniverse_usd/usd_assets/microduck.urdf` (ROS / Isaac URDF Importer ready)
  2. `02_blender_omniverse_usd/usd_assets/microduck_simready.usda` (Omniverse SimReady OpenUSD
     with `UsdPhysics.ArticulationRootAPI`, `UsdPhysics.RigidBodyAPI`, `UsdPhysics.MassAPI`,
     `UsdPhysics.CollisionAPI`, `UsdPhysics.RevoluteJoint`, and `UsdPhysics.DriveAPI:angular`)
  3. `02_blender_omniverse_usd/usd_assets/terrain_course.usda` (OpenUSD Stage referencing
     Stage A parametric stairs, hurdles, and robot articulation).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


STAGE_B_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_B_DIR.parent
DEFAULT_MJCF_PATH = (
    PROJECT_ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "robot_groundcontact.xml"
)
DEFAULT_STAGE_A_MANIFEST = PROJECT_ROOT / "01_freecad_cad" / "exports" / "cad_export_manifest.json"
DEFAULT_OUTPUT_DIR = STAGE_B_DIR / "usd_assets"

# Dynamixel XL330-M288-T physical & BAM M6 parameters
XL330_MAX_EFFORT_NM = 0.963
XL330_MAX_VELOCITY_RAD_S = 9.95
XL330_BAM_STIFFNESS_NM_RAD = 6.91
XL330_BAM_DAMPING_NMS_RAD = 0.258
XL330_ARMATURE_KGM2 = 0.0018

CANONICAL_14_SERVOS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


@dataclass
class GeomSpec:
    name: str
    mesh_name: str
    geom_class: str  # "visual", "collision", "self_collision_only"
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass
class JointSpec:
    name: str
    parent_body: str
    child_body: str
    axis: tuple[float, float, float]
    lower_rad: float
    upper_rad: float
    effort_nm: float = XL330_MAX_EFFORT_NM
    velocity_rad_s: float = XL330_MAX_VELOCITY_RAD_S
    stiffness: float = XL330_BAM_STIFFNESS_NM_RAD
    damping: float = XL330_BAM_DAMPING_NMS_RAD
    armature: float = XL330_ARMATURE_KGM2
    is_passive: bool = False


@dataclass
class BodySpec:
    name: str
    parent_name: str | None
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    mass_kg: float
    com_pos: tuple[float, float, float]
    fullinertia: tuple[float, float, float, float, float, float]
    joint: JointSpec | None = None
    geoms: list[GeomSpec] = field(default_factory=list)


def _parse_floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(x) for x in text.strip().split())


def _quat_wxyz_to_rpy(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    import warnings

    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rot = Rotation.from_quat([x, y, z, w])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rpy = rot.as_euler("xyz", degrees=False)
    return (float(rpy[0]), float(rpy[1]), float(rpy[2]))


def parse_mjcf_kinematic_tree(mjcf_path: Path) -> tuple[list[BodySpec], list[JointSpec]]:
    """Parse all bodies, inertial properties, joints, and visual/collision geoms from MJCF."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in {mjcf_path}")

    bodies: list[BodySpec] = []
    joints: list[JointSpec] = []

    def _walk_body(elem: ET.Element, parent_name: str | None) -> None:
        b_name = elem.attrib["name"]
        pos = _parse_floats(elem.attrib.get("pos"), (0.0, 0.0, 0.0))
        quat = _parse_floats(elem.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0))

        inertial = elem.find("inertial")
        if inertial is not None:
            mass = float(inertial.attrib.get("mass", "0.001"))
            com = _parse_floats(inertial.attrib.get("pos"), (0.0, 0.0, 0.0))
            full_i = _parse_floats(
                inertial.attrib.get("fullinertia"), (1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0)
            )
        else:
            mass = 0.001
            com = (0.0, 0.0, 0.0)
            full_i = (1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0)

        joint_spec: JointSpec | None = None
        joint_elem = elem.find("joint")
        if joint_elem is not None and parent_name is not None:
            j_name = joint_elem.attrib["name"]
            axis = _parse_floats(joint_elem.attrib.get("axis"), (0.0, 0.0, 1.0))
            j_range = _parse_floats(joint_elem.attrib.get("range"), (-1.5708, 1.5708))
            is_passive = j_name.startswith("passive_")
            joint_spec = JointSpec(
                name=j_name,
                parent_body=parent_name,
                child_body=b_name,
                axis=(axis[0], axis[1], axis[2]),
                lower_rad=float(j_range[0]),
                upper_rad=float(j_range[1]),
                is_passive=is_passive,
            )
            joints.append(joint_spec)

        geoms: list[GeomSpec] = []
        for g_idx, g_elem in enumerate(elem.findall("geom")):
            mesh_name = g_elem.attrib.get("mesh", "")
            if not mesh_name:
                continue
            g_class = g_elem.attrib.get("class", "visual")
            g_name = g_elem.attrib.get("name", f"{b_name}_{g_class}_{g_idx}")
            g_pos = _parse_floats(g_elem.attrib.get("pos"), (0.0, 0.0, 0.0))
            g_quat = _parse_floats(g_elem.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0))
            geoms.append(
                GeomSpec(
                    name=g_name,
                    mesh_name=mesh_name,
                    geom_class=g_class,
                    pos=(g_pos[0], g_pos[1], g_pos[2]),
                    quat_wxyz=(g_quat[0], g_quat[1], g_quat[2], g_quat[3]),
                )
            )

        body_spec = BodySpec(
            name=b_name,
            parent_name=parent_name,
            pos=(pos[0], pos[1], pos[2]),
            quat_wxyz=(quat[0], quat[1], quat[2], quat[3]),
            mass_kg=mass,
            com_pos=(com[0], com[1], com[2]),
            fullinertia=(full_i[0], full_i[1], full_i[2], full_i[3], full_i[4], full_i[5]),
            joint=joint_spec,
            geoms=geoms,
        )
        bodies.append(body_spec)

        for child_body in elem.findall("body"):
            _walk_body(child_body, b_name)

    for top_body in worldbody.findall("body"):
        _walk_body(top_body, None)

    return bodies, joints


def export_urdf(
    bodies: list[BodySpec],
    joints: list[JointSpec],
    urdf_path: Path,
    mesh_rel_prefix: str = "../../src/mjlab_microduck/robot/microduck/assets",
) -> None:
    """Write standard `microduck.urdf` compatible with NVIDIA Isaac Lab `UrdfConverter`."""
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!-- Auto-generated by Argon Stage B Bridge from robot_groundcontact.xml -->',
        '<robot name="microduck">',
    ]

    for b in bodies:
        lines.append(f'  <link name="{b.name}">')
        cx, cy, cz = b.com_pos
        ixx, iyy, izz, ixy, ixz, iyz = b.fullinertia
        lines.append("    <inertial>")
        lines.append(f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{b.mass_kg:.6f}"/>')
        lines.append(
            f'      <inertia ixx="{ixx:.8e}" ixy="{ixy:.8e}" ixz="{ixz:.8e}" '
            f'iyy="{iyy:.8e}" iyz="{iyz:.8e}" izz="{izz:.8e}"/>'
        )
        lines.append("    </inertial>")

        for g in b.geoms:
            gx, gy, gz = g.pos
            rr, rp, ry = _quat_wxyz_to_rpy(g.quat_wxyz)
            mesh_uri = f"{mesh_rel_prefix}/{g.mesh_name}.stl"
            if g.geom_class == "visual":
                lines.append(f'    <visual name="{g.name}">')
                lines.append(
                    f'      <origin xyz="{gx:.6f} {gy:.6f} {gz:.6f}" rpy="{rr:.6f} {rp:.6f} {ry:.6f}"/>'
                )
                lines.append("      <geometry>")
                lines.append(f'        <mesh filename="{mesh_uri}" scale="1.0 1.0 1.0"/>')
                lines.append("      </geometry>")
                lines.append("    </visual>")
            elif g.geom_class == "collision":
                lines.append(f'    <collision name="{g.name}">')
                lines.append(
                    f'      <origin xyz="{gx:.6f} {gy:.6f} {gz:.6f}" rpy="{rr:.6f} {rp:.6f} {ry:.6f}"/>'
                )
                lines.append("      <geometry>")
                lines.append(f'        <mesh filename="{mesh_uri}" scale="1.0 1.0 1.0"/>')
                lines.append("      </geometry>")
                lines.append("    </collision>")

        lines.append("  </link>")

    body_by_name = {b.name: b for b in bodies}
    for j in joints:
        child = body_by_name[j.child_body]
        bx, by, bz = child.pos
        rr, rp, ry = _quat_wxyz_to_rpy(child.quat_wxyz)
        ax, ay, az = j.axis
        lines.append(f'  <joint name="{j.name}" type="revolute">')
        lines.append(f'    <parent link="{j.parent_body}"/>')
        lines.append(f'    <child link="{j.child_body}"/>')
        lines.append(
            f'    <origin xyz="{bx:.6f} {by:.6f} {bz:.6f}" rpy="{rr:.6f} {rp:.6f} {ry:.6f}"/>'
        )
        lines.append(f'    <axis xyz="{ax:.4f} {ay:.4f} {az:.4f}"/>')
        lines.append(
            f'    <limit lower="{j.lower_rad:.6f}" upper="{j.upper_rad:.6f}" '
            f'effort="{j.effort_nm:.4f}" velocity="{j.velocity_rad_s:.4f}"/>'
        )
        lines.append(f'    <dynamics damping="{j.damping:.4f}" friction="0.0048"/>')
        lines.append("  </joint>")

    lines.append("</robot>")
    urdf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_simready_robot_usda(
    bodies: list[BodySpec],
    joints: list[JointSpec],
    usda_path: Path,
    mesh_rel_prefix: str = "../../src/mjlab_microduck/robot/microduck/assets",
) -> None:
    """Write an Omniverse SimReady OpenUSD ASCII (`.usda`) articulation file for Microduck.

    Applies `PhysicsArticulationRootAPI`, `PhysicsRigidBodyAPI`, `PhysicsMassAPI`,
    `PhysicsCollisionAPI`, `PhysicsRevoluteJoint`, and `PhysicsDriveAPI:angular`
    conforming to OpenUSD `UsdPhysics` schema.
    """
    lines: list[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Microduck"',
        '    doc = "SimReady OpenUSD Articulation for Microduck Bipedal Robot (14 XL330 Servos, BAM M6)"',
        "    metersPerUnit = 1.0",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Microduck" (',
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI"]',
        ")",
        "{",
        '    def PhysicsScene "PhysicsScene"',
        "    {",
        "        vector3f physics:gravityDirection = (0.0, 0.0, -1.0)",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
        "",
        '    def Scope "Links"',
        "    {",
    ]

    for b in bodies:
        px, py, pz = b.pos
        qw, qx, qy, qz = b.quat_wxyz
        cx, cy, cz = b.com_pos
        ixx, iyy, izz, _, _, _ = b.fullinertia
        lines.extend(
            [
                f'        def Xform "{b.name}" (',
                '            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
                "        )",
                "        {",
                f"            double3 xformOp:translate = ({px:.6f}, {py:.6f}, {pz:.6f})",
                f"            quatd xformOp:orient = ({qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                "            bool physics:rigidBodyEnabled = true",
                f"            float physics:mass = {b.mass_kg:.6f}",
                f"            point3f physics:centerOfMass = ({cx:.6f}, {cy:.6f}, {cz:.6f})",
                f"            float3 physics:diagonalInertia = ({ixx:.8e}, {iyy:.8e}, {izz:.8e})",
            ]
        )

        for g in b.geoms:
            gx, gy, gz = g.pos
            gqw, gqx, gqy, gqz = g.quat_wxyz
            is_col = g.geom_class == "collision"
            schema_str = (
                ' (prepend apiSchemas = ["PhysicsCollisionAPI"])' if is_col else ""
            )
            safe_gname = g.name.replace("-", "_").replace(".", "_")
            lines.extend(
                [
                    f'            def Xform "{safe_gname}"{schema_str}',
                    "            {",
                    f'                asset info:meshAsset = @{mesh_rel_prefix}/{g.mesh_name}.stl@',
                    f'                string info:purpose = "{g.geom_class}"',
                    f"                double3 xformOp:translate = ({gx:.6f}, {gy:.6f}, {gz:.6f})",
                    f"                quatd xformOp:orient = ({gqw:.6f}, {gqx:.6f}, {gqy:.6f}, {gqz:.6f})",
                    '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                    "            }",
                ]
            )

        lines.append("        }")

    lines.append("    }")
    lines.append("")
    lines.append('    def Scope "Joints"')
    lines.append("    {")

    for j in joints:
        lower_deg = math.degrees(j.lower_rad)
        upper_deg = math.degrees(j.upper_rad)
        axis_token = "Z"
        if abs(j.axis[0]) > 0.9:
            axis_token = "X"
        elif abs(j.axis[1]) > 0.9:
            axis_token = "Y"

        lines.extend(
            [
                f'        def PhysicsRevoluteJoint "{j.name}" (',
                '            prepend apiSchemas = ["PhysicsDriveAPI:angular"]',
                "        )",
                "        {",
                f'            uniform token physics:axis = "{axis_token}"',
                f"            rel physics:body0 = </Microduck/Links/{j.parent_body}>",
                f"            rel physics:body1 = </Microduck/Links/{j.child_body}>",
                f"            float physics:lowerLimit = {lower_deg:.4f}",
                f"            float physics:upperLimit = {upper_deg:.4f}",
                f"            float physics:armature = {j.armature:.6f}",
                '            uniform token drive:angular:physics:type = "force"',
                f"            float drive:angular:physics:maxForce = {j.effort_nm:.4f}",
                f"            float drive:angular:physics:stiffness = {j.stiffness:.4f}",
                f"            float drive:angular:physics:damping = {j.damping:.4f}",
                "        }",
            ]
        )

    lines.append("    }")
    lines.append("}")
    usda_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_terrain_course_usda(
    stage_a_manifest_path: Path,
    terrain_usda_path: Path,
) -> dict[str, Any]:
    """Compile Stage A parametric CAD primitives into an Omniverse SimReady Terrain USD Stage."""
    if not stage_a_manifest_path.exists():
        raise FileNotFoundError(
            f"Stage A manifest not found at {stage_a_manifest_path}. Run Stage A first."
        )

    manifest = json.loads(stage_a_manifest_path.read_text(encoding="utf-8"))
    stairs_model = manifest["models"].get("nc_high_stairs_course", {})
    hurdles_model = manifest["models"].get("step_over_hurdles", {})
    vision_model = manifest["models"].get("vision_see_and_walk_course", {})

    lines: list[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    doc = "Argon Stage B OpenUSD Terrain & Microduck SimReady Environment (with OpenGL Head Camera & ToF)"',
        "    metersPerUnit = 1.0",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "MicroduckRobot" (',
        '        prepend references = @./microduck_simready.usda@',
        "    )",
        "    {",
        "        double3 xformOp:translate = (0.0, 0.0, 0.125)",
        '        uniform token[] xformOpOrder = ["xformOp:translate"]',
        '        def Scope "OnboardPerceptionSensors"',
        "        {",
        '            def Camera "head_camera"',
        "            {",
        "                float focalLength = 18.0",
        "                float horizontalAperture = 20.955",
        "                float verticalAperture = 15.2908",
        "                float2 clippingRange = (0.02, 10.0)",
        "                double3 xformOp:translate = (0.0155, -0.000091, -0.0733)",
        '                uniform token[] xformOpOrder = ["xformOp:translate"]',
        "            }",
        '            def Xform "tof_rangefinder"',
        "            {",
        "                double3 xformOp:translate = (0.0135, 0.022409, -0.0733)",
        '                uniform token[] xformOpOrder = ["xformOp:translate"]',
        "            }",
        "        }",
        "    }",
        "",
        '    def Scope "ParametricTerrain"',
        "    {",
    ]

    prim_count = 0
    for group_name, model_data in [
        ("HighStairsCourse", stairs_model),
        ("StepOverHurdles", hurdles_model),
        ("VisionSeeAndWalkCourse", vision_model),
    ]:
        lines.append(f'        def Xform "{group_name}"')
        lines.append("        {")
        for prim in model_data.get("primitives", []):
            prim_count += 1
            pname = prim["name"]
            cx, cy, cz = prim["center_xyz"]
            lx, ly, lz = prim["extents_xyz"]
            r, g, b, _ = prim.get("rgba", [0.5, 0.5, 0.5, 1.0])
            lines.extend(
                [
                    f'            def Cube "{pname}" (',
                    '                prepend apiSchemas = ["PhysicsCollisionAPI"]',
                    "            )",
                    "            {",
                    "                double size = 1.0",
                    f"                double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})",
                    f"                float3 xformOp:scale = ({lx:.6f}, {ly:.6f}, {lz:.6f})",
                    '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                    f"                color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]",
                    "            }",
                ]
            )
        lines.append("        }")

    lines.extend(
        [
            "    }",
            "}",
        ]
    )
    usda_text = "\n".join(lines) + "\n"
    terrain_usda_path.write_text(usda_text, encoding="utf-8")
    vision_usda_path = terrain_usda_path.parent / "vision_obstacle_course.usda"
    vision_usda_path.write_text(usda_text, encoding="utf-8")
    return {
        "terrain_usda": str(terrain_usda_path),
        "vision_obstacle_course_usda": str(vision_usda_path),
        "collider_primitives": prim_count,
    }


def build_urdf_and_openusd_assets(
    mjcf_path: Path = DEFAULT_MJCF_PATH,
    stage_a_manifest: Path = DEFAULT_STAGE_A_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run full Stage B conversion: MJCF + CAD -> URDF + SimReady OpenUSD (`.usda`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bodies, joints = parse_mjcf_kinematic_tree(mjcf_path)

    actuated_joints = [j for j in joints if not j.is_passive]
    actuated_names = tuple(j.name for j in actuated_joints)
    total_mass_kg = sum(b.mass_kg for b in bodies)
    collision_geoms_count = sum(
        1 for b in bodies for g in b.geoms if g.geom_class == "collision"
    )
    visual_geoms_count = sum(1 for b in bodies for g in b.geoms if g.geom_class == "visual")

    # Audit against canonical 14 servo order
    servo_set_matches = set(actuated_names) == set(CANONICAL_14_SERVOS)
    all_limits_valid = all(j.lower_rad < j.upper_rad for j in actuated_joints)

    urdf_path = output_dir / "microduck.urdf"
    robot_usda_path = output_dir / "microduck_simready.usda"
    terrain_usda_path = output_dir / "terrain_course.usda"

    export_urdf(bodies, joints, urdf_path)
    export_simready_robot_usda(bodies, joints, robot_usda_path)
    terrain_info = export_terrain_course_usda(stage_a_manifest, terrain_usda_path)

    audit_summary = {
        "stage": "Stage B (Part 2) - URDF & Omniverse OpenUSD SimReady Compiler",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_mjcf": str(mjcf_path),
        "total_links": len(bodies),
        "total_joints": len(joints),
        "actuated_servo_count": len(actuated_joints),
        "canonical_14_servo_match": servo_set_matches,
        "all_joint_limits_valid": all_limits_valid,
        "total_robot_mass_kg": round(total_mass_kg, 5),
        "visual_geoms_count": visual_geoms_count,
        "collision_geoms_count": collision_geoms_count,
        "xl330_drive_spec": {
            "max_effort_nm": XL330_MAX_EFFORT_NM,
            "max_velocity_rad_s": XL330_MAX_VELOCITY_RAD_S,
            "bam_stiffness_nm_rad": XL330_BAM_STIFFNESS_NM_RAD,
            "bam_damping_nms_rad": XL330_BAM_DAMPING_NMS_RAD,
        },
        "outputs": {
            "urdf": str(urdf_path),
            "robot_simready_usda": str(robot_usda_path),
            "terrain_course_usda": terrain_info["terrain_usda"],
            "terrain_collider_primitives": terrain_info["collider_primitives"],
        },
        "joints_detail": [asdict(j) for j in actuated_joints],
    }

    report_path = output_dir / "usd_conversion_manifest.json"
    report_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage B (Part 2): Convert Microduck MJCF & Stage A CAD to URDF and OpenUSD."
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF_PATH)
    parser.add_argument("--stage-a-manifest", type=Path, default=DEFAULT_STAGE_A_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = build_urdf_and_openusd_assets(
        mjcf_path=args.mjcf,
        stage_a_manifest=args.stage_a_manifest,
        output_dir=args.output_dir,
    )
    print(
        f"[Stage B - USD Bridge] Links: {summary['total_links']} | "
        f"Servos: {summary['actuated_servo_count']} (14-Servo Match={summary['canonical_14_servo_match']}) | "
        f"Total Mass: {summary['total_robot_mass_kg']:.4f} kg | "
        f"Outputs: microduck.urdf, microduck_simready.usda, terrain_course.usda"
    )


if __name__ == "__main__":
    main()
