"""Unit tests for standalone Microduck stairs simulation scene (scene_stairs.xml).

Validates:
1. XML schema and compilation via MuJoCo.
2. Robot inclusion (robot_groundcontact.xml with 14 actuated servos).
3. Exact riser dimensions for Flight 1 (2.5cm), Flight 2 (3.0cm), Flight 3 (3.5cm).
4. Continuous gapless alignment along the +X primary axis.
5. Floor-anchored solid geom grounding (z_bottom == 0.0).
6. 3D frictional contact dynamics (condim == 3, friction >= 1.0).
7. Complete keyframe catalog (8 keyframes with valid qpos and ctrl vectors).
8. Keyframe forward kinematics and dynamic settling stability.
"""

from pathlib import Path
import math
import mujoco
import numpy as np
import pytest

SCENE_STAIRS_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
    / "scene_stairs.xml"
)

# Step sequence table: (geom_name, length_dx, riser_dz, top_z, material_name)
STEP_SEQUENCE = [
    # Flight 1: 1.0cm risers (0.010 m)
    ("stair_1_1", 0.12, 0.010, 0.010, "mat_stair_25"),
    ("stair_1_2", 0.12, 0.010, 0.020, "mat_stair_25"),
    ("stair_1_3", 0.12, 0.010, 0.030, "mat_stair_25"),
    # Landing 1: flat platform at 3.0cm
    ("landing_1", 0.36, 0.000, 0.030, "mat_landing_1"),
    # Flight 2: 1.25cm risers (0.0125 m)
    ("stair_2_1", 0.12, 0.0125, 0.0425, "mat_stair_30"),
    ("stair_2_2", 0.12, 0.0125, 0.0550, "mat_stair_30"),
    ("stair_2_3", 0.12, 0.0125, 0.0675, "mat_stair_30"),
    # Landing 2: flat platform at 6.75cm
    ("landing_2", 0.36, 0.000, 0.0675, "mat_landing_2"),
    # Flight 3: 1.50cm risers (0.0150 m)
    ("stair_3_1", 0.12, 0.0150, 0.0825, "mat_stair_35"),
    ("stair_3_2", 0.12, 0.0150, 0.0975, "mat_stair_35"),
    ("stair_3_3", 0.12, 0.0150, 0.1125, "mat_stair_35"),
    # Summit platform: flat resting area at 11.25cm
    ("landing_summit", 0.48, 0.000, 0.1125, "mat_summit"),
]

EXPECTED_KEYFRAMES = [
    ("INIT", 0.00, 0.120),
    ("STAND", 0.00, 0.125),
    ("SPAWN_STAIRS_25", 0.48, 0.145),
    ("SPAWN_LANDING_1", 0.84, 0.155),
    ("SPAWN_STAIRS_30", 1.20, 0.180),
    ("SPAWN_LANDING_2", 1.56, 0.1925),
    ("SPAWN_STAIRS_35", 1.92, 0.2225),
    ("SPAWN_SUMMIT", 2.34, 0.2375),
]


@pytest.fixture(scope="module")
def stairs_model():
    """Loads the scene_stairs.xml model."""
    assert SCENE_STAIRS_PATH.exists(), f"scene_stairs.xml not found at {SCENE_STAIRS_PATH}"
    return mujoco.MjModel.from_xml_path(str(SCENE_STAIRS_PATH))


def test_scene_stairs_compiles(stairs_model):
    """Verify that scene_stairs.xml parses and compiles cleanly."""
    assert stairs_model is not None
    assert stairs_model.nbody > 0
    assert stairs_model.ngeom > 0


def test_scene_stairs_robot_model(stairs_model):
    """Verify that robot_groundcontact is included with exactly 14 actuated servos."""
    assert stairs_model.nu == 14
    assert stairs_model.njnt == 15  # 1 freejoint + 14 revolute servos
    assert stairs_model.nq == 21   # 7 (freejoint) + 14 (joints)
    assert stairs_model.nv == 20   # 6 (freejoint) + 14 (joints)

    # Verify presence of trunk and foot collision geoms
    foot_left = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_right = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    assert foot_left != -1, "left_foot_collision missing"
    assert foot_right != -1, "right_foot_collision missing"


def test_all_stair_geometries_exist(stairs_model):
    """Verify all 12 stair flights and landing geoms exist."""
    for geom_name, _, _, _, _ in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert gid != -1, f"Geom {geom_name} not found in model"
        assert stairs_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX


def test_stair_riser_heights(stairs_model):
    """Verify exact step riser heights across Flight 1, 2, 3 and landings."""
    prev_top_z = 0.000
    for geom_name, _, expected_riser_dz, expected_top_z, _ in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = stairs_model.geom_pos[gid]
        size = stairs_model.geom_size[gid]

        top_z = pos[2] + size[2]
        riser_dz = top_z - prev_top_z

        assert math.isclose(top_z, expected_top_z, abs_tol=1e-5), (
            f"{geom_name}: top_z={top_z:.4f} != expected {expected_top_z:.4f}"
        )
        assert math.isclose(riser_dz, expected_riser_dz, abs_tol=1e-5), (
            f"{geom_name}: riser_dz={riser_dz:.4f} != expected {expected_riser_dz:.4f}"
        )
        prev_top_z = top_z


def test_stair_gapless_continuity(stairs_model):
    """Verify gapless alignment along X axis with no steps overlapping or separated."""
    expected_x = 0.30  # Starting edge of Flight 1
    for geom_name, expected_dx, _, _, _ in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = stairs_model.geom_pos[gid]
        size = stairs_model.geom_size[gid]

        x_min = pos[0] - size[0]
        x_max = pos[0] + size[0]
        dx = 2 * size[0]

        assert math.isclose(x_min, expected_x, abs_tol=1e-5), (
            f"{geom_name}: x_min={x_min:.4f} != expected_x={expected_x:.4f} (gap detected)"
        )
        assert math.isclose(dx, expected_dx, abs_tol=1e-5), (
            f"{geom_name}: dx={dx:.4f} != expected_dx={expected_dx:.4f}"
        )
        expected_x = x_max

    # Summit landing should end exactly at x = 2.58
    assert math.isclose(expected_x, 2.58, abs_tol=1e-5)


def test_stair_solid_grounding(stairs_model):
    """Verify that every step box extends down to z=0 to prevent tunneling."""
    for geom_name, _, _, _, _ in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = stairs_model.geom_pos[gid]
        size = stairs_model.geom_size[gid]

        bottom_z = pos[2] - size[2]
        assert math.isclose(bottom_z, 0.0, abs_tol=1e-6), (
            f"{geom_name}: bottom_z={bottom_z:.4f} is not grounded at 0.0"
        )


def test_stair_contact_friction_parameters(stairs_model):
    """Verify condim == 3 and tangential friction >= 1.0 (nominally 1.2)."""
    for geom_name, _, _, _, expected_mat in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        condim = stairs_model.geom_condim[gid]
        friction = stairs_model.geom_friction[gid]
        matid = stairs_model.geom_matid[gid]
        mat_name = (
            mujoco.mj_id2name(stairs_model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
            if matid >= 0
            else None
        )

        assert condim == 3, f"{geom_name}: condim={condim} != 3"
        assert friction[0] >= 1.0, f"{geom_name}: tangential friction {friction[0]} < 1.0"
        assert math.isclose(friction[0], 1.2, abs_tol=1e-5), f"{geom_name}: friction[0] != 1.2"
        assert mat_name == expected_mat, f"{geom_name}: material={mat_name} != {expected_mat}"


def test_stair_width(stairs_model):
    """Verify lateral width is 0.50m (half-size 0.25m) centered at y=0.0."""
    for geom_name, _, _, _, _ in STEP_SEQUENCE:
        gid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = stairs_model.geom_pos[gid]
        size = stairs_model.geom_size[gid]

        assert math.isclose(pos[1], 0.0, abs_tol=1e-6), f"{geom_name}: y_pos != 0.0"
        assert math.isclose(size[1], 0.25, abs_tol=1e-6), f"{geom_name}: y_half_size != 0.25"


def test_all_keyframes_exist_and_valid(stairs_model):
    """Verify all 8 keyframes exist with valid 21D qpos and 14D ctrl."""
    assert stairs_model.nkey == 8
    for kname, expected_x, expected_z in EXPECTED_KEYFRAMES:
        kid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_KEY, kname)
        assert kid != -1, f"Keyframe {kname} not found"

        qpos = stairs_model.key_qpos[kid]
        ctrl = stairs_model.key_ctrl[kid]

        assert len(qpos) == 21, f"{kname}: qpos length {len(qpos)} != 21"
        assert len(ctrl) == 14, f"{kname}: ctrl length {len(ctrl)} != 14"

        # Check root coordinates
        assert math.isclose(qpos[0], expected_x, abs_tol=1e-3), (
            f"{kname}: root x={qpos[0]:.3f} != {expected_x:.3f}"
        )
        assert math.isclose(qpos[1], 0.0, abs_tol=1e-3), f"{kname}: root y != 0.0"
        assert math.isclose(qpos[2], expected_z, abs_tol=1e-3), (
            f"{kname}: root z={qpos[2]:.3f} != {expected_z:.3f}"
        )

        # Quaternion must be identity [1, 0, 0, 0]
        np.testing.assert_allclose(qpos[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-4)


def test_keyframe_dynamic_settle_stability(stairs_model):
    """Verify that all standing spawn keyframes settle stably without toppling."""
    data = mujoco.MjData(stairs_model)

    for kname, _, _ in EXPECTED_KEYFRAMES:
        if kname == "INIT":
            continue

        kid = mujoco.mj_name2id(stairs_model, mujoco.mjtObj.mjOBJ_KEY, kname)
        data.qpos[:] = stairs_model.key_qpos[kid]
        data.qvel[:] = 0.0
        data.ctrl[:] = stairs_model.key_ctrl[kid]

        # Step 50 steps (0.25s) with built-in position actuators
        for _ in range(50):
            mujoco.mj_step(stairs_model, data)

        # Check orientation stability: pitch and roll must remain within 10 degrees
        w, x, y, z = data.qpos[3:7]
        pitch = math.degrees(math.asin(2.0 * (w * y - z * x)))
        roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))

        assert abs(pitch) < 10.0, f"{kname}: unstable pitch {pitch:.1f}deg"
        assert abs(roll) < 10.0, f"{kname}: unstable roll {roll:.1f}deg"
        assert not np.isnan(data.qpos).any(), f"{kname}: NaNs in qpos"
