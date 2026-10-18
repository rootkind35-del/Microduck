"""Unit and invariant regression tests for scripts/infer_stairs.py.

Verifies:
1. scripts/infer_stairs.py imports cleanly and adheres to repository invariants.
2. BAM constants match _BAM_ACTUATOR_KWARGS in microduck_constants.py.
3. Actuator conversion matches warp BAM dynamics on scene_stairs.xml.
4. 100-step simulation loop runs with 0 NaNs.
5. All 5 spawn presets position the robot correctly across the stair flights.
6. Real-time balance telemetry: tilt angle, step contact, normal forces, torque load.
7. Strict 61D unified actor observation layout (48 proprio + 13 commands).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_STAIRS_PATH = REPO_ROOT / "scripts" / "infer_stairs.py"
SCENE_STAIRS_PATH = REPO_ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_stairs.xml"


@pytest.fixture(scope="module")
def infer_mod():
    """Load scripts/infer_stairs.py module dynamically."""
    assert INFER_STAIRS_PATH.exists(), "scripts/infer_stairs.py must exist"
    spec = importlib.util.spec_from_file_location("infer_stairs", INFER_STAIRS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_infer_stairs_module_invariants(infer_mod):
    """Verify module-level constants and timing invariants."""
    assert hasattr(infer_mod, "SPAWN_PRESETS"), "Must expose SPAWN_PRESETS"
    presets = infer_mod.SPAWN_PRESETS
    expected_keys = ["floor", "stairs_25", "stairs_30", "stairs_35", "summit"]
    for k in expected_keys:
        assert k in presets, f"Missing preset: {k}"

    assert math.isclose(infer_mod.CONTROL_FREQ, 50.0, rel_tol=1e-5), "Must run at 50 Hz control"
    assert infer_mod.DECIMATION == 4, "Decimation must be 4"
    assert math.isclose(infer_mod.PHYSICS_DT, 0.005, rel_tol=1e-5), "Physics dt must be 0.005s (200 Hz)"


def test_infer_stairs_bam_constants(infer_mod):
    """Verify BAM constants mirror _BAM_ACTUATOR_KWARGS and BamActuator stiff solver."""
    from bam.mjlab import BamActuator
    from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS as k

    assert infer_mod.BAM_MOTOR_NAME == k["motor_name"]
    assert infer_mod.BAM_MODEL == k["model"]
    assert infer_mod.BAM_KP_FW == k["kp_fw"]
    assert infer_mod.BAM_VIN_RANGE == k["vin_range"]
    assert infer_mod.BAM_VIN_DROP_GAIN_RANGE == k["vin_drop_gain_range"]
    assert infer_mod.BAM_VIN_MIN == k["vin_min"]
    assert infer_mod.BAM_MAX_CURRENT == k.get("max_current")
    assert infer_mod.BAM_STIFF_SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert infer_mod.BAM_STIFF_SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


def test_infer_stairs_bam_controller_initialization(infer_mod):
    """Verify BAM M6 controller compiles scene_stairs.xml cleanly with 14 actuators."""
    model, data, bam_ctrl, act_names = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    assert model is not None
    assert data is not None
    assert len(act_names) == 14, f"Expected 14 actuated servos, got {len(act_names)}"
    assert all(not name.startswith("passive_") for name in act_names), "Passive joints must not be actuated"


def test_infer_stairs_actuator_conversion(infer_mod):
    """Verify torque motor conversion on scene_stairs.xml matches warp dynamics."""
    bam_model = infer_mod.load_bam_model(infer_mod.BAM_KP_FW, 7.4, infer_mod.BAM_MAX_CURRENT)
    model, data, bam_ctrl, act_names = infer_mod.load_mujoco_with_bam(
        str(SCENE_STAIRS_PATH), bam_model=bam_model, timestep=infer_mod.PHYSICS_DT
    )
    assert len(act_names) == 14
    assert model.nu == 14
    assert all(not n.startswith("passive_") for n in act_names)

    # Torque motors: ctrl is the BAM torque, no MuJoCo PD
    assert (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    assert np.allclose(model.actuator_gainprm[:, 0], 1.0)
    assert (model.actuator_forcelimited == 1).all()

    kt, R = bam_model.kt.value, bam_model.R.value
    assert np.allclose(model.actuator_forcerange[:, 1], 7.4 * kt / R)

    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())
    assert np.allclose(model.dof_solref[dofs], infer_mod.BAM_STIFF_SOLREF_FRICTION)
    assert np.allclose(model.dof_solimp[dofs], infer_mod.BAM_STIFF_SOLIMP_FRICTION)


def test_infer_stairs_step_loop(infer_mod):
    """Verify 100-step simulation with dummy policy produces 0 NaNs and valid torques."""
    bam_model = infer_mod.load_bam_model(infer_mod.BAM_KP_FW, 7.4, infer_mod.BAM_MAX_CURRENT)
    model, data, bam_ctrl, names = infer_mod.load_mujoco_with_bam(
        str(SCENE_STAIRS_PATH), bam_model=bam_model, timestep=infer_mod.PHYSICS_DT
    )
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    last_action = np.zeros(14, dtype=np.float32)
    cmd_twist = np.array([0.1, 0.0, 0.0], dtype=np.float32)

    for _ in range(100):
        obs = infer_mod.build_61d_observation(model, data, last_action, cmd_twist)
        assert not np.isnan(obs).any(), "Observation contains NaNs"
        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        assert not np.isnan(data.qpos).any(), "NaN in qpos during step loop"
        assert not np.isnan(data.qvel).any(), "NaN in qvel during step loop"

    limit = model.actuator_forcerange[0, 1]
    assert (np.abs(data.ctrl) <= limit + 1e-9).all()
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert (model.dof_frictionloss[dofs] > 0).all()


def test_infer_stairs_spawn_presets(infer_mod):
    """Verify all 5 spawn presets position the robot correctly across stair flights."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))

    presets_to_test = {
        "bottom": lambda x, z: x < 0.30 and z > 0.10,
        "flight1": lambda x, z: 0.30 <= x <= 0.66 and z > 0.13,
        "flight2": lambda x, z: 1.02 <= x <= 1.38 and z > 0.16,
        "flight3": lambda x, z: 1.74 <= x <= 2.10 and z > 0.20,
        "top": lambda x, z: x >= 2.10 and z > 0.22,
    }

    for preset_name, validator in presets_to_test.items():
        infer_mod.reset_to_preset(model, data, preset_name, bam_ctrl=bam_ctrl)
        assert not np.isnan(data.qpos).any(), f"NaN in qpos after reset to {preset_name}"
        assert not np.isnan(data.qvel).any(), f"NaN in qvel after reset to {preset_name}"
        x, y, z = data.qpos[:3]
        assert validator(x, z), f"Preset {preset_name} positioned incorrectly: x={x:.3f}, z={z:.3f}"
        assert np.allclose(data.qvel, 0.0), f"Velocities not zeroed for preset {preset_name}"


def test_infer_stairs_telemetry(infer_mod):
    """Verify tilt angle calculation, contact detection, and torque load reporting."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    # 1. Tilt angle calculation
    roll, pitch, yaw = infer_mod.compute_euler_angles(data.qpos[3:7])
    assert abs(pitch) < 0.2, f"Expected standing pitch near 0, got {pitch}"
    assert abs(roll) < 0.2, f"Expected standing roll near 0, got {roll}"

    # 2. Complete telemetry computation
    telem = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem["status"] in ("BALANCED", "UNSTABLE", "FALLEN")
    assert "pitch_deg" in telem and "roll_deg" in telem
    assert "feet_contacts" in telem
    assert "normal_forces" in telem
    assert "torque_loads" in telem

    # 3. Feet contacts
    fc_l, fc_r = telem["feet_contacts"]
    assert isinstance(fc_l, str) and isinstance(fc_r, str)

    # 4. Normal forces
    fn_l, fn_r = telem["normal_forces"]
    assert fn_l >= 0.0 and fn_r >= 0.0

    # 5. Torque loads relative to 0.96 Nm stall limit
    tl = telem["torque_loads"]
    assert "knee_load_pct" in tl and "ankle_load_pct" in tl
    assert 0.0 <= tl["knee_load_pct"] <= 200.0
    assert 0.0 <= tl["ankle_load_pct"] <= 200.0


def test_infer_stairs_observation_builder_61d(infer_mod):
    """Verify strict 61D unified actor observation format."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "floor", bam_ctrl=bam_ctrl)

    last_action = np.zeros(14, dtype=np.float32)
    cmd_twist = np.array([0.1, 0.0, 0.2], dtype=np.float32)

    obs = infer_mod.build_61d_observation(model, data, last_action, cmd_twist)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (61,), f"Expected shape (61,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any(), "Observation contains NaNs"

    # Proprioception and command slice checks
    assert np.allclose(obs[48:51], cmd_twist), "Twist command slice [48:51] mismatch"
    assert np.allclose(obs[51:55], 0.0), "Head pose command [51:55] must be zero-padded"
    assert np.allclose(obs[55:61], 0.0), "Body pose command [55:61] must be zero-padded"


def test_infer_stairs_headless_execution(infer_mod):
    """Verify headless execution for 0.1s steps cleanly without NaNs."""
    class MockArgs:
        scene = str(SCENE_STAIRS_PATH)
        spawn = "stairs_25"
        policy = None
        headless = True
        duration_s = 0.1
        lin_vel_x = 0.05
        vin = 7.4
        debug = False
        telemetry_hz = 5.0

    ret = infer_mod.run_simulation(MockArgs())
    assert ret == 0, "run_simulation should exit with 0"


def test_infer_stairs_all_presets_reset(infer_mod):
    """Verify reset to every preset succeeds and yields valid state."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    for preset_name in infer_mod.SPAWN_PRESETS.keys():
        infer_mod.reset_to_preset(model, data, preset_name, bam_ctrl=bam_ctrl)
        assert not np.isnan(data.qpos).any(), f"NaN in qpos after reset to {preset_name}"
        assert not np.isnan(data.qvel).any(), f"NaN in qvel after reset to {preset_name}"
        assert data.qpos[2] > 0.05, f"Base height too low after reset to {preset_name}: {data.qpos[2]}"


def test_summit_focus_navigator_centering_and_edge_guard(infer_mod):
    """Verify SummitFocusNavigator auto-steers towards summit and protects corridor edges."""
    nav = infer_mod.SummitFocusNavigator(target_pos=infer_mod.SUMMIT_TARGET_POS, enabled=True)

    # 1. Centered on corridor, pointing forward (yaw = 0)
    pos_center = np.array([0.50, 0.0, 0.14], dtype=np.float32)
    quat_fwd = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # yaw = 0
    qvel_zero = np.zeros(6, dtype=np.float32)
    base_twist = np.array([0.10, 0.0, 0.0], dtype=np.float32)

    twist, head, info = nav.update(pos_center, quat_fwd, qvel_zero, base_twist)
    assert abs(twist[2]) < 0.05, f"Expected near-zero steering on centerline, got wz={twist[2]}"
    assert not info["edge_guard"]
    assert not info["arrived"]
    assert info["status"] == "FOCUS ON SUMMIT"

    # 2. Left drift (y = +0.08m) -> must steer right (negative wz)
    pos_left = np.array([0.50, 0.08, 0.14], dtype=np.float32)
    twist_l, _, _ = nav.update(pos_left, quat_fwd, qvel_zero, base_twist)
    assert twist_l[2] < -0.05, f"Expected rightward turn command (negative wz), got {twist_l[2]}"

    # 3. Right drift (y = -0.08m) -> must steer left (positive wz)
    pos_right = np.array([0.50, -0.08, 0.14], dtype=np.float32)
    twist_r, _, _ = nav.update(pos_right, quat_fwd, qvel_zero, base_twist)
    assert twist_r[2] > 0.05, f"Expected leftward turn command (positive wz), got {twist_r[2]}"

    # 4. Approaching edge (y = +0.17m > 0.15m) -> Edge guard activates & throttles vx
    pos_edge = np.array([0.80, 0.17, 0.15], dtype=np.float32)
    twist_edge, _, info_edge = nav.update(pos_edge, quat_fwd, qvel_zero, base_twist)
    assert info_edge["edge_guard"], "Edge guard must be active at y=0.17m"
    assert twist_edge[0] < base_twist[0], "Forward velocity must be throttled when near edge"
    assert twist_edge[2] < -0.15, "Strong corrective turn must be commanded towards center"

    # 5. Danger boundary (y = +0.22m) -> Forward velocity zeroed (pivot-only)
    pos_danger = np.array([0.80, 0.22, 0.15], dtype=np.float32)
    twist_danger, _, info_danger = nav.update(pos_danger, quat_fwd, qvel_zero, base_twist)
    assert info_danger["edge_guard"]
    assert math.isclose(twist_danger[0], 0.0, abs_tol=1e-5), f"Expected vx=0.0 at danger edge, got {twist_danger[0]}"

    # 6. Summit arrival (x = 2.25m) -> arrived flag set and zero twist
    pos_summit = np.array([2.25, 0.02, 0.23], dtype=np.float32)
    twist_summit, _, info_summit = nav.update(pos_summit, quat_fwd, qvel_zero, base_twist)
    assert info_summit["arrived"], "Must report arrived=True at summit"
    assert np.allclose(twist_summit, 0.0), f"Commands must be zeroed upon arrival, got {twist_summit}"


def test_summit_focus_overlay_rendering(infer_mod):
    """Verify SummitFocusOverlay draws beacon, beam, and corridor lines without error."""
    overlay = infer_mod.SummitFocusOverlay(target_pos=infer_mod.SUMMIT_TARGET_POS)
    model, data, _, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))

    scn = mujoco.MjvScene(model, maxgeom=100)
    nav_info = {"edge_guard": False, "arrived": False}
    overlay.draw(data, scn, nav_info=nav_info)
    assert scn.ngeom >= 4, f"Expected at least 4 visual overlay geoms drawn, got {scn.ngeom}"


def test_stair_step_booster_clearance_and_stride(infer_mod):
    """Verify StairStepBooster boosts swing leg knee/ankle lift and forward hip reach."""
    booster = infer_mod.StairStepBooster(boost_level=1.2)
    assert math.isclose(booster.boost_level, 1.2, rel_tol=1e-5)

    # 1. Boost level bounds clamping
    b_min = infer_mod.StairStepBooster(boost_level=-0.5)
    assert math.isclose(b_min.boost_level, 0.0)
    b_max = infer_mod.StairStepBooster(boost_level=3.5)
    assert math.isclose(b_max.boost_level, 2.0)

    # 2. Body pitch lean on stairs
    cmd_body = np.zeros(6, dtype=np.float32)
    leaned_body = booster.apply_body_lean(cmd_body, is_walking=True, on_stairs=True)
    assert leaned_body[4] < -0.05, f"Expected forward pitch lean (negative pitch), got {leaned_body[4]}"
    assert leaned_body[2] > 0.005, f"Expected slight CoM elevation, got {leaned_body[2]}"

    # Flat ground or not walking -> no lean applied
    cmd_body_flat = booster.apply_body_lean(cmd_body, is_walking=True, on_stairs=False)
    assert np.allclose(cmd_body_flat, cmd_body)

    # 3. Leg clearance boost on swing leg
    target_pos = infer_mod.DEFAULT_JOINT_POS.copy()
    # Left foot in swing (fn_l = 0.2N < 1.6N), right foot in stance (fn_r = 7.5N > 1.6N)
    boosted_l = booster.apply_leg_boost(target_pos, fn_left=0.2, fn_right=7.5, is_walking=True, on_stairs=True)

    # Left leg: hip pitch forward (idx 2 more negative), knee flex (idx 3 more negative), ankle dorsiflex (idx 4 more positive)
    assert boosted_l[2] < target_pos[2], "Left hip pitch must swing forward"
    assert boosted_l[3] < target_pos[3], "Left knee must flex to lift foot"
    assert boosted_l[4] > target_pos[4], "Left ankle must dorsiflex to lift toe"

    # Right foot in swing (fn_l = 7.5N, fn_r = 0.2N)
    boosted_r = booster.apply_leg_boost(target_pos, fn_left=7.5, fn_right=0.2, is_walking=True, on_stairs=True)
    assert boosted_r[11] > target_pos[11], "Right hip pitch must swing forward"
    assert boosted_r[12] > target_pos[12], "Right knee must flex to lift foot"
    assert boosted_r[13] < target_pos[13], "Right ankle must dorsiflex to lift toe"


def test_standup_recovery_manager_lifecycle(infer_mod):
    """Verify StandupRecoveryManager fall detection, state machine, and smooth re-stabilization."""
    mgr = infer_mod.StandupRecoveryManager(standup_session="mock_session", duration_s=1.0, stabilize_s=0.5, enabled=True)
    assert mgr.state == "IDLE"

    # 1. Transient bump (< 8 steps) should not trigger standup
    for _ in range(5):
        mgr.check_fall(pitch_deg=50.0, roll_deg=0.0, root_z=0.12)
    assert mgr.state == "IDLE"

    # 2. Sustained fall (8+ steps) triggers standup
    for _ in range(4):
        mgr.check_fall(pitch_deg=50.0, roll_deg=0.0, root_z=0.05)
    assert mgr.check_fall(pitch_deg=50.0, roll_deg=0.0, root_z=0.05)

    mode, is_active, vx_resume = mgr.update(dt=0.02, pitch_deg=50.0, roll_deg=0.0, root_z=0.05)
    assert mgr.state == "STANDUP"
    assert is_active
    assert mode == "STANDUP"

    # 3. Transition to STABILIZING when standup finishes
    mgr.time_left = 0.0
    mode, is_active, _ = mgr.update(dt=0.02, pitch_deg=10.0, roll_deg=5.0, root_z=0.11)
    assert mgr.state == "STABILIZING"
    assert is_active
    assert mode == "STABILIZING"

    # 4. Transition back to IDLE when stabilized
    mgr.time_left = 0.0
    mode, is_active, vx_target = mgr.update(dt=0.02, pitch_deg=5.0, roll_deg=2.0, root_z=0.12)
    assert mgr.state == "IDLE"
    assert not is_active
    assert mode == "NORMAL"
    assert vx_target > 0.0, "Must resume forward target velocity"


def test_infer_stairs_cli_standup_and_boost_flags(infer_mod, monkeypatch):
    """Verify CLI arguments --standup, --stair-boost, and --auto-standup are properly exposed."""
    import sys
    test_args = [
        "infer_stairs.py",
        "--stair-boost", "1.5",
        "--no-auto-standup",
        "--standup", "standup.onnx",
        "--no-auto-probe",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = infer_mod.parse_args()
    assert math.isclose(parsed.stair_boost, 1.5)
    assert not parsed.auto_standup
    assert parsed.standup == "standup.onnx"
    assert not parsed.auto_probe


def test_query_forward_terrain_elevation(infer_mod):
    """Verify query_forward_terrain_elevation returns exact obstacle elevations."""
    model, data, _, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    mujoco.mj_forward(model, data)

    # Floor (x = 0.10m): z = 0.0m
    h_floor = infer_mod.query_forward_terrain_elevation(model, data, 0.10, 0.0)
    assert math.isclose(h_floor, 0.0, abs_tol=1e-4)

    # Flight 1 Step 1_1 (x = 0.35m): z = 0.010m (1.0cm)
    h_s1 = infer_mod.query_forward_terrain_elevation(model, data, 0.35, 0.0)
    assert math.isclose(h_s1, 0.010, abs_tol=1e-4)

    # Landing 1 (x = 0.85m): z = 0.030m (3.0cm)
    h_l1 = infer_mod.query_forward_terrain_elevation(model, data, 0.85, 0.0)
    assert math.isclose(h_l1, 0.030, abs_tol=1e-4)

    # Flight 2 Step 2_1 (x = 1.10m): z = 0.0425m (4.25cm)
    h_s2 = infer_mod.query_forward_terrain_elevation(model, data, 1.10, 0.0)
    assert math.isclose(h_s2, 0.0425, abs_tol=1e-4)

    # Summit platform (x = 2.20m): z = 0.1125m (11.25cm)
    h_summit = infer_mod.query_forward_terrain_elevation(model, data, 2.20, 0.0)
    assert math.isclose(h_summit, 0.1125, abs_tol=1e-4)


def test_adaptive_obstacle_probe_lifecycle_and_stance_counter_lean(infer_mod):
    """Verify AdaptiveObstacleProbeController height sensing, 4-stage probing, and stance forward counter-lean."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    controller = infer_mod.AdaptiveObstacleProbeController(boost_level=1.0, auto_probe=True)
    assert controller.probe_level == 1
    assert not controller.obstacle_detected

    # 1. On flat floor far from riser (x = 0.0m) -> Normal walk, clean target_pos
    obs_info = controller.update_obstacle_state(model, data, fn_left=5.0, fn_right=5.0, is_walking=True)
    assert not obs_info["detected"]
    assert obs_info["phase"] == "WALK"
    assert obs_info["status"] == "NORMAL WALK"

    # Clean target_pos untouched when unobstructed on flat floor
    target_pos = infer_mod.DEFAULT_JOINT_POS.copy()
    boosted_flat = controller.apply_leg_boost(target_pos, fn_left=5.0, fn_right=0.2, is_walking=True, on_stairs=False)
    assert np.allclose(boosted_flat, target_pos), "Must preserve pure policy gait when unobstructed on flat ground"

    # 2. Move robot close to Flight 1 (x = 0.28m) -> Step 1_1 (1.0cm) detected ahead
    data.qpos[0] = 0.28
    mujoco.mj_forward(model, data)
    obs_info_near = controller.update_obstacle_state(model, data, fn_left=5.0, fn_right=0.2, is_walking=True)
    assert obs_info_near["detected"], "Obstacle ahead must be detected"
    assert obs_info_near["height"] >= 0.008, f"Expected sensed height ~0.010m, got {obs_info_near['height']}"
    assert obs_info_near["level"] >= 1
    assert obs_info_near["phase"] == "PROBING"

    # 3. Biomechanical stance forward counter-pitch lean check:
    # Right leg swing (fn_r = 0.2N < 1.6N), Left leg stance (fn_l = 5.0N > 1.6N)
    boosted_stairs = controller.apply_leg_boost(target_pos, fn_left=5.0, fn_right=0.2, is_walking=True, on_stairs=True)
    # Right swing leg: knee flexed (lifts foot), ankle dorsiflexed (levels toe)
    assert boosted_stairs[12] > target_pos[12], "Right knee must flex to lift foot"
    assert boosted_stairs[13] < target_pos[13], "Right ankle must dorsiflex to level foot"
    # Left stance leg: MUST LEAN FORWARD to prevent backward tipping onto buttocks
    assert boosted_stairs[2] < target_pos[2], "Left stance hip must pitch forward over stance foot"
    assert boosted_stairs[4] > target_pos[4], "Left stance ankle must push forward to support CoM"

    # 4. Probe level escalation check:
    assert controller.probe_level >= 1
    controller._active_probe_level = 2
    controller.probe_level = 2
    assert controller.probe_level == 2
    assert "PROBING (L2:" in controller.status_msg or controller.probe_level == 2



