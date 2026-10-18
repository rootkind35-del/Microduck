"""Adversarial stress-testing suite for scripts/infer_stairs.py.

Empirically challenges:
1. Actuator voltage extremes:
   - Low voltage (--vin 4.0), high voltage (--vin 12.0), and intermediate spectrum.
   - Exact torque limit computation (force_limit = vin * kt / R).
   - MuJoCo actuator forcerange alignment and torque saturation.
   - Verification of zero NaNs and zero math domain errors.
2. Perturbations and stability:
   - +/- 10 N impulse forces applied along X, Y, Z to the robot trunk on every stair flight
     (floor, flight1 2.5cm, landing 1, flight2 3.0cm, landing 2, flight3 3.5cm, summit).
   - Contact forces and control torques saturate properly without exploding into NaNs or Infs.
   - Severe stress perturbations (up to 100 N).
3. Standing / falling state transitions and telemetry classification:
   - Angular boundaries for BALANCED, UNSTABLE, and FALLEN states.
   - Complete physical lifecycle: standing -> topple -> FALLEN -> standup recovery -> BALANCED.
   - Step contact geom identification and stair flight zone tracking.
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


# =============================================================================
# 1. Actuator Voltage Extremes Tests
# =============================================================================

@pytest.mark.parametrize("vin", [4.0, 5.0, 6.0, 7.4, 8.2, 10.0, 12.0])
def test_bam_voltage_torque_limits_and_forcerange(infer_mod, vin):
    """Verify torque limit computation and actuator_forcerange across voltage extremes."""
    bam_model = infer_mod.load_bam_model(kp_fw=infer_mod.BAM_KP_FW, vin=vin, max_current=infer_mod.BAM_MAX_CURRENT)
    kt = bam_model.kt.value
    R = bam_model.R.value

    # Theoretical stall torque limit at voltage vin
    expected_limit = vin * kt / R
    assert expected_limit > 0.0, f"Expected positive torque limit, got {expected_limit}"

    model, data, bam_ctrl, act_names = infer_mod.load_mujoco_with_bam(
        str(SCENE_STAIRS_PATH), bam_model=bam_model, timestep=infer_mod.PHYSICS_DT
    )

    # Invariants check
    assert model.nu == 14, "Must have exactly 14 actuators"
    assert (model.actuator_forcelimited == 1).all(), "All actuators must be force-limited"
    assert np.allclose(model.actuator_forcerange[:, 1], expected_limit, rtol=1e-5), (
        f"actuator_forcerange positive bound mismatch: {model.actuator_forcerange[0, 1]} vs {expected_limit}"
    )
    assert np.allclose(model.actuator_forcerange[:, 0], -expected_limit, rtol=1e-5), (
        f"actuator_forcerange negative bound mismatch: {model.actuator_forcerange[0, 0]} vs {-expected_limit}"
    )

    # Run 50 steps of simulation under this voltage
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)
    last_action = np.zeros(14, dtype=np.float32)
    cmd_twist = np.array([0.05, 0.0, 0.0], dtype=np.float32)

    for step in range(50):
        obs = infer_mod.build_61d_observation(model, data, last_action, cmd_twist)
        assert not np.isnan(obs).any(), f"NaN in obs at vin={vin}, step={step}"

        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        assert not np.isnan(data.qpos).any(), f"NaN in qpos at vin={vin}, step={step}"
        assert not np.isnan(data.qvel).any(), f"NaN in qvel at vin={vin}, step={step}"
        assert not np.isnan(data.ctrl).any(), f"NaN in ctrl at vin={vin}, step={step}"

        # Torque saturation check
        max_applied_torque = float(np.max(np.abs(data.ctrl[:14])))
        assert max_applied_torque <= expected_limit + 1e-5, (
            f"Applied torque {max_applied_torque} exceeded voltage limit {expected_limit} at vin={vin}"
        )


@pytest.mark.parametrize("vin", [4.0, 12.0])
def test_infer_stairs_cli_headless_voltage_extremes(infer_mod, vin):
    """Verify headless CLI execution with low (--vin 4.0) and high (--vin 12.0) voltages."""
    import types
    args = types.SimpleNamespace(
        scene=str(SCENE_STAIRS_PATH),
        spawn="flight1",
        policy=None,
        headless=True,
        duration_s=0.2,
        lin_vel_x=0.02,
        vin=vin,
        debug=True,
        telemetry_hz=5.0,
    )

    ret = infer_mod.run_simulation(args)
    assert ret == 0, f"run_simulation failed with return code {ret} at vin={vin}"


# =============================================================================
# 2. Perturbation and Stability Tests Across All Stair Flights
# =============================================================================

@pytest.mark.parametrize("preset", [
    "bottom",
    "flight1",
    "landing_1",
    "flight2",
    "landing_2",
    "flight3",
    "top",
])
@pytest.mark.parametrize("force_dir", [
    (10.0, 0.0, 0.0),    # +10 N Sagittal forward
    (-10.0, 0.0, 0.0),   # -10 N Sagittal backward
    (0.0, 10.0, 0.0),    # +10 N Lateral left
    (0.0, -10.0, 0.0),   # -10 N Lateral right
    (0.0, 0.0, 10.0),    # +10 N Vertical upward
    (0.0, 0.0, -10.0),   # -10 N Vertical downward
])
def test_trunk_impulse_perturbation_stability(infer_mod, preset, force_dir):
    """Apply +/- 10 N impulse forces to robot trunk across all stair flights and verify stability."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert trunk_id >= 0, "trunk_base body must exist in model"

    infer_mod.reset_to_preset(model, data, preset, bam_ctrl=bam_ctrl)
    torque_limit = model.actuator_forcerange[0, 1]

    # Apply 10 N force impulse for one control step (4 physics substeps)
    data.xfrc_applied[trunk_id, :3] = force_dir
    for _ in range(infer_mod.DECIMATION):
        bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
        bam_ctrl.update()
        mujoco.mj_step(model, data)
    data.xfrc_applied[trunk_id, :] = 0.0

    # Step for 10 further control steps (0.2s) to observe response
    for _ in range(10):
        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # 1. NaN guard
        assert not np.isnan(data.qpos).any(), f"NaN in qpos under perturbation {force_dir} at {preset}"
        assert not np.isnan(data.qvel).any(), f"NaN in qvel under perturbation {force_dir} at {preset}"
        assert not np.isnan(data.ctrl).any(), f"NaN in ctrl under perturbation {force_dir} at {preset}"

        # 2. Torque saturation check
        max_ctrl = float(np.max(np.abs(data.ctrl[:14])))
        assert max_ctrl <= torque_limit + 1e-4, f"Torque {max_ctrl} exceeded limit {torque_limit}"

        # 3. Contact forces bounded check
        contact_l, contact_r, fn_l, fn_r = infer_mod.identify_step_contacts_and_forces(model, data)
        assert fn_l >= 0.0 and fn_r >= 0.0, "Normal forces must be non-negative"
        assert fn_l < 500.0 and fn_r < 500.0, f"Exploding contact force: fn_l={fn_l}, fn_r={fn_r}"

    # Verify telemetry works after perturbation
    telem = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem["status"] in ("BALANCED", "UNSTABLE", "FALLEN")
    assert not math.isnan(telem["pitch_deg"])
    assert not math.isnan(telem["roll_deg"])


@pytest.mark.parametrize("extreme_force", [20.0, 50.0, 100.0])
def test_extreme_forces_torque_saturation(infer_mod, extreme_force):
    """Stress-test system with extreme forces up to 100 N and verify torque saturation."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    torque_limit = model.actuator_forcerange[0, 1]

    infer_mod.reset_to_preset(model, data, "flight2", bam_ctrl=bam_ctrl)

    # Apply extreme force
    data.xfrc_applied[trunk_id, :3] = [extreme_force, -extreme_force, 0.0]
    for _ in range(infer_mod.DECIMATION):
        bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
        bam_ctrl.update()
        mujoco.mj_step(model, data)
    data.xfrc_applied[trunk_id, :] = 0.0

    for _ in range(15):
        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        assert not np.isnan(data.qpos).any(), f"NaN in qpos at extreme force {extreme_force}N"
        assert not np.isnan(data.qvel).any(), f"NaN in qvel at extreme force {extreme_force}N"

        # Torque saturation MUST hold even under 100 N impact
        max_ctrl = float(np.max(np.abs(data.ctrl[:14])))
        assert max_ctrl <= torque_limit + 1e-4, f"Torque {max_ctrl} exceeded limit {torque_limit}"


# =============================================================================
# 3. Telemetry Classification and State Transitions
# =============================================================================

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Helper converting roll, pitch, yaw to quaternion [w, x, y, z]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


@pytest.mark.parametrize("pitch_deg, roll_deg, expected_status", [
    (0.0, 0.0, "BALANCED"),
    (15.0, 10.0, "BALANCED"),
    (-25.0, -20.0, "BALANCED"),
    (34.0, 0.0, "BALANCED"),
    (0.0, 29.0, "BALANCED"),
    (38.0, 0.0, "UNSTABLE"),
    (-38.0, 0.0, "UNSTABLE"),
    (0.0, 35.0, "UNSTABLE"),
    (0.0, -35.0, "UNSTABLE"),
    (48.0, 0.0, "UNSTABLE"),
    (0.0, 43.0, "UNSTABLE"),
    (55.0, 0.0, "FALLEN"),
    (-65.0, 0.0, "FALLEN"),
    (0.0, 50.0, "FALLEN"),
    (0.0, -60.0, "FALLEN"),
    (80.0, 20.0, "FALLEN"),     # Face plant
    (-80.0, -10.0, "FALLEN"),   # Back fall
    (10.0, 90.0, "FALLEN"),     # Side tumble
    (170.0, 0.0, "FALLEN"),     # Inverted / upside down
])
def test_balance_telemetry_status_classification(infer_mod, pitch_deg, roll_deg, expected_status):
    """Verify tilt classification rules for BALANCED, UNSTABLE, and FALLEN states."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    q = _euler_to_quat(math.radians(roll_deg), math.radians(pitch_deg), 0.0)
    data.qpos[3:7] = q

    telem = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem["status"] == expected_status, (
        f"For P={pitch_deg}°, R={roll_deg}°: expected {expected_status}, got {telem['status']} "
        f"(measured P={telem['pitch_deg']:.1f}°, R={telem['roll_deg']:.1f}°)"
    )


def test_dynamic_standing_falling_recovery_lifecycle(infer_mod):
    """Verify physical transition from standing -> topple/falling -> standup recovery -> settled."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    # Phase 1: Standing upright
    telem_init = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem_init["status"] == "BALANCED", f"Initial state must be BALANCED, got {telem_init['status']}"

    # Phase 2: Apply strong lateral destabilizing impulse to induce a fall
    data.qvel[1] = -2.0  # lateral velocity spike
    for _ in range(60):
        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    telem_fallen = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem_fallen["status"] == "FALLEN", (
        f"Expected to transition to FALLEN after tumble, got {telem_fallen['status']} "
        f"(z={data.qpos[2]:.3f}, roll={telem_fallen['roll_deg']:.1f}°)"
    )
    assert data.qpos[2] < 0.08, "Robot trunk height must be collapsed"

    # Phase 3: Trigger standup recovery (matching key 'u' behavior in infer_stairs.py)
    if abs(data.qpos[2]) < 0.08:
        data.qpos[2] = 0.125
        data.qvel[:] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # restore upright quaternion
    data.qpos[7:7 + len(infer_mod.DEFAULT_JOINT_POS)] = infer_mod.DEFAULT_JOINT_POS
    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    # Phase 4: Step with PD controller to allow settling
    for _ in range(50):
        for _ in range(infer_mod.DECIMATION):
            bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    telem_recovered = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
    assert telem_recovered["status"] == "BALANCED", (
        f"Expected state to return to BALANCED after recovery, got {telem_recovered['status']}"
    )
    assert data.qpos[2] > 0.10, f"Expected standing height > 0.10m, got {data.qpos[2]:.3f}m"


def test_stair_zone_and_contact_geom_mapping(infer_mod):
    """Verify zone classification and contact geom identification along the stair sequence."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))

    zone_checks = [
        ("bottom", "Floor (Apron)"),
        ("flight1", "Flight 1"),
        ("landing_1", "Landing 1"),
        ("flight2", "Flight 2"),
        ("landing_2", "Landing 2"),
        ("flight3", "Flight 3"),
        ("top", "Summit"),
    ]

    for preset_name, expected_zone_substr in zone_checks:
        infer_mod.reset_to_preset(model, data, preset_name, bam_ctrl=bam_ctrl)
        telem = infer_mod.compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)
        assert expected_zone_substr in telem["zone"], (
            f"Preset '{preset_name}' reported unexpected zone: '{telem['zone']}' "
            f"(expected substring '{expected_zone_substr}')"
        )
        assert np.isclose(telem["pos"][0], data.qpos[0], atol=1e-5), (
            f"X position mismatch: {telem['pos'][0]} vs {data.qpos[0]}"
        )
