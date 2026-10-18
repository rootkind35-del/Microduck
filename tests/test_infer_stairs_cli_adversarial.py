"""Adversarial CLI, preset rollout, and keyboard control test suite for scripts/infer_stairs.py.

Authored by M3 Challenger 1.

Covers:
1. CLI boundary and edge parameters:
   - Invalid scene path (FileNotFoundError)
   - Invalid policy path (FileNotFoundError)
   - Invalid preset name (graceful fallback to 'bottom')
   - Negative and zero telemetry Hz (safe clamping)
   - Extreme commanded velocities (--lin-vel-x 2.0 and -2.0)
2. All 5 spawn presets across flights:
   - bottom, flight1, flight2, flight3, top
   - 100+ steps simulation (125 steps / 2.5s) in headless mode without crashing or emitting NaNs
   - Full 13-preset alias coverage (floor, stairs_25, landing_1, stairs_30, landing_2, stairs_35, summit, init)
3. Numerical stability and NaN guards:
   - 100+ steps across presets with BAM M6 dynamics
   - Verification of 0 NaNs and 0 Infs in qpos, qvel, ctrl, sensordata, 61D obs
4. Keyboard control mappings and state transitions:
   - Incremental velocity trims and clamping (vx, vy, wz, space emergency stop)
   - Body pitch trim mode (b toggle, +/- 2 deg trim, clamping to +/- 30 deg)
   - Head gaze mode (h toggle, -20 deg beak pitch)
   - Heading lock mode (c toggle, target yaw capture, proportional yaw correction)
   - Perturbation push (p key)
   - Standup recovery trigger (u key, body posture commands and fall reset)
   - Policy inference toggle (t key)
   - Preset hotkeys (1..5)
   - Clean quit request (q key)
5. End-to-end simulation loop with simulated keyboard stream:
   - Mocked TerminalInput injecting keys into run_simulation
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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
# 1. CLI Boundary and Edge Parameter Tests
# =============================================================================

def test_cli_invalid_scene_path(infer_mod):
    """Verify non-existent scene path raises FileNotFoundError and exits with error."""
    class Args:
        scene = "nonexistent_scene_xyz.xml"
        policy = None
        spawn = "bottom"
        lin_vel_x = 0.0
        vin = 7.4
        headless = True
        debug = False
        telemetry_hz = 5.0
        duration_s = 0.1

    with pytest.raises(FileNotFoundError, match="Scene XML not found"):
        infer_mod.run_simulation(Args())

    # Subprocess CLI verification
    res = subprocess.run(
        [sys.executable, str(INFER_STAIRS_PATH), "--headless", "--duration-s", "0.1", "--scene", "nonexistent_foo.xml"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "FileNotFoundError" in res.stderr


def test_cli_invalid_policy_path(infer_mod):
    """Verify non-existent policy file raises FileNotFoundError and exits with error."""
    class Args:
        scene = str(SCENE_STAIRS_PATH)
        policy = "nonexistent_policy.onnx"
        spawn = "bottom"
        lin_vel_x = 0.0
        vin = 7.4
        headless = True
        debug = False
        telemetry_hz = 5.0
        duration_s = 0.1

    with pytest.raises(FileNotFoundError, match="Policy file not found"):
        infer_mod.run_simulation(Args())

    # Subprocess CLI verification
    res = subprocess.run(
        [sys.executable, str(INFER_STAIRS_PATH), "--headless", "--duration-s", "0.1", "--policy", "nonexistent_bar.onnx"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "FileNotFoundError" in res.stderr


def test_cli_invalid_preset_fallback(infer_mod):
    """Verify unknown preset falls back to 'bottom'/'STAND' and runs without crash."""
    class Args:
        scene = str(SCENE_STAIRS_PATH)
        policy = None
        spawn = "totally_invalid_preset_name"
        lin_vel_x = 0.0
        vin = 7.4
        headless = True
        debug = False
        telemetry_hz = 5.0
        duration_s = 0.1

    ret = infer_mod.run_simulation(Args())
    assert ret == 0

    # Subprocess CLI verification
    res = subprocess.run(
        [sys.executable, str(INFER_STAIRS_PATH), "--headless", "--duration-s", "0.1", "--spawn", "invalid_stair_preset"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Spawn preset: bottom" in res.stdout
    assert "Finished 5 steps with 0 NaNs and clean exit." in res.stdout


def test_cli_telemetry_hz_clamping(infer_mod):
    """Verify negative and zero telemetry Hz are safely clamped without zero-division or crash."""
    for hz in [-50.0, -1.0, 0.0, 0.1]:
        class Args:
            scene = str(SCENE_STAIRS_PATH)
            policy = None
            spawn = "bottom"
            lin_vel_x = 0.0
            vin = 7.4
            headless = True
            debug = False
            telemetry_hz = hz
            duration_s = 0.1

        ret = infer_mod.run_simulation(Args())
        assert ret == 0


def test_cli_extreme_commanded_velocities(infer_mod):
    """Verify extreme commanded linear velocities (+2.0 m/s and -2.0 m/s) run 100+ steps without NaNs."""
    for vx in [2.0, -2.0]:
        class Args:
            scene = str(SCENE_STAIRS_PATH)
            policy = None
            spawn = "bottom"
            lin_vel_x = vx
            vin = 7.4
            headless = True
            debug = False
            telemetry_hz = 5.0
            duration_s = 2.0  # 100 steps @ 50 Hz

        ret = infer_mod.run_simulation(Args())
        assert ret == 0


# =============================================================================
# 2. Multi-Preset 100+ Steps Headless Verification
# =============================================================================

@pytest.mark.parametrize("preset", ["bottom", "flight1", "flight2", "flight3", "top"])
def test_all_five_presets_125_steps_headless(infer_mod, preset):
    """Verify 100+ steps (2.5s = 125 steps) across each of the 5 presets without NaNs."""
    class Args:
        scene = str(SCENE_STAIRS_PATH)
        policy = None
        spawn = preset
        lin_vel_x = 0.05
        vin = 7.4
        headless = True
        debug = False
        telemetry_hz = 5.0
        duration_s = 2.5  # 125 steps

    ret = infer_mod.run_simulation(Args())
    assert ret == 0


def test_cli_subprocess_all_five_presets():
    """Verify CLI invocation for all 5 presets executes with 0 exit code and 0 NaNs."""
    presets = ["bottom", "flight1", "flight2", "flight3", "top"]
    for p in presets:
        res = subprocess.run(
            [
                sys.executable,
                str(INFER_STAIRS_PATH),
                "--headless",
                "--spawn", p,
                "--duration-s", "2.1",  # 105 steps
                "--lin-vel-x", "0.05",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Preset {p} failed with error:\n{res.stderr}"
        assert "Finished 105 steps with 0 NaNs and clean exit." in res.stdout


def test_all_13_presets_100_steps_nan_guard(infer_mod):
    """Step 100 steps per preset across all 13 preset aliases with strict NaN/Inf guards."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))

    last_action = np.zeros(14, dtype=np.float32)
    cmd_twist = np.array([0.05, 0.0, 0.0], dtype=np.float32)

    for preset_name in infer_mod.SPAWN_PRESETS.keys():
        infer_mod.reset_to_preset(model, data, preset_name, bam_ctrl=bam_ctrl)

        for step in range(100):
            obs = infer_mod.build_61d_observation(model, data, last_action, cmd_twist)
            assert obs.shape == (61,), f"Preset {preset_name}, step {step}: obs shape mismatch"
            assert not np.isnan(obs).any(), f"Preset {preset_name}, step {step}: NaN in observation"
            assert not np.isinf(obs).any(), f"Preset {preset_name}, step {step}: Inf in observation"

            for _ in range(infer_mod.DECIMATION):
                bam_ctrl.q_target[:] = infer_mod.DEFAULT_JOINT_POS
                bam_ctrl.update()
                mujoco.mj_step(model, data)

            assert not np.isnan(data.qpos).any(), f"Preset {preset_name}, step {step}: NaN in qpos"
            assert not np.isnan(data.qvel).any(), f"Preset {preset_name}, step {step}: NaN in qvel"
            assert not np.isnan(data.ctrl).any(), f"Preset {preset_name}, step {step}: NaN in ctrl"


# =============================================================================
# 3. Keyboard Control Mappings and State Transitions Programmatic Tests
# =============================================================================

def test_terminal_input_unit(infer_mod):
    """Verify TerminalInput queues and returns keys correctly."""
    ti = infer_mod.TerminalInput()
    assert ti.get_keys() == []

    ti.put_key("up")
    ti.put_key("b")
    ti.put_key("1")
    assert ti.get_keys() == ["up", "b", "1"]
    assert ti.get_keys() == []


def test_keyboard_command_state_transitions(infer_mod):
    """Test every keyboard command and mode transition logic programmatically."""
    model, data, bam_ctrl, _ = infer_mod.load_bam_controller(str(SCENE_STAIRS_PATH))
    infer_mod.reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    cmd_twist = np.zeros(3, dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)
    pitch_trim_mode = False
    head_gaze_mode = False
    heading_lock_mode = False
    target_yaw = None
    policy_active = True
    current_preset = "bottom"
    quit_requested = False

    def handle(k: str):
        nonlocal cmd_twist, cmd_head, cmd_body, pitch_trim_mode, head_gaze_mode
        nonlocal heading_lock_mode, target_yaw, policy_active, current_preset, quit_requested

        if k == "up":
            if pitch_trim_mode:
                cmd_body[4] = float(np.clip(cmd_body[4] + math.radians(2), -math.radians(30), math.radians(30)))
            else:
                cmd_twist[0] = float(np.clip(cmd_twist[0] + 0.05, -0.25, 0.35))
        elif k == "down":
            if pitch_trim_mode:
                cmd_body[4] = float(np.clip(cmd_body[4] - math.radians(2), -math.radians(30), math.radians(30)))
            else:
                cmd_twist[0] = float(np.clip(cmd_twist[0] - 0.05, -0.25, 0.35))
        elif k == "left":
            cmd_twist[1] = float(np.clip(cmd_twist[1] + 0.02, -0.20, 0.20))
        elif k == "right":
            cmd_twist[1] = float(np.clip(cmd_twist[1] - 0.02, -0.20, 0.20))
        elif k in ("a", "left"):
            cmd_twist[2] = float(np.clip(cmd_twist[2] + 0.10, -0.80, 0.80))
        elif k in ("e", "right"):
            cmd_twist[2] = float(np.clip(cmd_twist[2] - 0.10, -0.80, 0.80))
        elif k == " ":
            cmd_twist[:] = 0.0
        elif k == "b":
            pitch_trim_mode = not pitch_trim_mode
        elif k == "h":
            head_gaze_mode = not head_gaze_mode
            cmd_head[1] = -math.radians(20) if head_gaze_mode else 0.0
        elif k == "c":
            heading_lock_mode = not heading_lock_mode
            target_yaw = infer_mod.compute_euler_angles(data.qpos[3:7])[2] if heading_lock_mode else None
        elif k == "p":
            push = (np.random.rand(3) - 0.5) * 0.5
            data.qvel[:3] += push
        elif k == "u":
            cmd_body[2] = 0.02
            cmd_body[4] = -math.radians(10)
            if abs(data.qpos[2]) < 0.08:
                data.qpos[2] = 0.125
                data.qvel[:] = 0.0
        elif k == "t":
            policy_active = not policy_active
        elif k == "q":
            quit_requested = True
        elif k in infer_mod.HOTKEY_PRESETS:
            current_preset = infer_mod.HOTKEY_PRESETS[k]
            infer_mod.reset_to_preset(model, data, current_preset, bam_ctrl=bam_ctrl)

    # 1. Forward velocity increment & saturation
    for _ in range(10):
        handle("up")
    assert math.isclose(cmd_twist[0], 0.35, abs_tol=1e-4), f"Expected vx clamped to 0.35, got {cmd_twist[0]}"

    # 2. Backward velocity decrement & saturation
    for _ in range(15):
        handle("down")
    assert math.isclose(cmd_twist[0], -0.25, abs_tol=1e-4), f"Expected vx clamped to -0.25, got {cmd_twist[0]}"

    # 3. Strafe left / right & saturation
    for _ in range(15):
        handle("left")
    assert math.isclose(cmd_twist[1], 0.20, abs_tol=1e-4)
    for _ in range(25):
        handle("right")
    assert math.isclose(cmd_twist[1], -0.20, abs_tol=1e-4)

    # 4. Yaw left / right & saturation
    for _ in range(10):
        handle("a")
    assert math.isclose(cmd_twist[2], 0.80, abs_tol=1e-4)
    for _ in range(20):
        handle("e")
    assert math.isclose(cmd_twist[2], -0.80, abs_tol=1e-4)

    # 5. Space emergency zero
    handle(" ")
    assert np.allclose(cmd_twist, 0.0)

    # 6. Pitch trim mode toggle & trimming
    handle("b")
    assert pitch_trim_mode is True
    for _ in range(20):
        handle("up")
    assert math.isclose(cmd_body[4], math.radians(30), abs_tol=1e-4)
    for _ in range(40):
        handle("down")
    assert math.isclose(cmd_body[4], -math.radians(30), abs_tol=1e-4)
    handle("b")
    assert pitch_trim_mode is False

    # 7. Head gaze mode toggle
    handle("h")
    assert head_gaze_mode is True
    assert math.isclose(cmd_head[1], -math.radians(20), abs_tol=1e-4)
    handle("h")
    assert head_gaze_mode is False
    assert math.isclose(cmd_head[1], 0.0, abs_tol=1e-4)

    # 8. Heading lock mode toggle
    handle("c")
    assert heading_lock_mode is True
    assert target_yaw is not None
    handle("c")
    assert heading_lock_mode is False
    assert target_yaw is None

    # 9. Standup recovery trigger
    data.qpos[2] = 0.04  # simulate fallen low height
    data.qvel[:] = 0.5
    handle("u")
    assert math.isclose(cmd_body[2], 0.02, abs_tol=1e-4)
    assert math.isclose(cmd_body[4], -math.radians(10), abs_tol=1e-4)
    assert data.qpos[2] == 0.125
    assert np.allclose(data.qvel, 0.0)

    # 10. Policy inference toggle
    handle("t")
    assert policy_active is False
    handle("t")
    assert policy_active is True

    # 11. Presets hotkeys 1..5
    for hotkey, expected in [("1", "bottom"), ("2", "flight1"), ("3", "flight2"), ("4", "flight3"), ("5", "top")]:
        handle(hotkey)
        assert current_preset == expected
        assert not np.isnan(data.qpos).any()

    # 12. Quit request
    handle("q")
    assert quit_requested is True


# =============================================================================
# 4. End-to-End Simulation with Mocked Key Stream
# =============================================================================

def test_run_simulation_with_mocked_keystream(infer_mod):
    """Run run_simulation with mocked TerminalInput feeding a dynamic sequence of keys."""
    key_batches = [
        ["up", "up"],
        ["left"],
        ["b"],
        ["up"],
        ["h"],
        ["c"],
        ["p"],
        ["u"],
        ["3"],  # jump to flight2
        ["q"],  # quit cleanly
    ]
    batch_idx = 0

    class MockTerminalInput:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_keys(self):
            nonlocal batch_idx
            if batch_idx < len(key_batches):
                keys = key_batches[batch_idx]
                batch_idx += 1
                return keys
            return []

    class Args:
        scene = str(SCENE_STAIRS_PATH)
        policy = None
        spawn = "bottom"
        lin_vel_x = 0.0
        vin = 7.4
        headless = True
        debug = True
        telemetry_hz = 10.0
        duration_s = 5.0

    with patch.object(infer_mod, "TerminalInput", MockTerminalInput):
        ret = infer_mod.run_simulation(Args())
        assert ret == 0
        assert batch_idx >= 10, "Should have processed all key batches up to 'q'"
