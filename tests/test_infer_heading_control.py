import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_keyboard_command_isolation_and_heading_lock(ip):
    scene_path = str(REPO / "src/mjlab_microduck/robot/microduck/scene.xml")
    walking_path = str(REPO / "walking_rough.onnx")

    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(scene_path, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)

    policy = ip.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=walking_path,
        new_cmd_obs=True,
    )

    # 1. UP arrow commands pure straight walking [vel_max_x, 0, 0]
    policy.set_vel_cmd(policy.vel_max_x, 0.0, 0.0)
    assert np.allclose(policy.vel_cmd, [policy.vel_max_x, 0.0, 0.0])
    obs = policy.get_observations()
    # In 61D obs layout: twist command is at indices [48:51]
    assert np.allclose(obs[48:51], [policy.vel_max_x, 0.0, 0.0])

    # 2. Steering with A key commands yaw turn
    policy.set_vel_cmd(policy.nominal_vel_x, 0.0, policy.vel_max_ang)
    assert np.allclose(policy.vel_cmd, [policy.vel_max_x, 0.0, policy.vel_max_ang])

    # 3. Pressing UP again MUST cleanly clear the turn command
    policy.set_vel_cmd(policy.vel_max_x, 0.0, 0.0)
    assert np.allclose(policy.vel_cmd, [policy.vel_max_x, 0.0, 0.0])
    obs = policy.get_observations()
    assert np.allclose(obs[48:51], [policy.vel_max_x, 0.0, 0.0])

    # 4. Heading lock disengages during manual yaw-rate steering and re-engages
    #    on straight walk. Heading-Lock is ON by default (laser-straight manual
    #    walking); force it OFF first so the toggle path is exercised from a
    #    known state.
    policy.heading_lock_enabled = False
    policy.target_yaw = None
    policy.toggle_heading_lock()
    assert policy.heading_lock_enabled is True
    assert policy.target_yaw is not None

    # Manual yaw-rate command:
    policy.set_vel_cmd(policy.nominal_vel_x, 0.0, policy.vel_max_ang)
    assert policy.target_yaw is None

    # Straight walk:
    policy.set_vel_cmd(policy.vel_max_x, 0.0, 0.0)
    assert policy.target_yaw is not None

    # 5. Heading steering (LEFT/RIGHT under Heading-Lock): nudges the locked
    #    target and leaves NO persistent yaw-rate command behind.
    assert policy.steer_heading(math.radians(10.0)) is True
    assert policy.nominal_vel_ang == 0.0
    assert policy.vel_cmd[2] == 0.0
    assert policy.target_yaw is not None
    # Disabled lock declines steering (caller falls back to yaw-rate commands)
    # and toggling off clears the lock state:
    policy.toggle_heading_lock()
    assert policy.heading_lock_enabled is False
    assert policy.steer_heading(math.radians(10.0)) is False
    assert policy.target_yaw is None
    assert policy.vel_cmd[2] == 0.0
