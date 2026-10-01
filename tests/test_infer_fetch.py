import importlib.util
from pathlib import Path
import math
import numpy as np
import mujoco
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


def test_fetch_scene_and_ball_physics(ip):
    scene_path = str(REPO / "src/mjlab_microduck/robot/microduck/scene_fetch.xml")
    walking_path = str(REPO / "logs/rsl_rl/velocity/2026-09-27_23-17-48_velocity/2026-09-27_23-17-48_velocity.onnx")
    ground_pick_path = str(REPO / "logs/rsl_rl/ground_pick/2026-09-30_02-43-42_ground_pick/2026-09-30_02-43-42_ground_pick.onnx")

    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(scene_path, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)

    policy = ip.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=walking_path,
        ground_pick_onnx_path=ground_pick_path,
        new_cmd_obs=True,
        ground_pick_period=4.0
    )

    assert policy.ball_qpos_adr is not None
    assert policy.ball_qvel_adr is not None
    assert policy.mouth_tip_site_id >= 0
    assert policy.human_hand_geom_id >= 0
    assert policy.ball_geom_id >= 0

    # Test ball toss
    policy.toss_ball()
    assert policy.fetch_mode is True
    assert policy.fetch_state == "BALL_IN_FLIGHT"
    assert np.allclose(policy.vel_cmd, [0.0, 0.0, 0.0])

    # Check ball has initial forward velocity
    ball_vel = data.qvel[policy.ball_qvel_adr:policy.ball_qvel_adr + 3]
    assert ball_vel[0] > 1.0  # vx > 1 m/s
    assert ball_vel[2] > 0.4  # vz > 0.4 m/s (upward arc)


def test_fetch_approach_and_pick(ip):
    scene_path = str(REPO / "src/mjlab_microduck/robot/microduck/scene_fetch.xml")
    walking_path = str(REPO / "logs/rsl_rl/velocity/2026-09-27_23-17-48_velocity/2026-09-27_23-17-48_velocity.onnx")
    ground_pick_path = str(REPO / "logs/rsl_rl/ground_pick/2026-09-30_02-43-42_ground_pick/2026-09-30_02-43-42_ground_pick.onnx")

    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(scene_path, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)

    policy = ip.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=walking_path,
        ground_pick_onnx_path=ground_pick_path,
        new_cmd_obs=True,
        ground_pick_period=4.0
    )

    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    # Place ball at a known resting spot in front: x=0.35, y=0.05
    data.qpos[policy.ball_qpos_adr:policy.ball_qpos_adr + 3] = [0.35, 0.05, 0.035]
    data.qvel[policy.ball_qvel_adr:policy.ball_qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    policy.trigger_fetch()
    assert policy.fetch_mode is True
    assert policy.fetch_state == "NAVIGATE"

    # Step loop verifying transition to APPROACH and PICK
    decimation = 4
    control_dt = 0.02
    picked = False

    for _ in range(500):  # up to 10s
        policy.update_ground_pick_phase(control_dt)
        policy.update_fetch(control_dt)
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(decimation):
            if policy.ball_held and policy.ball_qpos_adr is not None and policy.mouth_tip_site_id >= 0:
                hold_z = max(0.035, float(data.site_xpos[policy.mouth_tip_site_id][2]))
                rot = ip._quat2mat(data.xquat[policy.trunk_base_id])
                fwd_off = rot[:2, 0] * 0.015
                data.qpos[policy.ball_qpos_adr:policy.ball_qpos_adr + 3] = [
                    data.site_xpos[policy.mouth_tip_site_id][0] + fwd_off[0],
                    data.site_xpos[policy.mouth_tip_site_id][1] + fwd_off[1],
                    hold_z,
                ]
                data.qvel[policy.ball_qvel_adr:policy.ball_qvel_adr + 6] = 0.0
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

        if policy.ball_held:
            picked = True
            break

    assert picked, "Robot should successfully navigate, approach, and grasp the ball!"
