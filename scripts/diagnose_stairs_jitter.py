#!/usr/bin/env python3
"""Diagnose stairs policy failure: captures frames and telemetry to analyze tilt, jitter and shaking."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image

from scripts.infer_stairs import (
    DEFAULT_SCENE_XML,
    BAM_KP_FW,
    BAM_VIN,
    BAM_MAX_CURRENT,
    PHYSICS_DT,
    DECIMATION,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    load_bam_model,
    load_mujoco_with_bam,
    reset_to_preset,
    build_61d_observation,
    compute_euler_angles,
)

ARTIFACT_DIR = Path("/home/dhp01/.gemini/antigravity/brain/a21b19a9-726d-4110-aa09-6c8ad3dd61d8")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

JOINT_NAMES = [
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
]


def run_diagnostics(policy_path: str, cmd_vx: float = 0.20):
    print(f"Loading policy: {policy_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(os.path.abspath(policy_path), sess_options=opts, providers=["CPUExecutionProvider"])

    bam_model = load_bam_model(BAM_KP_FW, BAM_VIN, BAM_MAX_CURRENT)
    model, data, bam_ctrl, act_names = load_mujoco_with_bam(
        DEFAULT_SCENE_XML, bam_model=bam_model, timestep=PHYSICS_DT
    )

    # Renderer setup with custom camera view
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Look from side-front at the robot
    camera.lookat = [0.15, 0.0, 0.10]
    camera.distance = 0.65
    camera.elevation = -15.0
    camera.azimuth = 135.0  # side perspective

    reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    # Settle feet onto floor
    for _ in range(15):
        bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    cmd_twist = np.array([cmd_vx, 0.0, 0.0], dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)

    capture_steps = {
        0: "frame_00_stand_init.png",       # t = 0.00s (Initial stand)
        3: "frame_01_step3_tilt_start.png",  # t = 0.06s (Action starts tilting)
        6: "frame_02_step6_fall.png",        # t = 0.12s (Fall / losing balance)
        15: "frame_03_step15_ground.png",    # t = 0.30s (Impact with floor)
        50: "frame_04_step50_jitter.png",    # t = 1.00s (Lying & jittering)
    }

    step_telemetry = []

    for step in range(60):
        t = step * CONTROL_DT
        obs = build_61d_observation(model, data, last_action, cmd_twist, cmd_head, cmd_body)
        raw_out = session.run(None, {"obs": obs.reshape(1, -1)})[0]
        action = raw_out.flatten()

        cur_x = float(data.qpos[0])
        cur_y = float(data.qpos[1])
        cur_z = float(data.qpos[2])
        roll, pitch, yaw = compute_euler_angles(data.qpos[3:7])
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        cur_qpos = data.qpos[7:21].copy()
        q_err = cur_qpos - DEFAULT_JOINT_POS

        # Measure action delta (action rate)
        action_rate = np.linalg.norm(action - last_action)

        info = {
            "step": step,
            "t": t,
            "pos": (cur_x, cur_y, cur_z),
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "action_norm": float(np.linalg.norm(action)),
            "action_rate": float(action_rate),
            "action": action.tolist(),
            "q_err": q_err.tolist(),
            "cur_qpos": cur_qpos.tolist(),
        }
        step_telemetry.append(info)

        # Capture image if scheduled
        if step in capture_steps:
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            img_path = ARTIFACT_DIR / capture_steps[step]
            Image.fromarray(rgb).save(img_path)
            print(f"Captured {capture_steps[step]} at step {step} (t={t:.2f}s): pos=({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f}), Roll={roll_deg:+.1f}°, Pitch={pitch_deg:+.1f}°")

        # Step physics
        last_action = action.copy()
        q_target = DEFAULT_JOINT_POS + action
        bam_ctrl.q_target[:] = q_target

        for _ in range(DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    return step_telemetry


if __name__ == "__main__":
    policy = "logs/rsl_rl/stairs_climb/2026-10-15_02-53-15_stairs_climb/2026-10-15_02-53-15_stairs_climb.onnx"
    telem = run_diagnostics(policy, cmd_vx=0.20)

