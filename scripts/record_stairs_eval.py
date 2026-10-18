"""Script to record simulation video and keyframe images of stairs_climb_latest.onnx.
Saves MP4 video and PNG keyframes into the artifact directory for visual diagnosis.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as ort

DEFAULT_JOINT_POS = np.array([
    0.0, -0.087266, -0.457924, -0.004940, 0.452984,  # Left leg (0..4)
    0.349066, 0.349066, 0.0, 0.0,                    # Neck & Head (5..8)
    0.0, 0.087266, 0.457924, 0.004940, -0.452984,    # Right leg (9..13)
], dtype=np.float32)

ARTIFACT_DIR = Path("/home/dhp01/.gemini/antigravity/brain/a21b19a9-726d-4110-aa09-6c8ad3dd61d8")
SCENE_XML = Path("/home/dhp01/microduck_rl/src/mjlab_microduck/robot/microduck/scene_stairs.xml")
POLICY_PATH = Path("/home/dhp01/microduck_rl/stairs_climb_latest.onnx")

PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 1.0


def quat_apply_inverse(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = vec[0], vec[1], vec[2]
    # conjugate
    rx = w * vx - y * vz + z * vy
    ry = w * vy - z * vx + x * vz
    rz = w * vz - x * vy + y * vx
    rw = x * vx + y * vy + z * vz
    return np.array([
        rw * (-x) + rx * w - ry * (-z) + rz * (-y),
        rw * (-y) + ry * w - rz * (-x) + rx * (-z),
        rw * (-z) + rz * w - rx * (-y) + ry * (-x),
    ], dtype=np.float32)


def build_61d_obs(model, data, last_action, twist, head_cmd=None, body_cmd=None):
    if head_cmd is None:
        head_cmd = np.zeros(4, dtype=np.float32)
    if body_cmd is None:
        body_cmd = np.zeros(6, dtype=np.float32)

    root_quat = data.qpos[3:7]
    grav_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    proj_grav = quat_apply_inverse(root_quat, grav_w)

    ang_vel_w = data.qvel[3:6]
    ang_vel_b = quat_apply_inverse(root_quat, ang_vel_w)

    joint_pos = (data.qpos[7:21] - DEFAULT_JOINT_POS).astype(np.float32)
    joint_vel = data.qvel[6:20].astype(np.float32)

    obs = np.concatenate([
        ang_vel_b,        # 3
        proj_grav,        # 3
        joint_pos,        # 14
        joint_vel,        # 14
        last_action,      # 14
        twist,            # 3
        head_cmd,         # 4
        body_cmd,         # 6
    ]).astype(np.float32)
    assert obs.shape == (61,), f"Expected 61D obs, got {obs.shape}"
    return obs


def run_recording(pure_policy: bool = True, output_prefix: str = "eval_pure"):
    print(f"--- Running Recording (pure_policy={pure_policy}) ---")
    import sys
    sys.path.insert(0, str(Path("/home/dhp01/microduck_rl")))
    from scripts.infer_stairs import load_mujoco_with_bam, reset_to_preset

    model, data, bam_ctrl, _ = load_mujoco_with_bam(str(SCENE_XML))
    reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    # Load ONNX session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(POLICY_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    renderer = mujoco.Renderer(model, height=480, width=640)
    # Camera setup: side/diagonal view tracking robot
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat = np.array([0.4, 0.0, 0.10])
    cam.distance = 1.2
    cam.elevation = -15
    cam.azimuth = 135

    frames = []
    last_action = np.zeros(14, dtype=np.float32)
    twist = np.array([0.08, 0.0, 0.0], dtype=np.float32)

    num_seconds = 3.0
    num_steps = int(num_seconds / CONTROL_DT)

    saved_images = []

    for step in range(num_steps):
        t = step * CONTROL_DT
        # Build observation
        obs = build_61d_obs(model, data, last_action, twist)
        action = session.run(None, {input_name: obs.reshape(1, -1)})[0].flatten()
        action = np.clip(action, -3.0, 3.0)
        last_action = action.copy()

        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        # Substep physics (4x at 200 Hz)
        for _ in range(DECIMATION):
            bam_ctrl.q_target[:] = target_pos
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Update camera lookat to follow robot
        cam.lookat[0] = data.qpos[0] + 0.2
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = data.qpos[2]

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        frames.append(pixels)

        # Save keyframes at specific moments (t=0.0s, 0.2s, 0.5s, 1.0s, 2.0s)
        if step in (0, 4, 10, 20, 40):
            img_path = ARTIFACT_DIR / f"{output_prefix}_frame_step{step}_t{t:.2f}s.png"
            imageio.imwrite(str(img_path), pixels)
            saved_images.append(img_path)
            print(f"Saved keyframe: {img_path.name} | pos: {data.qpos[:3]} | qvel[0]: {data.qvel[0]:.2f}")

    video_path = ARTIFACT_DIR / f"{output_prefix}_video.mp4"
    imageio.mimwrite(str(video_path), frames, fps=int(1.0 / CONTROL_DT))
    print(f"Saved video to: {video_path}")
    return video_path, saved_images


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_recording(pure_policy=True, output_prefix="stairs_run_pure")
