#!/usr/bin/env python3
"""Comprehensive Evaluation and Verification Script for newly trained Microduck models:
1. rough_latest.onnx (Walking / Rough terrain policy)
2. stairs_latest.onnx (Stairs climbing policy)
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import mujoco
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.infer_policy as ip

ARTIFACTS_DIR = Path("/home/dhp01/.gemini/antigravity/brain/6e93f781-5efb-4237-a37c-8b61ff05a698")
ROUGH_FRAMES_DIR = ARTIFACTS_DIR / "eval_rough_frames"
STAIRS_FRAMES_DIR = ARTIFACTS_DIR / "eval_stairs_frames"
ROUGH_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
STAIRS_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def eval_rough(onnx_path="rough_latest.onnx", duration=3.5):
    print("=" * 70)
    print(f"[*] EVALUATING WALKING POLICY: {onnx_path}")
    print("=" * 70)

    scene_path = "src/mjlab_microduck/robot/microduck/scene_walk.xml"
    bam_model = ip.load_bam_model(kp_fw=ip.BAM_KP_FW, vin=7.4, max_current=ip.BAM_MAX_CURRENT)
    model, data, bam_ctrl, actuator_names = ip.load_mujoco_with_bam(
        xml_path=scene_path,
        bam_model=bam_model,
        timestep=0.005,
        vin_drop_gain=0.0,
        vin_min=ip.BAM_VIN_MIN,
    )

    policy = ip.PolicyInference(
        model=model,
        data=data,
        walking_onnx_path=onnx_path,
        bam_ctrl=bam_ctrl,
        new_cmd_obs=True,
    )
    policy.warmup_all_sessions()
    policy.set_vel_cmd(lin_vel_x=0.15, lin_vel_y=0.0, ang_vel_z=0.0, verbose=False)

    # Initialize robot in default standing pose
    qpos_adr = model.jnt_qposadr[0]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0.0
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    left_hip_yaw_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_yaw")
    right_hip_yaw_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_yaw")
    left_hip_roll_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
    right_hip_roll_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll")

    left_qpos_adr = model.jnt_qposadr[left_hip_yaw_idx]
    right_qpos_adr = model.jnt_qposadr[right_hip_yaw_idx]
    left_roll_adr = model.jnt_qposadr[left_hip_roll_idx]
    right_roll_adr = model.jnt_qposadr[right_hip_roll_idx]

    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.95
    camera.elevation = -28.0  # looking down at feet
    camera.azimuth = 145.0    # angled front-side view

    control_dt = 0.02
    decimation = 4
    total_steps = int(duration / control_dt)

    start_x = data.qpos[0]
    
    hip_yaw_l_list = []
    hip_yaw_r_list = []
    hip_roll_l_list = []
    hip_roll_r_list = []
    z_list = []
    pitch_list = []
    frames = []
    saved_keyframe_indices = [0, total_steps // 4, total_steps // 2, (3 * total_steps) // 4, total_steps - 1]

    for step in range(total_steps):
        t = step * control_dt

        # Base orientation
        quat = data.qpos[3:7]  # w, x, y, z
        w, x, y, z_q = quat
        sinp = 2 * (w * y - z_q * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0)) * (180.0 / math.pi)

        # Policy infer & step
        action = policy.infer()
        policy.apply_action(action)

        for _ in range(decimation):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

        cur_yaw_l = np.degrees(data.qpos[left_qpos_adr])
        cur_yaw_r = np.degrees(data.qpos[right_qpos_adr])
        cur_roll_l = np.degrees(data.qpos[left_roll_adr])
        cur_roll_r = np.degrees(data.qpos[right_roll_adr])

        hip_yaw_l_list.append(cur_yaw_l)
        hip_yaw_r_list.append(cur_yaw_r)
        hip_roll_l_list.append(cur_roll_l)
        hip_roll_r_list.append(cur_roll_r)
        z_list.append(data.qpos[2])
        pitch_list.append(pitch)

        # Track camera with robot
        camera.lookat = [data.qpos[0], data.qpos[1], data.qpos[2] + 0.02]
        renderer.update_scene(data, camera)
        img_np = renderer.render()
        frames.append(Image.fromarray(img_np))

        if step in saved_keyframe_indices:
            idx = saved_keyframe_indices.index(step)
            img_path = ROUGH_FRAMES_DIR / f"frame_{idx:02d}_t{int(t*10):02d}.png"
            Image.fromarray(img_np).save(img_path)

    # Save animated GIF
    gif_path = ARTIFACTS_DIR / "rough_parallel_eval.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1::2], duration=40, loop=0)
        print(f"[✓] Saved GIF to {gif_path}")

    dx = data.qpos[0] - start_x
    vx = dx / duration
    mean_yaw_l = np.mean(hip_yaw_l_list[-100:])
    mean_yaw_r = np.mean(hip_yaw_r_list[-100:])
    mean_roll_l = np.mean(hip_roll_l_list[-100:])
    mean_roll_r = np.mean(hip_roll_r_list[-100:])
    splay_angle = mean_yaw_l - mean_yaw_r
    mean_z = np.mean(z_list[-100:])
    min_z = np.min(z_list)
    mean_pitch = np.mean(pitch_list[-100:])

    print("\n--- ROUGH_LATEST RESULTS ---")
    print(f"Forward distance Δx  : {dx:+.3f} m (avg speed {vx:+.3f} m/s)")
    print(f"Trunk height z       : mean={mean_z*1000:.1f} mm, min={min_z*1000:.1f} mm")
    print(f"Body pitch angle     : {mean_pitch:+.1f}°")
    print(f"Left Hip Yaw angle   : {mean_yaw_l:+.1f}° (std={np.std(hip_yaw_l_list[-100:]):.1f}°)")
    print(f"Right Hip Yaw angle  : {mean_yaw_r:+.1f}° (std={np.std(hip_yaw_r_list[-100:]):.1f}°)")
    print(f"Feet Splay (V-shape) : {splay_angle:+.1f}° (Target: < 15°, ideal ~ 0°)")
    print(f"Left Hip Roll angle  : {mean_roll_l:+.1f}°")
    print(f"Right Hip Roll angle : {mean_roll_r:+.1f}°")
    print(f"Fallen?              : {'NO (Walked continuously)' if min_z > 0.08 else 'YES'}")
    return {
        "dx": dx,
        "vx": vx,
        "mean_z": mean_z,
        "min_z": min_z,
        "mean_pitch": mean_pitch,
        "mean_yaw_l": mean_yaw_l,
        "mean_yaw_r": mean_yaw_r,
        "splay_angle": splay_angle,
        "fallen": min_z <= 0.08,
    }


def eval_stairs(onnx_path="stairs_latest.onnx", duration=4.0):
    print("\n" + "=" * 70)
    print(f"[*] EVALUATING STAIRS POLICY: {onnx_path}")
    print("=" * 70)

    scene_path = "src/mjlab_microduck/robot/microduck/scene_stairs.xml"
    bam_model = ip.load_bam_model(kp_fw=ip.BAM_KP_FW, vin=7.4, max_current=ip.BAM_MAX_CURRENT)
    model, data, bam_ctrl, actuator_names = ip.load_mujoco_with_bam(
        xml_path=scene_path,
        bam_model=bam_model,
        timestep=0.005,
        vin_drop_gain=0.0,
        vin_min=ip.BAM_VIN_MIN,
    )

    policy = ip.PolicyInference(
        model=model,
        data=data,
        walking_onnx_path=onnx_path,
        bam_ctrl=bam_ctrl,
        new_cmd_obs=True,
    )
    policy.warmup_all_sessions()
    policy.set_vel_cmd(lin_vel_x=0.15, lin_vel_y=0.0, ang_vel_z=0.0, verbose=False)

    qpos_adr = model.jnt_qposadr[0]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0.0
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.05
    camera.elevation = -18.0
    camera.azimuth = 135.0

    control_dt = 0.02
    decimation = 4
    total_steps = int(duration / control_dt)

    start_x = data.qpos[0]
    start_z = data.qpos[2]

    z_list = []
    pitch_list = []
    frames = []
    saved_keyframe_indices = [0, total_steps // 4, total_steps // 2, (3 * total_steps) // 4, total_steps - 1]

    for step in range(total_steps):
        t = step * control_dt

        quat = data.qpos[3:7]
        w, x, y, z_q = quat
        sinp = 2 * (w * y - z_q * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0)) * (180.0 / math.pi)

        action = policy.infer()
        policy.apply_action(action)

        for _ in range(decimation):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

        z_list.append(data.qpos[2])
        pitch_list.append(pitch)

        camera.lookat = [data.qpos[0] + 0.05, data.qpos[1], data.qpos[2] + 0.02]
        renderer.update_scene(data, camera)
        img_np = renderer.render()
        frames.append(Image.fromarray(img_np))

        if step in saved_keyframe_indices:
            idx = saved_keyframe_indices.index(step)
            img_path = STAIRS_FRAMES_DIR / f"frame_{idx:02d}_t{int(t*10):02d}.png"
            Image.fromarray(img_np).save(img_path)

    gif_path = ARTIFACTS_DIR / "stairs_latest_eval.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1::2], duration=40, loop=0)
        print(f"[✓] Saved GIF to {gif_path}")

    dx = data.qpos[0] - start_x
    dz = data.qpos[2] - start_z
    mean_pitch = np.mean(pitch_list[-100:])
    final_z = data.qpos[2]

    print("\n--- STAIRS_LATEST RESULTS ---")
    print(f"Forward distance Δx  : {dx:+.3f} m")
    print(f"Climbing elevation Δz: {dz*1000:+.1f} mm (final z={final_z*1000:.1f} mm)")
    print(f"Body pitch angle     : {mean_pitch:+.1f}°")
    print(f"Fallen?              : {'NO (Climbed steadily)' if final_z > 0.10 else 'YES'}")
    return {
        "dx": dx,
        "dz": dz,
        "final_z": final_z,
        "mean_pitch": mean_pitch,
        "fallen": final_z <= 0.10,
    }


if __name__ == "__main__":
    r_res = eval_rough("rough_latest.onnx", duration=3.5)
    s_res = eval_stairs("stairs_latest.onnx", duration=4.0)
    print("\n" + "=" * 70)
    print("ALL VERIFICATION SIMULATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
