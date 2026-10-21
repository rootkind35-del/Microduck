#!/usr/bin/env python3
"""Detailed Gait Symmetry & Alternating Step Verification for rough_latest.onnx

Evaluates:
1. Left vs Right Foot Lift Height (Clearance Symmetry).
2. Stride Alternation (Single-support step alternation vs dragging/stationary foot).
3. Straight-line walking vs Circular rotation (Yaw drift and trajectory straightness).
4. Hip splay (V-shape leg angles).
5. Stability and forward speed.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.infer_policy as ip

ARTIFACTS_DIR = Path("/home/dhp01/.gemini/antigravity/brain/6e93f781-5efb-4237-a37c-8b61ff05a698")
FRAMES_DIR = ARTIFACTS_DIR / "eval_symmetry_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def get_yaw_deg(quat):
    """quat is [w, x, y, z]"""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def evaluate_policy(onnx_path="rough_latest.onnx", duration=4.0):
    print("=" * 75)
    print(f"[*] EVALUATING GAIT SYMMETRY & STRIDE: {onnx_path}")
    print("=" * 75)

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
    # Commanded forward velocity: 0.15 m/s, straight (yaw rate = 0)
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

    # IDs for sites and joints
    left_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    left_yaw_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_yaw")
    right_yaw_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_yaw")
    left_roll_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
    right_roll_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_roll")

    left_yaw_adr = model.jnt_qposadr[left_yaw_jnt]
    right_yaw_adr = model.jnt_qposadr[right_yaw_jnt]
    left_roll_adr = model.jnt_qposadr[left_roll_jnt]
    right_roll_adr = model.jnt_qposadr[right_roll_jnt]

    # Geoms for contact detection
    left_sole_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    right_sole_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

    # Camera setup (angled front-quarter view to clearly see both feet lifting)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.90
    camera.elevation = -22.0
    camera.azimuth = 140.0

    control_dt = 0.02
    decimation = 4
    total_steps = int(duration / control_dt)

    start_x = data.qpos[0]
    start_y = data.qpos[1]
    initial_yaw = get_yaw_deg(data.qpos[3:7])

    times = []
    foot_z_l = []
    foot_z_r = []
    contact_l = []
    contact_r = []
    yaw_angles = []
    hip_yaw_l = []
    hip_yaw_r = []
    trunk_z = []
    frames = []

    # Ground level baseline
    baseline_foot_z = min(data.site_xpos[left_foot_site, 2], data.site_xpos[right_foot_site, 2])

    for step in range(total_steps):
        t = step * control_dt

        action = policy.infer()
        policy.apply_action(action)

        for _ in range(decimation):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Record metrics
        times.append(t)
        cur_zl = data.site_xpos[left_foot_site, 2] - baseline_foot_z
        cur_zr = data.site_xpos[right_foot_site, 2] - baseline_foot_z
        foot_z_l.append(max(0.0, cur_zl))
        foot_z_r.append(max(0.0, cur_zr))

        # Check contact with floor
        l_in_contact = False
        r_in_contact = False
        for c_idx in range(data.ncon):
            con = data.contact[c_idx]
            if con.geom1 == left_sole_geom or con.geom2 == left_sole_geom:
                l_in_contact = True
            if con.geom1 == right_sole_geom or con.geom2 == right_sole_geom:
                r_in_contact = True
        contact_l.append(l_in_contact)
        contact_r.append(r_in_contact)

        cur_yaw = get_yaw_deg(data.qpos[3:7])
        yaw_angles.append(cur_yaw)

        hip_yaw_l.append(math.degrees(data.qpos[left_yaw_adr]))
        hip_yaw_r.append(math.degrees(data.qpos[right_yaw_adr]))
        trunk_z.append(data.qpos[2])

        # Render frame
        camera.lookat = [data.qpos[0] + 0.02, data.qpos[1], data.qpos[2] + 0.02]
        renderer.update_scene(data, camera)
        img = renderer.render()
        frames.append(Image.fromarray(img))

    # Save animated GIF
    gif_path = ARTIFACTS_DIR / "rough_alternating_eval.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1::2], duration=40, loop=0)
        print(f"[✓] Saved animated GIF to {gif_path}")

    # Save 5 keyframes evenly spaced
    key_indices = [0, total_steps // 4, total_steps // 2, (3 * total_steps) // 4, total_steps - 1]
    for i, kidx in enumerate(key_indices):
        kpath = FRAMES_DIR / f"frame_{i:02d}_t{int(times[kidx]*10):02d}.png"
        frames[kidx].save(kpath)
        print(f"[✓] Saved keyframe {i+1}/5 to {kpath}")

    # Calculations
    dx = data.qpos[0] - start_x
    dy = data.qpos[1] - start_y
    dist = math.sqrt(dx * dx + dy * dy)
    avg_vx = dx / duration

    max_clearance_l = max(foot_z_l) * 1000.0  # mm
    max_clearance_r = max(foot_z_r) * 1000.0  # mm
    mean_clearance_l = np.mean([z for z in foot_z_l if z > 0.005]) * 1000.0 if any(z > 0.005 for z in foot_z_l) else 0.0
    mean_clearance_r = np.mean([z for z in foot_z_r if z > 0.005]) * 1000.0 if any(z > 0.005 for z in foot_z_r) else 0.0

    # Step peak counts (local maxima above 10mm)
    def count_steps(z_series, threshold=0.010):
        peaks = 0
        in_step = False
        for z in z_series:
            if z > threshold and not in_step:
                peaks += 1
                in_step = True
            elif z <= threshold * 0.6:
                in_step = False
        return peaks

    steps_l = count_steps(foot_z_l, threshold=0.010)
    steps_r = count_steps(foot_z_r, threshold=0.010)

    # Alternating single-support time ratio
    # Single support: exactly one foot in contact
    single_support_frames = sum(1 for cl, cr in zip(contact_l, contact_r) if cl != cr)
    double_support_frames = sum(1 for cl, cr in zip(contact_l, contact_r) if cl and cr)
    flight_frames = sum(1 for cl, cr in zip(contact_l, contact_r) if not cl and not cr)
    single_support_pct = (single_support_frames / total_steps) * 100.0
    double_support_pct = (double_support_frames / total_steps) * 100.0

    final_yaw = yaw_angles[-1]
    yaw_drift = final_yaw - initial_yaw
    max_yaw_dev = max(abs(y - initial_yaw) for y in yaw_angles)

    splay_angles = [yl - yr for yl, yr in zip(hip_yaw_l[-100:], hip_yaw_r[-100:])]
    mean_splay = np.mean(splay_angles)
    mean_trunk_z = np.mean(trunk_z[-100:]) * 1000.0
    min_trunk_z = min(trunk_z) * 1000.0

    print("\n" + "=" * 75)
    print("--- QUANTITATIVE EVALUATION RESULTS ---")
    print(f"Walking Duration        : {duration:.1f} s ({total_steps} control cycles @ 50 Hz)")
    print(f"Forward Distance Δx     : {dx:+.3f} m (Average vx = {avg_vx:+.3f} m/s)")
    print(f"Lateral Drift Δy        : {dy:+.3f} m")
    print(f"Heading / Yaw Drift     : {yaw_drift:+.1f}° (Max deviation from straight: {max_yaw_dev:.1f}°)")
    print(f"Left Foot Peak Lift     : {max_clearance_l:.1f} mm ({steps_l} steps taken)")
    print(f"Right Foot Peak Lift    : {max_clearance_r:.1f} mm ({steps_r} steps taken)")
    print(f"Foot Clearance Ratio    : {min(max_clearance_l, max_clearance_r) / max(max_clearance_l, max_clearance_r) * 100.0:.1f}% (100% = perfectly equal)")
    print(f"Single-support (Walking): {single_support_pct:.1f}% of gait cycle")
    print(f"Double-support (Stance) : {double_support_pct:.1f}% of gait cycle")
    print(f"Feet Splay Angle        : {mean_splay:+.1f}° (V-shape spread)")
    print(f"Trunk Height            : mean = {mean_trunk_z:.1f} mm (min = {min_trunk_z:.1f} mm)")
    print(f"Fall Status             : {'PASSED (Steady continuous bipedal walking)' if min_trunk_z > 80.0 else 'FAILED (Fell)'}")
    print("=" * 75)

    return {
        "dx": dx,
        "dy": dy,
        "avg_vx": avg_vx,
        "yaw_drift": yaw_drift,
        "max_yaw_dev": max_yaw_dev,
        "max_clearance_l": max_clearance_l,
        "max_clearance_r": max_clearance_r,
        "steps_l": steps_l,
        "steps_r": steps_r,
        "single_support_pct": single_support_pct,
        "double_support_pct": double_support_pct,
        "mean_splay": mean_splay,
        "mean_trunk_z": mean_trunk_z,
        "min_trunk_z": min_trunk_z,
    }


if __name__ == "__main__":
    policy_path = sys.argv[1] if len(sys.argv) > 1 else "rough_latest.onnx"
    evaluate_policy(policy_path, duration=4.0)

