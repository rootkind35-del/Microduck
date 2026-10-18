#!/usr/bin/env python3
"""Frame-by-Frame High Stairs Kinematic and Contact Verification Script.

Evaluates Microduck stair-stepping performance strictly from origin coordinates
(x=0, y=0, z=0.125 on floor), recording frame-by-frame telemetry at 50 Hz:
- Root position & velocity
- Foot clearance height & pitch angle relative to world horizontal
- Step edge collision detection at x=0.30m (stair riser)
- Horizontal reaction force (Fx < 0 backward impulse) and pitching moment
- Overall balance stability and progression onto the stairs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_stairs import (
    BAM_VIN,
    BAM_MAX_CURRENT,
    BAM_KP_FW,
    PHYSICS_DT,
    DECIMATION,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    load_bam_model,
    load_mujoco_with_bam,
    reset_to_preset,
    build_61d_observation,
    identify_step_contacts_and_forces,
    compute_euler_angles,
)


def run_frame_by_frame_eval(
    policy_path: str = "walking.onnx",
    scene_path: str = "src/mjlab_microduck/robot/microduck/scene_stairs.xml",
    duration_s: float = 4.5,
    cmd_vx: float = 0.20,
    enable_adaptive_reflex: bool = True,
    verbose: bool = True,
) -> dict:
    """Run frame-by-frame simulation strictly from origin coordinates."""
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX not found: {policy_path}")

    bam_model = load_bam_model(BAM_KP_FW, BAM_VIN, BAM_MAX_CURRENT)
    model, data, bam_ctrl, act_names = load_mujoco_with_bam(
        scene_path, bam_model=bam_model, timestep=PHYSICS_DT
    )

    # STRICTLY RESET TO ORIGIN / FLOOR (đầu tọa độ)
    reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    # Settle feet onto floor
    for _ in range(15):
        bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
        bam_ctrl.update()
        mujoco.mj_step(model, data)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(os.path.abspath(policy_path), sess_options=opts, providers=["CPUExecutionProvider"])

    cmd_twist = np.array([cmd_vx, 0.0, 0.0], dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)

    l_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    r_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    total_steps = int(duration_s / CONTROL_DT)
    frames = []

    stair_edge_hit_count = 0
    max_backward_impulse_fx = 0.0
    min_x_after_approach = 999.0
    max_x_reached = -999.0
    fell_backward = False

    for step in range(total_steps):
        t = (step + 1) * CONTROL_DT
        cur_x = float(data.qpos[0])
        cur_z = float(data.qpos[2])
        roll, pitch, yaw = compute_euler_angles(data.qpos[3:7])
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)

        c_l, c_r, fn_l, fn_r = identify_step_contacts_and_forces(model, data)

        obs = build_61d_observation(
            model, data, last_action, cmd_twist, cmd_head=cmd_head, cmd_body=cmd_body
        )
        ort_input = {session.get_inputs()[0].name: obs.reshape(1, -1).astype(np.float32)}
        action = session.run(None, ort_input)[0].flatten()
        last_action = np.clip(action, -3.0, 3.0)
        target_pos = DEFAULT_JOINT_POS + last_action

        # Foot positions & orientations
        l_foot_pos = data.site_xpos[l_foot_site].copy()
        r_foot_pos = data.site_xpos[r_foot_site].copy()
        l_mat = data.site_xmat[l_foot_site].reshape(3, 3)
        r_mat = data.site_xmat[r_foot_site].reshape(3, 3)
        l_pitch_deg = math.degrees(math.atan2(l_mat[0, 2], l_mat[0, 0]))
        r_pitch_deg = math.degrees(math.atan2(r_mat[0, 2], r_mat[0, 0]))

        # Adaptive clearance & foot-leveling reflex
        if enable_adaptive_reflex:
            approaching_stair = (cur_x >= 0.20) or ("stair" in c_l or "stair" in c_r)
            if approaching_stair and cmd_twist[0] > 0.01:
                # Slight forward lean to counterbalance elevation gain
                cmd_body[4] = -math.radians(5.0)

                # Left leg swinging (airborne)
                if fn_l < 1.5:
                    d_hip = -0.25
                    d_knee = +0.45
                    d_ankle = d_knee - d_hip  # Level foot: +0.70 rad
                    target_pos[2] += d_hip
                    target_pos[3] += d_knee
                    target_pos[4] += d_ankle

                # Right leg swinging (airborne)
                if fn_r < 1.5:
                    d_hip = +0.25
                    d_knee = -0.45
                    d_ankle = d_knee - d_hip  # Level foot: -0.70 rad
                    target_pos[11] += d_hip
                    target_pos[12] += d_knee
                    target_pos[13] += d_ankle

        # Physics decimation
        for _ in range(DECIMATION):
            bam_ctrl.q_target[:] = target_pos
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Inspect contacts on this frame
        stair_contacts = []
        fx_net_backward = 0.0
        for i in range(data.ncon):
            con = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
            if "stair" in g1 or "stair" in g2:
                efc = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, efc)
                fn = efc[0]
                norm_x = con.frame[0]
                # If contact normal points along -X, normal force pushes backward
                if norm_x < -0.5:
                    fx_backward = fn * abs(norm_x)
                    fx_net_backward += fx_backward
                stair_contacts.append({
                    "g1": g1,
                    "g2": g2,
                    "pos": con.pos.tolist(),
                    "norm": con.frame[:3].tolist(),
                    "fn": float(fn),
                })

        if fx_net_backward > 0.5:
            stair_edge_hit_count += 1
            max_backward_impulse_fx = max(max_backward_impulse_fx, fx_net_backward)

        max_x_reached = max(max_x_reached, cur_x)
        if cur_x >= 0.22:
            min_x_after_approach = min(min_x_after_approach, cur_x)

        if pitch_deg > 25.0 or cur_z < 0.07:
            fell_backward = True

        frame_data = {
            "step": step,
            "t": t,
            "x": cur_x,
            "z": cur_z,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "l_foot_z": float(l_foot_pos[2]),
            "r_foot_z": float(r_foot_pos[2]),
            "l_pitch_deg": l_pitch_deg,
            "r_pitch_deg": r_pitch_deg,
            "fn_l": float(fn_l),
            "fn_r": float(fn_r),
            "stair_contacts": len(stair_contacts),
            "fx_net_backward": float(fx_net_backward),
        }
        frames.append(frame_data)

        if verbose and (step % 10 == 0 or fx_net_backward > 0.5 or len(stair_contacts) > 0):
            stair_str = f"STAIR({len(stair_contacts)}) Fx_back={fx_net_backward:.1f}N" if stair_contacts else "clean"
            print(
                f"[{t:4.2f}s | step {step:3d}] pos:[{cur_x:.3f}, {cur_z:.3f}] "
                f"pitch:{pitch_deg:+5.1f}° | feet_z:[L:{l_foot_pos[2]*100:4.1f}cm, R:{r_foot_pos[2]*100:4.1f}cm] "
                f"foot_pitch:[L:{l_pitch_deg:+5.1f}°, R:{r_pitch_deg:+5.1f}°] | {stair_str}"
            )

    result = {
        "total_frames": len(frames),
        "duration_s": duration_s,
        "max_x_reached": max_x_reached,
        "min_x_after_approach": min_x_after_approach,
        "stair_edge_hit_count": stair_edge_hit_count,
        "max_backward_impulse_fx": max_backward_impulse_fx,
        "fell_backward": fell_backward,
        "final_x": float(data.qpos[0]),
        "final_z": float(data.qpos[2]),
        "final_pitch_deg": math.degrees(compute_euler_angles(data.qpos[3:7])[1]),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Frame-by-Frame High Stairs Kinematic and Contact Verification")
    parser.add_argument("--policy", type=str, default="walking.onnx", help="Path to policy ONNX")
    parser.add_argument("--scene", type=str, default="src/mjlab_microduck/robot/microduck/scene_stairs.xml", help="Scene XML")
    parser.add_argument("--duration-s", type=float, default=4.5, help="Simulation duration in seconds")
    parser.add_argument("--vx", type=float, default=0.20, help="Forward commanded speed")
    parser.add_argument("--no-reflex", action="store_true", help="Disable adaptive foot leveling reflex (baseline)")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    args = parser.parse_args()

    print("==================================================================")
    print(" Microduck Frame-by-Frame Stairs Evaluation (Origin Coordinates) ")
    print("==================================================================")
    print(f" Policy: {args.policy}")
    print(f" Adaptive Reflex: {not args.no_reflex}")
    print(f" Start Position: Origin (0, 0, 0.125m) on Floor")
    print("==================================================================")

    res = run_frame_by_frame_eval(
        policy_path=args.policy,
        scene_path=args.scene,
        duration_s=args.duration_s,
        cmd_vx=args.vx,
        enable_adaptive_reflex=not args.no_reflex,
        verbose=not args.quiet,
    )

    print("\n----------------- EVALUATION REPORT -----------------")
    print(f" Max X Reached: {res['max_x_reached']:.3f} m")
    print(f" Stair Edge Riser Hits (Fx < -0.5N): {res['stair_edge_hit_count']} frames")
    print(f" Max Backward Reaction Force Fx: {res['max_backward_impulse_fx']:.2f} N")
    print(f" Fell Backward: {res['fell_backward']}")
    print(f" Final Position: x={res['final_x']:.3f}m, z={res['final_z']:.3f}m, pitch={res['final_pitch_deg']:+.1f}°")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    main()
