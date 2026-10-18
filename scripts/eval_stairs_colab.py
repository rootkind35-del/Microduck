#!/usr/bin/env python3
"""Automated Verification & Evaluation Script for Colab / Local Stairs Training.

Finds the latest stairs checkpoint in logs/rsl_rl/stairs_climb/, exports to ONNX
(with baked observation normalizer), evaluates climbing performance on
scene_stairs.xml with BAM M6 actuators, records an animated GIF, and prints a
full empirical acceptance report.
"""

import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
import mujoco
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mjlab_microduck.export import ExportConfig, run_export
import scripts.infer_policy as ip

def find_latest_stairs_checkpoint():
    """Find the newest model_*.pt in logs/rsl_rl/stairs_climb/."""
    candidates = glob.glob("logs/rsl_rl/stairs_climb/*/model_*.pt")
    if not candidates:
        raise FileNotFoundError("No checkpoint found in logs/rsl_rl/stairs_climb/*/")
    # Sort by modification time
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0])

def main():
    print("=" * 78)
    print("🦆 MICRODUCK STAIRS CLIMBING POLICY: AUTOMATED VERIFICATION & TEST")
    print("=" * 78)

    # 1. Locate checkpoint
    latest_pt = find_latest_stairs_checkpoint()
    print(f"[*] Found newest trained checkpoint: {latest_pt}")
    print(f"    File size: {latest_pt.stat().st_size / (1024*1024):.2f} MB")

    # 2. Export to ONNX with baked normalizer
    onnx_path = Path("stairs_climb_latest.onnx")
    print(f"\n[*] Exporting to ONNX: {onnx_path} ...")
    export_cfg = ExportConfig(
        onnx_file=str(onnx_path),
        checkpoint_file=str(latest_pt),
    )
    run_export("Mjlab-StairsClimb-MicroDuck", export_cfg)
    print(f"[✓] ONNX export successful: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # 3. Setup MuJoCo physical test
    scene_stairs_path = "src/mjlab_microduck/robot/microduck/scene_stairs.xml"
    print(f"\n[*] Initializing MuJoCo with BAM M6 actuators ({scene_stairs_path}) ...")
    bam_model = ip.load_bam_model(kp_fw=ip.BAM_KP_FW, vin=7.4, max_current=ip.BAM_MAX_CURRENT)
    model, data, bam_ctrl, actuator_names = ip.load_mujoco_with_bam(
        xml_path=scene_stairs_path,
        bam_model=bam_model,
        timestep=0.005,
        vin_drop_gain=0.0,
        vin_min=ip.BAM_VIN_MIN,
    )

    # 4. Multi-trial physical evaluation
    num_trials = 5
    trial_duration_s = 4.0
    control_dt = 0.02
    decimation = 4
    steps_per_trial = int(trial_duration_s / control_dt)

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    trunk_id = 1
    qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = []
    print(f"\n[*] Running {num_trials} stochastic physical rollouts (duration {trial_duration_s}s, vx=0.15 m/s) ...")
    print(f"{'Trial':>5} | {'Final X':>8} | {'Final Z':>8} | {'Max Pitch':>10} | {'Max Roll':>9} | {'Peak LF_Z':>9} | {'Peak RF_Z':>9} | {'Status':>12}")
    print("-" * 88)

    for trial in range(1, num_trials + 1):
        # Reset robot to spawn
        mujoco.mj_resetData(model, data)
        # Random initial perturbation
        data.qpos[qpos_adr + 0] = np.random.uniform(-0.01, 0.01)
        data.qpos[qpos_adr + 1] = np.random.uniform(-0.01, 0.01)
        data.qpos[qpos_adr + 2] = 0.125
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        for i, qidx in enumerate(model.jnt_qposadr[1:15]):
            data.qpos[qidx] = ip.DEFAULT_POSE[i]
        if bam_ctrl is not None:
            bam_ctrl.reset(data.qpos)
        mujoco.mj_forward(model, data)

        max_pitch = 0.0
        max_roll = 0.0
        peak_lf_z = 0.0
        peak_rf_z = 0.0
        fallen = False
        prev_action = np.zeros(14, dtype=np.float32)

        for step in range(steps_per_trial):
            # Read sensors
            ang_vel = data.sensordata[:3] if len(data.sensordata) >= 3 else data.qvel[3:6]
            rot = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rot, data.qpos[qpos_adr+3:qpos_adr+7])
            proj_grav = np.array([rot[6], rot[7], rot[8]], dtype=np.float32)

            pitch = abs(math.asin(max(-1.0, min(1.0, -rot[2])))) * 180 / np.pi
            roll = abs(math.atan2(rot[5], rot[8])) * 180 / np.pi
            if pitch > max_pitch: max_pitch = pitch
            if roll > max_roll: max_roll = roll

            q_joints = data.qpos[7:21].astype(np.float32)
            v_joints = data.qvel[6:20].astype(np.float32)
            cmd_twist = np.array([0.15, 0.0, 0.0], dtype=np.float32)
            cmd_head = np.zeros(4, dtype=np.float32)
            cmd_body = np.zeros(6, dtype=np.float32)

            obs = np.concatenate([ang_vel, proj_grav, q_joints, v_joints, prev_action, cmd_twist, cmd_head, cmd_body]).reshape(1, 61).astype(np.float32)
            action = session.run([output_name], {input_name: obs})[0][0]
            prev_action = action.copy()

            target_pos = ip.DEFAULT_POSE + action * 1.0
            if bam_ctrl is not None:
                bam_ctrl.q_target[:] = target_pos
            else:
                data.ctrl[:] = target_pos

            for _ in range(decimation):
                if bam_ctrl is not None:
                    bam_ctrl.update()
                mujoco.mj_step(model, data)

            lf_z = data.site_xpos[left_foot_id][2] if left_foot_id >= 0 else 0.0
            rf_z = data.site_xpos[right_foot_id][2] if right_foot_id >= 0 else 0.0
            if lf_z > peak_lf_z: peak_lf_z = lf_z
            if rf_z > peak_rf_z: peak_rf_z = rf_z

            rz = data.qpos[qpos_adr + 2]
            if rz < 0.075 or max_pitch > 60.0 or max_roll > 60.0:
                fallen = True
                break

        rx = float(data.qpos[qpos_adr + 0])
        rz = float(data.qpos[qpos_adr + 2])
        status = "PASSED" if (not fallen and rx > 0.20) else ("FALLEN" if fallen else "STALLED")
        results.append({
            "trial": trial,
            "rx": rx,
            "rz": rz,
            "max_pitch": max_pitch,
            "max_roll": max_roll,
            "peak_lf_z": peak_lf_z,
            "peak_rf_z": peak_rf_z,
            "status": status,
        })
        print(f"{trial:5d} | {rx:+7.3f}m | {rz:7.3f}m | {max_pitch:9.1f}° | {max_roll:8.1f}° | {peak_lf_z*1000:7.1f}mm | {peak_rf_z*1000:7.1f}mm | {status:>12}")

    # Summary
    success_count = sum(1 for r in results if r["status"] == "PASSED")
    success_rate = (success_count / num_trials) * 100
    avg_x = np.mean([r["rx"] for r in results])
    avg_lf = np.mean([r["peak_lf_z"] for r in results]) * 1000
    avg_rf = np.mean([r["peak_rf_z"] for r in results]) * 1000

    print("-" * 88)
    print(f"📊 SUMMARY REPORT:")
    print(f"   ► Climbing Success Rate : {success_rate:.1f}% ({success_count}/{num_trials})")
    print(f"   ► Mean Forward Progress : {avg_x:+.3f} m")
    print(f"   ► Average Foot Clearance: LF={avg_lf:.1f} mm, RF={avg_rf:.1f} mm")
    print(f"   ► Deployment Artifact  : {onnx_path.resolve()}")
    print("=" * 88)

    # 5. Record visual GIF demonstration
    gif_output = "stairs_climb_eval.gif"
    print(f"\n[*] Generating high-resolution GIF verification: {gif_output} ...")
    cmd = (
        f"uv run scripts/infer_policy.py --walking {onnx_path} --scene {scene_stairs_path} "
        f"--new-cmd-obs --headless --record-gif {gif_output} --record-duration 4.0 --lin-vel-x 0.15"
    )
    os.system(cmd)
    if os.path.exists(gif_output):
        print(f"[✓] GIF recorded successfully: {gif_output} ({os.path.getsize(gif_output)/(1024*1024):.2f} MB)")
        try:
            # Check if running in Google Colab
            import google.colab
            from google.colab import files
            from IPython.display import Image, display
            print("\n[Colab Detected] Displaying visual GIF inline...")
            display(Image(gif_output))
            print("[Colab Detected] Triggering automatic download of ONNX model to your PC...")
            files.download(str(onnx_path))
        except Exception:
            pass

if __name__ == "__main__":
    main()

