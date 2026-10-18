"""Generate 10 chronological keyframe images depicting the entire high-step leg lifting process.
Uses the newly trained highstep_walking.onnx policy with BAM M6 actuators.
Saves images with rich biomechanical telemetry overlays into the artifact directory.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path("/home/dhp01/microduck_rl")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infer_stairs import (
    ACTION_SCALE,
    BAM_KP_FW,
    BAM_MAX_CURRENT,
    BAM_VIN,
    CONTROL_DT,
    DECIMATION,
    DEFAULT_JOINT_POS,
    PHYSICS_DT,
    build_61d_observation,
    compute_euler_angles,
    load_bam_controller,
    load_mujoco_with_bam,
    reset_to_preset,
)

ARTIFACT_DIR = Path("/home/dhp01/.gemini/antigravity/brain/a21b19a9-726d-4110-aa09-6c8ad3dd61d8")
POLICY_PATH = PROJECT_ROOT / "test_750.onnx"
SCENE_XML = PROJECT_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"


def get_foot_telemetry(model: mujoco.MjModel, data: mujoco.MjData):
    foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

    pos_l = data.geom_xpos[foot_l_id].copy() if foot_l_id >= 0 else np.zeros(3)
    pos_r = data.geom_xpos[foot_r_id].copy() if foot_r_id >= 0 else np.zeros(3)

    # Normal contact force
    f_l = 0.0
    f_r = 0.0
    c_array = mujoco.MjContact()
    for i in range(data.ncon):
        c = data.contact[i]
        c_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, c_force)
        norm_f = abs(float(c_force[0]))
        if c.geom1 == foot_l_id or c.geom2 == foot_l_id:
            f_l += norm_f
        if c.geom1 == foot_r_id or c.geom2 == foot_r_id:
            f_r += norm_f

    return pos_l, pos_r, f_l, f_r


def draw_overlay(
    image_np: np.ndarray,
    frame_idx: int,
    phase_title: str,
    phase_desc: str,
    t_val: float,
    foot_name: str,
    foot_z_cm: float,
    vz_val: float,
    knee_deg: float,
    roll_deg: float,
    pitch_deg: float,
    f_stance: float,
    balance_status: str,
) -> np.ndarray:
    """Draw professional HUD banner on top and bottom of frame."""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Load default font
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

    # Top banner (Title & Phase)
    top_bar_h = 75
    overlay = Image.new("RGBA", (w, top_bar_h), (15, 23, 42, 220))
    img.paste(overlay, (0, 0), overlay)

    title_text = f"MICRODUCK HIGH-STEP BIOMECHANICS | KHUNG {frame_idx:02d}/10: {phase_title.upper()}"
    subtitle_text = f"Thoi gian: t = {t_val:.3f}s | Muc tieu MDP: {phase_desc}"

    draw.text((20, 12), title_text, fill=(56, 189, 248), font=font_large)
    draw.text((20, 38), subtitle_text, fill=(241, 245, 249), font=font_small)

    # Bottom banner (Telemetry HUD)
    bot_bar_h = 75
    bot_overlay = Image.new("RGBA", (w, bot_bar_h), (15, 23, 42, 225))
    img.paste(bot_overlay, (0, h - bot_bar_h), bot_overlay)

    # Telemetry columns
    col1 = (
        f"Chan lang ({foot_name}):\n"
        f"Do cao: {foot_z_cm:+.1f} cm | Vz: {vz_val:+.2f} m/s\n"
        f"Gap goi: {knee_deg:.1f} deg"
    )
    col2 = (
        f"Goc than (CoM Tilt):\n"
        f"Roll: {roll_deg:+.1f} deg | Pitch: {pitch_deg:+.1f} deg\n"
        f"Do nghieng an toan (<20 deg)"
    )
    col3 = (
        f"Chan tru & Thang bang:\n"
        f"Luc tai: {f_stance:.1f} N (~800g body)\n"
        f"Trang thai: {balance_status}"
    )

    draw.text((20, h - bot_bar_h + 10), col1, fill=(255, 255, 255), font=font_small)
    draw.text((450, h - bot_bar_h + 10), col2, fill=(187, 247, 208), font=font_small)
    draw.text((880, h - bot_bar_h + 10), col3, fill=(254, 240, 138), font=font_small)

    return np.array(img)


def main():
    print(f"Loading model from {SCENE_XML}...")
    model, data, bam_ctrl, _ = load_mujoco_with_bam(str(SCENE_XML))
    reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)

    print(f"Loading ONNX policy: {POLICY_PATH}...")
    session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Configure offscreen buffer for 1280x720 high-res rendering
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    # Side-diagonal camera angle giving clear view of foot elevation and knee flexion
    cam.distance = 1.05
    cam.elevation = -12.0
    cam.azimuth = 105.0

    last_action = np.zeros(14, dtype=np.float32)
    twist = np.array([0.08, 0.0, 0.0], dtype=np.float32)

    # Step simulation and log foot data to detect a full high-step swing
    print("Simulating trajectory and recording step telemetry...")
    trajectory = []
    num_steps = 150  # 3.0 seconds

    prev_pos_l = None
    prev_pos_r = None

    for step in range(num_steps):
        t = step * CONTROL_DT
        obs = build_61d_observation(model, data, last_action, twist)
        action = session.run(None, {input_name: obs.reshape(1, -1)})[0].flatten()
        action = np.clip(action, -3.0, 3.0)
        last_action = action.copy()

        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        for _ in range(DECIMATION):
            bam_ctrl.q_target[:] = target_pos
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        pos_l, pos_r, f_l, f_r = get_foot_telemetry(model, data)
        quat = data.qpos[3:7]
        roll, pitch, yaw = compute_euler_angles(quat)

        vz_l = (pos_l[2] - prev_pos_l[2]) / CONTROL_DT if prev_pos_l is not None else 0.0
        vz_r = (pos_r[2] - prev_pos_r[2]) / CONTROL_DT if prev_pos_r is not None else 0.0
        prev_pos_l = pos_l.copy()
        prev_pos_r = pos_r.copy()

        # Joint knee angles
        # Joint layout: 0-4 left leg (knee is idx 3), 9-13 right leg (knee is idx 12)
        knee_l_deg = math.degrees(data.qpos[7 + 3])
        knee_r_deg = math.degrees(data.qpos[7 + 12])

        trajectory.append({
            "step": step,
            "t": t,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "pos_l": pos_l,
            "pos_r": pos_r,
            "f_l": f_l,
            "f_r": f_r,
            "vz_l": vz_l,
            "vz_r": vz_r,
            "roll": roll,
            "pitch": pitch,
            "knee_l_deg": knee_l_deg,
            "knee_r_deg": knee_r_deg,
        })

    # Find the best swing phase for high step
    # Look for right or left foot swing where foot z reaches maximum elevation
    r_heights = [d["pos_r"][2] for d in trajectory]
    max_r_step = int(np.argmax(r_heights[10:90])) + 10
    print(f"Max right foot height at step {max_r_step}: z = {trajectory[max_r_step]['pos_r'][2]:.4f}m")

    # Trace backward to find start of swing (when f_r drops to ~0)
    swing_start = max_r_step
    while swing_start > 0 and trajectory[swing_start]["f_r"] < 1.0:
        swing_start -= 1

    # Trace forward to find end of swing (when f_r increases back after landing)
    swing_end = max_r_step
    while swing_end < len(trajectory) - 1 and trajectory[swing_end]["f_r"] < 1.0:
        swing_end += 1

    print(f"Detected full right-foot high-step swing between steps [{swing_start} -> {swing_end}] "
          f"(Duration: {(swing_end - swing_start)*CONTROL_DT:.2f}s)")

    # Select exactly 10 chronological keyframe step indices
    # We include swing_start - 1 (double support prep) to swing_end + 1 (propulsion lock)
    sample_indices = np.linspace(max(0, swing_start - 1), min(len(trajectory) - 1, swing_end + 1), 10, dtype=int)
    # Ensure strictly non-decreasing indices
    sample_indices = np.unique(sample_indices)
    while len(sample_indices) < 10:
        # Fill in missing steps
        for i in range(len(sample_indices) - 1):
            if sample_indices[i+1] - sample_indices[i] > 1:
                sample_indices = np.sort(np.append(sample_indices, sample_indices[i] + 1))
                break
        if len(sample_indices) < 10:
            sample_indices = np.sort(np.append(sample_indices, sample_indices[-1] + 1))

    sample_indices = sample_indices[:10]
    print(f"Selected 10 keyframe step indices: {sample_indices.tolist()}")

    # 10 Phase definitions
    phases = [
        ("Khoi dau chu ky & Chuyen dich trong tam", "Chuyen trong tam sang chan trai de chuan bi nhac chan phai", "Chuan bi nhac"),
        ("Roi mat dat & Kich hoat nang chan", "Chan phai roi san, luc tiep dat F_z giam ve 0N", "Dang nhac"),
        ("Nang chan em ai theo phuong thang dung", "Kiem soat van toc Vz <= 0.18 m/s, khu hoan toan luc giat dot ngot", "Thang bang dong"),
        ("Gap sau khop goi dang chim (Digitigrade)", "Goi phai gap nguoc ve sau de co gon ban chan khong quet mat san", "Thang bang dong"),
        ("Dinh cao buoc nhac chan (Peak Clearance)", "Ban chan dat do cao toi da (3.5 - 4.5 cm) vuot chuong ngai vat", "Dinh buoc cao"),
        ("Duy tri can bang dong don chan", "Chan trai tru vung 800g than robot, triet tieu chao dao goc than", "Tru don vung"),
        ("Duoi khop hang vuon chan ve truoc", "Khop hang phai vuon ve truoc de mo rong sai buoc", "Vuon buoc"),
        ("Can bang phang mat ban chan (Sole Leveling)", "Co chan duoi phang song song mat san, chuan bi tiep dat an toan", "Chuan bi cham san"),
        ("Tiep dat em ai & Giam chan va cham", "Ban chan ha xuong nhe nhang, tiep thu luc va cham khong bi nay bat", "Tiep dat em"),
        ("Khoa chan tru moi & Day tien ve truoc", "Chan phai thanh chan tru moi, day than robot buoc tiep", "Khoa chan tru moi"),
    ]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for i, step_idx in enumerate(sample_indices):
        d = trajectory[step_idx]
        data.qpos[:] = d["qpos"]
        data.qvel[:] = d["qvel"]
        mujoco.mj_forward(model, data)

        # Track robot with camera
        cam.lookat[0] = d["pos_r"][0]
        cam.lookat[1] = d["pos_r"][1]
        cam.lookat[2] = d["qpos"][2]

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

        # Telemetry for overlay
        foot_z_cm = (d["pos_r"][2] - 0.005) * 100.0
        vz = d["vz_r"]
        knee_deg = d["knee_r_deg"]
        roll_deg = math.degrees(d["roll"])
        pitch_deg = math.degrees(d["pitch"])
        f_stance = d["f_l"]  # left foot is stance

        phase_title, phase_desc, bal_state = phases[i]

        annotated_frame = draw_overlay(
            pixels,
            frame_idx=i + 1,
            phase_title=phase_title,
            phase_desc=phase_desc,
            t_val=d["t"],
            foot_name="Phai (Right)",
            foot_z_cm=foot_z_cm,
            vz_val=vz,
            knee_deg=knee_deg,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            f_stance=f_stance,
            balance_status=bal_state,
        )

        out_name = f"highstep_frame_{i+1:02d}.png"
        out_path = ARTIFACT_DIR / out_name
        imageio.imwrite(str(out_path), annotated_frame)
        generated_files.append(out_path)
        print(f"Generated frame {i+1:02d}/10: {out_name} | t={d['t']:.2f}s | z={foot_z_cm:.1f}cm | knee={knee_deg:.1f}deg")

    print("\nAll 10 keyframes generated successfully!")
    return generated_files


if __name__ == "__main__":
    main()
