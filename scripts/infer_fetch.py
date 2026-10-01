#!/usr/bin/env python3
"""Interactive One-Button Ball Fetch Simulation for Microduck.

Chỉ cần 1 nút bấm (SPACE hoặc F):
  1. Bàn tay người ném bóng vào sân (hiệu ứng ném parabol 3D thực tế).
  2. Màn hình Camera góc trên bên phải (Top-Right HUD) hiển thị góc nhìn từ
     head_camera của robot và nhận diện quả bóng thời gian thực (Bounding Box,
     khoảng cách, góc lệch).
  3. Robot tự động chạy tới bóng, giảm tốc tiếp cận, cúi gập người ngậm bóng
     (sử dụng model ground_pick đã train), quay đầu mang bóng về trả tận chân
     người ném và nhả bóng.
"""

import argparse
import math
import multiprocessing as mp
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageTk

# File paths
SCENE_FETCH_XML = "src/mjlab_microduck/robot/microduck/scene_fetch.xml"
DEFAULT_WALK_ONNX = "logs/rsl_rl/velstand/2026-09-29_17-02-18_velstand/2026-09-29_17-02-18_velstand.onnx"
DEFAULT_PICK_ONNX = "logs/rsl_rl/ground_pick/2026-09-30_02-43-42_ground_pick/2026-09-30_02-43-42_ground_pick.onnx"

# BAM Actuator parameters
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

# Human zone and Hand coordinates
HUMAN_ZONE_POS = np.array([-0.4, 0.28, 0.0], dtype=np.float32)
HAND_LAUNCH_POS = np.array([-0.20, 0.23, 0.34], dtype=np.float32)
BALL_RADIUS = 0.035

HOME_LEG_CTRL = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # Left leg
    0.3491, 0.3491, 0.0, 0.0,                 # Neck/head
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,     # Right leg
]


def load_bam_controller(xml_path: str, kp_fw: float = BAM_KP_FW, vin: float = BAM_VIN):
    """Compile MuJoCo model and initialize BAM M6 voltage-controlled actuators."""
    from bam.model import load_model
    from bam.mujoco import MujocoController

    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(xml_path)
    # Ensure head camera faces directly forward
    for cam in spec.cameras:
        if cam.name == "head_camera":
            cam.quat = [0.707107, 0, 0.707107, 0]

    # Set realistic ball rolling friction
    for g in spec.geoms:
        if g.name == "ball_geom":
            g.friction = [0.8, 0.03, 0.008]

    names = []
    for act in spec.actuators:
        tgt_name = act.target.name if hasattr(act.target, "name") else str(act.target)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(
        bam_model, names, model, data,
        vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN
    )
    return model, data, bam_ctrl, names


class TerminalInput:
    """Non-blocking keyboard reader."""
    def __init__(self):
        self.old_settings = None
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.reader_thread = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            self.reader_thread = threading.Thread(target=self._reader, daemon=True)
            self.reader_thread.start()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def _reader(self):
        while not self.stop_event.is_set():
            if select.select([sys.stdin], [], [], 0.02)[0]:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n", " "):
                    self.queue.put("space")
                else:
                    self.queue.put(ch.lower())

    def get_keys(self):
        keys = []
        while not self.queue.empty():
            try:
                keys.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return keys


def camera_hud_process(frame_queue: mp.Queue, stop_event: mp.Event):
    """Floating HUD Process at Top-Right Corner of screen."""
    import tkinter as tk

    try:
        root = tk.Tk()
        root.title("Microduck Vision HUD [Head Camera]")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        hud_w, hud_h = 360, 270
        screen_w = root.winfo_screenwidth()
        pos_x = max(20, screen_w - hud_w - 30)
        pos_y = 40
        root.geometry(f"{hud_w}x{hud_h}+{pos_x}+{pos_y}")

        canvas = tk.Canvas(root, width=hud_w, height=hud_h, bg="#0d1117", highlightthickness=2, highlightbackground="#30363d")
        canvas.pack(fill=tk.BOTH, expand=True)

        photo_container = [None]

        while not stop_event.is_set():
            try:
                msg = frame_queue.get(timeout=0.03)
                if msg is None:
                    break
                img_rgb, hud_info = msg

                pil_img = Image.fromarray(img_rgb).resize((hud_w, hud_h), Image.Resampling.BILINEAR)
                draw = ImageDraw.Draw(pil_img, "RGBA")

                # Top status header
                draw.rectangle([(0, 0), (hud_w, 28)], fill=(13, 17, 23, 220))
                state_text = f"AUTO-FETCH: {hud_info.get('state', 'IDLE')}"
                draw.text((10, 6), state_text, fill="#58a6ff")
                draw.text((hud_w - 130, 6), "[BAT 7.4V | 50Hz]", fill="#8b949e")

                # Detection Bounding Box
                detected = hud_info.get("detected", False)
                if detected:
                    bbox = hud_info["bbox"]
                    bx0 = int(bbox[0] * hud_w)
                    by0 = int(bbox[1] * hud_h)
                    bx1 = int(bbox[2] * hud_w)
                    by1 = int(bbox[3] * hud_h)
                    cx = (bx0 + bx1) // 2
                    cy = (by0 + by1) // 2

                    color = (57, 211, 83, 255)
                    bracket_len = max(6, min(14, (bx1 - bx0) // 4))

                    draw.line([(bx0, by0), (bx0 + bracket_len, by0)], fill=color, width=2)
                    draw.line([(bx0, by0), (bx0, by0 + bracket_len)], fill=color, width=2)
                    draw.line([(bx1, by0), (bx1 - bracket_len, by0)], fill=color, width=2)
                    draw.line([(bx1, by0), (bx1, by0 + bracket_len)], fill=color, width=2)
                    draw.line([(bx0, by1), (bx0 + bracket_len, by1)], fill=color, width=2)
                    draw.line([(bx0, by1), (bx0, by1 - bracket_len)], fill=color, width=2)
                    draw.line([(bx1, by1), (bx1 - bracket_len, by1)], fill=color, width=2)
                    draw.line([(bx1, by1), (bx1, by1 - bracket_len)], fill=color, width=2)

                    draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 230, 0, 220), width=1)
                    draw.line([(cx, cy - 6), (cx, cy + 6)], fill=(255, 230, 0, 220), width=1)

                    dist_m = hud_info.get("dist_m", 0.0)
                    bearing_deg = hud_info.get("bearing_deg", 0.0)
                    info_lbl = f"LOCK: {dist_m:.2f}m | {bearing_deg:+.1f}°"
                    draw.rectangle([(bx0, max(30, by0 - 18)), (bx0 + 130, max(46, by0 - 2))], fill=(13, 17, 23, 190))
                    draw.text((bx0 + 4, max(32, by0 - 16)), info_lbl, fill="#39d353")
                else:
                    draw.rectangle([(10, hud_h - 26), (hud_w - 10, hud_h - 8)], fill=(13, 17, 23, 160))
                    draw.text((16, hud_h - 24), "[SEARCHING FOR BALL IN FOV...]", fill="#e3b341")

                held = hud_info.get("held", False)
                if held:
                    draw.rectangle([(10, hud_h - 28), (hud_w - 10, hud_h - 6)], fill=(35, 134, 54, 230))
                    draw.text((18, hud_h - 24), "★ BALL SECURED IN BEAK -> RETURNING ★", fill="#ffffff")

                tk_photo = ImageTk.PhotoImage(pil_img)
                photo_container[0] = tk_photo
                canvas.create_image(0, 0, anchor=tk.NW, image=tk_photo)
                root.update_idletasks()
                root.update()
            except queue.Empty:
                root.update()
            except Exception:
                break
        root.destroy()
    except Exception as e:
        print(f"HUD Process terminated: {e}")


class FetchPolicyController:
    """Handles observations, policy inference, and skill swapping."""
    def __init__(self, model, data, walk_onnx_path, pick_onnx_path, bam_ctrl):
        self.model = model
        self.data = data
        self.bam_ctrl = bam_ctrl

        self.walk_session = ort.InferenceSession(walk_onnx_path) if Path(walk_onnx_path).exists() else None
        self.pick_session = ort.InferenceSession(pick_onnx_path) if Path(pick_onnx_path).exists() else None

        self.current_session = self.walk_session
        self.policy_name = "walk"
        self.n_joints = 14

        self.trunk_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qpos_adr = model.jnt_qposadr[self.trunk_jnt]
        self.qvel_adr = model.jnt_dofadr[self.trunk_jnt]

        self.ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_jnt = model.body_jntadr[self.ball_body]
        self.ball_qpos_adr = model.jnt_qposadr[self.ball_jnt]
        self.ball_qvel_adr = model.jnt_dofadr[self.ball_jnt]

        self.mouth_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.head_cmd = np.zeros(4, dtype=np.float32)
        self.body_cmd = np.zeros(6, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)

        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.default_joint_pos = np.array(HOME_LEG_CTRL, dtype=np.float32)

        self.ground_pick_active = False
        self.ground_pick_phase = 0.0
        self.ground_pick_period = 4.0

    def quat_rotate_inverse(self, q, v):
        w, x, y, z = q
        q_conj = np.array([w, -x, -y, -z], dtype=np.float32)
        t = 2.0 * np.cross(q_conj[1:4], v)
        return v + q_conj[0] * t + np.cross(q_conj[1:4], t)

    def get_observations(self):
        quat = self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7]
        omega_w = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6]
        omega_b = self.quat_rotate_inverse(quat, omega_w)

        grav_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        proj_grav = self.quat_rotate_inverse(quat, grav_w)

        joint_pos = self.data.qpos[7:7 + self.n_joints] - self.default_joint_pos
        joint_vel = self.data.qvel[6:6 + self.n_joints]

        self.command[:3] = self.vel_cmd
        self.command[3:7] = self.head_cmd
        self.command[7:13] = self.body_cmd

        return np.concatenate([
            omega_b, proj_grav, joint_pos, joint_vel, self.last_action, self.command
        ]).astype(np.float32)

    def infer_and_apply(self):
        obs = self.get_observations().reshape(1, -1)
        if self.current_session is not None:
            out = self.current_session.run(None, {"obs": obs})[0]
            action = out.flatten()
        else:
            action = np.zeros(self.n_joints, dtype=np.float32)

        self.last_action = action.copy()
        targets = self.default_joint_pos + action * 0.25
        if self.bam_ctrl is not None:
            self.bam_ctrl.q_target[:] = targets
        else:
            self.data.ctrl[:] = targets

    def trigger_ground_pick(self):
        if self.pick_session is None:
            return
        self.ground_pick_active = True
        self.ground_pick_phase = 0.0
        self.current_session = self.pick_session
        self.policy_name = "pick"

    def update_ground_pick(self, dt: float):
        if not self.ground_pick_active:
            return
        self.ground_pick_phase += dt / self.ground_pick_period
        self.vel_cmd[0] = np.cos(2.0 * np.pi * self.ground_pick_phase)
        self.vel_cmd[1] = np.sin(2.0 * np.pi * self.ground_pick_phase)
        self.vel_cmd[2] = 0.0

        if self.ground_pick_phase >= 0.75:
            self.ground_pick_active = False
            self.current_session = self.walk_session
            self.policy_name = "walk"
            self.vel_cmd[:] = 0.0


class AutonomousFetchEngine:
    """Autonomous State Machine for 1-Button Fetching."""
    STATE_IDLE = "IDLE (SẴN SÀNG)"
    STATE_TRACK_FLIGHT = "BÓNG ĐANG BAY"
    STATE_NAVIGATE = "DI CHUYỂN TỚI BÓNG"
    STATE_APPROACH = "TIẾP CẬN CHÍNH XÁC"
    STATE_GRASP = "CÚI GẮP BÓNG"
    STATE_RETURN = "MANG BÓNG VỀ"
    STATE_RELEASE = "NHẢ BÓNG HOÀN TẤT"

    def __init__(self, controller: FetchPolicyController, model, data):
        self.controller = controller
        self.model = model
        self.data = data
        self.state = self.STATE_IDLE
        self.ball_held = False
        self.state_time = 0.0

    def toss_ball(self, speed_fwd=1.2, lateral=-0.18, speed_up=0.85):
        """Simulate human hand throwing ball into arena."""
        b_qpos = self.controller.ball_qpos_adr
        b_qvel = self.controller.ball_qvel_adr

        self.data.qpos[b_qpos:b_qpos + 3] = HAND_LAUNCH_POS
        self.data.qpos[b_qpos + 3:b_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[b_qvel:b_qvel + 3] = [speed_fwd, lateral, speed_up]
        self.data.qvel[b_qvel + 3:b_qvel + 6] = np.random.uniform(-2, 2, 3)

        self.ball_held = False
        self.state = self.STATE_TRACK_FLIGHT
        self.state_time = 0.0
        print(f"\n[QUĂNG BÓNG!] Bàn tay người ném bóng vào sân -> Robot bắt đầu tự đi theo nhặt!")

    def update(self, dt: float, ball_detected: bool, ball_info: dict):
        self.state_time += dt

        robot_pos = self.data.qpos[self.controller.qpos_adr:self.controller.qpos_adr + 2]
        robot_quat = self.data.qpos[self.controller.qpos_adr + 3:self.controller.qpos_adr + 7]
        w, x, y, z = robot_quat
        robot_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        ball_pos_3d = self.data.qpos[self.controller.ball_qpos_adr:self.controller.ball_qpos_adr + 3]
        ball_pos_xy = ball_pos_3d[:2]
        ball_vel_3d = self.data.qvel[self.controller.ball_qvel_adr:self.controller.ball_qvel_adr + 3]
        ball_speed = float(np.linalg.norm(ball_vel_3d))

        mouth_pos_3d = self.data.site_xpos[self.controller.mouth_site]
        dist_mouth_to_ball = float(np.linalg.norm(mouth_pos_3d - ball_pos_3d))

        vec_to_ball = ball_pos_xy - robot_pos
        dist_robot_to_ball = float(np.linalg.norm(vec_to_ball))
        angle_to_ball = math.atan2(vec_to_ball[1], vec_to_ball[0])
        yaw_err_to_ball = (angle_to_ball - robot_yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Ball latch in beak
        if self.ball_held:
            hold_pos = mouth_pos_3d + np.array([0.02, 0.0, -0.01])
            self.data.qpos[self.controller.ball_qpos_adr:self.controller.ball_qpos_adr + 3] = hold_pos
            self.data.qvel[self.controller.ball_qvel_adr:self.controller.ball_qvel_adr + 6] = 0.0

        if self.state == self.STATE_IDLE:
            self.controller.vel_cmd[:] = 0.0
            self.controller.head_cmd[:] = 0.0

        elif self.state == self.STATE_TRACK_FLIGHT:
            self.controller.head_cmd[2] = np.clip(yaw_err_to_ball * 0.8, -0.5, 0.5)
            # When ball touches ground or 1 second passes
            if (ball_pos_3d[2] < 0.06 and ball_speed < 0.5) or self.state_time > 1.2:
                self.state = self.STATE_NAVIGATE
                self.state_time = 0.0
                print("[Auto-Fetch] Bóng đã chạm đất -> Robot di chuyển tới bóng!")

        elif self.state == self.STATE_NAVIGATE:
            self.controller.head_cmd[2] = np.clip(yaw_err_to_ball * 0.7, -0.4, 0.4)
            self.controller.head_cmd[1] = np.clip(-0.3, -0.5, 0.0)

            cmd_yaw = np.clip(yaw_err_to_ball * 1.8, -0.8, 0.8)
            fwd_speed = 0.22 if abs(yaw_err_to_ball) < 0.5 else 0.08
            self.controller.vel_cmd = np.array([fwd_speed, 0.0, cmd_yaw], dtype=np.float32)

            if dist_robot_to_ball < 0.22:
                self.state = self.STATE_APPROACH
                self.state_time = 0.0
                print("[Auto-Fetch] Đến gần quả bóng (< 22cm) -> Giảm tốc và căn chỉnh mỏ!")

        elif self.state == self.STATE_APPROACH:
            self.controller.head_cmd[1] = -0.4
            cmd_yaw = np.clip(yaw_err_to_ball * 2.2, -0.5, 0.5)
            cmd_fwd = np.clip((dist_robot_to_ball - 0.09) * 0.8, 0.0, 0.08)
            self.controller.vel_cmd = np.array([cmd_fwd, 0.0, cmd_yaw], dtype=np.float32)

            if dist_mouth_to_ball < 0.08 or (dist_robot_to_ball < 0.12 and abs(yaw_err_to_ball) < 0.2):
                self.state = self.STATE_GRASP
                self.state_time = 0.0
                self.controller.trigger_ground_pick()
                print("[Auto-Fetch] Mỏ robot đã vào vị trí -> Kích hoạt Ground Pick cúi gắp bóng!")

        elif self.state == self.STATE_GRASP:
            if self.controller.ground_pick_phase > 0.35 and not self.ball_held:
                self.ball_held = True
                print("[Auto-Fetch] ★ ĐÃ GẮP ĐƯỢC BÓNG VÀO MỎ! ★")

            if not self.controller.ground_pick_active and self.state_time > 2.5:
                self.state = self.STATE_RETURN
                self.state_time = 0.0
                print("[Auto-Fetch] Đứng dậy thành công -> Mang bóng về vị trí người ném!")

        elif self.state == self.STATE_RETURN:
            vec_to_human = HUMAN_ZONE_POS[:2] - robot_pos
            dist_to_human = float(np.linalg.norm(vec_to_human))
            angle_to_human = math.atan2(vec_to_human[1], vec_to_human[0])
            yaw_err_human = (angle_to_human - robot_yaw + math.pi) % (2.0 * math.pi) - math.pi

            cmd_yaw = np.clip(yaw_err_human * 1.8, -0.8, 0.8)
            fwd_speed = 0.20 if abs(yaw_err_human) < 0.6 else 0.06
            self.controller.vel_cmd = np.array([fwd_speed, 0.0, cmd_yaw], dtype=np.float32)

            if dist_to_human < 0.28:
                self.state = self.STATE_RELEASE
                self.state_time = 0.0
                print("[Auto-Fetch] Đã về tới chân người ném -> Thả bóng!")

        elif self.state == self.STATE_RELEASE:
            self.controller.vel_cmd[:] = 0.0
            if self.ball_held and self.state_time > 0.5:
                self.ball_held = False
                self.data.qvel[self.controller.ball_qvel_adr:self.controller.ball_qvel_adr + 3] = [0.05, 0.0, -0.1]
                print("[Auto-Fetch] ★ ĐÃ THẢ BÓNG VÀO VÙNG TRẢ! NHIỆM VỤ HOÀN TẤT! ★")

            if self.state_time > 2.0:
                self.state = self.STATE_IDLE
                print("[Auto-Fetch] Robot sẵn sàng cho lượt ném tiếp theo! (Bấm SPACE để ném bóng tiếp)")


def detect_ball_in_frame(img_rgb: np.ndarray, model, data, cam_id, ball_pos_3d):
    """Detect ball in RGB image via color segmentation and optical projection."""
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    mask = (r > 150) & (g > 60) & (b < 90) & (r > g * 1.15)
    pixel_count = int(np.sum(mask))

    h, w = img_rgb.shape[:2]
    cam_pos = data.cam_xpos[cam_id]
    dist_3d = float(np.linalg.norm(ball_pos_3d - cam_pos))

    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    p_cam = cam_mat.T @ (ball_pos_3d - cam_pos)
    bearing_deg = math.degrees(math.atan2(p_cam[0], -p_cam[2])) if abs(p_cam[2]) > 1e-4 else 0.0

    if pixel_count >= 15:
        ys, xs = np.where(mask)
        x0, x1 = float(xs.min()) / w, float(xs.max()) / w
        y0, y1 = float(ys.min()) / h, float(ys.max()) / h
        return True, {
            "bbox": [x0, y0, x1, y1],
            "dist_m": dist_3d,
            "bearing_deg": bearing_deg,
            "pixels": pixel_count,
        }
    return False, {"dist_m": dist_3d, "bearing_deg": bearing_deg, "pixels": 0}


def main():
    parser = argparse.ArgumentParser(description="Microduck 1-Button Ball Fetch Simulation")
    parser.add_argument("--scene", type=str, default=SCENE_FETCH_XML, help="Path to scene XML")
    parser.add_argument("--walking", type=str, default=DEFAULT_WALK_ONNX, help="Path to walking ONNX policy")
    parser.add_argument("--ground-pick", type=str, default=DEFAULT_PICK_ONNX, help="Path to ground pick ONNX policy")
    args = parser.parse_args()

    print("=" * 72)
    print("   MICRODUCK 1-BUTTON BALL FETCHING SIMULATION")
    print("=" * 72)
    print("  Chỉ cần nhấn 1 NÚT DUY NHẤT (SPACE hoặc ENTER hoặc F):")
    print("   -> Bàn tay người tự động quăng bóng đi theo hình vòng cung parabol.")
    print("   -> Màn hình Camera góc trên bên phải nhận diện quả bóng.")
    print("   -> Robot tự động chạy tới bóng, cúi gắp và mang về chân người ném!")
    print("  Nhấn 'Q' để thoát.")
    print("=" * 72 + "\n")

    # 1. Load scene and BAM actuators
    model, data, bam_ctrl, act_names = load_bam_controller(args.scene)
    renderer = mujoco.Renderer(model, height=180, width=240)

    # 2. Initialize controller and Fetch state machine
    policy = FetchPolicyController(model, data, args.walking, args.ground_pick, bam_ctrl)
    engine = AutonomousFetchEngine(policy, model, data)

    # 3. Floating Camera HUD process in top-right corner
    frame_queue = mp.Queue(maxsize=2)
    hud_stop_event = mp.Event()
    hud_proc = mp.Process(target=camera_hud_process, args=(frame_queue, hud_stop_event), daemon=True)
    hud_proc.start()

    control_dt = 0.02
    decimation = int(control_dt / model.opt.timestep)

    # Initial ball placement
    data.qpos[policy.ball_qpos_adr:policy.ball_qpos_adr + 3] = [-0.2, 0.0, BALL_RADIUS]

    first_throw_done = False
    start_sim_time = time.time()

    with TerminalInput() as term, \
         mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

        viewer.cam.azimuth = 145
        viewer.cam.elevation = -22
        viewer.cam.distance = 2.2
        viewer.cam.lookat = [0.3, 0.0, 0.2]
        viewer.sync()

        step_counter = 0
        last_time = time.time()

        try:
            while viewer.is_running():
                t0 = time.time()
                dt = t0 - last_time
                last_time = t0

                # Auto throw first ball after 1.5s for instant demonstration
                if not first_throw_done and (time.time() - start_sim_time > 1.5):
                    first_throw_done = True
                    speed = np.random.uniform(1.1, 1.35)
                    lat = np.random.uniform(-0.20, -0.12)
                    up = np.random.uniform(0.75, 0.95)
                    engine.toss_ball(speed_fwd=speed, lateral=lat, speed_up=up)

                # Process keyboard inputs
                for k in term.get_keys():
                    if k in ("space", "f"):
                        # THE 1-BUTTON TRIGGER!
                        speed = np.random.uniform(1.1, 1.4)
                        lat = np.random.uniform(-0.22, -0.10)
                        up = np.random.uniform(0.75, 0.95)
                        engine.toss_ball(speed_fwd=speed, lateral=lat, speed_up=up)
                    elif k == "q":
                        break

                policy.update_ground_pick(control_dt)

                # Render robot head camera
                renderer.update_scene(data, camera="head_camera")
                cam_rgb = renderer.render()

                ball_pos_3d = data.qpos[policy.ball_qpos_adr:policy.ball_qpos_adr + 3]
                detected, ball_info = detect_ball_in_frame(cam_rgb, model, data, policy.cam_id, ball_pos_3d)

                # Step autonomous fetch engine
                engine.update(control_dt, detected, ball_info)

                # Send frame to floating HUD
                if step_counter % 2 == 0 and not frame_queue.full():
                    hud_dict = {
                        "state": engine.state,
                        "detected": detected,
                        "held": engine.ball_held,
                        **ball_info
                    }
                    try:
                        frame_queue.put_nowait((cam_rgb, hud_dict))
                    except queue.Full:
                        pass

                policy.infer_and_apply()

                for _ in range(decimation):
                    bam_ctrl.update()
                    mujoco.mj_step(model, data)

                viewer.sync()
                step_counter += 1

                elapsed = time.time() - t0
                to_sleep = control_dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

        except KeyboardInterrupt:
            pass

    hud_stop_event.set()
    frame_queue.put(None)
    hud_proc.join(timeout=1.0)
    print("\nMô phỏng Fetch kết thúc.")


if __name__ == "__main__":
    main()

