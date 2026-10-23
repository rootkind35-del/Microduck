#!/usr/bin/env python3
"""Interactive & Headless Rehearsal Simulation for Microduck High Stairs RL.

Runs standalone CPU simulation of Microduck on the high stairs environment
(`src/mjlab_microduck/robot/microduck/scene_stairs.xml`) with BAM M6
voltage-controlled actuator dynamics at 50 Hz control rate.

Features:
- BAM M6 XL330 voltage-controlled actuators matching warp training.
- 50 Hz control loop (physics dt=0.005s, decimation=4 -> 0.020s).
- Strict 61D unified actor observation layout (48D proprio + 13D commands).
- Stair spawn presets across consecutive flights (2.5cm, 3.0cm, 3.5cm) and platforms.
- Real-time keyboard controls:
  UP/DOWN forward velocity, LEFT/RIGHT strafe, A/E yaw, SPACE zero,
  B body pitch trim, H head gaze, C heading lock, P perturbation push,
  U standup recovery, T toggle policy, Q quit.
- Real-time balance telemetry: trunk pitch/roll tilt, foot contact per step,
  normal forces, knee/ankle torque load relative to 0.96 Nm stall limit.
- In-viewer HUD overlay via `viewer.set_texts`.
- Headless execution mode (--headless, --debug) for automated verification and CI.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import select
import sys
import termios
import threading
import time
from pathlib import Path
from typing import Any

# Force WSLg Mesa D3D12 to use dedicated NVIDIA GPU instead of integrated Intel GPU
# (prevents window freeze / 'Not Responding' white-screen in WSL2)
if "MESA_D3D12_DEFAULT_ADAPTER_NAME" not in os.environ:
    os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"

import mujoco
import numpy as np

try:
    import mujoco.viewer
    HAS_MUJOCO_VIEWER = True
except ImportError:
    HAS_MUJOCO_VIEWER = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class _HeadlessViewer:
    user_scn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def is_running(self):
        return True

    def sync(self):
        pass

    def close(self):
        pass


# Default paths
DEFAULT_SCENE_XML = "src/mjlab_microduck/robot/microduck/scene_stairs.xml"

# BAM M6 defaults (mirroring src/mjlab_microduck/robot/microduck_constants.py)
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_RANGE = (6.5, 8.2)
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)
BAM_VIN_MIN = 6.0
BAM_MAX_CURRENT = None
BAM_STALL_TORQUE = 0.96  # Nm stall limit for Dynamixel XL330
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

# Timing invariants
PHYSICS_DT = 0.005  # 200 Hz
DECIMATION = 4      # 4 substeps -> 50 Hz control
CONTROL_DT = PHYSICS_DT * DECIMATION
CONTROL_FREQ = 1.0 / CONTROL_DT  # 50.0 Hz

# Default joint positions (HOME pose: 14 actuated servos)
DEFAULT_JOINT_POS = np.array([
    0.0, -0.087266, -0.457924, -0.004940, 0.452984,  # Left leg (0..4)
    0.349066, 0.349066, 0.0, 0.0,                    # Neck & Head (5..8)
    0.0, 0.087266, 0.457924, 0.004940, -0.452984,    # Right leg (9..13)
], dtype=np.float32)

ACTION_SCALE = 1.0

# Spawn presets contract: mapping name -> keyframe name in scene_stairs.xml
SPAWN_PRESETS: dict[str, str] = {
    "bottom": "STAND",
    "flight1": "SPAWN_STAIRS_25",
    "flight2": "SPAWN_STAIRS_30",
    "flight3": "SPAWN_STAIRS_35",
    "top": "SPAWN_SUMMIT",
    "floor": "STAND",
    "stairs_25": "SPAWN_STAIRS_25",
    "landing_1": "SPAWN_LANDING_1",
    "stairs_30": "SPAWN_STAIRS_30",
    "landing_2": "SPAWN_LANDING_2",
    "stairs_35": "SPAWN_STAIRS_35",
    "summit": "SPAWN_SUMMIT",
    "init": "INIT",
}

HOTKEY_PRESETS: dict[str, str] = {
    "1": "bottom",
    "2": "flight1",
    "3": "flight2",
    "4": "flight3",
    "5": "top",
}


def load_bam_model(kp_fw: float = BAM_KP_FW, vin: float = BAM_VIN, max_current: Any = BAM_MAX_CURRENT):
    """Build the BAM M6 model + XL330 voltage-controlled actuator."""
    from bam.model import load_model
    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = max_current if (max_current and max_current > 0) else None
    return bam_model


def load_mujoco_with_bam(
    xml_path: str,
    bam_model: Any = None,
    timestep: float = PHYSICS_DT,
    vin_drop_gain: float = BAM_VIN_DROP_GAIN,
    vin_min: float = BAM_VIN_MIN,
):
    """Load scene and attach BAM M6 MujocoController to non-passive actuators."""
    from bam.mujoco import MujocoController

    if bam_model is None:
        bam_model = load_bam_model(BAM_KP_FW, BAM_VIN, BAM_MAX_CURRENT)

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(xml_path)
    spec.option.timestep = timestep

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
    model.opt.timestep = timestep
    data = mujoco.MjData(model)

    bam_ctrl = MujocoController(
        bam_model,
        names,
        model,
        data,
        vin_drop_gain=vin_drop_gain,
        vin_min=vin_min,
    )
    _orig_update = bam_ctrl.update
    def _safe_update():
        _orig_update()
        np.clip(data.ctrl[:len(names)], -force_limit, force_limit, out=data.ctrl[:len(names)])
    bam_ctrl.update = _safe_update
    return model, data, bam_ctrl, names


def load_bam_controller(xml_path: str, kp_fw: float = BAM_KP_FW, vin: float = BAM_VIN):
    """Compatibility wrapper returning model, data, bam_ctrl, act_names."""
    bam_model = load_bam_model(kp_fw=kp_fw, vin=vin, max_current=BAM_MAX_CURRENT)
    return load_mujoco_with_bam(xml_path, bam_model=bam_model, timestep=PHYSICS_DT)


def reset_to_preset(model: mujoco.MjModel, data: mujoco.MjData, preset_name: str, bam_ctrl: Any = None) -> None:
    """Reset simulation state to the requested stair keyframe preset."""
    key_name = SPAWN_PRESETS.get(preset_name, "STAND")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
        data.qpos[7:7 + len(DEFAULT_JOINT_POS)] = DEFAULT_JOINT_POS
        data.ctrl[:len(DEFAULT_JOINT_POS)] = DEFAULT_JOINT_POS

    data.qvel[:] = 0.0
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)


def compute_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector in body frame: R^T * [0, 0, -1]."""
    w, x, y, z = quat
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.array([gx, gy, gz], dtype=np.float32)


def compute_euler_angles(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to roll, pitch, yaw in radians."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def build_61d_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    last_action: np.ndarray,
    cmd_twist: np.ndarray,
    cmd_head: np.ndarray | None = None,
    cmd_body: np.ndarray | None = None,
) -> np.ndarray:
    """Build the strict 61D unified actor observation vector:
    48D base proprioception + 13D command block [twist(3), head_pose(4), body_pose(6)].
    """
    # 1. Base angular velocity in body frame (3D)
    imu_ang_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if imu_ang_id >= 0:
        sensor_adr = model.sensor_adr[imu_ang_id]
        base_ang_vel = np.array(data.sensordata[sensor_adr:sensor_adr + 3], dtype=np.float32)
    else:
        base_ang_vel = np.array(data.qvel[3:6], dtype=np.float32)

    # 2. Projected gravity (3D)
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    if trunk_body_id >= 0:
        base_quat = np.array(data.xquat[trunk_body_id], dtype=np.float32)
    else:
        base_quat = np.array(data.qpos[3:7], dtype=np.float32)

    norm = np.linalg.norm(base_quat)
    if norm > 1e-6:
        base_quat /= norm
    else:
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    projected_gravity = compute_projected_gravity(base_quat)

    # 3. Joint positions relative to default (14D)
    num_servos = 14
    joint_pos = np.array(data.qpos[7:7 + num_servos], dtype=np.float32) - DEFAULT_JOINT_POS

    # 4. Joint velocities (14D)
    joint_vel = np.array(data.qvel[6:6 + num_servos], dtype=np.float32)

    # 5. Last actions (14D)
    actions = np.array(last_action, dtype=np.float32)

    # 6. Command block (13D)
    twist_cmd = np.array(cmd_twist, dtype=np.float32)
    head_pose_cmd = np.zeros(4, dtype=np.float32) if cmd_head is None else np.array(cmd_head, dtype=np.float32)
    body_pose_cmd = np.zeros(6, dtype=np.float32) if cmd_body is None else np.array(cmd_body, dtype=np.float32)

    obs = np.concatenate([
        base_ang_vel,         # 3
        projected_gravity,    # 3
        joint_pos,            # 14
        joint_vel,            # 14
        actions,              # 14
        twist_cmd,            # 3
        head_pose_cmd,        # 4
        body_pose_cmd,        # 6
    ])
    assert obs.shape == (61,), f"Observation dimension mismatch: expected 61, got {obs.shape}"
    return obs


def get_stair_zone(x: float) -> str:
    """Identify which stair flight or landing zone the robot is currently on."""
    if x < 0.30:
        return "Floor (Apron)"
    elif x < 0.42:
        return "Flight 1 (Step 1_1, 1.0cm)"
    elif x < 0.54:
        return "Flight 1 (Step 1_2, 1.0cm)"
    elif x < 0.66:
        return "Flight 1 (Step 1_3, 1.0cm)"
    elif x < 1.02:
        return "Landing 1 (3.0cm elev)"
    elif x < 1.14:
        return "Flight 2 (Step 2_1, 1.25cm)"
    elif x < 1.26:
        return "Flight 2 (Step 2_2, 1.25cm)"
    elif x < 1.38:
        return "Flight 2 (Step 2_3, 1.25cm)"
    elif x < 1.74:
        return "Landing 2 (6.75cm elev)"
    elif x < 1.86:
        return "Flight 3 (Step 3_1, 1.5cm)"
    elif x < 1.98:
        return "Flight 3 (Step 3_2, 1.5cm)"
    elif x < 2.10:
        return "Flight 3 (Step 3_3, 1.5cm)"
    else:
        return "Summit (11.25cm elev)"


def identify_step_contacts_and_forces(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[str, str, float, float]:
    """Identify which stair geom each foot is contacting, and total normal forces."""
    foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

    contact_l_geom = "air"
    contact_r_geom = "air"
    fn_left = 0.0
    fn_right = 0.0
    c_force = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        mujoco.mj_contactForce(model, data, i, c_force)
        fn = float(abs(c_force[0]))

        if g1 == foot_l_id or g2 == foot_l_id:
            other_id = g2 if g1 == foot_l_id else g1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id) or f"geom_{other_id}"
            contact_l_geom = name
            fn_left += fn
        elif g1 == foot_r_id or g2 == foot_r_id:
            other_id = g2 if g1 == foot_r_id else g1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id) or f"geom_{other_id}"
            contact_r_geom = name
            fn_right += fn

    return contact_l_geom, contact_r_geom, fn_left, fn_right


def compute_torque_loads(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Compute knee and ankle servo torque loads relative to 0.96 Nm stall limit."""
    # Joint mapping:
    # 3: left_knee, 4: left_ankle, 12: right_knee, 13: right_ankle
    tau_knee_l = float(abs(data.ctrl[3])) if len(data.ctrl) > 3 else 0.0
    tau_ankle_l = float(abs(data.ctrl[4])) if len(data.ctrl) > 4 else 0.0
    tau_knee_r = float(abs(data.ctrl[12])) if len(data.ctrl) > 12 else 0.0
    tau_ankle_r = float(abs(data.ctrl[13])) if len(data.ctrl) > 13 else 0.0

    max_knee = max(tau_knee_l, tau_knee_r)
    max_ankle = max(tau_ankle_l, tau_ankle_r)

    knee_load_pct = (max_knee / BAM_STALL_TORQUE) * 100.0
    ankle_load_pct = (max_ankle / BAM_STALL_TORQUE) * 100.0

    return {
        "tau_knee_l": tau_knee_l,
        "tau_knee_r": tau_knee_r,
        "tau_ankle_l": tau_ankle_l,
        "tau_ankle_r": tau_ankle_r,
        "knee_load_pct": knee_load_pct,
        "ankle_load_pct": ankle_load_pct,
    }


def compute_balance_telemetry(
    model: mujoco.MjModel, data: mujoco.MjData, bam_ctrl: Any = None
) -> dict[str, Any]:
    """Compute complete real-time balance and stairs navigation telemetry."""
    roll, pitch, yaw = compute_euler_angles(data.qpos[3:7])
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)

    if abs(pitch_deg) > 50.0 or abs(roll_deg) > 45.0:
        status = "FALLEN"
    elif abs(pitch_deg) > 35.0 or abs(roll_deg) > 30.0:
        status = "UNSTABLE"
    else:
        status = "BALANCED"

    contact_l, contact_r, fn_l, fn_r = identify_step_contacts_and_forces(model, data)
    torque_info = compute_torque_loads(model, data)
    pos = np.array(data.qpos[:3], dtype=np.float32)
    zone = get_stair_zone(pos[0])

    return {
        "pos": pos,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": math.degrees(yaw),
        "status": status,
        "feet_contacts": (contact_l, contact_r),
        "normal_forces": (fn_l, fn_r),
        "torque_loads": torque_info,
        "zone": zone,
    }


# =============================================================================
# Summit Autonomous Focus, Line-of-Sight Guidance & Corridor Anti-Drift Guard
# =============================================================================

SUMMIT_TARGET_POS = np.array([2.34, 0.0, 0.1125], dtype=np.float32)
SUMMIT_SITE_POS = np.array([2.34, 0.0, 0.1175], dtype=np.float32)
SUMMIT_ARRIVAL_X = 2.15
CORRIDOR_HALF_WIDTH = 0.25        # Physical staircase lateral edges at y = +/- 0.25m
SAFE_CORRIDOR_HALF_WIDTH = 0.15    # Inner safe envelope
DANGER_CORRIDOR_HALF_WIDTH = 0.20   # Active edge guard threshold


class SummitFocusNavigator:
    """Autonomous Summit Target Focus, Line-of-Sight Guidance & Corridor Anti-Drift Guard."""

    def __init__(
        self,
        target_pos: np.ndarray = SUMMIT_TARGET_POS,
        enabled: bool = True,
        kp_heading: float = 1.4,
        kd_heading: float = 0.10,
        kp_lateral: float = 2.2,
        max_yaw_rate: float = 0.45,
    ):
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.enabled = enabled
        self.kp_heading = kp_heading
        self.kd_heading = kd_heading
        self.kp_lateral = kp_lateral
        self.max_yaw_rate = max_yaw_rate
        self.summit_reached = False
        self.edge_guard_active = False
        self.last_yaw = 0.0
        self.status_msg = "FOCUS ON SUMMIT"

    def update(
        self,
        robot_pos: np.ndarray,
        robot_quat: np.ndarray,
        robot_qvel: np.ndarray,
        base_cmd_twist: np.ndarray,
        dt: float = CONTROL_DT,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Compute auto-aim heading correction, corridor centering, edge guard, and head gaze."""
        x, y, z = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
        roll, pitch, cur_yaw = compute_euler_angles(robot_quat)
        yaw_rate = float(robot_qvel[5]) if len(robot_qvel) > 5 else (cur_yaw - self.last_yaw) / max(1e-4, dt)
        self.last_yaw = cur_yaw

        cmd_twist = base_cmd_twist.copy()
        cmd_head = np.zeros(4, dtype=np.float32)

        dx = float(self.target_pos[0] - x)
        dy = float(self.target_pos[1] - y)  # 0.0 - y = -y
        dist_to_summit = math.hypot(dx, dy)

        # 1. Summit arrival check: robot reached summit platform (x >= 2.15m)
        if x >= SUMMIT_ARRIVAL_X and abs(y) <= CORRIDOR_HALF_WIDTH:
            self.summit_reached = True
            self.status_msg = "SUMMIT REACHED (BALANCING)"
            cmd_twist[:] = 0.0
            cmd_head[:] = 0.0
            nav_info = {
                "dist": dist_to_summit,
                "lat_dev": y,
                "heading_err_deg": 0.0,
                "status": self.status_msg,
                "arrived": True,
                "edge_guard": False,
                "gaze_pitch_deg": 0.0,
                "gaze_yaw_deg": 0.0,
            }
            return cmd_twist, cmd_head, nav_info

        if not self.enabled:
            self.status_msg = "FOCUS MANUAL"
            nav_info = {
                "dist": dist_to_summit,
                "lat_dev": y,
                "heading_err_deg": 0.0,
                "status": self.status_msg,
                "arrived": False,
                "edge_guard": False,
                "gaze_pitch_deg": 0.0,
                "gaze_yaw_deg": 0.0,
            }
            return cmd_twist, cmd_head, nav_info

        # 2. Closed-loop Path Guidance & Corridor Centering
        user_vx = float(cmd_twist[0])
        abs_y = abs(y)
        cross_track_err = -y  # target y = 0.0, error = 0.0 - y

        # Pure-pursuit lookahead towards corridor centerline y=0
        lookahead = float(np.clip(dx * 0.45, 0.18, 0.45))
        desired_yaw = math.atan2(cross_track_err * self.kp_lateral, lookahead)
        heading_err = (desired_yaw - cur_yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Yaw rate steering regulation
        wz_steer = self.kp_heading * heading_err - self.kd_heading * yaw_rate
        wz_steer = float(np.clip(wz_steer, -self.max_yaw_rate, self.max_yaw_rate))

        # Lateral strafe trim
        vy_trim = float(np.clip(0.3 * cross_track_err, -0.03, 0.03))

        # 3. Anti-Fall Edge Guard
        # If robot drifts close to the edge of the stairs (|y| > SAFE_CORRIDOR_HALF_WIDTH = 0.15m)
        self.edge_guard_active = False
        if abs_y > SAFE_CORRIDOR_HALF_WIDTH and x < SUMMIT_ARRIVAL_X:
            self.edge_guard_active = True
            urgency = float(np.clip((abs_y - SAFE_CORRIDOR_HALF_WIDTH) / (CORRIDOR_HALF_WIDTH - SAFE_CORRIDOR_HALF_WIDTH), 0.0, 1.0))
            steer_sign = 1.0 if cross_track_err > 0 else -1.0
            wz_steer += steer_sign * 0.30 * urgency
            wz_steer = float(np.clip(wz_steer, -0.65, 0.65))
            # Throttle forward velocity to prioritize turning over falling off
            user_vx *= (1.0 - 0.75 * urgency)
            if abs_y >= DANGER_CORRIDOR_HALF_WIDTH:
                user_vx = 0.0  # Stop forward motion and pivot back to center!
                self.status_msg = f"EDGE GUARD: PIVOT TO CENTER (|y|={abs_y*100:.1f}cm)"
            else:
                self.status_msg = f"EDGE GUARD: STEERING TO CENTER (|y|={abs_y*100:.1f}cm)"
        else:
            self.status_msg = "FOCUS ON SUMMIT"

        # Apply guidance when moving forward or when edge guard intervenes
        if abs(user_vx) > 0.01 or self.edge_guard_active:
            cmd_twist[0] = user_vx
            cmd_twist[1] = vy_trim
            cmd_twist[2] = wz_steer

        # 4. Head Gaze Focus on Summit Center
        head_pitch_target = math.atan2(float(self.target_pos[2] - z - 0.15), max(0.20, dx))
        cmd_head[1] = float(np.clip(head_pitch_target - 0.12, -0.40, 0.15))
        cmd_head[2] = float(np.clip(heading_err * 0.75, -0.35, 0.35))

        nav_info = {
            "dist": dist_to_summit,
            "lat_dev": y,
            "heading_err_deg": math.degrees(heading_err),
            "status": self.status_msg,
            "arrived": self.summit_reached,
            "edge_guard": self.edge_guard_active,
            "gaze_pitch_deg": math.degrees(cmd_head[1]),
            "gaze_yaw_deg": math.degrees(cmd_head[2]),
        }
        return cmd_twist, cmd_head, nav_info


class SummitFocusOverlay:
    """Renders 3D summit target beacon, sight beam, and corridor boundary lines in viewer.user_scn."""

    TARGET_COLOR = (0.0, 0.9, 0.9, 0.85)    # Cyan target
    BEAM_COLOR = (1.0, 0.8, 0.0, 0.55)      # Golden sight beam
    CORRIDOR_COLOR = (0.2, 0.8, 0.2, 0.40)  # Green safe centerline
    EDGE_COLOR = (0.9, 0.2, 0.2, 0.45)      # Red warning edges

    def __init__(self, target_pos: np.ndarray = SUMMIT_TARGET_POS):
        self.target_pos = np.asarray(target_pos, dtype=np.float64)

    def draw(self, data: mujoco.MjData, scn: Any, nav_info: dict[str, Any] | None = None) -> None:
        if scn is None or not hasattr(scn, "ngeom"):
            return

        # 1. Target Disc / Beacon at Summit
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([0.08, 0.005, 0.0], dtype=np.float64),
                np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.005], dtype=np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                np.array(self.TARGET_COLOR, dtype=np.float32),
            )
            scn.ngeom += 1

        # Vertical beacon pin pole
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            p1 = np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.005], dtype=np.float64)
            p2 = np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.180], dtype=np.float64)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(self.TARGET_COLOR, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p1, p2)
            scn.ngeom += 1

        # Beacon top sphere
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.015, 0.0, 0.0], dtype=np.float64),
                np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.180], dtype=np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                np.array(self.TARGET_COLOR, dtype=np.float32),
            )
            scn.ngeom += 1

        # 2. Sight Beam connecting robot head to summit target
        robot_pos = data.qpos[:3].copy()
        head_pos = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.16], dtype=np.float64)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            beam_color = (1.0, 0.3, 0.3, 0.6) if (nav_info and nav_info.get("edge_guard")) else self.BEAM_COLOR
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(beam_color, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, head_pos, self.target_pos)
            scn.ngeom += 1

        # 3. Corridor Guideline (Centerline y=0)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            p_start = np.array([0.30, 0.0, 0.015], dtype=np.float64)
            p_end = np.array([2.10, 0.0, 0.115], dtype=np.float64)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(self.CORRIDOR_COLOR, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, p_start, p_end)
            scn.ngeom += 1


def query_forward_terrain_elevation(
    model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float
) -> float:
    """Raycast query with direct box geom fallback for terrain elevation at (x, y)."""
    pnt = np.array([x, y, 0.40], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, 1, geomid)
    if dist >= 0 and geomid[0] >= 0 and model.geom_bodyid[geomid[0]] == 0:
        return float(pnt[2] - dist)

    # Geometry lookup fallback for static worldbody boxes
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0:
            pos = model.geom_pos[gid]
            size = model.geom_size[gid]
            gtype = model.geom_type[gid]
            if gtype == 6:  # Box
                if (pos[0] - size[0] <= x <= pos[0] + size[0]) and (pos[1] - size[1] <= y <= pos[1] + size[1]):
                    return float(pos[2] + size[2])
    return 0.0


def detect_riser_contact(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[bool, float, float]:
    """Detect horizontal obstacle/riser collision forces on left and right feet."""
    foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    f_xy_l = 0.0
    f_xy_r = 0.0
    c_force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if g1 in (foot_l_id, foot_r_id) or g2 in (foot_l_id, foot_r_id):
            mujoco.mj_contactForce(model, data, i, c_force)
            f_tan = math.hypot(c_force[1], c_force[2])
            n_x = con.frame[0]
            n_y = con.frame[1]
            f_norm_xy = abs(c_force[0]) * math.hypot(n_x, n_y)
            f_tot_horiz = max(f_tan, f_norm_xy)
            if g1 == foot_l_id or g2 == foot_l_id:
                f_xy_l += f_tot_horiz
            if g1 == foot_r_id or g2 == foot_r_id:
                f_xy_r += f_tot_horiz
    is_riser_hit = (f_xy_l > 0.4 or f_xy_r > 0.4)
    return is_riser_hit, f_xy_l, f_xy_r


class StairStepBooster:
    """Adaptive stair clearance boost & progressive obstacle height probing controller.

    Addresses user request and root-cause audit:
    - Normal walk when unobstructed: pure walking.onnx gait.
    - Senses forward obstacle elevation (mj_ray) and riser contact (F_xy > 0.4N).
    - Progressively probes clearance levels 1-4 to receive height data and clear obstacles.
    - Applies Stance Forward Counter-Pitch Lean to prevent backward tipping onto buttocks.
    """

    def __init__(self, boost_level: float = 1.0, auto_probe: bool = True):
        self.boost_level = float(np.clip(boost_level, 0.0, 2.0))
        self.auto_probe = auto_probe
        self.probe_level: int = 1
        self._active_probe_level: int = 1
        self.obstacle_detected: bool = False
        self.sensed_height: float = 0.0
        self.stalled_steps: int = 0
        self.cleared_steps: int = 0
        self.phase: str = "WALK"
        self.status_msg: str = "NORMAL WALK"

    def update_obstacle_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fn_left: float,
        fn_right: float,
        is_walking: bool = True,
    ) -> dict[str, Any]:
        """Detect forward obstacle height, riser contact, and update progressive probing levels."""
        if not self.auto_probe:
            return {
                "detected": False,
                "height": 0.0,
                "level": self.probe_level,
                "phase": "MANUAL",
                "status": "PROBE OFF",
                "riser_hit": False,
            }

        foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        xl, yl, zl = (
            data.geom_xpos[foot_l_id]
            if foot_l_id >= 0
            else (data.qpos[0], data.qpos[1] - 0.04, data.qpos[2])
        )
        xr, yr, zr = (
            data.geom_xpos[foot_r_id]
            if foot_r_id >= 0
            else (data.qpos[0], data.qpos[1] + 0.04, data.qpos[2])
        )

        current_foot_z = min(zl, zr)
        lead_x = max(xl, xr)
        lead_y = yl if xl >= xr else yr

        # Probe terrain elevation ahead of leading foot (4.5 cm ahead of toe)
        probe_x = lead_x + 0.045
        elev_ahead = query_forward_terrain_elevation(model, data, probe_x, lead_y)
        delta_h = float(elev_ahead - current_foot_z)

        # Detect horizontal riser collision
        is_riser_hit, f_xy_l, f_xy_r = detect_riser_contact(model, data)

        if delta_h > 0.005 or is_riser_hit:
            self.obstacle_detected = True
            self.sensed_height = max(delta_h, 0.010)
            self.cleared_steps = 0

            # Baseline probe level from sensed obstacle height
            if self.sensed_height <= 0.011:
                base_lvl = 1
            elif self.sensed_height <= 0.016:
                base_lvl = 2
            elif self.sensed_height <= 0.021:
                base_lvl = 3
            else:
                base_lvl = 4

            if self._active_probe_level < base_lvl:
                self._active_probe_level = base_lvl

            # Progressive probing: if foot contacts riser or stalled, escalate level
            if is_riser_hit:
                self.stalled_steps += 1
                if self.stalled_steps >= 6:
                    if self._active_probe_level < 4:
                        self._active_probe_level += 1
                        self.stalled_steps = 0
            else:
                self.stalled_steps = max(0, self.stalled_steps - 1)

            self.probe_level = self._active_probe_level
            clearance_cm = 1.0 + 0.4 * self.probe_level
            self.phase = "PROBING"
            self.status_msg = f"PROBING (L{self.probe_level}: {clearance_cm:.1f}cm | H:{self.sensed_height*100:.1f}cm)"

        else:
            if self.obstacle_detected:
                self.cleared_steps += 1
                if self.cleared_steps > 15:
                    self.obstacle_detected = False
                    self._active_probe_level = 1
                    self.probe_level = 1
                    self.phase = "WALK"
                    self.status_msg = "NORMAL WALK"
                else:
                    self.phase = "PUSHOFF"
                    self.status_msg = f"STEP CLEARED (Holding L{self.probe_level})"
            else:
                self.phase = "WALK"
                self.status_msg = "NORMAL WALK"

        return {
            "detected": self.obstacle_detected,
            "height": self.sensed_height,
            "level": self.probe_level,
            "phase": self.phase,
            "status": self.status_msg,
            "riser_hit": is_riser_hit,
        }

    def apply_body_lean(
        self, cmd_body: np.ndarray, is_walking: bool = True, on_stairs: bool = True
    ) -> np.ndarray:
        """Apply forward body lean and slight CoM elevation when ascending stairs or clearing obstacles."""
        active = (on_stairs or self.obstacle_detected)
        if not active or self.boost_level <= 0.0 or not is_walking:
            return cmd_body
        res = cmd_body.copy()
        eff_boost = self.boost_level * (1.0 if not self.obstacle_detected else (0.7 + 0.25 * self.probe_level))
        res[4] = float(np.clip(res[4] - math.radians(6.5 * eff_boost), -math.radians(25), math.radians(25)))
        res[2] = float(np.clip(res[2] + 0.010 * eff_boost, -0.05, 0.05))
        return res

    def apply_leg_boost(
        self,
        target_pos: np.ndarray,
        fn_left: float,
        fn_right: float,
        is_walking: bool = True,
        on_stairs: bool = True,
    ) -> np.ndarray:
        """Boost swing knee/ankle lift & stride, with stance forward counter-pitch lean."""
        active = (on_stairs or self.obstacle_detected)
        if not active or self.boost_level <= 0.0 or not is_walking:
            return target_pos

        res = target_pos.copy()
        lvl_mult = (0.7 + 0.3 * self.probe_level) if self.obstacle_detected else 1.0
        b = self.boost_level * lvl_mult

        # Stance Forward Counter-Pitch Lean (Root-cause fix for backward tipping):
        # Lean pelvis/trunk forward over the stance foot by slightly pitching hips & ankles forward
        lean_hip = math.radians(2.0 * b)
        lean_ankle = math.radians(1.5 * b)
        res[2] -= lean_hip
        res[11] += lean_hip
        res[4] -= lean_ankle
        res[13] += lean_ankle

        # Left leg (0..4): hip_yaw(0), hip_roll(1), hip_pitch(2), knee(3), ankle(4)
        # Right leg (9..13): hip_yaw(9), hip_roll(10), hip_pitch(11), knee(12), ankle(13)
        left_swing = fn_left < 1.6
        right_swing = fn_right < 1.6

        if left_swing and not right_swing:
            # Left leg swing
            res[2] -= 0.14 * b   # Hip pitch swing forward
            res[3] -= 0.18 * b   # Knee flex (lifts foot over riser)
            res[4] += 0.10 * b   # Ankle dorsiflex (levels toe/sole parallel to tread)
            # Right stance leg counter-pitch forward lean & stabilization
            res[11] += 0.06 * b  # Right hip pitch lean forward over stance foot
            res[13] -= 0.05 * b  # Right ankle push forward
            res[12] += 0.04 * b  # Right knee extension support
        elif right_swing and not left_swing:
            # Right leg swing
            res[11] += 0.14 * b  # Hip pitch swing forward
            res[12] += 0.18 * b  # Knee flex (lifts foot over riser)
            res[13] -= 0.10 * b  # Ankle dorsiflex (levels toe/sole parallel to tread)
            # Left stance leg counter-pitch forward lean & stabilization
            res[2] -= 0.06 * b   # Left hip pitch lean forward over stance foot
            res[4] += 0.05 * b   # Left ankle push forward
            res[3] -= 0.04 * b   # Left knee extension support

        return res


AdaptiveObstacleProbeController = StairStepBooster


class StandupRecoveryManager:
    """Manages automatic fall detection, neural standup recovery via standup.onnx, and smooth re-stabilization."""

    def __init__(
        self,
        standup_session: Any = None,
        duration_s: float = 3.5,
        stabilize_s: float = 0.8,
        enabled: bool = True,
    ):
        self.standup_session = standup_session
        self.duration_s = duration_s
        self.stabilize_s = stabilize_s
        self.enabled = enabled

        self.state: str = "IDLE"  # "IDLE", "STANDUP", "STABILIZING"
        self.time_left: float = 0.0
        self.fallen_step_count: int = 0
        self.trigger_count: int = 0
        self.pre_fall_vx: float = 0.0
        self.ramp_progress: float = 1.0

    def check_fall(self, pitch_deg: float, roll_deg: float, root_z: float) -> bool:
        """Detect sustained fall state."""
        is_fallen = abs(pitch_deg) > 42.0 or abs(roll_deg) > 40.0 or root_z < 0.075
        if is_fallen:
            self.fallen_step_count += 1
        else:
            self.fallen_step_count = max(0, self.fallen_step_count - 1)
        # Require 8 consecutive physics steps (~0.16s) to trigger auto-standup
        return self.fallen_step_count >= 8

    def trigger(self, current_vx: float = 0.0, force: bool = False) -> bool:
        """Trigger standup recovery sequence."""
        if self.standup_session is None and not force:
            return False
        if self.state != "IDLE" and not force:
            return False

        self.state = "STANDUP"
        self.time_left = self.duration_s
        self.pre_fall_vx = max(0.05, current_vx)
        self.fallen_step_count = 0
        self.ramp_progress = 0.0
        self.trigger_count += 1
        mode_str = "Neural Standup" if self.standup_session is not None else "Kinematic Standup"
        print(f"\n[STANDUP #{self.trigger_count}] ⚠️ Fall detected! {mode_str} Recovery engaged ({self.duration_s:.1f}s)...")
        return True

    def update(
        self,
        dt: float,
        pitch_deg: float,
        roll_deg: float,
        root_z: float,
    ) -> tuple[str, bool, float]:
        """Advance standup recovery sequence and return (active_policy_tag, is_active, target_vx_override)."""
        if self.state == "IDLE":
            if self.enabled and self.check_fall(pitch_deg, roll_deg, root_z):
                self.trigger(current_vx=0.08)
                return "STANDUP", True, 0.0
            return "NORMAL", False, -1.0

        self.time_left -= dt

        # Stage 1: Active Neural Standup via standup.onnx
        if self.state == "STANDUP":
            # Early success check if robot is already upright on feet
            if self.time_left < (self.duration_s - 1.2) and abs(pitch_deg) < 14.0 and abs(roll_deg) < 14.0 and root_z > 0.11:
                print(f"[STANDUP] Upright position reached early at t_rem={self.time_left:.2f}s! Entering stabilization.")
                self.state = "STABILIZING"
                self.time_left = self.stabilize_s
                return "STABILIZING", True, 0.0
            elif self.time_left <= 0.0:
                print("[STANDUP] Standup motion complete. Entering stabilization.")
                self.state = "STABILIZING"
                self.time_left = self.stabilize_s
                return "STABILIZING", True, 0.0
            return "STANDUP", True, 0.0

        # Stage 2: Post-Recovery Stabilization (prevents 'tự ngã như trúng độc')
        if self.state == "STABILIZING":
            is_stable = (abs(pitch_deg) < 15.0 and abs(roll_deg) < 15.0 and root_z > 0.095)
            if (self.time_left <= 0.0 and is_stable) or self.time_left < -1.5:
                self.state = "IDLE"
                self.time_left = 0.0
                self.ramp_progress = 0.0
                print(f"[STANDUP] 🏁 Stabilized & balanced on feet! Resuming walk (target vx={self.pre_fall_vx:+.2f} m/s).")
                return "NORMAL", False, self.pre_fall_vx
            return "STABILIZING", True, 0.0

        return "NORMAL", False, -1.0


class TerminalInput:
    """Non-blocking keyboard reader on stdin and /dev/tty (cbreak mode, background thread)."""

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}
    _VN_MAP = {
        "ư": "w", "ứ": "w", "ừ": "w", "ử": "w", "ữ": "w", "ự": "w",
        "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
        "đ": "d",
        "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    }

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._fds: list[int] = []
        self._tty_fd: int | None = None
        self._old_attrs: dict[int, Any] = {}
        self.enabled = False
        self._stop = threading.Event()

    def __enter__(self):
        # 1. Try sys.stdin if it is a TTY
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                fd = sys.stdin.fileno()
                self._fds.append(fd)
            except Exception:
                pass
        # 2. Also open /dev/tty directly if not already same fd (bypasses uv run stdin wrappers)
        try:
            tty_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            if not self._fds or os.path.sameopenfile(tty_fd, self._fds[0]) is False:
                self._tty_fd = tty_fd
                self._fds.append(tty_fd)
            else:
                os.close(tty_fd)
        except Exception:
            pass

        for fd in list(self._fds):
            try:
                self._old_attrs[fd] = termios.tcgetattr(fd)
                new_attrs = termios.tcgetattr(fd)
                new_attrs[3] = new_attrs[3] & ~(termios.ICANON | termios.ECHO)
                new_attrs[6][termios.VMIN] = 1
                new_attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, new_attrs)
                self.enabled = True
            except Exception:
                pass

        if self.enabled:
            threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for fd, attrs in self._old_attrs.items():
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            except Exception:
                pass
        if self._tty_fd is not None:
            try:
                os.close(self._tty_fd)
            except Exception:
                pass

    def _reader(self):
        while not self._stop.is_set():
            try:
                r, _, _ = select.select(self._fds, [], [], 0.02)
                if not r or self._stop.is_set():
                    continue
                for fd in r:
                    try:
                        data = os.read(fd, 128)
                    except (OSError, ValueError):
                        continue
                    if not data:
                        continue
                    text = data.decode("utf-8", errors="ignore")
                    i = 0
                    n = len(text)
                    while i < n:
                        if text[i] == "\x1b":
                            if i + 2 < n and text[i + 1] in ("[", "O") and text[i + 2] in self._ARROWS:
                                self._queue.put(self._ARROWS[text[i + 2]])
                                i += 3
                            else:
                                i += 1
                        elif text[i] in ("\x7f", "\x08"):
                            self._queue.put("backspace")
                            i += 1
                        else:
                            ch = text[i].lower()
                            ch = self._VN_MAP.get(ch, ch)
                            if ch in (" ", "\n", "\r"):
                                self._queue.put(" ")
                            else:
                                self._queue.put(ch)
                            i += 1
            except Exception:
                break

    def get_keys(self) -> list[str]:
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return keys

    def put_key(self, key: str) -> None:
        self._queue.put(key)


class InViewerPiPCamera:
    """Renders the robot's head/eye camera directly inside the MuJoCo viewer top-right corner."""

    def __init__(self, model: mujoco.MjModel, width: int = 240, height: int = 160, enabled: bool = True):
        self.model = model
        self.width = width
        self.height = height
        self.enabled = enabled
        self.renderer = None
        self.cam = None
        self.head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
        if self.head_bid < 0:
            self.head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_assembly")
        self.trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if self.enabled:
            try:
                self.renderer = mujoco.Renderer(model, height=height, width=width)
                self.cam = mujoco.MjvCamera()
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            except Exception as e:
                print(f"[PiP Camera] Offscreen renderer unavailable ({e}), disabling PiP inset.")
                self.enabled = False

    def update_and_draw(self, data: mujoco.MjData, viewer_ctx: Any) -> None:
        if not self.enabled or self.renderer is None or self.cam is None:
            if hasattr(viewer_ctx, "clear_images"):
                try:
                    viewer_ctx.clear_images()
                except Exception:
                    pass
            return
        if not hasattr(viewer_ctx, "set_images") or not hasattr(viewer_ctx, "viewport"):
            return

        try:
            if self.head_bid >= 0:
                head_pos = data.xpos[self.head_bid]
                head_rot = data.xmat[self.head_bid].reshape(3, 3)
                head_fwd = -head_rot[:, 2]
                head_up = head_rot[:, 0]
                cam_pos = head_pos + head_fwd * 0.08 + head_up * 0.015
                head_yaw_deg = math.degrees(math.atan2(head_fwd[1], head_fwd[0]))
                gaze_target = cam_pos + head_fwd * 0.60 - head_up * 0.18
            else:
                robot_pos = data.qpos[:3]
                cam_pos = robot_pos + np.array([0.08, 0.0, 0.16])
                head_yaw_deg = 0.0
                gaze_target = cam_pos + np.array([0.60, 0.0, -0.18])

            self.cam.lookat = [float(gaze_target[0]), float(gaze_target[1]), float(gaze_target[2])]
            self.cam.distance = 0.55
            self.cam.elevation = -20.0
            self.cam.azimuth = head_yaw_deg

            self.renderer.update_scene(data, self.cam)
            rgb = self.renderer.render().copy()

            # Draw sleek border & center crosshair directly in numpy (zero PIL overhead)
            h, w, _ = rgb.shape
            ch, cw = h // 2, w // 2
            # Crosshair
            rgb[ch, max(0, cw - 10):max(0, cw - 3)] = [0, 240, 255]
            rgb[ch, min(w, cw + 4):min(w, cw + 11)] = [0, 240, 255]
            rgb[max(0, ch - 10):max(0, ch - 3), cw] = [0, 240, 255]
            rgb[min(h, ch + 4):min(h, ch + 11), cw] = [0, 240, 255]
            # Border (2px cyan-slate)
            rgb[:2, :] = [14, 165, 233]
            rgb[-2:, :] = [14, 165, 233]
            rgb[:, :2] = [14, 165, 233]
            rgb[:, -2:] = [14, 165, 233]

            vp = viewer_ctx.viewport
            left = max(0, int(vp.width) - w - 14)
            bottom = max(0, int(vp.height) - h - 14)
            rect = mujoco.MjrRect(left, bottom, w, h)
            viewer_ctx.set_images([(rect, rgb)])
        except Exception:
            pass


def parse_args():
    default_walking = (
        "rough_latest.onnx"
        if os.path.exists("rough_latest.onnx")
        else (
            "stairs_climb.onnx"
            if os.path.exists("stairs_climb.onnx")
            else ("walking.onnx" if os.path.exists("walking.onnx") else None)
        )
    )
    default_standing = "standing.onnx" if os.path.exists("standing.onnx") else None
    default_standup = "standup.onnx" if os.path.exists("standup.onnx") else None
    parser = argparse.ArgumentParser(
        description="Interactive & Headless Rehearsal Simulation for Microduck High Stairs"
    )
    parser.add_argument(
        "--policy", "--walking",
        type=str,
        default=default_walking,
        dest="policy",
        help=f"Path to trained ONNX walking/stairs policy file (default: {default_walking})",
    )
    parser.add_argument(
        "--standing", "-s",
        type=str,
        default=default_standing,
        dest="standing",
        help=f"Path to standing idle balance policy ONNX file (default: {default_standing})",
    )
    parser.add_argument(
        "--standup", "-u",
        type=str,
        default=default_standup,
        dest="standup",
        help=f"Path to neural standup recovery policy ONNX file (default: {default_standup})",
    )
    parser.add_argument(
        "--stair-boost",
        type=float,
        default=1.0,
        dest="stair_boost",
        help="Adaptive stair clearance & stride boost factor [0.0 - 2.0] (default: 1.0)",
    )
    parser.add_argument(
        "--auto-standup",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="auto_standup",
        help="Enable/disable automatic standup recovery when robot falls (default: True)",
    )
    parser.add_argument(
        "--auto-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="auto_probe",
        help="Enable/disable automatic forward obstacle height sensing and clearance probing (default: True)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=DEFAULT_SCENE_XML,
        help="Path to scene XML (default: scene_stairs.xml)",
    )
    parser.add_argument(
        "--spawn", "--preset",
        type=str,
        default="bottom",
        dest="spawn",
        help="Initial spawn preset on stairs: bottom, flight1, landing1, flight2, landing2, flight3, top (default: bottom)",
    )
    parser.add_argument(
        "--lin-vel-x", "--vx",
        type=float,
        default=0.0,
        dest="lin_vel_x",
        help="Initial commanded forward velocity (default: 0.0 m/s for stable standing idle)",
    )
    parser.add_argument(
        "--vin",
        type=float,
        default=BAM_VIN,
        help=f"Battery voltage for BAM M6 actuators (default: {BAM_VIN}V)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical viewer (for testing/CI)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output and diagnostics",
    )
    parser.add_argument(
        "--telemetry-hz",
        type=float,
        default=5.0,
        help="Telemetry logging frequency in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--duration-s", "--record-duration",
        type=float,
        default=None,
        dest="duration_s",
        help="Run duration in seconds before terminating (default: None)",
    )
    parser.add_argument(
        "--no-focus",
        action="store_true",
        help="Disable autonomous summit focus navigation and corridor edge guard (manual steering only)",
    )
    parser.add_argument(
        "--hud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the embedded PiP Robot Eye Camera in the top-right corner of the viewer (default: True)",
    )
    return parser.parse_args()


def run_simulation(args) -> int:
    """Main simulation loop handling BAM dynamics, policy inference, and telemetry."""
    scene_path = args.scene
    if not os.path.isabs(scene_path):
        scene_path = os.path.abspath(scene_path)

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    # Initialize MuJoCo and BAM controller
    bam_model = load_bam_model(BAM_KP_FW, args.vin, BAM_MAX_CURRENT)
    model, data, bam_ctrl, act_names = load_mujoco_with_bam(scene_path, bam_model=bam_model, timestep=PHYSICS_DT)

    preset_name = args.spawn if args.spawn in SPAWN_PRESETS else "bottom"
    reset_to_preset(model, data, preset_name, bam_ctrl=bam_ctrl)

    # Initialize ONNX policy sessions (walking + standing balance + neural standup)
    walking_session = None
    standing_session = None
    standup_session = None
    if HAS_ORT:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        if args.policy is not None:
            if not os.path.exists(args.policy):
                raise FileNotFoundError(f"Policy file not found: {args.policy}")
            print(f"[infer_stairs] Loading walking policy: {args.policy}")
            walking_session = ort.InferenceSession(os.path.abspath(args.policy), sess_options=opts, providers=["CPUExecutionProvider"])
        if getattr(args, "standing", None) is not None:
            if not os.path.exists(args.standing):
                raise FileNotFoundError(f"Standing policy file not found: {args.standing}")
            print(f"[infer_stairs] Loading standing idle policy: {args.standing}")
            standing_session = ort.InferenceSession(os.path.abspath(args.standing), sess_options=opts, providers=["CPUExecutionProvider"])
        if getattr(args, "standup", None) is not None:
            if not os.path.exists(args.standup):
                print(f"[infer_stairs] Standup policy file not found: {args.standup} (will use kinematic recovery)")
            else:
                print(f"[infer_stairs] Loading neural standup recovery policy: {args.standup}")
                standup_session = ort.InferenceSession(os.path.abspath(args.standup), sess_options=opts, providers=["CPUExecutionProvider"])

    # Settle feet directly without uncontrolled physics stepping
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
        bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
    else:
        data.ctrl[:14] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)

    # Control state variables
    cmd_twist = np.array([args.lin_vel_x, 0.0, 0.0], dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)
    target_pos = DEFAULT_JOINT_POS.copy()

    # Control flags
    pitch_trim_mode = False
    head_gaze_mode = False
    heading_lock_mode = False
    target_yaw: float | None = None
    policy_active = True
    current_preset = preset_name
    quit_requested = False

    # Standup Recovery & Stair Clearance Boost Managers
    standup_mgr = StandupRecoveryManager(
        standup_session=standup_session,
        duration_s=3.5,
        stabilize_s=0.8,
        enabled=getattr(args, "auto_standup", True),
    )
    step_booster = StairStepBooster(
        boost_level=getattr(args, "stair_boost", 1.0),
        auto_probe=getattr(args, "auto_probe", True),
    )

    # Autonomous Summit Focus Navigation & Corridor Guard
    focus_navigator = SummitFocusNavigator(target_pos=SUMMIT_TARGET_POS, enabled=not getattr(args, "no_focus", False))
    focus_overlay = SummitFocusOverlay(target_pos=SUMMIT_TARGET_POS)
    nav_info: dict[str, Any] = {
        "dist": float(np.linalg.norm(SUMMIT_TARGET_POS[:2] - data.qpos[:2])),
        "lat_dev": float(data.qpos[1]),
        "heading_err_deg": 0.0,
        "status": "FOCUS ON SUMMIT" if focus_navigator.enabled else "FOCUS MANUAL",
        "arrived": False,
        "edge_guard": False,
        "gaze_pitch_deg": 0.0,
        "gaze_yaw_deg": 0.0,
    }

    sim_time = 0.0
    step_count = 0
    last_telemetry_time = 0.0
    telemetry_period = 1.0 / max(0.5, args.telemetry_hz)

    # In debug mode without explicit duration, default to 100 steps (2.0s) for autonomous test execution
    duration_s = args.duration_s
    if duration_s is None and args.debug and args.headless:
        duration_s = 2.0

    print("=================================================================")
    print(" Microduck High Stairs Rehearsal Simulation (BAM M6, 50 Hz)       ")
    print("=================================================================")
    print(f" Scene: {scene_path}")
    print(f" Spawn preset: {preset_name}")
    print(f" Voltage: {args.vin}V")
    print(f" Policy: {args.policy or 'Rest PD Controller'}")
    print(f" Standup: {'ENABLED (Neural standup.onnx)' if standup_session is not None else ('ENABLED (Kinematic)' if standup_mgr.enabled else 'DISABLED')}")
    print(f" Stair Clearance Boost: {step_booster.boost_level:.1f}x (Adjust: [ / ])")
    print(f" Obstacle Height Probe: {'ENABLED (Auto-Detect & Progressive Probing)' if step_booster.auto_probe else 'DISABLED'}")
    print(f" Headless: {args.headless} | Debug: {args.debug}")
    print(f" Summit Focus Nav: {'ENABLED (Target: [2.34, 0.00, 0.11]m)' if focus_navigator.enabled else 'DISABLED'}")
    if duration_s is not None:
        print(f" Duration: {duration_s:.1f}s (target: {int(duration_s / CONTROL_DT)} steps)")
    print(" Controls (work in BOTH Terminal and 3D Viewer Window):")
    print("   UP / W / I     : Walk Forward / Speed up (+0.05 m/s)")
    print("   DOWN / S / K   : Slow down / Walk Backward (-0.05 m/s)")
    print("   LEFT / A / J   : Turn Left (+0.15 rad/s)")
    print("   RIGHT / D / L  : Turn Right (-0.15 rad/s)")
    print("   , / .          : Strafe Left / Right (+/- 0.03 m/s)")
    print("   [ / ]          : Stair Clearance Boost - / + 0.2x")
    print("   O              : Toggle Obstacle Height Probing Mode")
    print("   SPACE          : Stop (zero all velocities)")
    print("   BACKSPACE / 0  : Reset robot to bottom of stairs")
    print("   F              : Toggle Summit Focus / Auto-Navigation (Default: ON)")
    print("   V              : Toggle In-Viewer PiP Robot Eye Camera HUD (Default: ON)")
    print("   B              : Toggle Body Pitch Trim Mode (UP/DOWN trims pitch +/- 2°)")
    print("   H              : Toggle Head Gaze Mode (tilt beak pitch down)")
    print("   C              : Toggle Heading Lock")
    print("   P              : Apply Lateral Perturbation Push")
    print("   U              : Standup Recovery Trigger")
    print("   T              : Toggle Policy Inference (pause/resume)")
    print("   Q              : Quit Cleanly")
    print("   1..5           : Jump to Spawn Presets (1=bottom, 2=flight1, ... 5=top)")
    print("=================================================================")

    use_viewer = (not args.headless) and HAS_MUJOCO_VIEWER
    pip_camera = InViewerPiPCamera(model, width=240, height=160, enabled=(use_viewer and getattr(args, "hud", True)))
    last_key_info = "NONE"

    def handle_command_key(k: str) -> None:
        nonlocal cmd_twist, cmd_head, cmd_body, pitch_trim_mode, head_gaze_mode
        nonlocal heading_lock_mode, target_yaw, policy_active, current_preset, quit_requested
        nonlocal focus_navigator, step_booster, standup_mgr, last_key_info

        last_key_info = k.upper()
        if k in ("up", "w", "i"):
            if pitch_trim_mode:
                cmd_body[4] = float(np.clip(cmd_body[4] + math.radians(2), -math.radians(30), math.radians(30)))
                print(f"🎮 [TRIM] Pitch trim -> {math.degrees(cmd_body[4]):+.1f}°")
            else:
                if cmd_twist[0] < 0.05:
                    cmd_twist[0] = 0.12
                else:
                    cmd_twist[0] = float(np.clip(cmd_twist[0] + 0.05, -0.25, 0.35))
                focus_state = "ON" if focus_navigator.enabled else "OFF"
                print(f"🎮 [CMD: TIẾN] vx={cmd_twist[0]:+.2f} m/s, vy={cmd_twist[1]:+.2f}, wz={cmd_twist[2]:+.2f} (Focus: {focus_state})")
        elif k in ("down", "s", "k"):
            if pitch_trim_mode:
                cmd_body[4] = float(np.clip(cmd_body[4] - math.radians(2), -math.radians(30), math.radians(30)))
                print(f"🎮 [TRIM] Pitch trim -> {math.degrees(cmd_body[4]):+.1f}°")
            else:
                cmd_twist[0] = float(np.clip(cmd_twist[0] - 0.06, -0.25, 0.35))
                if cmd_twist[0] <= 0.0 and focus_navigator.enabled:
                    focus_navigator.enabled = False
                    print("🎮 [MANUAL] Chuyển sang Lái Thủ Công để đi lùi (Bấm F để bật lại Auto-Focus)")
                print(f"🎮 [CMD: LÙI/GIẢM TỐC] vx={cmd_twist[0]:+.2f} m/s, vy={cmd_twist[1]:+.2f}, wz={cmd_twist[2]:+.2f}")
        elif k in ("left", "a", "j"):
            if focus_navigator.enabled:
                focus_navigator.enabled = False
                print("🎮 [MANUAL] Bàn phím can thiệp -> Tắt tự lái Summit Focus, trao toàn quyền rẽ trái/phải (Bấm F để bật lại)")
            if abs(cmd_twist[0]) < 0.04:
                cmd_twist[0] = 0.08
            cmd_twist[2] = float(np.clip(cmd_twist[2] + 0.15, -0.80, 0.80))
            print(f"🎮 [CMD: RẼ TRÁI] vx={cmd_twist[0]:+.2f} m/s, wz={cmd_twist[2]:+.2f} rad/s")
        elif k in ("right", "d", "e", "l"):
            if focus_navigator.enabled:
                focus_navigator.enabled = False
                print("🎮 [MANUAL] Bàn phím can thiệp -> Tắt tự lái Summit Focus, trao toàn quyền rẽ trái/phải (Bấm F để bật lại)")
            if abs(cmd_twist[0]) < 0.04:
                cmd_twist[0] = 0.08
            cmd_twist[2] = float(np.clip(cmd_twist[2] - 0.15, -0.80, 0.80))
            print(f"🎮 [CMD: RẼ PHẢI] vx={cmd_twist[0]:+.2f} m/s, wz={cmd_twist[2]:+.2f} rad/s")
        elif k == ",":
            if focus_navigator.enabled:
                focus_navigator.enabled = False
            cmd_twist[1] = float(np.clip(cmd_twist[1] + 0.03, -0.20, 0.20))
            print(f"🎮 [CMD: BƯỚC NGANG TRÁI] vy={cmd_twist[1]:+.2f} m/s")
        elif k == ".":
            if focus_navigator.enabled:
                focus_navigator.enabled = False
            cmd_twist[1] = float(np.clip(cmd_twist[1] - 0.03, -0.20, 0.20))
            print(f"🎮 [CMD: BƯỚC NGANG PHẢI] vy={cmd_twist[1]:+.2f} m/s")
        elif k in ("[", "-"):
            step_booster.boost_level = float(np.clip(step_booster.boost_level - 0.2, 0.0, 2.0))
            print(f"🎮 [BOOST] Stair clearance boost factor -> {step_booster.boost_level:.1f}x")
        elif k in ("]", "=", "+"):
            step_booster.boost_level = float(np.clip(step_booster.boost_level + 0.2, 0.0, 2.0))
            print(f"🎮 [BOOST] Stair clearance boost factor -> {step_booster.boost_level:.1f}x")
        elif k == "o":
            step_booster.auto_probe = not step_booster.auto_probe
            state_str = "ON (Adaptive Obstacle Height Probing)" if step_booster.auto_probe else "OFF (Manual)"
            print(f"🎮 [MODE] Obstacle Height Probing: {state_str}")
        elif k == "v":
            pip_camera.enabled = not pip_camera.enabled
            print(f"📷 [PiP CAMERA] Khung camera mắt robot góc trên phải: {'BẬT (ON)' if pip_camera.enabled else 'TẮT (OFF)'}")
        elif k == " ":
            cmd_twist[:] = 0.0
            focus_navigator.summit_reached = False
            print("🛑 [CMD: DỪNG] Zero velocity (Robot đứng giữ thăng bằng)")
        elif k in ("backspace", "0"):
            cmd_twist[:] = 0.0
            focus_navigator.summit_reached = False
            reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)
            print("🔄 [RESET] Đã đưa robot về lại vị trí xuất phát chân cầu thang!")
        elif k == "f":
            focus_navigator.enabled = not focus_navigator.enabled
            if focus_navigator.enabled and cmd_twist[0] < 0.08:
                cmd_twist[0] = 0.20
            state_str = "ON (Auto-Steer & Corridor Edge Guard)" if focus_navigator.enabled else "OFF (Manual Steering)"
            print(f"🎯 [MODE] Summit Focus Navigation: {state_str}")
        elif k == "b":
            pitch_trim_mode = not pitch_trim_mode
            print(f"🎮 [MODE] Body Pitch Trim: {'ON (UP/DOWN trims angle)' if pitch_trim_mode else 'OFF'}")
        elif k == "h":
            head_gaze_mode = not head_gaze_mode
            cmd_head[1] = -math.radians(20) if head_gaze_mode else 0.0
            print(f"🎮 [MODE] Head Gaze: {'LOOKING DOWN (-20°)' if head_gaze_mode else 'FORWARD (0°)'}")
        elif k == "c":
            heading_lock_mode = not heading_lock_mode
            target_yaw = compute_euler_angles(data.qpos[3:7])[2] if heading_lock_mode else None
            state_str = f"ON (locked at {math.degrees(target_yaw):.1f}°)" if heading_lock_mode else "OFF"
            print(f"🎮 [MODE] Heading Lock: {state_str}")
        elif k == "p":
            push = (np.random.rand(3) - 0.5) * 0.5
            data.qvel[:3] += push
            print(f"💥 [PUSH] Perturbation push applied: [{push[0]:+.2f}, {push[1]:+.2f}, {push[2]:+.2f}] m/s")
        elif k == "u":
            if standup_session is not None:
                standup_mgr.trigger(current_vx=cmd_twist[0], force=True)
            else:
                print("🎮 [RECOVERY] Standup recovery triggered (kinematic fallback)!")
                cmd_body[2] = 0.02  # elevate CoM
                cmd_body[4] = -math.radians(10)  # lean forward
                if abs(data.qpos[2]) < 0.08:
                    data.qpos[2] = 0.125
                    data.qvel[:] = 0.0
        elif k == "t":
            policy_active = not policy_active
            print(f"🎮 [POLICY] Inference {'RESUMED' if policy_active else 'PAUSED (Holding target)'}")
        elif k == "q":
            print("👋 [EXIT] Quit requested by user.")
            quit_requested = True
        elif k in HOTKEY_PRESETS:
            current_preset = HOTKEY_PRESETS[k]
            print(f"📍 [SPAWN] Preset {k} -> {current_preset}")
            reset_to_preset(model, data, current_preset, bam_ctrl=bam_ctrl)

    def viewer_key_callback(keycode: int):
        # Map GLFW keys from 3D viewer window directly into command queue
        if keycode == 265:
            term_in.put_key("up")
        elif keycode == 264:
            term_in.put_key("down")
        elif keycode == 263:
            term_in.put_key("left")
        elif keycode == 262:
            term_in.put_key("right")
        elif keycode in (259, 261):  # Backspace / Delete
            term_in.put_key("backspace")
        elif keycode in (257, 335):  # Enter / KP_Enter
            term_in.put_key("up")
        elif 32 <= keycode <= 126:
            char = chr(keycode).lower()
            term_in.put_key(char)

    # When launched interactively in terminal without explicit --vx/--lin-vel-x, auto-start smooth climb (0.24 m/s)
    if use_viewer and abs(cmd_twist[0]) < 1e-5 and ("--lin-vel-x" not in sys.argv and "--vx" not in sys.argv):
        cmd_twist[0] = 0.24
        print("[AUTO-CRUISE] Interactive terminal launch detected: starting smooth autonomous climb at vx=+0.24 m/s (SPACE to pause, Arrow/WASD/IJKL to steer).")

    viewer_ctx = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=viewer_key_callback
    ) if use_viewer else _HeadlessViewer()

    summit_hold_timer = 0.0
    frame_drop_count = 0

    with TerminalInput() as term_in, viewer_ctx as viewer_context:
        # Simulation loop with monotonic drift-free frame pacer
        next_step_clock = time.perf_counter()
        while True:
            step_start_clock = time.perf_counter()
            if quit_requested:
                break
            if duration_s is not None and sim_time >= duration_s:
                print(f"[infer_stairs] Target duration {duration_s:.1f}s reached ({step_count} steps). Exiting cleanly.")
                break
            if not viewer_context.is_running():
                print("[infer_stairs] Viewer closed. Exiting.")
                break

            # Process pending keyboard commands
            for key in term_in.get_keys():
                handle_command_key(key)

            # Fall detection & Neural Standup State Machine
            r_cur, p_cur, y_cur = compute_euler_angles(data.qpos[3:7])
            pitch_deg = math.degrees(p_cur)
            roll_deg = math.degrees(r_cur)
            root_z = float(data.qpos[2])

            rec_mode, is_recovering, vx_resume = standup_mgr.update(
                CONTROL_DT, pitch_deg, roll_deg, root_z
            )
            if vx_resume >= 0.0:
                cmd_twist[0] = vx_resume

            # Smooth speed ramp after standup stabilization (prevents sudden tip-over)
            if not is_recovering and standup_mgr.ramp_progress < 1.0:
                standup_mgr.ramp_progress = min(1.0, standup_mgr.ramp_progress + CONTROL_DT / 0.5)
                effective_twist_vx = float(cmd_twist[0] * standup_mgr.ramp_progress)
            else:
                effective_twist_vx = float(cmd_twist[0])

            # Heading Lock runtime regulation
            if heading_lock_mode and target_yaw is not None and abs(cmd_twist[0]) > 0.01:
                cur_yaw = y_cur
                yaw_err = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
                cmd_twist[2] = float(np.clip(0.8 * yaw_err, -0.4, 0.4))

            # Summit Autonomous Focus & Anti-Drift Corridor Navigation
            working_twist = cmd_twist.copy()
            working_twist[0] = effective_twist_vx
            active_twist, active_head, nav_info = focus_navigator.update(
                data.qpos[:3], data.qpos[3:7], data.qvel, working_twist, dt=CONTROL_DT
            )
            if focus_navigator.enabled:
                if not head_gaze_mode:
                    cmd_head = active_head
                effective_twist = active_twist
            else:
                effective_twist = working_twist

            if nav_info.get("arrived", False):
                if abs(cmd_twist[0]) > 0.001:
                    print(f"[{sim_time:5.2f}s] [SUMMIT] Microduck reached summit platform! Holding standing balance.")
                    cmd_twist[:] = 0.0
                    summit_hold_timer = 0.0
                elif use_viewer:
                    summit_hold_timer += CONTROL_DT
                    if summit_hold_timer >= 2.5:
                        print(f"[{sim_time:5.2f}s] [AUTO-LOOP] Summit hold complete -> looping smoothly back to bottom!")
                        focus_navigator.summit_reached = False
                        summit_hold_timer = 0.0
                        reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)
                        cmd_twist[:] = [0.24, 0.0, 0.0]
            else:
                summit_hold_timer = 0.0

            # State-dependent policy selection & command shaping
            on_stairs = (data.qpos[0] >= 0.25)
            is_walking = (abs(effective_twist[0]) > 0.02 and not is_recovering)

            if is_recovering:
                effective_twist[:] = 0.0
                cmd_head[:] = 0.0
                cmd_body[:] = 0.0
                if rec_mode == "STANDUP":
                    active_session = standup_session or standing_session or walking_session
                    policy_tag = "STANDUP"
                else:  # "STABILIZING"
                    active_session = standing_session or walking_session
                    policy_tag = "STABILIZE"
                    # Crucial fix for 'tự ngã như trúng độc': zero out violent standup thrash actions
                    last_action[:] = 0.0
            else:
                # Apply forward trunk lean and CoM elevation on stairs
                cmd_body = step_booster.apply_body_lean(cmd_body, is_walking=is_walking, on_stairs=on_stairs)

                cmd_mag = float(np.linalg.norm(effective_twist))
                active_session = (
                    standing_session
                    if (cmd_mag <= 0.05 and standing_session is not None)
                    else (walking_session or standing_session)
                )
                policy_tag = "STAND" if (active_session is standing_session and standing_session is not None) else "CLIMB"

            # 50 Hz Control Step
            obs = build_61d_observation(
                model, data, last_action, effective_twist, cmd_head=cmd_head, cmd_body=cmd_body
            )

            if active_session is not None and policy_active:
                obs_input = obs.reshape(1, -1).astype(np.float32)
                ort_inputs = {active_session.get_inputs()[0].name: obs_input}
                action = active_session.run(None, ort_inputs)[0].flatten()
                last_action = np.clip(action, -3.0, 3.0)
                target_pos = DEFAULT_JOINT_POS + last_action * ACTION_SCALE
            else:
                target_pos = DEFAULT_JOINT_POS.copy()

            # Boost swing knee lift & hip stride extension over step risers with progressive height probing
            _, _, fn_l, fn_r = identify_step_contacts_and_forces(model, data)
            obs_info = step_booster.update_obstacle_state(
                model, data, fn_l, fn_r, is_walking=is_walking
            )
            target_pos = step_booster.apply_leg_boost(
                target_pos,
                fn_left=fn_l,
                fn_right=fn_r,
                is_walking=is_walking,
                on_stairs=on_stairs,
            )

            # Decimation physics substeps (4 @ 200 Hz = 50 Hz control rate)
            for _ in range(DECIMATION):
                if bam_ctrl is not None:
                    bam_ctrl.q_target[:] = target_pos
                    bam_ctrl.update()
                else:
                    data.ctrl[:14] = target_pos
                mujoco.mj_step(model, data)

            sim_time += CONTROL_DT
            step_count += 1
            step_compute_ms = (time.perf_counter() - step_start_clock) * 1000.0

            # Check NaN guard
            if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
                raise FloatingPointError(f"Simulation NaN detected at step {step_count}, t={sim_time:.3f}s")

            # Telemetry computation & display
            if sim_time - last_telemetry_time >= telemetry_period or (args.debug and step_count % 10 == 0):
                last_telemetry_time = sim_time
                telem = compute_balance_telemetry(model, data, bam_ctrl=bam_ctrl)

                # Terminal log
                p = telem["pos"]
                fc_l, fc_r = telem["feet_contacts"]
                fn_l, fn_r = telem["normal_forces"]
                tl = telem["torque_loads"]
                probe_tag = obs_info["status"] if step_booster.auto_probe else f"Boost:{step_booster.boost_level:.1f}x"
                print(
                    f"[{sim_time:5.2f}s] [{policy_tag:9s}] pos:[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] | "
                    f"tilt: P:{telem['pitch_deg']:+5.1f}° R:{telem['roll_deg']:+5.1f}° [{telem['status']:8s}] | "
                    f"[{probe_tag}] | "
                    f"Nav: [{nav_info['status']}] D:{nav_info['dist']:.2f}m Lat:{nav_info['lat_dev']:+.2f}m | "
                    f"{telem['zone']} | 50Hz ({step_compute_ms:.2f}ms, Drops:{frame_drop_count})"
                )

                # In-Viewer HUD Overlay (throttled to telemetry_hz to eliminate GLFW string-render stalls)
                if viewer_context is not None and hasattr(viewer_context, "set_texts"):
                    hud_l = (
                        f"Step: {step_count} | Sim: {sim_time:.2f}s | Policy: [{policy_tag}] | Key: [{last_key_info}]\n"
                        f"Pos: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] | Zone: {telem['zone']}\n"
                        f"Tilt: P:{telem['pitch_deg']:+.1f}° R:{telem['roll_deg']:+.1f}° | [{telem['status']}]\n"
                        f"Probe: [{obs_info['status']}] (Toggle: O, Level: [ / ], PiP: V)"
                    )
                    hud_r = (
                        f"Nav: [{nav_info['status']}] | Dist: {nav_info['dist']:.2f}m\n"
                        f"Lat Dev: {nav_info['lat_dev']:+.3f}m | Head Err: {nav_info['heading_err_deg']:+.1f}°\n"
                        f"Cmd: [vx:{effective_twist[0]:+.2f}, vy:{effective_twist[1]:+.2f}, wz:{effective_twist[2]:+.2f}]"
                    )
                    try:
                        viewer_context.set_texts([
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud_l, hud_r)
                        ])
                    except Exception:
                        pass

            # Decoupled Viewer Sync (syncs smoothly on every 2nd control step in interactive mode to prevent D3D12/GLFW stalls)
            if viewer_context is not None:
                if (not use_viewer) or (step_count % 2 == 0):
                    if getattr(viewer_context, "user_scn", None) is not None:
                        viewer_context.user_scn.ngeom = 0
                        focus_overlay.draw(data, viewer_context.user_scn, nav_info)
                    if use_viewer and (step_count % 4 == 0):
                        pip_camera.update_and_draw(data, viewer_context)
                    viewer_context.sync()
                next_step_clock += CONTROL_DT
                now_clock = time.perf_counter()
                sleep_time = next_step_clock - now_clock
                if sleep_time > 0.0005:
                    time.sleep(sleep_time)
                elif sleep_time < -CONTROL_DT:
                    # Resync clock without spiraling if OS window manager paused
                    next_step_clock = now_clock

    print(f"[infer_stairs] Finished {step_count} steps with 0 NaNs and clean exit.")
    return 0


def main():
    args = parse_args()
    return run_simulation(args)


if __name__ == "__main__":
    sys.exit(main())
