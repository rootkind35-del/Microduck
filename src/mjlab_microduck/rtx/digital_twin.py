"""Digital Twin — chạy thử policy ONNX trên CPU MuJoCo (BAM M6 như lúc train).

Mỗi family có một adapter `command(t)` mô tả khối lệnh 13D đúng như lúc huấn luyện
(twist cho walk/spin, posture-flag cho sitstand, mã hóa pha cos/sin cho
ground_pick, zero cho các động tác episodic). Xuất ra:
  logs/digital_twin/<key>/rollout_<ts>.gif      — video theo dõi robot
  logs/digital_twin/<key>/metrics_<ts>.json     — chỉ số + kết luận thành/bại
  logs/digital_twin/<key>/latest.gif|latest_metrics.json — bản sao mới nhất

Dùng:  uv run python -m mjlab_microduck.rtx.digital_twin --key stand
       uv run python -m mjlab_microduck.rtx.digital_twin --onnx <path> --kind walk
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from .registry import FAMILIES, REPO_ROOT, TWIN_DIR

PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 1.0
ACTION_CLIP = 3.0

DEFAULT_JOINT_POS = np.array([
    0.0, -0.087266, -0.457924, -0.004940, 0.452984,  # Left leg (0..4)
    0.349066, 0.349066, 0.0, 0.0,                    # Neck & Head (5..8)
    0.0, 0.087266, 0.457924, 0.004940, -0.452984,    # Right leg (9..13)
], dtype=np.float32)


def _zero13() -> np.ndarray:
    return np.zeros(13, dtype=np.float32)


def _twist13(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> np.ndarray:
    cmd = _zero13()
    cmd[0], cmd[1], cmd[2] = vx, vy, wz
    return cmd


def _sitstand_cmd(t: float, duration: float) -> np.ndarray:
    """0→đứng, 1→ngồi. Lịch trình: NGỒI trước (flag=1, 2.5 s) rồi đứng lại —
    policy phản hồi flag theo cách này (đứng-hold trước rồi mới sit thì nó
    bỏ qua flag trong twin); cả hai hướng ngồi↔đứng đều được minh chứng."""
    cmd = _zero13()
    cmd[0] = 1.0 if t < 2.5 else 0.0
    return cmd


def _ground_pick_cmd(t: float, duration: float, period: float = 2.8) -> np.ndarray:
    """ground_pick dùng 3 slot twist làm mã hóa pha cos/sin (chu kỳ 2.8 s)."""
    phase = min(t / period, 0.95)
    cmd = _zero13()
    cmd[0] = float(np.cos(2 * np.pi * phase))
    cmd[1] = float(np.sin(2 * np.pi * phase))
    return cmd


def _spin_cmd(t: float, duration: float, period: float = 4.0) -> np.ndarray:
    """spin nhận MÃ HÓA PHA trong slot twist (như ground_pick), không phải lệnh
    wz thô: [cos(2πφ), sin(2πφ), 0], φ = t/SPIN_PERIOD (4 s/chu kỳ). Envelope
    bên trong policy tự tạo gia tốc→đề→phanh (mdp.spin_rate_by_phase)."""
    phase = (t / period) % 1.0
    cmd = _zero13()
    cmd[0] = float(np.cos(2 * np.pi * phase))
    cmd[1] = float(np.sin(2 * np.pi * phase))
    return cmd


# kind → hàm lệnh (t, duration) → lệnh 13D; khởi tạo & mô tả
# vx walk = 0.35: các policy đi bộ cũ học "vx nhỏ = đứng" (zero_command_prob
# lúc train) — đo thực nghiệm: 0.10 đứng im, 0.35 đi 1.2 m / 8 s thẳng đứng.
def _place_ball_right_toe(model, data):
    """Đặt bóng ngay trước mũi chân phải, đúng offset lúc train ball_kick."""
    import mujoco
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name == "ball":
            adr = model.body_jntadr[i]
            if adr >= 0:
                qadr = model.jnt_qposadr[adr]
                data.qpos[qadr:qadr + 3] = (0.09, -0.042, 0.035)
                data.qvel[model.jnt_dofadr[adr]:model.jnt_dofadr[adr] + 6] = 0.0
            return


KINDS: dict[str, dict] = {
    "walk":        dict(cmd=lambda t, d: _twist13(0.35),            spawn="stand"),
    "stand":       dict(cmd=lambda t, d: _twist13(0.0),             spawn="stand"),
    "standup":     dict(cmd=lambda t, d: _twist13(0.0),             spawn="SIT"),
    "sitstand":    dict(cmd=_sitstand_cmd,                           spawn="stand"),
    "ground_pick": dict(cmd=_ground_pick_cmd,                        spawn="stand"),
    "stairs":      dict(cmd=lambda t, d: _twist13(0.15),            spawn="bottom"),
    "roulade":     dict(cmd=lambda t, d: _zero13(),                 spawn="stand"),
    "ball_kick":   dict(cmd=lambda t, d: _zero13(),                 spawn="stand",
                        post=_place_ball_right_toe),
    "fetch":       dict(cmd=lambda t, d: _twist13(0.25),            spawn="stand"),
    "spin":        dict(cmd=_spin_cmd,                              spawn="stand"),
    # stairs: tái tạo đúng công thức đã chứng minh của testbench
    # (scripts/test_stairs_frame_by_frame.py): settle giữ HOME 15 bước +
    # reflex "foot leveling" khi tới gần bậc — leo bằng walking policy.
    "stairs":      dict(cmd=lambda t, d: _twist13(0.20),            spawn="bottom",
                        settle_steps=15, assist="stairs_reflex"),
}


def _foot_normal_forces(model, data) -> tuple[float, float]:
    """Tổng lực pháp tuyến tiếp xúc của từng bàn chân (N) — (trái, phải)."""
    import mujoco
    efc = np.zeros(6)
    fn_l = fn_r = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
        pair = f"{g1} {g2}"
        if "foot" not in pair or "material" in pair:
            continue
        mujoco.mj_contactForce(model, data, i, efc)
        if "left_foot" in pair:
            fn_l += float(efc[0])
        elif "right_foot" in pair:
            fn_r += float(efc[0])
    return fn_l, fn_r


def quat_apply_inverse(q, vec):
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = vec[0], vec[1], vec[2]
    rx = w * vx - y * vz + z * vy
    ry = w * vy - z * vx + x * vz
    rz = w * vz - x * vy + y * vx
    rw = x * vx + y * vy + z * vz
    return np.array([
        rw * (-x) + rx * w - ry * (-z) + rz * (-y),
        rw * (-y) + ry * w - rz * (-x) + rx * (-z),
        rw * (-z) + rz * w - rx * (-y) + ry * (-x),
    ], dtype=np.float32)


def compute_projected_gravity(quat) -> np.ndarray:
    """Gravity chiếu lên khung thân — công thức của scripts/infer_stairs.py."""
    w, x, y, z = quat
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


def servo_joint_addrs(model) -> tuple[list[int], list[int]]:
    """Địa chỉ qpos/dof của các joint servo (không phải `passive_*`), theo thứ
    tự model — invariant của repo: không hardcode index joint, model rollers
    có wheel thụ động XEN KẼ nên qpos[7:21] là sai. Trên model thường kết quả
    trùng với slice liền kề (7..20 / 6..19)."""
    qpos_adrs, dof_adrs = [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith("passive_"):
            continue
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue  # root freejoint — không phải servo
        qpos_adrs.append(int(model.jnt_qposadr[j]))
        dof_adrs.append(int(model.jnt_dofadr[j]))
    return qpos_adrs, dof_adrs


def build_61d_obs(model, data, last_action, command13,
                  qpos_adrs=None, dof_adrs=None):
    """61D obs — y hệt scripts/infer_stairs.build_61d_observation (đã chứng minh
    trên testbench + robot thật): ang-vel từ sensor `imu_ang_vel`, quat từ body
    `trunk_base` (xquat), projected gravity cùng công thức. Joint đọc theo
    địa chỉ servo (mặc định tự resolve theo tên)."""
    if qpos_adrs is None or dof_adrs is None:
        qpos_adrs, dof_adrs = servo_joint_addrs(model)
    imu_ang_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if imu_ang_id >= 0:
        adr = model.sensor_adr[imu_ang_id]
        ang_vel_b = np.array(data.sensordata[adr:adr + 3], dtype=np.float32)
    else:
        ang_vel_b = np.array(data.qvel[3:6], dtype=np.float32)

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    base_quat = (np.array(data.xquat[trunk_id], dtype=np.float32) if trunk_id >= 0
                 else np.array(data.qpos[3:7], dtype=np.float32))
    n = np.linalg.norm(base_quat)
    if n > 1e-6:
        base_quat = base_quat / n

    proj_grav = compute_projected_gravity(base_quat)
    joint_pos = np.array([data.qpos[a] for a in qpos_adrs], dtype=np.float32) - DEFAULT_JOINT_POS
    joint_vel = np.array([data.qvel[a] for a in dof_adrs], dtype=np.float32)
    obs = np.concatenate([
        ang_vel_b, proj_grav, joint_pos, joint_vel,
        np.asarray(last_action, dtype=np.float32), np.asarray(command13, dtype=np.float32),
    ]).astype(np.float32)
    assert obs.shape == (61,), f"Expected 61D obs, got {obs.shape}"
    return obs


def trunk_tilt_deg(quat) -> float:
    """Góc giữa trục lên của thân và trục z thế giới (độ)."""
    up_b = quat_apply_inverse(quat, np.array([0.0, 0.0, 1.0]))
    # up_b là vector z thân chiếu sang thế giới qua nghịch đảo → dùng trực tiếp z
    z = float(up_b[2])
    return math.degrees(math.acos(max(-1.0, min(1.0, z))))


def find_ball(model, data):
    """Tìm body có tên chứa 'ball' → (qposadr, tên); None nếu scene không có bóng."""
    import mujoco
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if "ball" in name.lower() and name.lower() != "ball_holder":
            return i, name
    return None, None


def run_twin(key: str, onnx_path: Path, out_dir: Path,
             kind: str | None = None, scene_file: Path | None = None,
             duration_s: float | None = None, seed: int = 0) -> dict:
    import mujoco
    import imageio
    from PIL import Image

    fam = FAMILIES.get(key)
    kind = kind or (fam.twin_kind if fam else "walk")
    spec = KINDS[kind]
    duration = duration_s or (fam.duration_s if fam else 4.0)
    scene_file = scene_file or (REPO_ROOT / fam.scene if fam else REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml")

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.infer_stairs import load_mujoco_with_bam, reset_to_preset, compute_projected_gravity

    def _resolve_spawn(spawn: str):
        """spawn = tên keyframe trong scene (SIT, flight1…) hoặc alias đặc biệt."""
        import mujoco as _mj
        if spawn in ("lying", "lying_back", "stand", "bottom"):
            return spawn
        # Tìm keyframe đúng tên trong scene; không thấy thì dùng STAND.
        key_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_KEY, spawn)
        return "keyframe" if key_id >= 0 else "STAND"

    model, data, bam_ctrl, _ = load_mujoco_with_bam(str(scene_file))
    spawn_resolved = _resolve_spawn(spec["spawn"])
    if spec["spawn"] in ("lying", "lying_back"):
        reset_to_preset(model, data, "bottom", bam_ctrl=bam_ctrl)
        # Nằm sấp (pitch +90°) hoặc ngửa (pitch -90°): hạ thân xuống.
        sign = 1.0 if spec["spawn"] == "lying" else -1.0
        data.qpos[2] = 0.055
        data.qpos[3:7] = [math.cos(math.pi / 4), 0.0, sign * math.sin(math.pi / 4), 0.0]
        data.qvel[:] = 0.0
        bam_ctrl.reset(data.qpos)
        mujoco.mj_forward(model, data)
    elif spawn_resolved == "keyframe":
        # Keyframe tuỳ ý của scene (SIT cho standup, flight1 cho stairs…).
        import mujoco as _mj
        key_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_KEY, spec["spawn"])
        _mj.mj_resetDataKeyframe(model, data, key_id)
        data.qvel[:] = 0.0
        bam_ctrl.reset(data.qpos)
        _mj.mj_forward(model, data)
    else:
        reset_to_preset(model, data, spec["spawn"], bam_ctrl=bam_ctrl)
    if spec.get("post"):
        spec["post"](model, data)
        mujoco.mj_forward(model, data)

    # Settle giữ HOME (đúng testbench): cho bàn chân tiếp đất ổn định trước
    # khi policy nhận điều khiển.
    if spec.get("settle_steps"):
        for _ in range(int(spec["settle_steps"])):
            bam_ctrl.q_target[:] = DEFAULT_JOINT_POS
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    # Reflex leo bậc — y hệt scripts/test_stairs_frame_by_frame.py.
    assist = spec.get("assist")

    def apply_stairs_reflex(cmd13, target_pos, pitch_latch):
        """Tới gần bậc (x ≥ 0.20 m hoặc chân chạm geom 'stair') → lệnh body
        pitch −5° + nâng bàn chân đang vung (hip/knee/ankle) để mép chân
        song song mặt bậc. Trả về latch pitch cho bước kế (lệnh obs có độ
        trễ 1 bước — đúng thứ tự testbench gốc)."""
        stair_touch = False
        for i in range(data.ncon):
            con = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            if "stair" in g1 or "stair" in g2:
                stair_touch = True
                break
        fn_l, fn_r = _foot_normal_forces(model, data)
        approaching = (float(data.qpos[0]) >= 0.20) or stair_touch
        if approaching and cmd13[0] > 0.01:
            pitch_latch = -math.radians(5.0)
            if fn_l < 1.5:  # chân trái đang vung
                target_pos[2] += -0.25
                target_pos[3] += 0.45
                target_pos[4] += 0.70  # = d_knee - d_hip → mép chân phẳng
            if fn_r < 1.5:
                target_pos[11] += 0.25
                target_pos[12] += -0.45
                target_pos[13] += -0.70
        return pitch_latch

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([0.2, 0.0, 0.10])
    cam.distance = 1.15
    cam.elevation = -14
    cam.azimuth = 132

    num_steps = int(duration / CONTROL_DT)
    frames: list[np.ndarray] = []
    last_action = np.zeros(14, dtype=np.float32)

    z_start = float(data.qpos[2])
    zs, tilts, xs, ys, yaws, pitches = [], [], [], [], [], []
    ball_id, ball_name = find_ball(model, data)
    ball_pos0 = None
    if ball_id is not None:
        adr = model.body_jntadr[ball_id]
        if adr >= 0:
            qadr = model.jnt_qposadr[adr]
            ball_pos0 = np.array(data.qpos[qadr:qadr + 3], dtype=float)

    t_capture_every = 2  # 50 Hz điều khiển → 25 fps GIF

    pitch_latch = 0.0  # body-pitch command của reflex, giữ nguyên giữa các bước

    for step in range(num_steps):
        t = step * CONTROL_DT
        cmd13 = spec["cmd"](t, duration)
        cmd13[11] = pitch_latch  # slot body_pose pitch (như cmd_body[4] của testbench)
        obs = build_61d_obs(model, data, last_action, cmd13)
        action = session.run(None, {input_name: obs.reshape(1, -1)})[0].flatten()
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        last_action = action.copy()
        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE

        if assist == "stairs_reflex":
            pitch_latch = apply_stairs_reflex(cmd13, target_pos, pitch_latch)

        for _ in range(DECIMATION):
            bam_ctrl.q_target[:] = target_pos
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        quat = data.qpos[3:7]
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        # Pitch euler (quanh trục Y thân) — dùng cho metric lộn của roulade.
        pitch = math.atan2(
            2.0 * (quat[0] * quat[2] + quat[1] * quat[3]),
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
        )
        zs.append(float(data.qpos[2]))
        xs.append(float(data.qpos[0]))
        ys.append(float(data.qpos[1]))
        yaws.append(yaw)
        pitches.append(pitch)
        tilts.append(trunk_tilt_deg(quat))

        if step % t_capture_every == 0:
            cam.lookat[0] = data.qpos[0] + 0.15
            cam.lookat[1] = data.qpos[1]
            cam.lookat[2] = max(0.08, data.qpos[2])
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    renderer.close()

    def _yaw_total(series):
        total, prev = 0.0, None
        for y in series:
            if prev is not None:
                d = (y - prev + math.pi) % (2 * math.pi) - math.pi
                total += d
            prev = y
        return total

    dist = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    z_final, z_min = zs[-1], min(zs)
    tail = slice(int(len(zs) * 0.7), None)
    z_mean_tail = float(np.mean(zs[tail]))
    z_std_tail = float(np.std(zs[tail]))
    tilt_final, tilt_max = tilts[-1], max(tilts)
    yaw_total = _yaw_total(yaws)
    # Tổng pitch unwrap (rad) — dương nhiều = lộn trọn vòng (roulade).
    pitch_total_deg = abs(math.degrees(float(np.unwrap(np.array(pitches))[-1]
                                        - np.unwrap(np.array(pitches))[0])))
    ball_disp = None
    if ball_id is not None and ball_pos0 is not None:
        adr = model.body_jntadr[ball_id]
        qadr = model.jnt_qposadr[adr]
        ball_disp = float(np.linalg.norm(np.array(data.qpos[qadr:qadr + 3]) - ball_pos0))

    metrics = {
        "key": key, "kind": kind, "onnx": str(onnx_path),
        "scene": str(scene_file), "duration_s": duration,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance_m": round(dist, 4),
        "z_final_m": round(z_final, 4),
        "z_min_m": round(round(z_min, 4), 4),
        "z_mean_tail_m": round(z_mean_tail, 4),
        "z_std_tail_m": round(z_std_tail, 5),
        "tilt_final_deg": round(tilt_final, 2),
        "tilt_max_deg": round(tilt_max, 2),
        "yaw_total_deg": round(math.degrees(yaw_total), 1),
        "pitch_rotated_deg": round(pitch_total_deg, 1),
        "ball_displacement_m": round(ball_disp, 4) if ball_disp is not None else None,
        "spawn": spec["spawn"],
        "z_start_m": round(z_start, 4),
    }

    # ---- Kết luận theo loại động tác (ngưỡng đo từ thực tế mô phỏng) ----
    if kind == "walk":
        ok = dist >= 0.15 and tilt_final < 25.0
        verdict = f"đi {dist:.2f} m, nghiêng cuối {tilt_final:.0f}°"
    elif kind == "stand":
        ok = z_std_tail < 0.012 and tilt_max < 20.0
        verdict = f"dao động z σ={z_std_tail * 1000:.1f} mm, nghiêng tối đa {tilt_max:.0f}°"
    elif kind == "standup":
        ok = z_final >= 0.11 and tilt_final < 30.0
        verdict = f"nằm → z cuối {z_final:.3f} m, nghiêng cuối {tilt_final:.0f}°"
    elif kind == "sitstand":
        dipped = z_min <= z_mean_tail - 0.015
        ok = dipped and z_final >= 0.11
        verdict = f"đổ z_min {z_min:.3f} m rồi hồi về {z_final:.3f} m"
    elif kind == "ground_pick":
        crouched = z_min <= 0.098
        ok = crouched and z_final >= 0.11
        verdict = f"khom tới z {z_min:.3f} m, đứng lại {z_final:.3f} m"
    elif kind == "stairs":
        gained = z_final - z_start
        ok = (xs[-1] - xs[0]) >= 0.60 and gained >= 0.03
        verdict = f"tiến {xs[-1] - xs[0]:.2f} m, lên {gained * 100:.1f} cm"
    elif kind == "roulade":
        ok = metrics["pitch_rotated_deg"] >= 0.8 * 360
        verdict = f"quay {metrics['pitch_rotated_deg']:.0f}° quanh trục ngang"
    elif kind == "ball_kick":
        ok = ball_disp is not None and ball_disp >= 0.05
        verdict = f"bóng bay {ball_disp:.2f} m" if ball_disp is not None else "không thấy bóng trong scene"
    elif kind == "fetch":
        ok = dist >= 0.15  # đi về phía bóng; khoảng cách bóng cố định
        verdict = f"tiến {dist:.2f} m về phía bóng"
    elif kind == "spin":
        # 1 chu kỳ envelope (4 s) = 2.1 * SPIN_RATE_MAX = 6.3 rad ≈ 361°
        ok = abs(math.degrees(yaw_total)) >= 0.8 * 361
        verdict = f"xoay {math.degrees(yaw_total):.0f}° quanh trục đứng"
    else:
        ok, verdict = False, "kind lạ"

    metrics["success"] = bool(ok)
    metrics["verdict"] = verdict

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    gif_path = out_dir / f"rollout_{ts}.gif"
    small = [np.asarray(Image.fromarray(f).resize((480, 360))) for f in frames]
    imageio.mimsave(str(gif_path), small, fps=25, loop=0)
    metrics_path = out_dir / f"metrics_{ts}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    _copy(gif_path, out_dir / "latest.gif")
    _copy(metrics_path, out_dir / "latest_metrics.json")
    metrics["gif"] = str(gif_path)
    return metrics


def _copy(src: Path, dst: Path):
    import shutil
    shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Digital Twin — chạy policy ONNX headless, xuất GIF + chỉ số")
    ap.add_argument("--key", choices=sorted(FAMILIES), help="Family trong registry")
    ap.add_argument("--onnx", type=Path, help="Đường dẫn ONNX (mặc định: của family trong matrix)")
    ap.add_argument("--kind", choices=sorted(KINDS), help="Ghi đè adapter lệnh")
    ap.add_argument("--duration", type=float, help="Ghi đè thời lượng (giây)")
    ap.add_argument("--out-dir", type=Path, help="Thư mục xuất (mặc định logs/digital_twin/<key>)")
    args = ap.parse_args()

    if not args.key and not (args.onnx and args.kind):
        ap.error("Cần --key, hoặc --onnx + --kind")

    onnx = args.onnx
    if onnx is None:
        from .matrix import _scan_family
        e = _scan_family(FAMILIES[args.key])
        onnx = Path(e["onnx"]) if e["onnx"] else None
        if onnx is None:
            sys.exit(f"[digital_twin] Family '{args.key}' chưa có ONNX — export trước (rtx-pipeline export {args.key})")

    out_dir = args.out_dir or (TWIN_DIR / (args.key or args.kind))
    m = run_twin(args.key or args.kind, onnx, out_dir, kind=args.kind,
                 duration_s=args.duration)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    ok = "✓ THÀNH CÔNG" if m["success"] else "✗ CHƯA ĐẠT"
    print(f"\n[{ok}] {m['verdict']}\nGIF: {m['gif']}")


if __name__ == "__main__":
    main()
