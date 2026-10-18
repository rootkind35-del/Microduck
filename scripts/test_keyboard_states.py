#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath("."))
import scripts.infer_policy as ip

def main():
    print("=" * 75)
    print("🚀 BẮT ĐẦU TEST TỰ ĐỘNG CÁC PHÍM (KÈM STANDING KHI DỪNG)")
    print("=" * 75)

    bam_model = ip.load_bam_model(
        kp_fw=ip.BAM_KP_FW,
        vin=7.4,
        max_current=ip.BAM_MAX_CURRENT,
    )
    model, data, bam_ctrl, actuator_names = ip.load_mujoco_with_bam(
        xml_path="src/mjlab_microduck/robot/microduck/scene_stairs.xml",
        bam_model=bam_model,
        timestep=0.005,
        vin_drop_gain=0.0,
        vin_min=ip.BAM_VIN_MIN,
    )

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)

    policy = ip.PolicyInference(
        model=model,
        data=data,
        bam_ctrl=bam_ctrl,
        stairs_onnx_path="stairs_climb.onnx",
        standing_onnx_path="standing.onnx",
        standup_onnx_path="standup.onnx",
        new_cmd_obs=True,
    )

    test_sequence = [
        ("KHỞI ĐẦU TỰ ĐỘNG (0s - 3s)", None),
        ("BẤM 'UP' (Tiến lên bậc)", "up"),
        ("BẤM 'SPACE' (Dừng lại - chuyển sang Standing)", " "),
        ("BẤM 'P' (Thử lực đẩy Push ngẫu nhiên)", "p"),
        ("BẤM 'SPACE' (Giữ thăng bằng)", " "),
    ]

    qpos_adr = policy._trunk_qpos_adr
    control_dt = 0.02
    sim_steps_per_ctrl = 4

    for step_idx, (desc, key_to_press) in enumerate(test_sequence):
        if key_to_press:
            if key_to_press == "up":
                policy.set_vel_cmd(policy.vel_max_x, 0.0, 0.0)
            elif key_to_press == " ":
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif key_to_press == "p":
                data.qvel[qpos_adr:qpos_adr+3] += np.random.uniform(-0.3, 0.3, size=3)

        # Chạy 3.0 giây mô phỏng
        for _ in range(150):
            policy.update_behavior(control_dt)
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(sim_steps_per_ctrl):
                mujoco.mj_step(model, data)

        trunk_z = float(data.qpos[qpos_adr + 2])
        qw, qx, qy, qz = data.qpos[qpos_adr + 3:qpos_adr + 7]
        cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
        cos_tilt = max(-1.0, min(1.0, cos_tilt))
        tilt_deg = math.degrees(math.acos(cos_tilt))

        if trunk_z < 0.07 or tilt_deg > 50.0:
            posture_status = "❌ NẰM BẸT TRÊN SÀN (Fallen)"
        elif trunk_z < 0.095:
            posture_status = "⚠️ NGỒI BỆP / CÚI THẤP (Crouched)"
        else:
            posture_status = "✅ ĐỨNG THẲNG THĂNG BẰNG (Upright)"

        mode = policy.current_policy
        cmd = policy.vel_cmd

        print(f"\n[{step_idx+1:02d}/{len(test_sequence)}] Thao tác: {desc}")
        print(f"   ► Policy đang chạy: [{mode}] | Lệnh Cmd: vx={cmd[0]:+.2f}, vy={cmd[1]:+.2f}, yaw={cmd[2]:+.2f}")
        print(f"   ► Chiều cao thân   : {trunk_z*1000:6.1f} mm  (chuẩn đứng thăng bằng: ~116 mm)")
        print(f"   ► Góc nghiêng thân : {tilt_deg:6.1f}°   (ngã: >50°)")
        print(f"   ► Trạng thái robot : {posture_status}")

if __name__ == "__main__":
    main()
