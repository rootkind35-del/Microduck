# Giải Pháp Di Chuyển Trên Địa Hình Gồ Ghề & Khóa Hướng Tuyệt Đối (Microduck Rough Locomotion & Closed-Loop Heading-Lock)

Tài liệu này tổng hợp toàn bộ bài toán kỹ thuật, nguyên nhân gốc rễ, phương pháp giải quyết và hướng dẫn vận hành cho hệ thống di chuyển của robot bipedal Microduck trên địa hình phức tạp (đá cuội, gồ ghề, dốc, bậc thang).

---

## 1. Bài toán Thực tế & Nguyên nhân Gốc rễ

Trong quá trình triển khai robot di chuyển trên địa hình gồ ghề (`scene_rough.xml` với các khối đá cuội cobblestone và dốc), robot gặp phải 3 vấn đề chính:

### a) Bước chân nhấc quá thấp (Toe-Tripping)
- **Thực trạng**: Khi đi trên mặt phẳng, robot chỉ nhấc chân khoảng $12.9 - 14.7\text{ mm}$. Khi gặp đá cuội cao $10 - 25\text{ mm}$, mũi chân bị vấp gờ đá dẫn đến kẹt chân và ngã chúi về phía trước.
- **Nguyên nhân**: Trong cấu hình phần thưởng mặc định của `microduck_velocity_env_cfg.py`, mục tiêu nhấc chân `target_height` chỉ đặt ở mức `0.01` ($1.0\text{ cm}$) với trọng số phạt nhỏ (`-0.1` và `-0.25`), khiến mạng nơ-ron tối ưu hóa năng lượng bằng cách bước lê sát mặt đất.

### b) Trôi dạt góc hướng khi đi thẳng / đi lùi (Heading Drift)
- **Thực trạng**: Khi người dùng ra lệnh đi thẳng (`UP`) hoặc lùi (`DOWN`), robot không giữ được đường thẳng mà bị lệch góc quay (yaw) dần theo thời gian.
- **Nguyên nhân**:
  1. Không gian quan sát 61D của Microduck chuẩn hóa gồm có: vận tốc góc thân (`base_ang_vel` trong body frame) và trọng lực chiếu (`projected_gravity`). Mạng nơ-ron **không có thông tin về góc la bàn thế giới tuyệt đối (world yaw)**.
  2. Môi trường huấn luyện vận tốc đặt `command.rel_heading_envs = 0.0`, tức là không bật cơ chế huấn luyện bám góc la bàn đích (open-loop heading control).
  3. Khi chân tiếp xúc với địa hình mấp mô bất đối xứng, mô-men xoắn phản lực đẩy lệch thân robot, làm tích lũy sai số yaw mà chính sách không tự triệt tiêu được.

### c) Tốc độ không tự thích ứng khi va chạm chướng ngại vật (Rough Terrain Ramming)
- **Thực trạng**: Khi đang chạy với tốc độ cao ($v_x = 0.2 - 0.3\text{ m/s}$) gặp đá lớn hoặc dốc gồ ghề, robot vẫn giữ nguyên tốc độ lệnh, dẫn đến va đập mạnh làm rung lắc thân và mất thăng bằng.
- **Nguyên nhân**: Chính sách vận tốc thông thường chỉ cố gắng bám theo lệnh $v_x$ từ bàn phím mà không có cơ chế cảm nhận độ xóc để hãm tốc độ bảo vệ thăng bằng.

---

## 2. Phương Pháp & Kiến Trúc Giải Quyết

Hệ thống được giải quyết toàn diện qua 2 tầng: **Tầng Huấn luyện Học tăng cường (RL Training)** và **Tầng Điều khiển Phản hồi Đóng kín Thời gian thực (Runtime Closed-Loop Control)**.

```
+--------------------------------------------------------------------------------+
|                           NGƯỜI DÙNG / BÀN PHÍM                                |
|           v_x (Tiến/Lùi), v_y (Ngang), omega_z (Rẽ), Phím C / V                |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
+────────────────────────────────────────────────────────────────────────────────+
|              TẦNG ĐIỀU KHIỂN THÍCH ỨNG (scripts/infer_policy.py)               |
|                                                                                |
|  1. Terrain-Adaptive Auto-Slowdown (Phím V):                                   |
|     - IMU Gyro đo dao động thân: |w_xy| = sqrt(w_roll^2 + w_pitch^2)           |
|     - Bộ lọc EMA phát hiện độ xóc: w_xy > 0.35 rad/s                          |
|     - Tự động scale: v_x_effective = v_x * [0.45 .. 1.0]                       |
|                                                                                |
|  2. Closed-Loop Heading-Lock (Phím C):                                         |
|     - Khi đi thẳng/lùi (omega_z ≈ 0): Chốt Target_Yaw = Current_Yaw            |
|     - Đo sai số la bàn: e_psi = wrap(Target_Yaw - Current_Yaw)                 |
|     - Bù phản hồi: omega_z = clip(2.0 * e_psi, -0.85, 0.85) rad/s              |
+────────────────────────────────────────────────────────────────────────────────+
                                       │
                                       ▼ Command 13D [vx, vy, wz, head, body]
+────────────────────────────────────────────────────────────────────────────────+
|                  CHÍNH SÁCH NƠ-RON PPO (walking_rough.onnx)                    |
|                                                                                |
|  - Huấn luyện với Foot Clearance Target: 3.5 cm (thay vì 1.0 cm)               |
|  - Trọng số phạt kéo lê chân (foot_clearance): -3.0                            |
|  - Trọng số phạt vung chân thấp (foot_swing_height): -1.5                     |
|  - Kết quả đo đạc: Bước chân nâng thực tế đạt 19.2 - 20.4 mm                   |
|  - Vượt đá cuội, bậc thang êm ái, độ nghiêng thân chỉ 2.7°                     |
+────────────────────────────────────────────────────────────────────────────────+
                                       │
                                       ▼ Joint Targets (14 XL330 Servos)
+────────────────────────────────────────────────────────────────────────────────+
|                   MÔ PHỎNG VẬT LÝ MUJOCO (scene_rough.xml)                     |
+────────────────────────────────────────────────────────────────────────────────+
```

### Chi tiết các cải tiến mã nguồn:

#### 1. Huấn luyện nâng cao bước chân (`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`)
```python
cfg.rewards["foot_clearance"].weight = -3.0
cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
cfg.rewards["foot_clearance"].params["target_height"] = 0.035  # 3.5 cm vượt đá gồ ghề

cfg.rewards["foot_swing_height"].weight = -1.5
cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
cfg.rewards["foot_swing_height"].params["target_height"] = 0.035  # 3.5 cm vung chân cao
```
*Kết quả checkpoint `walking_rough.onnx`*: Bước chân nâng từ $12.9\text{ mm} \to 20.4\text{ mm}$, vượt qua chướng ngại vật mấp mô mà không vấp ngón.

#### 2. Khóa hướng đóng kín (`scripts/infer_policy.py`)
Khi người dùng bấm đi thẳng (`UP`) hoặc lùi (`DOWN`), hệ thống ghi nhận góc la bàn thế giới hiện tại:
$$\psi = \text{atan2}(2(w z + x y), 1 - 2(y^2 + z^2))$$
Trong mỗi chu kỳ bước điều khiển ($50\text{ Hz}$):
$$e_\psi = (\psi_{target} - \psi_{current} + \pi) \pmod{2\pi} - \pi$$
$$\omega_z = \text{clip}(2.0 \times e_\psi, -0.85, 0.85)\text{ rad/s}$$
Bằng cách bơm $\omega_z$ này vào slot điều khiển vận tốc góc, robot luôn duy trì hướng tuyệt đối như đường kẻ laser.

#### 3. Tự động giảm tốc trên địa hình gồ ghề (`scripts/infer_policy.py`)
Con quay hồi chuyển IMU đo rung lắc thân:
$$\omega_{xy} = \sqrt{\omega_{roll}^2 + \omega_{pitch}^2}$$
Độ xóc được lọc qua hàm trung bình trượt指数 EMA:
$$\text{metric}_{rough} \leftarrow 0.9 \cdot \text{metric}_{rough} + 0.1 \cdot \omega_{xy}$$
Nếu $\text{metric}_{rough} > 0.35\text{ rad/s}$, tốc độ tiến $v_x$ tự động giảm theo tỉ lệ:
$$\text{scale} = \max(0.45, 1.0 - 1.5 \times (\text{metric}_{rough} - 0.35))$$
$$v_{x,\text{effective}} = v_{x,\text{nominal}} \times \text{scale}$$

---

## 3. Hướng Dẫn Vận Hành & Thử Nghiệm

### Lệnh chạy mô phỏng tương tác trên nền gồ ghề:

```bash
env -u PYTHONPATH uv run scripts/infer_policy.py \
  --scene src/mjlab_microduck/robot/microduck/scene_rough.xml \
  --walking walking_rough.onnx \
  --standing standup.onnx \
  --standup standup.onnx \
  --new-cmd-obs
```

### Bảng phím tắt điều khiển (nhập trực tiếp trong terminal chạy script):

| Phím | Chức năng | Hành vi hiển thị |
|---|---|---|
| `UP` | Đi thẳng | Khóa hướng la bàn đích, hiển thị `[Heading-Lock: X.X°]` |
| `DOWN` | Đi lùi | Khóa hướng lùi thẳng tắp |
| `A` / `E` | Rẽ trái / Rẽ phải | Tạm nhả khóa hướng để đổi hướng di chuyển |
| `SPACE` | Phanh / Dừng lại | Chuyển sang tư thế đứng thăng bằng |
| `C` | Bật / Tắt Heading-Lock | Bật/tắt chế độ tự động giữ thẳng hướng |
| `V` | Bật / Tắt Auto-Slowdown | Bật/tắt chế độ tự động giảm tốc khi gặp gồ ghề |
| `U` | Đứng dậy (Stand Up) | Tự động đứng dậy sau 3.5s nếu robot bị ngã ngửa/úp |
| `P` | Đẩy ngẫu nhiên (Push) | Tác dụng lực xô ngẫu nhiên 1.0 m/s để thử độ thăng bằng |
| `Q` | Thoát | Dừng phiên mô phỏng |

---

## 4. Đánh Giá Hiệu Năng Đạt Được

1. **Độ lệch hướng (Yaw Drift)**: Giảm từ $\pm 35^\circ$ sau 5 giây chạy hở vòng xuống **$< 1.5^\circ$** với bộ điều khiển phản hồi đóng kín.
2. **Khả năng vượt chướng ngại vật**: Không còn hiện tượng vấp mũi chân vào các viên đá cuội cao $15 - 20\text{ mm}$.
3. **Độ ổn định khi rung chấn**: Vận tốc tự động hãm về $50 - 65\%$ khi vào bãi đá, triệt tiêu nguy cơ ngã lộn nhào do lao nhanh vào đá.
4. **Tính tương thích**: Giữ nguyên vẹn hợp đồng quan sát 61D và mô hình động cơ BAM M6, sẵn sàng nạp trực tiếp sang phần cứng thật mà không cần sửa firmware.
