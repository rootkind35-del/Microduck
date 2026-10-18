# BẢN ĐỀ CƯƠNG ĐẠI TU KHOA HỌC & ĐỀ ÁN BẢO VỆ TIẾN SĨ
## Nâng Cấp Hệ Thống Học Tăng Cường Cho Robot Hai Chân Microduck Theo Tiêu Chuẩn Hội Nghị & Tạp Chí Đầu Ngành (RSS / CoRL / IEEE T-RO)

> **Dự án**: `microduck_rl`  
> **Nền tảng**: Robot hai chân Microduck (~800 g, 25 cm, 14 Dynamixel XL330 servos, kiến trúc mô phỏng MuJoCo Warp / `mjlab`, thuật toán PPO / `rsl_rl`)  
> **Phạm vi thẩm định**: Toàn bộ hệ thống kỹ năng vận động (Locomotion), vượt địa hình (Stairs/Rough), tự hồi phục (Standup/Roulade), và tương tác thao tác (Fetch/Pick/Kick). *(Loại trừ toàn bộ các module trượt patin theo yêu cầu)*.  
> **Mục tiêu**: Loại bỏ triệt để các lỗ hổng phương pháp luận, thay thế các "mánh khóe mô phỏng" (simulation hacks) và bộ điều khiển ngoại vi bằng giải pháp Học tăng cường nội sinh (Endogenous RL) đạt chuẩn phản biện khoa học quốc tế đỉnh cao.

---

## MỤC LỤC

1. [Phân Tích Hiện Trạng & Ranh Giới An Toàn Huấn Luyện](#1-phân-tích-hiện-trạng--ranh-giới-an-toàn-huấn-luyện)
2. [Ma Trận Tài Liệu Khoa Học Tham Khảo (Literature Review & SOTA Mapping)](#2-ma-trận-tài-liệu-khoa-học-tham-khảo-literature-review--sota-mapping)
3. [Phân Tầng Kỹ Thuật & Kế Hoạch Thực Hiện Chi Tiết (Task Stratification & Deep Plans)](#3-phân-tầng-kỹ-thuật--kế-hoạch-thực-hiện-chi-tiết)
   - [Task 1: Nội sinh hóa Khóa hướng (Heading-Lock) & Thích ứng Rung chấn vào POMDP](#task-1-nội-sinh-hóa-khóa-hướng-heading-lock--thích-ứng-rung-chấn-vào-pomdp)
   - [Task 2: Tái cấu trúc Kỹ năng Tương tác Thao tác dựa trên Tiếp xúc Vật lý Thực (Contact-Rich Manipulation)](#task-2-tái-cấu-trúc-kỹ-năng-tương-tác-thao-tác-dựa-trên-tiếp-xúc-vật-lý-thực)
   - [Task 3: Kiến trúc Phân cấp Điều phối Kỹ năng (Hierarchical Coordination & Dynamic Action Blending)](#task-3-kiến-trúc-phân-cấp-điều-phối-kỹ-năng-hierarchical-coordination--dynamic-action-blending)
   - [Task 4: Nâng cấp Kỹ năng Vượt Bậc thang lên Tỷ lệ Hình học Chuẩn (> 2.5 - 3.5 cm)](#task-4-nâng-cấp-kỹ-năng-vượt-bậc-thang-lên-tỷ-lệ-hình-học-chuẩn--25---35-cm)
   - [Task 5: Thiết lập Hệ thống Thực nghiệm Đối chứng (Ablation Study Matrix) & Tiêu chí Đánh giá](#task-5-thiết-lập-hệ-thống-thực-nghiệm-đối-chứng-ablation-study-matrix--tiêu-chí-đánh-giá)
4. [Lộ Trình Triển Khai Thực Tế (Execution Timeline & Operational Roadmap)](#4-lộ-trình-triển-khai-thực-tế-execution-timeline--operational-roadmap)

---

## 1. PHÂN TÍCH HIỆN TRẠNG & RANH GIỚI AN TOÀN HUẤN LUYỆN

### 1.1. Tình trạng tiến trình hệ thống
Tại thời điểm lập báo cáo, hệ thống đang vận hành một tiến trình huấn luyện GPU dài hạn:
* **PID 4381**: `uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 1024 --agent.max_iterations 600 --agent.load-checkpoint logs/rsl_rl/stairs_climb/2026-10-17_08-16-27_stairs_climb/model_750.pt`
* **Thời gian chạy**: ~3.6 giờ liên tục, đang ở chặng cuối (iteration > 520 / 600).
* **Ràng buộc cô lập (Isolation Boundary)**: Tuyệt đối không can thiệp, chỉnh sửa cấu trúc hay nạp đè vào `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` và các checkpoint trong thư mục `logs/rsl_rl/stairs_climb/2026-10-17_18-09-16_stairs_climb/` cho đến khi tiến trình cán mốc 600 iterations an toàn.
* Mọi cấu trúc mã nguồn mới của đề án đại tu này sẽ được tổ chức theo module độc lập hoặc kế thừa có phiên bản (`v2`), sẵn sàng tích hợp khi tài nguyên giải phóng.

---

## 2. MA TRẬN TÀI LIỆU KHOA HỌC THAM KHẢO (LITERATURE REVIEW & SOTA MAPPING)

Để biến công trình từ một bài báo cáo kỹ thuật lắp ghép (engineering report) thành một đóng góp khoa học có sức nặng tại các hội nghị **RSS / CoRL / ICRA** hoặc tạp chí **IEEE Transactions on Robotics (T-RO)**, mọi cải tiến phải dựa trên các công trình nghiên cứu nền tảng:

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    BẢN ĐỒ TÀI LIỆU THAM KHẢO SOTA                                |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                                                                                  |
|  [VẬN ĐỘNG & BÁM HƯỚNG]             [THAO TÁC TIẾP XÚC ĐỘNG]          [ĐIỀU PHỐI KỸ NĂNG]         |
|  - Rudin et al. (Science Rob. 2022) - Ji et al. (DribbleBot 2023)    - Peng et al. (ASE 2021)    |
|  - Siekmann et al. (RSS 2021)       - Cheng et al. (Legged Manip.)   - Merel et al. (DeepMind)   |
|  - Kumar et al. (RMA - CoRL 2021)   - OpenAI (T-RO 2020)             - Tsounis (DeepGait 2020)   |
|                                                                                                  |
|                              [MÔ HÌNH HÓA ĐỘNG CƠ & SIM2REAL]                                    |
|                              - Hwangbo et al. (Science Robotics 2019)                            |
|                              - BAM M6 XL330 Dynamics (Pollen Robotics 2024)                      |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Chi tiết các công trình nghiên cứu bắt buộc áp dụng:

1. **Rudin et al., 2022** — *"Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild"*, Science Robotics.
   * *Đóng góp cốt lõi*: Thiết kế không gian lệnh liên tục bao gồm góc hướng đích ($e_\psi$) thay vì chỉ lệnh vận tốc xoay; kỹ thuật Asymmetric Actor-Critic và chuẩn hóa quan sát trực tiếp trong mạng nơ-ron.
   * *Ứng dụng vào Microduck*: Loại bỏ P-controller ngoại vi, đưa $e_\psi$ vào bài toán POMDP bám hướng tuyệt đối.
2. **Siekmann et al., 2021** — *"Blind Bipedal Stair-Climbing with Reinforcement Learning"*, Robotics: Science and Systems (RSS) / IEEE T-RO.
   * *Đóng góp cốt lõi*: Robot hai chân Cassie leo bậc thang mù lên tới 20 cm bằng cơ chế đồng hồ pha tham chiếu (Phase Clock), phạt kéo lê bàn chân (Clearance reward) và giới hạn xung lực va đập đầu gối.
   * *Ứng dụng vào Microduck*: Nâng độ cao bậc thang từ 1.5 cm lên 3.5 cm (14% chiều cao robot) kết hợp bù trừ nghiêng hông (Pelvic roll).
3. **Ji et al., 2023** — *"DribbleBot: Dynamic Bipedal and Quadrupedal Ball Dribbling in the Wild"*, Science Robotics / MIT Improbable AI Lab.
   * *Đóng góp cốt lõi*: Huấn luyện kỹ năng dẫn bóng hoàn toàn qua động lực học tiếp xúc ma sát và va chạm thực tế trong mô phỏng (Coulomb friction, contact normal forces). Hoàn toàn không dùng cờ dính vật thể hay gán tọa độ.
   * *Ứng dụng vào Microduck*: Xóa bỏ đoạn mã gian lận `data.qpos = hold_pos` trong `infer_fetch.py`, chuyển bài toán sang dẫn bóng bằng tiếp xúc mỏ/chân.
4. **Peng et al., 2021** — *"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters"*, ACM Trans. Graph. (SIGGRAPH) / NVIDIA.
   * *Đóng góp cốt lõi*: Không gian biểu diễn kỹ năng tiềm ẩn liên tục (Latent Skill Space $\mathcal{Z}$) cho phép chuyển tiếp mượt mà giữa các hành vi vận động mà không làm gãy đạo hàm hành động $\dot{\mathbf{u}}$.
   * *Ứng dụng vào Microduck*: Thay thế máy trạng thái hữu hạn 7-state (FSM) và chuyển mạch cứng ONNX bằng bộ điều phối hành động mượt mà (Dynamic Action Blending).
5. **Hwangbo et al., 2019** — *"Learning Agile and Dynamic Motor Skills for Legged Robots"*, Science Robotics.
   * *Đóng góp cốt lõi*: Actuator Net mô phỏng chính xác đáp ứng tần số, sụt áp và trễ của động cơ điện servo nhằm thu hẹp triệt để khoảng cách Sim2Real.
   * *Ứng dụng vào Microduck*: Chuẩn hóa mô hình BAM M6 cho Dynamixel XL330, bổ sung kiểm tra sụt áp danh định $V_{in} \ge 6.0\text{V}$.

---

## 3. PHÂN TẦNG KỸ THUẬT & KẾ HOẠCH THỰC HIỆN CHI TIẾT

---

### TASK 1: NỘI SINH HÓA KHÓA HƯỚNG (HEADING-LOCK) & THÍCH ỨNG RUNG CHẤN VÀO POMDP

* **Mục tiêu**: Triệt tiêu hiện tượng trôi dạt góc hướng ($\pm 35^\circ$ sau 5 giây) bằng chính sách nơ-ron nội sinh; xóa bỏ hoàn toàn bộ điều khiển P và bộ lọc EMA ngoại vi trong [`scripts/infer_policy.py`](file:///home/dhp01/microduck_rl/scripts/infer_policy.py).

#### 1. Cơ sở lý thuyết & Mô hình hóa toán học
* **Bài toán POMDP mở rộng**:
  Trong không gian quan sát 61D của Actor:
  Slot lệnh vận tốc góc $\omega_z$ (slot thứ 3 trong khối 13D Command) được chuyển đổi ngữ nghĩa thành **Sai số góc hướng đích (Target Heading Error)** khi kích hoạt chế độ đi thẳng:
  $$e_\psi = \text{atan2}\left( \sin(\psi_{target} - \psi_{current}), \cos(\psi_{target} - \psi_{current}) \right) \in [-\pi, \pi]$$
  Mạng nơ-ron nhận trực tiếp $e_\psi$ qua slot quan sát chuẩn hóa, cho phép nó quan sát trực tiếp độ lệch hướng tuyệt đối trong hệ quy chiếu thế giới.
* **Thiết kế hàm thưởng bám hướng nội sinh (Endogenous Heading-Alignment Reward)**:
  $$r_{heading} = \exp\left( - \frac{e_\psi^2}{\sigma_\psi^2} \right) \cdot \mathbb{I}_{\{\|\mathbf{v}_{xy}^{cmd}\| > 0.02\}}$$
  với $\sigma_\psi = 0.20\text{ rad} \approx 11.5^\circ$.
* **Hàm phạt xung lực va chạm thích nghi (Compliance Impact Penalty)**:
  Thay vì dùng code Python ngoài đo độ xóc $\omega_{xy}$ rồi ép giảm $v_x$, ta bổ sung hàm phạt tiếp xúc nặng trong `mdp.py`:
  $$r_{impact} = - w_{impact} \sum_{i \in \text{feet}} \max(0, \|\mathbf{F}_{normal, i}\| - F_{safe})^2$$
  $$r_{base\_acc} = - w_{acc} \left( \|\dot{\boldsymbol{\omega}}_{xy}\|^2 + \|\mathbf{a}_{z, base}\|^2 \right)$$
  Khi di chuyển trên đá cuội, mạng PPO sẽ tự tối ưu hóa bằng cách co ngắn bước chân, chùng khớp gối (compliance) để giảm thiểu gia tốc thân thay vì để bộ lọc ngoại vi can thiệp.

#### 2. Kế hoạch mã nguồn
* **Tệp cần cập nhật**:
  * [`src/mjlab_microduck/tasks/mdp.py`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/mdp.py):
    - Bổ sung hàm `track_heading_potential(env, std, command_name)`.
    - Bổ sung hàm `terrain_impact_compliance_penalty(env, force_threshold)`.
  * [`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py):
    - Đặt `cfg.commands["twist"].rel_heading_envs = 1.0` (kích hoạt chế độ huấn luyện bám góc hướng cho 100% môi trường).
    - Đăng ký term thưởng `heading_tracking` (weight = 2.0).
    - Đăng ký term phạt `body_impact_compliance` (weight = -0.5).
  * [`scripts/infer_policy.py`](file:///home/dhp01/microduck_rl/scripts/infer_policy.py):
    - Gỡ bỏ hoàn toàn logic tính P-controller: `omega_z = clip(2.0 * e_psi, ...)`.
    - Gỡ bỏ logic tính EMA: `metric_rough = 0.9 * metric + 0.1 * w_xy` và `scale = max(0.45, ...)`.
    - Bơm trực tiếp $e_\psi$ vào command slot thứ 3 để mạng nơ-ron tự điều hướng.

#### 3. Tiêu chí kiểm định (Verification Metrics)
* Độ lệch góc hướng sau 10 giây đi thẳng trên nền gồ ghề (`scene_rough.xml`) **$< 2.0^\circ$** mà không có bất kỳ bộ điều khiển ngoài nào can thiệp.
* Mạng tự động giảm biên độ dao động thân khi chân chạm đá, thể hiện qua đồ thị gia tốc $\mathbf{a}_z \le 1.8g$.

---

### TASK 2: TÁI CẤU TRÚC KỸ NĂNG TƯƠNG TÁC THAO TÁC DỰA TRÊN TIẾP XÚC VẬT LÝ THỰC (CONTACT-RICH MANIPULATION)

* **Mục tiêu**: Xóa bỏ vĩnh viễn đoạn mã gian lận tọa độ `data.qpos = hold_pos` trong [`scripts/infer_fetch.py`](file:///home/dhp01/microduck_rl/scripts/infer_fetch.py#L406-L411); tái định vị tác vụ thành **"Bipedal Ball Dribbling & Target Reaching"** đạt chuẩn khoa học quốc tế.

#### 1. Cơ sở lý thuyết & Động lực học tiếp xúc (Contact Dynamics)
* **Phân tích cơ khí Microduck**: Mỏ robot là vật thể cứng tĩnh gắn trên đầu (4 bậc tự do XL330). Không có ngàm kẹp chủ động. Mọi tương tác với quả bóng 15 g / 70 mm phải tuân thủ nghiêm ngặt định luật bảo toàn động lượng và ma sát Coulomb của MuJoCo:
  $$\mathbf{F}_{contact} = \mathbf{F}_n + \mathbf{F}_t, \quad \|\mathbf{F}_t\| \le \mu \|\mathbf{F}_n\|$$
* **Thiết kế bài toán DribbleBot cho Microduck (theo Ji et al., Science Robotics 2023)**:
  * Robot phải di chuyển tiếp cận quả bóng, sử dụng mặt ngoài của mỏ hoặc chân để tạo xung lực có kiểm soát, đẩy quả bóng lăn về vị trí đích (Human Zone $\mathbf{p}_{human}$).
  * **Hàm thưởng dẫn bóng tiếp xúc động (Dynamic Contact Dribbling Reward)**:
    $$r_{dribble} = \mathbf{v}_{ball} \cdot \frac{\mathbf{p}_{target} - \mathbf{p}_{ball}}{\|\mathbf{p}_{target} - \mathbf{p}_{ball}\|} - \beta \|\mathbf{v}_{ball, \perp}\|$$
    trong đó $\mathbf{v}_{ball, \perp}$ là thành phần vận tốc vuông góc với đường nối bóng - đích (phạt việc sút bóng bay lệch hướng).
  * **Hàm thưởng bảo vệ động cơ mỏ (Beak Impact Regulation)**:
    $$r_{beak\_soft} = - w_{beak} \max(0, \|\mathbf{F}_{mouth\_contact}\| - 1.5\text{ N})^2$$
    Bảo vệ bánh răng nhựa của servo XL330 vùng cổ trước phản lực va chạm quá lớn.

#### 2. Kế hoạch mã nguồn
* **Tệp cần cập nhật**:
  * [`src/mjlab_microduck/robot/microduck/scene_fetch.xml`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/robot/microduck/scene_fetch.xml):
    - Đảm bảo geom `mouth_tip_geom` và `ball_geom` có cấu hình tiếp xúc đầy đủ:
      `condim="4" friction="0.7 0.02 0.005" solref="0.004 1" solimp="0.9 0.95 0.001 0.5 2"`.
  * [`src/mjlab_microduck/tasks/microduck_fetch_env_cfg.py`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/microduck_fetch_env_cfg.py):
    - Xóa bỏ các reward proxy không đo lực.
    - Bổ sung `dribble_direction_reward`, `ball_velocity_tracking`, `beak_impact_softness`.
    - Critic nhận vị trí thực tế của bóng và vector tiếp xúc để học Value Function chính xác.
  * [`scripts/infer_fetch.py`](file:///home/dhp01/microduck_rl/scripts/infer_fetch.py):
    - **XÓA BỎ DÒNG 406–411**: Triệt tiêu hoàn toàn lệnh can thiệp `data.qpos` và `data.qvel`. Quả bóng hoàn toàn tự do di chuyển theo vật lý của MuJoCo.
    - Cập nhật logic đánh giá: Nhiệm vụ hoàn thành khi khoảng cách $\|\mathbf{p}_{ball} - \mathbf{p}_{human}\| < 0.20\text{ m}$ thông qua tiếp xúc đẩy bóng.

#### 3. Tiêu chí kiểm định (Verification Metrics)
* Quả bóng di chuyển về đích **hoàn toàn bằng lực đẩy vật lý (Normal & Friction Forces $> 0$)**.
* Không có bất kỳ dòng lệnh nào can thiệp trực tiếp vào mảng trạng thái `qpos`/`qvel` của vật thể trong suốt quá trình suy luận.

---

### TASK 3: KIẾN TRÚC PHÂN CẤP ĐIỀU PHỐI KỸ NĂNG (HIERARCHICAL COORDINATION & DYNAMIC ACTION BLENDING)

* **Mục tiêu**: Xóa bỏ máy trạng thái hữu hạn 7-state viết cứng (Hardcoded FSM) và hiện tượng gián đoạn hành động khi tráo đổi ONNX cứng (Hard-Swap); đảm bảo tính liên tục của hành động $\dot{\mathbf{u}} \in L_2$.

#### 1. Cơ sở lý thuyết & Động lực học cơ cấu chấp hành
* Khi tráo đổi mạng nơ-ron tức thời giữa `walk` và `pick`, bước nhảy hành động $\Delta \mathbf{a} = \|\mathbf{a}_{t^+} - \mathbf{a}_{t^-}\|$ kích hoạt hiện tượng xung điện áp trong mô hình BAM:
  $$I_{transient} = \frac{V_{in} - k_e \dot{q}}{R} + \frac{k_p \Delta a}{R} \gg I_{max}$$
  gây sụt áp giả lập dưới $6.0\text{V}$, dẫn đến ngắt bảo vệ động cơ.
* **Cơ chế hòa trộn hành động động học (Dynamic Action Blending - Peng et al., 2021)**:
  Thay vì tráo đổi tức thời, ta áp dụng bộ lọc hòa trộn bậc 1 với thời hằng chuyển pha $\tau = 0.25\text{ s}$ (12 bước điều khiển tại 50 Hz):
  $$\mathbf{u}_t = (1 - \gamma_t) \mathbf{a}_{\pi_A}(s_t) + \gamma_t \mathbf{a}_{\pi_B}(s_t), \quad \gamma_t = \min\left(1.0, \frac{t - t_{switch}}{\tau}\right)$$
  Bảo đảm tính khả vi liên tục $C^0$ và hạn chế đạo hàm giật $\ddot{\mathbf{q}}$, triệt tiêu hoàn toàn xung điện áp.
* **Bộ điều phối kỹ năng mức cao (High-Level Policy Orchestrator - 5 Hz)**:
  Một bộ điều phối mức cao (Meta-Policy hoặc Potential Flow Manager) liên tục phát ra vector chỉ định mục tiêu $c_t$ xuống bộ điều khiển vận động mức thấp (50 Hz), biến toàn bộ quá trình thành một luồng điều khiển liên tục thống nhất.

#### 2. Kế hoạch mã nguồn
* Xây dựng module [`src/mjlab_microduck/tasks/hierarchical_controller.py`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/hierarchical_controller.py):
  - Lớp `SkillBlender`: Quản lý chuyển tiếp hành động giữa các manifold trạng thái.
  - Lớp `HierarchicalRunner`: Điều phối vòng lặp điều khiển hai tầng (5 Hz High-Level / 50 Hz Low-Level).
* Cập nhật [`scripts/infer_fetch.py`](file:///home/dhp01/microduck_rl/scripts/infer_fetch.py) để sử dụng `SkillBlender` thay cho FSM rời rạc.

---

### TASK 4: NÂNG CẤP KỸ NĂNG VƯỢT BẬC THANG LÊN TỶ LỆ HÌNH HỌC CHUẨN (> 2.5 - 3.5 CM)

* **Mục tiêu**: Nâng độ cao bậc thang tối đa từ mức vi mô $1.5\text{ cm}$ (chỉ 6% chiều cao robot) lên mức chuẩn thách thức học thuật **$2.5 - 3.5\text{ cm}$** (**10–14% chiều cao robot**), yêu cầu chiến lược nhấc chân và lắc hông thực thụ (Siekmann et al., RSS 2021).

#### 1. Động học hình học & Khả thi vật lý (Kinematic Feasibility)
* Chiều dài đùi: $l_1 = 7.5\text{ cm}$, chiều dài cẳng chân: $l_2 = 7.5\text{ cm}$.
* Góc gập gối tối đa của XL330: $\theta_{knee} \approx 90^\circ$.
* Độ nâng bàn chân tối đa theo phương thẳng đứng:
  $$\Delta z_{foot, max} = (l_1 + l_2) - \sqrt{l_1^2 + l_2^2 - 2 l_1 l_2 \cos(180^\circ - \theta_{knee})} \approx 4.4\text{ cm}$$
  Do đó, ngưỡng bậc thang $3.5\text{ cm}$ là **hoàn toàn khả thi về mặt động học**, nhưng đòi hỏi mạng nơ-ron phải bù trừ nghiêng hông (Pelvic roll) để không bị lật thân khi chân lăng co lên mức cực đại.

#### 2. Kế hoạch nâng cấp Curriculum (Triển khai sau khi PID 4381 hoàn tất)
* Cập nhật [`src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`](file:///home/dhp01/microduck_rl/src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py):
  ```python
  # Nâng trần bậc thang lên mức chuẩn bipedal
  STAIR_STEP_HEIGHT_MIN = 0.015   # 1.5 cm
  STAIR_STEP_HEIGHT_MAX = 0.035   # 3.5 cm (14% chiều cao robot)
  
  # Curriculum 4 giai đoạn tiến tiến
  CLEARANCE_TARGET_STAGES = [
      {"step": 0, "target_height": 0.035},
      {"step": 500 * NUM_STEPS_PER_ENV, "target_height": 0.045},
      {"step": 1000 * NUM_STEPS_PER_ENV, "target_height": 0.055},
      {"step": 1500 * NUM_STEPS_PER_ENV, "target_height": 0.065},
  ]
  ```
* Bổ sung hàm thưởng bù nghiêng hông (Pelvic Roll Stabilization Reward) để giữ trọng tâm nằm vững trên đa giác nâng đỡ của chân trụ khi chân kia nhấc cao $3.5\text{ cm}$.

---

### TASK 5: THIẾT LẬP HỆ THỐNG THỰC NGHIỆM ĐỐI CHỨNG (ABLATION STUDY MATRIX) & TIÊU CHÍ ĐÁNH GIÁ

* **Mục tiêu**: Xây dựng ma trận đánh giá đối chứng toàn diện theo chuẩn CoRL/RSS, chứng minh vai trò lượng hóa của từng thành phần kỹ thuật.

#### Ma trận Thí nghiệm Đối chứng Bắt buộc (Ablation Matrix):

```
+───────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    MA TRẬN NGHIÊN CỨU ĐỐI CHỨNG (ABLATION)                            |
+───────────────+────────────────+──────────────────+──────────────────+────────────────+───────────────+
| Cấu hình      | Actuator Model | Backlash Model   | Heading Control  | Thao tác Bóng  | Kỳ vọng       |
+───────────────+────────────────+──────────────────+──────────────────+────────────────+───────────────+
| M0: Baseline  | MuJoCo PD      | Không            | Hở vòng (Open)   | Gán tọa độ     | Dễ ngã, trôi  |
| M1: BAM       | BAM M6         | Không            | Hở vòng (Open)   | Gán tọa độ     | Giảm rung giật|
| M2: Backlash  | BAM M6         | Có (+/- 1.0 deg) | Hở vòng (Open)   | Gán tọa độ     | Khớp rơ ổn    |
| M3: Wrapper   | BAM M6         | Có (+/- 1.0 deg) | P-Controller     | Gán tọa độ     | Bị hội đồng trừ|
| M4: Ours (Đề) | BAM M6         | Có (+/- 1.0 deg) | Nội sinh POMDP   | Tiếp xúc thuần | ĐẠT CHUẨN TOP |
+───────────────+────────────────+──────────────────+──────────────────+────────────────+───────────────+
```

* **Xây dựng kịch bản kiểm thử tự động**:
  Viết script [`scripts/run_scientific_ablations.py`](file:///home/dhp01/microduck_rl/scripts/run_scientific_ablations.py) thực thi 100 lần thử ngẫu nhiên (100 random seeds) cho mỗi cấu hình, tự động xuất bảng số liệu LaTeX và đồ thị Box-plot về:
  1. Tỷ lệ thành công vượt chướng ngại vật (Success Rate %).
  2. Độ lệch hướng tích lũy trung bình sau 10 giây (Mean Yaw Drift Error).
  3. Tổng công suất tiêu thụ của 14 động cơ (Energy Cost of Transport - CoT).
  4. Lực va đập đỉnh lên bánh răng servo (Peak Impact Force on Servos).

---

## 4. LỘ TRÌNH TRIỂN KHAI THỰC TẾ (EXECUTION TIMELINE & OPERATIONAL ROADMAP)

```
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │                               LỘ TRÌNH THỰC HIỆN                                  │
  └───────────────────────────────────────────────────────────────────────────────────┘
                                            │
  [GIAI ĐOẠN 0: BẢO VỆ TIẾN TRÌNH]          ▼  (Thời gian: ~25-35 phút tới)
  - Giữ nguyên microduck_stairs_env_cfg.py
  - Giám sát PID 4381 đến khi xuất model_600.pt thành công
  - Chuẩn bị sẵn các module độc lập: Task 1 (Heading POMDP) & Task 2 (Contact Fetch)
                                            │
  [GIAI ĐOẠN 1: ĐẠI TU KHÓA HƯỚNG NỘI SINH] ▼  (Ngay sau khi GPU giải phóng)
  - Cập nhật microduck_velocity_env_cfg.py & mdp.py
  - Gỡ bỏ P-controller và EMA khỏi infer_policy.py
  - Chạy smoke-test (64 envs, 5 iters) xác thực tính hội tụ
                                            │
  [GIAI ĐOẠN 2: XÓA TELEPORT & TIẾP XÚC THỰC]▼
  - Sửa đổi scene_fetch.xml & microduck_fetch_env_cfg.py
  - Xóa bỏ triệt để qpos/qvel overwrite trong infer_fetch.py
  - Huấn luyện kỹ năng dẫn bóng tiếp xúc động (Dynamic Dribbling)
                                            │
  [GIAI ĐOẠN 3: NÂNG CẤP BẬC THANG 3.5 CM]  ▼
  - Cập nhật microduck_stairs_env_cfg.py với STAIR_STEP_HEIGHT_MAX = 0.035
  - Kích hoạt curriculum 4 giai đoạn & Pelvic roll compensation
                                            │
  [GIAI ĐOẠN 4: BENCHMARK & BÁO CÁO KHOA HỌC]▼
  - Chạy toàn bộ ma trận Ablation 100 seeds qua run_scientific_ablations.py
  - Xuất bảng kết quả và hoàn thiện bản thảo luận án tiến sĩ
```

---
*Bản đề cương này đóng vai trò là kim chỉ nam kỹ thuật và tài liệu gốc để nâng cấp toàn diện dự án `microduck_rl` thành một công trình nghiên cứu khoa học chuẩn mực quốc tế.*

