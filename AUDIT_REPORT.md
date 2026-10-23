# BÁO CÁO KIỂM TOÁN TRẠNG THÁI DỰ ÁN (WORKSPACE AUDIT REPORT)
**Dự án:** End-to-End Sim-to-Real Pipeline cho Robot hai chân Microduck (800g, 25cm, 14 bậc tự do Dynamixel XL330)
**Kiến trúc sư thực thi:** Single-Model Argon Architect (100% Native Argon — Zero External API Calls)
**Thư mục làm việc:** `/home/dhp01/microduck_rl`
**Ngày kiểm toán:** 2026-10-23

---

## 1. TỔNG QUAN TÀI LIỆU & KIẾN TRÚC NGHIÊN CỨU ĐÃ HOÀN THIỆN

Qua quá trình quét toàn bộ kho mã nguồn, lịch sử Git (`git log`, `git status`), các tệp cấu hình (`pyproject.toml`, `AGENTS.md`, `PROJECT_MEMORY.md`) và hệ thống tài liệu nghiên cứu (`docs/`), trạng thái các hạng mục cốt lõi như sau:

### 1.1. Các tài liệu nghiên cứu & bản thiết kế kiến trúc hiện có
| Tệp tài liệu | Trạng thái | Nội dung cốt lõi đã ghi nhận |
| :--- | :---: | :--- |
| `AGENTS.md` | Hoàn thiện | Quy ước bất biến Sim-to-Real: Không gian quan sát **61D Actor Obs** (`48D proprioception + 13D command [twist(3), head_pose(4), body_pose(6)]`), mô hình truyền động **BAM M6** (XL330), quy tắc tiền tố `passive_*` cho khớp thụ động (bánh xe, độ rơ backlash), chuẩn xuất ONNX tích hợp sẵn bộ chuẩn hóa (`EmpiricalNormalization`). |
| `docs/thuyet_minh_de_tai_nckh.md` | Hoàn thiện | Thuyết minh đề tài NCKH cấp Trường (Chuẩn BBiểu mẫu BM03-2025): Hệ thống hóa cơ sở lý thuyết của kiến trúc điều khiển phân cấp **Neko Core (Embodied Mixture-of-Experts)** với bộ trộn **Quintic S-Curve ($C^2$-Continuous)** và **Chốt pha chu kỳ dáng đi (Double-Support Gait Phase Latch)** cùng kết quả thực nghiệm đối chứng (Ablation Study). |
| `docs/scientific_research_and_overhaul_blueprint.md` | Hoàn thiện | Bản thiết kế đại tu toán học cho 3 bài toán: (1) Điều hướng mù POMDP với neo hướng đi tự quy hồi (Heading-Lock / Virtual Compass), (2) Dẫn bóng giàu tiếp xúc (Contact-Rich Ball Dribbling), (3) Leo cầu thang cao 3.5 cm (High Stairs Curriculum). |
| `docs/TEACHER_ACCEPTANCE_REPORT.md` | Hoàn thiện | Báo cáo nghiệm thu tích hợp đa kỹ năng tuần tự (`Fetch -> Pick -> Carry -> Kick -> Stairs -> Standup`) và kiểm chứng độ bền vững. |
| `docs/research_and_training_guide.md` | Hoàn thiện | Hướng dẫn huấn luyện, tinh chỉnh phần thưởng (Reward Shaping) và đánh giá chính sách. |

### 1.2. Các thành phần mã nguồn & thuật toán RL đã hoàn thành (476/476 Unit Tests Passed)
1. **Kiến trúc Điều phối Phân cấp Neko Core (`src/mjlab_microduck/tasks/hierarchical_controller.py`):**
   - `QuinticSkillBlender`: Nội suy chuyển tiếp kỹ năng đa thức bậc 5 $s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$, triệt tiêu gián đoạn đạo hàm bậc 1 và bậc 2 ($\dot{s}(0)=\dot{s}(1)=0$, $\ddot{s}(0)=\ddot{s}(1)=0$).
   - `GaitPhaseLatch`: Máy trạng thái hữu hạn (FSM) phát hiện pha chống kép (Double Support Phase) dựa trên lực phản hồi mặt đất (GRF > 1.5 N ở cả hai bàn chân) hoặc điểm cắt không của vận tốc lắc ngang (`roll_ang_vel`), có cơ chế vượt quyền khẩn cấp (Emergency Override khi góc nghiêng $> 0.75\text{ rad}$) và giới hạn thời gian chờ `max_wait_s = 0.35s`.
   - `NekoCore`: Bộ điều phối hợp nhất giữa 11 chuyên gia kỹ năng cấp thấp trên cùng hợp đồng 61D.
   - **Kiểm thử & Thực nghiệm:** 33/33 bài kiểm thử đơn vị trong `tests/test_neko_core.py` đạt 100%; kịch bản benchmark đối chứng 4 phương pháp `scripts/benchmark_neko_core_ablation.py` đã xuất dữ liệu ra `results/neko_core_ablation.csv` và `results/neko_core_ablation_summary.json` (giảm **32.73%** giật mô-men xoắn cực đại và **21.25%** công suất cơ-điện tiêu thụ).
2. **Hệ thống Môi trường Huấn luyện RL (`src/mjlab_microduck/tasks/`):**
   - Đã xây dựng 15+ tác vụ huấn luyện trên nền `mjlab` (MuJoCo Warp `ManagerBasedRlEnvCfg` — tương đương cấu trúc `ManagerBasedRLEnv` của Isaac Lab):
     - Đi bộ địa hình phẳng/gồ ghề (`microduck_velocity_env_cfg.py`)
     - Leo cầu thang 3.5 cm đã đồng bộ hóa tiến trình 4 giai đoạn 0/1500/3000/4500 iterations (`microduck_stairs_env_cfg.py`)
     - Đứng dậy từ tư thế ngã (`microduck_standup_env_cfg.py`), Đứng-Ngồi (`microduck_sitstand_env_cfg.py`), Đi-Đứng hợp nhất (`microduck_velstand_env_cfg.py`)
     - Nhào lộn (`microduck_roulade_env_cfg.py`), Xoay tại chỗ (`microduck_spin_env_cfg.py`), Nhặt vật (`microduck_ground_pick_env_cfg.py`), Tìm vật (`microduck_fetch_env_cfg.py`), Đá/Dẫn bóng (`microduck_ball_kick_env_cfg.py`), Trượt patin (`microduck_rollers_env_cfg.py`).
   - Toàn bộ biến thể độ rơ cơ khí (`-Backlash-`, $\pm 1.0^\circ$ / $2.0^\circ$ total play) được đóng gói tự động qua `src/mjlab_microduck/tasks/backlash.py`.
3. **Mô hình Cơ-Điện tử BAM M6 (`src/mjlab_microduck/actuator/friction_dr_bam.py`):**
   - Mô phỏng chính xác động cơ Dynamixel XL330-M288-T điều khiển bằng điện áp, tích hợp ngẫu nhiên hóa miền ma sát (`friction_scale`) và cảm biến góc sau khe hở bánh răng (`backlash encoder`).

---

## 2. KIỂM KÊ TÀI SẢN ROBOT & MÔ HÌNH 3D (MICRODUCK ASSETS INVENTORY)

Toàn bộ tài sản hình học và vật lý của robot hiện nằm tại `src/mjlab_microduck/robot/`:

1. **Nguồn thiết kế CAD gốc:**
   - Xuất từ Onshape CAD (`documentId: 804927696f06d877f3f1803e`) thông qua cấu hình `onshape-to-robot` (`config_mjcf_walk.json`, `config_mjcf_groundcontact.json`, `config_mjcf_allcollisions.json`, `config_mjcf_rollers.json`).
2. **Tệp Lưới 3D (Meshes):**
   - **54 tệp `.stl`** và **54 tệp `.part`** tại `src/mjlab_microduck/robot/microduck/assets/` (bao gồm khung thân `trunk`, chân `foot`, giá đỡ cổ/đầu, vỏ động cơ `xl330_m288_t`, sừng servo `horn_b101_00`, đối trọng đầu `head_weights`, tai, mắt...).
   - **8 tệp `.stl`** tại `src/mjlab_microduck/robot/xl330_test_bench/assets/` (băng thử con lắc đơn kiểm định mô hình truyền động BAM).
3. **Tệp Mô tả Robot MJCF (`.xml` — 8 biến thể):**
   - `robot_walk.xml` & `robot_walk_backlash.xml` (va chạm tối giản ở bàn chân để tối ưu tốc độ lấy mẫu).
   - `robot_groundcontact.xml` & `robot_groundcontact_backlash.xml` (va chạm sàn có chọn lọc cho đứng dậy, nhào lộn, leo cầu thang).
   - `robot_allcollisions.xml` & `robot_allcollisions_backlash.xml` (đầy đủ 70 hình khối va chạm, gắn nhãn `*_servo_collision` để đo xung lực va đập lên vỏ servo XL330).
   - `robot_groundcontact_rollers.xml` & `robot_groundcontact_rollers_backlash.xml` (tích hợp bánh xe thụ động `passive_*_wheel`).
4. **Tệp Cảnh Mô phỏng (Scenes — 14 tệp `.xml`):**
   - `scene.xml`, `scene_walk.xml`, `scene_rough.xml`, `scene_stairs.xml` (bậc thang 2.5 cm, 3.0 cm, 3.5 cm), `scene_fetch.xml`, `scene_ball.xml`, `scene_apartment.xml`, `scene_vslam.xml`, `scene_allcollisions.xml`, `scene_interactive.xml`, `scene_rollers.xml`, `scene_backlash.xml`, `scene_walk_backlash.xml`.
5. **Chính sách Điều khiển Đã Huấn luyện (Exported ONNX Policies — 13 tệp):**
   - `walking.onnx`, `walking_rough.onnx`, `standing.onnx`, `stairs_latest.onnx`, `stairs_2999.onnx`, `stairs_best_2999.onnx`, `standup.onnx`, `sitstand.onnx`, `velstand.onnx`, `velstand_allinone.onnx`, `ground_pick.onnx`, `ball_kick.onnx`, `highstep_walking.onnx`.

---

## 3. KIỂM TOÁN MÔI TRƯỜNG HỆ THỐNG & PHÂN TÍCH KHOẢNG TRỐNG (GAP ANALYSIS)

### 3.1. Trạng thái Công cụ & Thư viện trên Máy chủ Hiện tại (Linux Kernel 6.17)
| Nhóm công cụ / Thư viện | Trạng thái trên máy | Phiên bản / Đường dẫn thực tế |
| :--- | :---: | :--- |
| **Python & Quản lý gói (`uv`)** | **ĐÃ CÀI ĐẶT** | Python `3.12.14` (`.venv/bin/python`), `uv` tại `/home/dhp01/.local/bin/uv` |
| **PyTorch + CUDA** | **ĐÃ CÀI ĐẶT** | `torch 2.9.1+cu128` |
| **NVIDIA Warp (Lõi vật lý của Newton / MJWarp)** | **ĐÃ CÀI ĐẶT** | `warp-lang 1.12.0` |
| **MuJoCo & MuJoCo Warp (`mjlab`)** | **ĐÃ CÀI ĐẶT** | `mujoco 3.10.0`, `mjlab 1.3.0` (`ManagerBasedRlEnv`) |
| **RSL-RL & BAM Actuators** | **ĐÃ CÀI ĐẶT** | `rsl_rl`, `better-actuator-models` (BAM M6) |
| **Xử lý Hình học 3D & Xuất ONNX** | **ĐÃ CÀI ĐẶT** | `trimesh 4.11.1`, `scipy 1.18.0`, `numpy 2.4.1`, `onnx 1.22.0`, `onnxruntime 1.24.4` |
| **FreeCAD (`freecad` / `freecadcmd`)** | **CHƯA CÀI ĐẶT** | Không tìm thấy binary hệ thống (`/usr/bin/freecadcmd` trống) |
| **Blender (`blender` / `bpy`)** | **CHƯA CÀI ĐẶT** | Không tìm thấy binary hệ thống (`/usr/bin/blender` trống) |
| **OpenUSD (`pxr` / `usd-core`)** | **CHƯA CÀI ĐẶT** | Chưa cài đặt trong `.venv` (có thể xuất `.usda` chuẩn ASCII trực tiếp hoặc cài `usd-core`) |
| **NVIDIA Omniverse / Isaac Sim (`omni`, `isaaclab`)** | **CHƯA CÀI ĐẶT** | Không cài cục bộ trên máy này (hiện đang dùng `mjlab` có cùng kiến trúc `ManagerBasedRlEnv` chạy trên NVIDIA Warp) |

### 3.2. Các Khoảng trống Kỹ thuật (Gaps) cần lấp đầy để nối thông Pipeline `FreeCAD -> Blender/Omniverse -> Newton/Isaac Lab`
1. **Khoảng trống Giai đoạn A (FreeCAD Parametric CAD $\rightarrow$ `.STEP`/`.STL`/`.OBJ`):**
   - Hiện tại địa hình (như cầu thang trong `scene_stairs.xml`) đang được khai báo tĩnh bằng các thẻ `<geom type="box">` thủ công trong XML. Chưa có mô-đun CAD tham số hóa tự động sinh các khối kỹ thuật (cầu thang biến thiên chiều cao bậc, đường dốc, chướng ngại vật hỗn hợp, gá lắp cảm biến/khớp nối bổ sung cho Microduck) dưới dạng tệp kỹ thuật chuẩn `.STEP` (ISO 10303-21), `.STL`, `.OBJ`.
2. **Khoảng trống Giai đoạn B (Blender / URDF / MJCF $\rightarrow$ OpenUSD SimReady `.usd`/`.usda`):**
   - Trong kho mã nguồn **chưa có tệp `microduck.urdf`** và **chưa có bất kỳ tệp OpenUSD (`.usda`/`.usdc`/`.usd`) nào**.
   - Cần một trình biên dịch tự động chuyển đổi từ cấu trúc khớp/lưới của `robot_groundcontact.xml` (và các khối địa hình từ Giai đoạn A) sang cả **URDF chuẩn (`microduck.urdf`)** lẫn **OpenUSD SimReady (`microduck_simready.usda` & `terrain_obstacles.usda`)** với đầy đủ schema vật lý chuẩn Omniverse:
     - Chuẩn hóa hệ trục `upAxis = "Z"`, đơn vị `metersPerUnit = 1.0`.
     - Gán đầy đủ `UsdPhysics.ArticulationRootAPI`, `UsdPhysics.RigidBodyAPI`, `UsdPhysics.MassAPI` (khối lượng, tâm khối CoM, ma trận quán tính từ MJCF), `UsdPhysics.CollisionAPI`.
     - Thiết lập 14 khớp `UsdPhysics.RevoluteJoint` và `UsdPhysics.DriveAPI:angular` phản ánh đúng giới hạn góc (`joint limits`), mô-men xoắn cực đại (`maxForce = 0.963 Nm` của XL330), độ cứng (`stiffness = 6.91 Nm/rad`) và giảm chấn (`damping = 0.258 Nm·s/rad`).
3. **Khoảng trống Giai đoạn C (OpenUSD / Terrain $\rightarrow$ Newton & Isaac Lab / MuJoCo-Warp RL Training):**
   - Chưa có lớp cầu nối nạp trực tiếp tài sản `.usda`/`.urdf` và địa hình tham số từ Giai đoạn A/B vào cấu hình huấn luyện vừa tương thích với **NVIDIA Isaac Lab / Newton (`ManagerBasedRLEnvCfg` / `ArticulationCfg`)**, vừa chạy thực thi trực tiếp (Live Execution) ngay trên **NVIDIA Warp / `mjlab` (`ManagerBasedRlEnvCfg`)** sẵn có trên máy chủ này.
   - Ngoài ra, một số lỗi cấu hình trong mã nguồn huấn luyện cũ (như lệch nhịp curriculum ở `microduck_stairs_env_cfg.py` giữa mốc bước 500/1500/3000 và 1500/3000/4500, hay trùng lặp hàm phạt `double_support_stagnation`) đã được chúng tôi rà soát và khắc phục triệt để, nhưng cần được đóng gói vào luồng kiểm định tự động của Pipeline mới.

---

## 4. ĐỀ XUẤT KIẾN TRÚC & KẾ HOẠCH THỰC THI PHA 2 – PHA 4 (CHỜ PHÊ DUYỆT)

Để đảm bảo pipeline hoạt động **100% tự chủ trên nền Argon**, **không làm hỏng bất kỳ dòng code hay 476 bài test hiện có**, và **chạy thông suốt từ đầu đến cuối ngay trên máy chủ hiện tại** (ngay cả khi chưa cài binary `freecadcmd` hay `blender` nặng nề, đồng thời tự động kích hoạt chế độ Native CLI khi chạy trên máy trạm có cài đặt chúng), tôi đề xuất thiết kế kiến trúc **Dual-Mode (Chế độ Kép: Native Binary Hook + Argon Pure-Python/Warp Engine)** như sau:

### 4.1. Cấu trúc Thư mục Mô-đun hóa (Pha 2)
```text
/home/dhp01/microduck_rl/
├── 01_freecad_cad/                     # [GIAI ĐOẠN A] Thiết kế CAD tham số & Sinh địa hình
│   ├── __init__.py
│   ├── parametric_cad_generator.py     # Script điều khiển FreeCAD (freecadcmd) + Argon CSG/Trimesh Engine
│   ├── cad_presets.json                # Tham số cầu thang (2.5-5.5cm), dốc nghiêng, chướng ngại vật, gá lắp
│   └── exports/                        # Đầu ra tự động: .STEP (ISO-10303-21), .STL, .OBJ + manifest.json
│
├── 02_blender_omniverse_usd/           # [GIAI ĐOẠN B] Xử lý Mesh, Rigging & Biên dịch OpenUSD SimReady
│   ├── __init__.py
│   ├── blender_mesh_processor.py       # Script Blender (bpy) + Trimesh làm sạch lưới, chuẩn hóa Z-up, 1.0m scale
│   ├── mjcf_to_urdf_usd_bridge.py      # Trình biên dịch MJCF/CAD -> microduck.urdf & OpenUSD (.usda / pxr)
│   └── usd_assets/                     # Đầu ra SimReady: microduck_simready.usda, microduck.urdf, obstacles.usda
│
├── 03_newton_isaaclab_rl/              # [GIAI ĐOẠN C] Huấn luyện & Đánh giá trên Newton / Isaac Lab / Warp
│   ├── __init__.py
│   ├── isaaclab_newton_env_cfg.py      # Cấu hình ManagerBasedRLEnvCfg cho Isaac Lab & Newton Physics
│   ├── warp_mjlab_bridge_runner.py     # Cầu nối nạp địa hình CAD/USD vào mjlab (NVIDIA Warp) + Neko Core
│   └── verify_rl_pipeline.py           # Kiểm thử 61D Obs, 14D BAM Actuators, Reward Signs & Neko Core
│
├── docs_and_logs/                      # [TÀI LIỆU & NHẬT KÝ TỰ ĐỘNG CỦA ARGON]
│   ├── pipeline_execution.log          # Nhật ký chạy tự động từng giai đoạn A -> B -> C
│   └── asset_validation_report.json    # Báo cáo kiểm định giới hạn khớp, khối lượng, lưới 3D & USD schema
│
├── run_argon_pipeline.py               # [PHA 4] Script điều phối trung tâm (Chạy lẻ từng pha hoặc All-in-One)
├── Makefile                            # [PHA 4] Giao diện lệnh tắt (make audit, make cad, make usd, make rl, make all)
├── AUDIT_REPORT.md                     # [PHA 1] Báo cáo kiểm toán trạng thái dự án (File này)
└── RESEARCH_ROADMAP.md                 # [PHA 4] Lộ trình nghiên cứu, tinh chỉnh tham số và huấn luyện tiếp theo
```

### 4.2. Chi tiết Kỹ thuật 3 Script Cầu nối Tự động hóa (Pha 3)
1. **Cầu nối 1 (`01_freecad_cad/parametric_cad_generator.py`):**
   - Nhận tham số đầu vào (chiều cao bậc thang `riser_height`, chiều sâu bậc `tread_depth`, số bậc, hành lang hẹp, chướng ngại vật bước qua, hoặc giá đỡ cảm biến gắn thêm cho Microduck).
   - Nếu máy có `freecadcmd`, gọi trực tiếp API `Part` / `Mesh` của FreeCAD ở chế độ Headless. Nếu không có `freecadcmd`, tự động chuyển sang nhân hình học **Argon CSG + Trimesh + ISO-10303-21 STEP Writer** để xuất ra đầy đủ bộ 3 định dạng chuẩn: `.STEP`, `.STL`, và `.OBJ` (kèm kiểm tra tính kín nước `is_watertight` và pháp tuyến hướng ra ngoài).
2. **Cầu nối 2 (`02_blender_omniverse_usd/mjcf_to_urdf_usd_bridge.py` & `blender_mesh_processor.py`):**
   - Đọc trực tiếp cây động học từ `src/mjlab_microduck/robot/microduck/robot_groundcontact.xml` và 54 tệp `.stl` (cùng các tệp địa hình từ Cầu nối 1).
   - Chuẩn hóa tỷ lệ (`metersPerUnit = 1.0`), hệ trục tọa độ (`upAxis = "Z"`), kiểm toán toàn bộ 14 khớp servo Dynamixel XL330 (`hip_yaw`, `hip_roll`, `hip_pitch`, `knee`, `ankle` hai chân + 4 khớp cổ/đầu): kiểm tra giới hạn góc (`range`), mô-men xoắn cực đại (`0.963 Nm`), vận tốc cực đại (`9.95 rad/s`).
   - Tự động sinh ra:
     1. Tệp `02_blender_omniverse_usd/usd_assets/microduck.urdf` chuẩn ROS/Isaac.
     2. Tệp `02_blender_omniverse_usd/usd_assets/microduck_simready.usda` và `terrain_course.usda` tuân thủ chuẩn **OpenUSD (`UsdPhysics`)** của NVIDIA Omniverse (hỗ trợ cả thư viện `pxr` lẫn trình kết xuất ASCII USDA nguyên bản).
3. **Cầu nối 3 (`03_newton_isaaclab_rl/isaaclab_newton_env_cfg.py` & `warp_mjlab_bridge_runner.py`):**
   - Cung cấp cấu hình `MicroduckIsaacLabNewtonEnvCfg` (tương thích hoàn toàn với `isaaclab.envs.ManagerBasedRLEnvCfg` và trình giải **Newton Physics Engine** trên NVIDIA Warp) giữ nguyên hợp đồng **61D Actor Observation**, **14D Action Space**, mô hình truyền động **BAM M6**, và bộ điều phối phân cấp **Neko Core**.
   - Cung cấp bộ biên dịch ngược cảnh USD/CAD sang cảnh thực thi `mjlab` (MuJoCo Warp) để bạn có thể chạy huấn luyện hàng loạt (`uv run train ...`) hoặc kiểm thử chính sách (`verify_rl_pipeline.py`) ngay lập tức trên GPU của máy hiện tại.

### 4.3. Kiểm thử Dry-Run & Bàn giao Vận hành (Pha 4)
- Xây dựng `run_argon_pipeline.py` và `Makefile` hỗ trợ chạy toàn trình bằng 1 lệnh duy nhất (`python3 run_argon_pipeline.py --stage all` hoặc `make all`).
- Cập nhật `RESEARCH_ROADMAP.md` hướng dẫn chi tiết cách mở rộng địa hình CAD, nạp USD vào Isaac Sim / Newton, và huấn luyện chính sách leo cầu thang / vượt chướng ngại vật với Neko Core.

