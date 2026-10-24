# Hệ thống Multi Mini Model — Microduck trên RTX

> Trạng thái: bản hoạt động đầu tiên (2026-10-24), nhánh `RTX`.
> Toàn bộ số liệu trong tài liệu này là kết quả đo thực nghiệm trên chính máy
> này (RTX 4060 Laptop 8 GB), không phải số liệu mục tiêu.

## 1. Ý tưởng

Thay vì một model lớn làm mọi việc, robot chạy **một bộ nhiều mini model** —
mỗi hành vi một model nhỏ chuyên biệt:

| Nhóm | Model | Nhiệm vụ |
|---|---|---|
| Di chuyển | `walk` | Đi bộ theo lệnh tốc độ |
| Di chuyển | `stairs` | Leo cầu thang bậc thấp |
| Phục hồi | `stand` | Đứng ổn định (velstand) |
| Phục hồi | `standup` | Tự đứng dậy khi ngã |
| Phục hồi | `sitstand` | Ngồi ↔ đứng theo lệnh |
| Thao tác | `ground_pick` | Khom nhặt đồ dưới đất |
| Thao tác | `fetch` | Đi lấy bóng |
| Biểu diễn | `roulade` | Lộn trước |
| Biểu diễn | `ball_kick` | Sút bóng bằng chân phải |

Tất cả model chia sẻ **một hợp đồng quan sát 61D** (48 proprioception +
13D lệnh `[twist(3), head_pose(4), body_pose(6)]`) nên runtime trên robot có
thể **hot-swap** giữa các model bằng cách thay file ONNX giữa chừng — đúng cơ
chế mà `scripts/infer_policy.py` đã rehearse từ trước.

## 2. Pipeline 3 tầng

Mỗi model đi qua ba tầng, tất cả nằm trong repo:

```
[1] HUẤN LUYỆN          [2] DIGITAL TWIN           [3] TRIỂN KHAI (rehearsal)
PPO 1024 envs            CPU MuJoCo + BAM M6        ONNX normalizer-baked
MuJoCo Warp trên GPU     chạy lại policy ONNX       + manifest schema-2
~5-8 h / 6000 iters      GIF + chỉ số + kết luận    (`uv run publish`)
```

Tầng 2 là điểm mới: policy ONNX được nạp vào mô phỏng CPU MuJoCo với **động cơ
BAM M6 y hệt cấu hình lúc huấn luyện**, chạy rollout headless, xuất GIF và chỉ
số đo được (quãng đường, độ cao, độ nghiêng, dao động z, góc lộn, khoảng cách
bóng di chuyển). Mỗi loại động tác có tiêu chí đỗ riêng được đo — không chấm
bằng mắt.

## 3. Trạng thái hiện tại

| Model | Train (iters) | ONNX | Digital twin | Kết luận đo được |
|---|---|---|---|---|
| walk | 5999/6000 ✓ | ✓ | ✓ | đi 1.63 m / 8 s, nghiêng 17° |
| stand | 5999/6000 ✓ | ✓ | ✓ | dao động z σ=0.0 mm, nghiêng ≤1° |
| sitstand | 3999/4000 ✓ | ✓ | ✓ | ngồi z=0.063 m → đứng lại 0.117 m |
| ground_pick | 3499/3000 ✓ | ✓ | ✓ | khom tới z=0.091 m, đứng lại |
| ball_kick | 2999/3000 ✓ | ✓ | ✓ | bóng bay 1.55 m |
| roulade | 3499/3500 ✓ | ✓ | ✓ | lộn 358° quanh trục ngang |
| stairs | 7500/6000 ✓ | ✓ | ✗ | tốt nhất 0.51 m vào flight 1 rồi ngã |
| standup | 10003/5000 ✓ | ✓ | ✗ | từ SIT chưa đứng dậy trong twin |
| fetch | 999/1000 ✓ | ✓ | ✗ | đứng im ở mọi lệnh vx |
| spin | đang train (8000) | — | — | chạy nền, tự export+twin xong |

Nhóm thí nghiệm bánh xe (`rollers`, `swizzle`, `spin` là robot rollers) nằm
ngoài bộ deploy, hiển thị riêng trên dashboard.

**Về 3 model chưa đạt twin** — trung thực ghi nhận, không xoá:

- `stairs`: harness CPU lịch sử (`docs_and_logs/microduck_stairs_climb_trajectory.csv`)
  từng leo trọn course lên đỉnh, nhưng với checkpoint khác các ONNX hiện có.
  Harness tinh chỉnh `scripts/test_stairs_frame_by_frame.py` (settle + reflex
  foot-leveling) với các ONNX hiện tại cũng chỉ đạt tối đa 0.23 m. Cần eval
  lại các checkpoint `.pt` (`scripts/eval_all_stairs_checkpoints.py`) và
  export lại đúng bản tốt. Twin đã tái tạo đúng công thức settle+reflex và
  giữ nó làm mặc định.
- `standup`: policy train với spawn nằm sấp/ngửa/ngồi + noise; twin chỉ mới
  thử spawn từ keyframe SIT. Cần tái tạo đúng phân phối spawn lúc train.
- `fetch`: 1000 iters là đủ theo cfg nhưng gait quá yếu (đứng im ở mọi vx).
  Nên warm-start từ walk expert hoặc kéo dài train.

## 4. Sử dụng

```bash
# Quét matrix model → logs/rtx/matrix.json
uv run rtx-pipeline scan

# Smoke test 64 envs × 5 iters (luôn chạy trước train dài)
uv run rtx-pipeline smoke <key>

# Train đầy đủ; --chain tự export ONNX + chạy digital twin sau khi xong
uv run rtx-pipeline train <key> --chain
uv run rtx-pipeline train <key> --resume          # nối tiếp checkpoint tốt nhất

# Export ONNX (bake normalizer) / chạy digital twin
uv run rtx-pipeline export <key>
uv run rtx-pipeline demo <key> [--duration S]

# Train tuần tự các model còn thiếu — tự chờ GPU rảnh
uv run rtx-pipeline queue [keys...]

# Dashboard: matrix + tiến trình train realtime + gallery twin + nút chạy
uv run rtx-dashboard                # http://127.0.0.1:8420

# Tra trạng thái JSON (tiến trình train, tốc độ, ETA, GPU, queue)
uv run rtx-pipeline live
```

`<key>` ∈ `walk | stand | standup | sitstand | ground_pick | stairs |
roulade | ball_kick | fetch | spin`.

Dashboard là stdlib thuần (không thêm dependency); các hành động cần GPU chỉ
cho phép một tiến trình mỗi lúc, trả lỗi 409 kèm PID nếu bận.

## 5. Bài học kỹ thuật (đã khoá bằng test)

Các bất biến dưới đây được khoá trong `tests/test_rtx_package.py` sau khi từng
gây kết quả twin sai:

1. **Lệnh vx nhỏ với policy đi bộ cũ = đứng yên.** Các run velocity trước
   đây học "vx≈0.10 = đứng" (do zero-command prob lúc train). Twin phải gửi
   vx=0.35 thì policy mới đi (đo: 0.10 → 0.01 m, 0.35 → 1.2 m/8 s).
2. **Obs builder phải đọc sensor `imu_ang_vel` + xquat `trunk_base`** (như
   `scripts/infer_stairs.py`), và projected gravity dùng công thức chuẩn
   legged-gym. Bản quaternion-conjugate thủ công sai терma gy → obs lệch lúc
   thân nghiêng → roulade không hoàn thành vòng lộn (358° → chỉ 73°).
3. **Leo cầu thang cần settle 15 bước giữ HOME + reflex foot-leveling**
   (nâng bàn chân đang vung + lệnh body-pitch −5° khi tới gần bậc) — công thức
   của `scripts/test_stairs_frame_by_frame.py`. Không có settle, mọi checkpoint
   ngã trong 0.2 s đầu.
4. **sitstand phản hồi lệnh theo lịch trình ngồi-trước**: flag=1 ngay từ đầu
   (2.5 s) rồi về 0 — policy ngồi rồi đứng lại. Lịch trình đứng-hold trước
   làm nó bỏ qua flag trong twin.
5. **Ánh xạ task_id → family phải gộp biến thể** `-Rough-`/`-Backlash-`,
   nếu không live status mất iteration/ETA của tiến trình đang chạy.
6. **GPU chỉ chạy một train**: `queue` chờ tiến trình `bin/train Mjlab-*` sống
   thoát mới mở stage kế tiếp; dashboard chặn hành động trùng (HTTP 409).

## 6. Đường train trên Colab (T4, phiên 5 h 30)

Notebook `notebooks/train_spin_colab.ipynb` train model còn thiếu (spin) trên
Google Colab T4 15 GB, phiên 5 h 30:

- Smoke test 64 envs × 5 iters bắt buộc trước train (quy tắc repo).
- Train theo **chunk 500 iters + resume**: Colab ngắt giữa chừng chỉ mất tối
  đa 1 chunk; chạy lại notebook tự khôi phục checkpoint từ Drive.
- Sau mỗi chunk chép checkpoint/logs lên `Drive/Microduck_RTX/spin/`.
- Khi hết ngân sách (còn 30 phút): tự dừng train → export ONNX baked → chạy
  digital twin spin ngay trên Colab → lưu full logs + model + GIF về Drive và
  tải ONNX về máy.

Yêu cầu: nhánh `RTX` đã push lên GitHub (`git push origin RTX`) vì notebook
clone từ đó.

## 7. Cấu trúc code

```
src/mjlab_microduck/rtx/
├── registry.py      # bảng model family (task, target iters, scene, adapter)
├── matrix.py        # quét checkpoint/ONNX/twin → logs/rtx/matrix.json
├── digital_twin.py  # rollout ONNX trên CPU MuJoCo + BAM → GIF + metrics
├── pipeline.py      # CLI: scan/smoke/train/export/demo/queue/live
├── dashboard.py     # web server stdlib + API
└── static/          # UI (index.html, style.css, app.js)
```

Scene mới: `src/mjlab_microduck/robot/microduck/scene_stairs_allcollisions.xml`
(scene_stairs với robot allcollisions — đúng collision model lúc train stairs;
hiện twin vẫn dùng scene groundcontact theo harness đã chứng minh, scene này
để dành cho việc eval lại checkpoint).
