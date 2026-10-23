"""RTX Control Center — hệ thống multi mini model cho Microduck.

Ba tầng của pipeline (như một buổi demo sim-to-real hoàn chỉnh, chạy gọn
trên một máy RTX 4060):
  1. HUẤN LUYỆN  — PPO 1024 envs trên GPU (mjlab / MuJoCo Warp)
  2. DIGITAL TWIN — rehearsal CPU MuJoCo (BAM M6 như lúc train) + xuất GIF
  3. TRIỂN KHAI   — ONNX normalizer-baked + manifest schema-2 (rehearsal;
                    chưa có robot thật nên dừng ở bước nạp policy ảo)

Các module:
  registry     — bảng model family (task_id, exp_dir, target, adapter twin)
  matrix       — kiểm kê checkpoint/ONNX/đánh giá → logs/rtx/matrix.json
  digital_twin — chạy policy ONNX headless, ghi GIF + chỉ số
  pipeline     — CLI điều phối smoke/train/export/demo/queue (chờ GPU)
  dashboard    — web UI "RTX Control Center" (stdlib, không thêm dependency)
"""

__all__ = ["registry", "matrix", "digital_twin", "pipeline", "dashboard"]
