"""Bảng đăng ký các mini model của hệ thống multi mini model.

Mỗi family = một model nhỏ chuyên biệt, cùng hợp đồng obs 61D để runtime
hot-swap trên robot thật. `target_iters` lấy theo max_iterations mặc định
của từng env cfg (ground truth khi train), không tự đặt số mới.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = REPO_ROOT / "logs"
RTX_STATE_DIR = LOGS_ROOT / "rtx"
TWIN_DIR = LOGS_ROOT / "digital_twin"

# Nhóm hiển thị trên dashboard
GROUPS = {
    "locomotion": "Di chuyển",
    "recovery": "Phục hồi & tư thế",
    "manipulation": "Thao tác",
    "trick": "Động tác biểu diễn",
    "experimental": "Thí nghiệm (bánh xe — chưa thuộc bộ deploy)",
}


@dataclass(frozen=True)
class ModelFamily:
    key: str
    label: str
    group: str
    task_id: str
    exp_dir: str
    target_iters: int
    twin_kind: str  # adapter trong digital_twin.py
    scene: str  # scene XML cho digital twin (tương đối theo repo root)
    duration_s: float = 4.0
    note: str = ""
    trainable: bool = True
    preferred_onnx: str = ""  # chỉ định ONNX demo khi run tốt nhất (theo iters) không phải bản twin tốt nhất


# kind → cách nạp lệnh của từng loại policy (xem digital_twin.py)
_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        key="walk", label="Đi bộ (velocity)", group="locomotion",
        task_id="Mjlab-Velocity-Flat-MicroDuck", exp_dir="velocity",
        target_iters=6000, twin_kind="walk", scene="src/mjlab_microduck/robot/microduck/scene.xml",
        duration_s=8.0,
        note="Bộ chủ đạo; bản Rough đang train dùng chung thư mục logs/velocity",
    ),
    ModelFamily(
        key="stand", label="Đứng ổn định (velstand)", group="recovery",
        task_id="Mjlab-VelStand-Flat-MicroDuck", exp_dir="velstand",
        target_iters=6000, twin_kind="stand", scene="src/mjlab_microduck/robot/microduck/scene.xml",
        duration_s=4.0,
    ),
    ModelFamily(
        key="standup", label="Tự đứng dậy (standup)", group="recovery",
        task_id="Mjlab-StandUp-Flat-MicroDuck", exp_dir="microduck_stand",
        target_iters=5000, twin_kind="standup", scene="src/mjlab_microduck/robot/microduck/scene_allcollisions.xml",
        duration_s=4.0, preferred_onnx="logs/rsl_rl/microduck_stand/2026-10-06_15-07-06_microduck_stand/2026-10-06_15-07-06_microduck_stand.onnx",
    ),
    ModelFamily(
        key="sitstand", label="Ngồi ↔ đứng (sitstand)", group="recovery",
        task_id="Mjlab-SitStand-Flat-MicroDuck", exp_dir="microduck_sitstand",
        target_iters=4000, twin_kind="sitstand", scene="src/mjlab_microduck/robot/microduck/scene.xml",
        duration_s=6.0,
    ),
    ModelFamily(
        key="ground_pick", label="Nhặt đồ dưới đất", group="manipulation",
        task_id="Mjlab-GroundPick-Flat-MicroDuck", exp_dir="ground_pick",
        target_iters=3000, twin_kind="ground_pick", scene="src/mjlab_microduck/robot/microduck/scene.xml",
        duration_s=3.2,
    ),
    ModelFamily(
        key="stairs", label="Leo cầu thang", group="locomotion",
        task_id="Mjlab-StairsClimb-MicroDuck", exp_dir="stairs_climb",
        target_iters=6000, twin_kind="stairs", scene="src/mjlab_microduck/robot/microduck/scene_stairs.xml",
        duration_s=10.0, preferred_onnx="logs/rsl_rl/stairs_climb/2026-10-17_08-16-27_stairs_climb/2026-10-17_08-16-27_stairs_climb.onnx",
        note="Twin CPU: tốt nhất 0.56 m vào flight 1 (settle+reflex như testbench); lịch sử từng đạt summit với checkpoint khác — cần eval lại .pt",
    ),
    ModelFamily(
        key="roulade", label="Lộn trước (roulade)", group="trick",
        task_id="Mjlab-Roulade-Flat-MicroDuck", exp_dir="microduck_roulade",
        target_iters=3500, twin_kind="roulade", scene="src/mjlab_microduck/robot/microduck/scene.xml",
        duration_s=3.0,
    ),
    ModelFamily(
        key="ball_kick", label="Sút bóng (chân phải)", group="trick",
        task_id="Mjlab-BallKick-Flat-MicroDuck", exp_dir="ball_kick_right",
        target_iters=3000, twin_kind="ball_kick", scene="src/mjlab_microduck/robot/microduck/scene_ball.xml",
        duration_s=3.0,
    ),
    ModelFamily(
        key="fetch", label="Đi lấy bóng (fetch)", group="manipulation",
        task_id="Mjlab-Fetch-Flat-MicroDuck", exp_dir="microduck_fetch",
        target_iters=1000, twin_kind="fetch", scene="src/mjlab_microduck/robot/microduck/scene_fetch.xml",
        duration_s=6.0,
    ),
    ModelFamily(
        key="spin", label="Xoay tại chỗ (spin, bánh xe)", group="experimental",
        task_id="Mjlab-Spin-Flat-MicroDuck", exp_dir="spin",
        target_iters=8000, twin_kind="spin", scene="src/mjlab_microduck/robot/microduck/scene_rollers.xml",
        duration_s=6.0,
        note="Task spin train trên robot ROLLERS — chưa từng train local",
    ),
    # --- Nhóm thí nghiệm: bánh xe / biến thể, chưa thuộc bộ deploy ---
    ModelFamily(
        key="rollers", label="Trượt bánh xe (rollers)", group="experimental",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers", exp_dir="velocity_rollers",
        target_iters=6000, twin_kind="walk", scene="src/mjlab_microduck/robot/microduck/scene_rollers.xml",
        duration_s=5.0, trainable=False, note="Biến thể bánh xe — ngoài phạm vi lần này",
    ),
    ModelFamily(
        key="swizzle", label="Swizzle", group="experimental",
        task_id="Mjlab-Velocity-Swizzle-MicroDuck", exp_dir="velocity_swizzle",
        target_iters=6000, twin_kind="walk", scene="src/mjlab_microduck/robot/microduck/scene_rollers.xml",
        duration_s=5.0, trainable=False, note="Biến thể bánh xe — ngoài phạm vi lần này",
    ),
)

FAMILIES: dict[str, ModelFamily] = {f.key: f for f in _FAMILIES}

# Bộ "robot cần thiết" cho deploy — dùng bởi `rtx-pipeline queue --essential`
ESSENTIAL_KEYS = [k for k, f in FAMILIES.items() if f.trainable]


def scene_path(fam: ModelFamily) -> Path:
    return REPO_ROOT / fam.scene


def exp_dir(fam: ModelFamily) -> Path:
    return LOGS_ROOT / "rsl_rl" / fam.exp_dir
