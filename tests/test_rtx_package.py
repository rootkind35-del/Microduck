"""Regression tests cho gói rtx (RTX Control Center).

CPU-only, không dựng mô phỏng: khóa bất biến registry/matrix/adapter — những
lỗi từng gây lệch kết quả digital twin (key lạ fallback sang kind "walk",
projected gravity sai công thức, ánh xạ task biến thể Rough/Backlash).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.rtx import digital_twin as dt
from mjlab_microduck.rtx import pipeline
from mjlab_microduck.rtx.registry import ESSENTIAL_KEYS, FAMILIES, REPO_ROOT


def test_family_keys_unique_and_essential():
    assert len(FAMILIES) == len(set(FAMILIES))
    # Mọi key thiết yếu đều nằm trong FAMILIES và trainable
    for k in ESSENTIAL_KEYS:
        assert k in FAMILIES
        assert FAMILIES[k].trainable


def test_family_scenes_exist():
    for key, fam in FAMILIES.items():
        assert (REPO_ROOT / fam.scene).is_file(), f"{key}: thiếu scene {fam.scene}"


def test_family_task_ids_resolve_via_family_of_task():
    for key, fam in FAMILIES.items():
        assert pipeline.family_of_task(fam.task_id) == key
    # Biến thể Rough/Backlash phải ánh xạ về đúng family (lỗi thật 2026-10-24:
    # Velocity-Rough không nhận ra family "walk" → live status mất iters/ETA)
    assert pipeline.family_of_task("Mjlab-Velocity-Rough-MicroDuck") == "walk"
    assert pipeline.family_of_task("Mjlab-StandUp-Rough-Backlash-MicroDuck") == "standup"
    assert pipeline.family_of_task("Mjlab-SitStand-Rough-MicroDuck") == "sitstand"
    assert pipeline.family_of_task("Mjlab-Task-Khong-Ton-Tai") is None


def test_kind_adapters_return_13d_commands():
    for kind, spec in dt.KINDS.items():
        c = spec["cmd"](1.0, 4.0)
        assert isinstance(c, np.ndarray) and c.shape == (13,), kind
        assert np.all(np.isfinite(c))


def test_sitstand_schedule_sits_first_then_stands():
    # Lịch trình sit-first (flag=1 → 0) là lịch trình duy nhất policy phản hồi
    # trong twin — demo cũ đứng-hold-trước làm policy bỏ qua flag.
    assert dt._sitstand_cmd(0.0, 6.0)[0] == 1.0
    assert dt._sitstand_cmd(2.4, 6.0)[0] == 1.0
    assert dt._sitstand_cmd(3.0, 6.0)[0] == 0.0


def test_walk_command_uses_measured_speed():
    # Đo thực nghiệm: vx=0.10 → đứng im (zero-command attractor); 0.35 → đi
    # ~1.2 m / 8 s. Khóa vx để không ai chỉnh về vùng "đứng" nữa.
    assert dt.KINDS["walk"]["cmd"](0.0, 1.0)[0] == pytest.approx(0.35)


def test_projected_gravity_matches_standard_formula():
    """Công thức phải khớp legged_gym / infer_stairs (lỗi thật 2026-10-24:
    bản quaternion-conjugate thủ công sai terma gy → obs lệch lúc nghiêng)."""
    # Đứng thẳng: gravity chiếu = [0, 0, -1]
    g = dt.compute_projected_gravity(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(g, [0.0, 0.0, -1.0], atol=1e-6)
    # Pitch +90° (nose-up): gravity trong khung thân = [+1, 0, 0]
    q = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    g = dt.compute_projected_gravity(q)
    assert np.allclose(g, [1.0, 0.0, 0.0], atol=1e-6)
    # Roll +90° (quat x dương): gravity chiếu = [0, -1, 0]
    q = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
    g = dt.compute_projected_gravity(q)
    assert np.allclose(g, [0.0, -1.0, 0.0], atol=1e-6)


def test_stairs_kind_keeps_testbench_recipe():
    """Kind stairs phải giữ settle + reflex (công thức đã chứng minh của
    scripts/test_stairs_frame_by_frame.py) — nếu không, mọi checkpoint ngã
    trong 0.2 s đầu vì bị đòi leo khi chân chưa ổn định."""
    spec = dt.KINDS["stairs"]
    assert spec.get("settle_steps", 0) > 0
    assert spec.get("assist") == "stairs_reflex"


def test_spin_command_uses_phase_encoding():
    """spin nhận mã hóa pha [cos(2πφ), sin(2πφ)] trong slot twist (như
    ground_pick) — KHÔNG phải lệnh wz thô (lỗi thật 2026-10-24: wz=1.0)."""
    import mujoco  # noqa: F401 — đảm bảo môi trường có mujoco

    period = 4.0  # mdp.SPIN_PERIOD
    c0 = dt.KINDS["spin"]["cmd"](0.0, 6.0)
    assert np.allclose(c0[:3], [1.0, 0.0, 0.0], atol=1e-6)
    c_quarter = dt.KINDS["spin"]["cmd"](1.0, 6.0)
    assert np.allclose(c_quarter[:3], [0.0, 1.0, 0.0], atol=1e-6)
    # Chu kỳ lặp lại sau 4 s
    assert np.allclose(dt.KINDS["spin"]["cmd"](period, 6.0)[:3], c0[:3], atol=1e-6)


def test_servo_joint_addrs_plain_and_rollers():
    """Obs phải đọc joint servo theo địa chỉ model: model thường = slice liền
    kề; model rollers có 4 wheel `passive_*` XEN KẼ nên slice cứng là sai
    (invariant AGENTS.md: không hardcode joint index)."""
    import mujoco

    plain = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"))
    qa, da = dt.servo_joint_addrs(plain)
    assert qa == list(range(7, 21)) and da == list(range(6, 20))

    rollers = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"))
    qa_r, da_r = dt.servo_joint_addrs(rollers)
    assert len(qa_r) == 14 and len(da_r) == 14
    # Root freejoint + 4 wheel passive bị loại
    assert rollers.njnt == 19  # 1 free + 14 servo + 4 wheel
    assert qa_r != list(range(7, 21))  # wheel xen kẽ → không liền kề
    # Đúng 14 joint không phải passive/free, theo thứ tự model
    for j in range(rollers.njnt):
        name = mujoco.mj_id2name(rollers, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        adr = rollers.jnt_qposadr[j]
        included = adr in qa_r
        assert included == (not name.startswith("passive_")
                            and rollers.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE), name


def test_matrix_scan_writes_and_classifies(tmp_path, monkeypatch):
    from mjlab_microduck.rtx import matrix as mtx
    from mjlab_microduck.rtx.registry import LOGS_ROOT, RTX_STATE_DIR

    doc = mtx.scan()
    assert set(doc["matrix"].keys()) == set(FAMILIES.keys())
    # Walk đã train ≥ 3000 iters → không bao giờ là "missing"
    assert doc["matrix"]["walk"]["status"] in ("ready", "weak", "training")
    assert doc["matrix"]["walk"]["iters"] >= 3000
    # Spin chưa từng train local (trừ khi run đang ghi) → missing hoặc training
    assert doc["matrix"]["spin"]["status"] in ("missing", "training", "weak")
    # File state cho dashboard phải được ghi
    assert (RTX_STATE_DIR / "matrix.json").is_file()


def test_live_status_reports_gpu_shape():
    live = pipeline.live_status()
    assert {"ts", "training", "gpu", "queue"} <= set(live.keys())
    if live["gpu"] is not None:
        assert {"name", "util_pct", "mem_used_mb", "mem_total_mb", "temp_c"} <= set(live["gpu"].keys())
    for t in live["training"]:
        # Mọi tiến trình train sống phải ánh xạ được family
        assert t["family"] is not None, f"task lạ: {t['task_id']}"


def test_find_runs_prefers_highest_iteration():
    """find_runs sắp theo (iters, mtime) giảm dần — chọn nhầm run cũ hơn từng
    khiến demo dùng checkpoint yếu hơn (lỗi thật với stairs)."""
    runs = pipeline.find_runs(FAMILIES["walk"])
    if not runs:
        pytest.skip("chưa có run walk nào")
    iters = [mi for mi, _ in runs]
    assert iters == sorted(iters, reverse=True)
