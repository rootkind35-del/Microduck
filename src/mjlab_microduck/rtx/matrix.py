"""Kiểm kê matrix multi-mini-model: task → checkpoint → ONNX → digital twin.

Quét logs/rsl_rl/<exp_dir>/*/model_*.pt, chọn run tốt nhất (theo iteration
cao nhất), đối chiếu ONNX đã bake và kết quả digital twin mới nhất trong
logs/digital_tunnel… (logs/digital_twin). Xuất logs/rtx/matrix.json cho
dashboard và in bảng ra console.

Phân loại trạng thái:
  missing    — chưa có checkpoint nào
  weak       — có checkpoint nhưng < 90% target_iters
  ready      — đạt >= 90% target_iters (coi như đủ theo cấu hình)
  training   — đang có tiến trình train sống của family này (do pipeline tự nhận)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .registry import FAMILIES, LOGS_ROOT, RTX_STATE_DIR, ModelFamily, exp_dir


def _run_max_iter(run_dir: Path) -> int:
    best = -1
    for p in run_dir.glob("model_*.pt"):
        try:
            best = max(best, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return best


def _scan_family(fam: ModelFamily) -> dict:
    entry = {"family": asdict(fam), "best_run": None, "onnx": None,
             "twin": None, "status": "missing", "iters": 0}
    base = exp_dir(fam)
    if not base.exists():
        return entry

    runs = []
    for run in base.iterdir():
        if not run.is_dir() or run.name.startswith("."):
            continue
        mi = _run_max_iter(run)
        if mi >= 0:
            runs.append((mi, run))
    if not runs:
        return entry

    # Ưu tiên iteration cao nhất; bằng điểm thì lấy run mới nhất (mtime).
    runs.sort(key=lambda t: (t[0], t[1].stat().st_mtime), reverse=True)
    mi, run = runs[0]
    entry["best_run"] = str(run)
    entry["iters"] = mi
    ckpt = run / f"model_{mi}.pt"
    entry["checkpoint"] = str(ckpt) if ckpt.exists() else None
    entry["ckpt_mtime"] = ckpt.stat().st_mtime if ckpt.exists() else None

    onnxs = sorted(run.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    if onnxs:
        entry["onnx"] = str(onnxs[-1])

    twin_dir = LOGS_ROOT / "digital_twin" / fam.key
    if twin_dir.exists():
        results = sorted(twin_dir.glob("metrics_*.json"),
                         key=lambda p: p.stat().st_mtime)
        if results:
            try:
                entry["twin"] = json.loads(results[-1].read_text())
                entry["twin"]["metrics_file"] = str(results[-1])
            except (json.JSONDecodeError, OSError):
                pass

    if mi >= 0.9 * fam.target_iters:
        entry["status"] = "ready"
    else:
        entry["status"] = "weak"
    return entry


def is_training_task_id(task_id: str) -> bool:
    """Có tiến trình `train <task_id>` nào đang sống không (ps quét nhanh)."""
    try:
        import subprocess
        out = subprocess.run(
            ["pgrep", "-af", r"bin/train .+"], capture_output=True, text=True,
            timeout=5,
        ).stdout
    except Exception:
        return False
    for line in out.splitlines():
        if task_id in line:
            return True
    return False


def scan(active_tasks: dict[str, str] | None = None) -> dict:
    """Quét toàn bộ families. `active_tasks`: map task_id → trạng thái đang train."""
    active_tasks = active_tasks or {}
    matrix = {}
    for key, fam in FAMILIES.items():
        entry = _scan_family(fam)
        if fam.task_id in active_tasks:
            entry["status"] = "training"
        matrix[key] = entry
    doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matrix": matrix,
    }
    RTX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    (RTX_STATE_DIR / "matrix.json").write_text(json.dumps(doc, indent=2))
    return doc


def main():
    doc = scan()
    hdr = f"{'key':<12} {'status':<9} {'iters':>6}/{'<5':<6} {'onnx':<4} twin"
    print(hdr)
    print("-" * len(hdr))
    for key, e in doc["matrix"].items():
        fam = e["family"]
        it = f"{e['iters']:>6}/{fam['target_iters']:<6}"
        on = "✓" if e["onnx"] else "–"
        tw = "✓" if e["twin"] else "–"
        print(f"{key:<12} {e['status']:<9} {it} {on:<4} {tw}")
    print(f"\n→ {RTX_STATE_DIR / 'matrix.json'}")


if __name__ == "__main__":
    main()
