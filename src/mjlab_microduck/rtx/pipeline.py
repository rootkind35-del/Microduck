"""RTX Pipeline — điều phối vòng đời mỗi mini model trên một GPU RTX 4060.

Luồng chuẩn cho một model (đúng bài học "smoke test trước khi train dài"):
  smoke (64 envs × 5 iters) → train (1024 envs) → export ONNX (bake normalizer)
  → digital twin (GIF + chỉ số) → cập nhật matrix.

Ba tầng tương ứng "3 máy tính" của một pipeline sim-to-real gọn:
  train (GPU)  →  digital twin (CPU MuJoCo + BAM)  →  deploy rehearsal (ONNX)

GPU chỉ chạy MỘT tiến trình train mỗi lúc (8 GB VRAM): `queue` tự chờ tiến trình
train đang sống thoát rồi mới mở stage kế tiếp.

Dùng:
  uv run rtx-pipeline scan
  uv run rtx-pipeline smoke spin
  uv run rtx-pipeline train spin
  uv run rtx-pipeline export walk
  uv run rtx-pipeline demo stand
  uv run rtx-pipeline queue spin --wait-gpu     # train tuần tự qua đêm
  uv run rtx-pipeline live                      # JSON trạng thái cho dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from .registry import ESSENTIAL_KEYS, FAMILIES, LOGS_ROOT, REPO_ROOT, RTX_STATE_DIR, ModelFamily, exp_dir

TRAIN_LOG_DIR = RTX_STATE_DIR / "train_logs"
QUEUE_STATE = RTX_STATE_DIR / "queue_state.json"
QUEUE_LOG = RTX_STATE_DIR / "queue.log"


def _base_env() -> dict:
    env = os.environ.copy()
    # ROS Humble leak vào PYTHONPATH làm hỏng import trong venv (đã gặp khi test).
    env.pop("PYTHONPATH", None)
    # Train local không cần wandb online; offline vẫn ghi run dir đầy đủ.
    env["WANDB_MODE"] = "offline"
    return env


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RTX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_LOG, "a") as f:
        f.write(line + "\n")


def find_runs(fam: ModelFamily) -> list[tuple[int, Path]]:
    base = exp_dir(fam)
    if not base.exists():
        return []
    out = []
    for run in base.iterdir():
        if not run.is_dir() or run.name.startswith("."):
            continue
        best = -1
        for p in run.glob("model_*.pt"):
            try:
                best = max(best, int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        if best >= 0:
            out.append((best, run))
    out.sort(key=lambda t: (t[0], t[1].stat().st_mtime), reverse=True)
    return out


def best_checkpoint(fam: ModelFamily) -> tuple[Path, Path] | None:
    """(ckpt, run_dir) của run tốt nhất; None nếu chưa có."""
    runs = find_runs(fam)
    if not runs:
        return None
    mi, run = runs[0]
    return run / f"model_{mi}.pt", run


def own_train_pids() -> set[int]:
    """PID các tiến trình `bin/train Mjlab-...` đang sống."""
    try:
        out = subprocess.run(["pgrep", "-f", r"bin/train Mjlab-"],
                             capture_output=True, text=True, timeout=5).stdout
        return {int(x) for x in out.split() if x.isdigit()}
    except Exception:
        return set()


def family_of_task(task_id: str) -> str | None:
    """task_id → family key, chấp nhận biến thể Rough/Backlash."""
    from .registry import FAMILIES
    norm = task_id.replace("-Rough-", "-Flat-").replace("Backlash-", "")
    for k, f in FAMILIES.items():
        if f.task_id in (task_id, norm):
            return k
    return None


def wait_for_gpu_free(timeout_s: float = 60 * 60 * 12, poll_s: float = 60.0,
                       note: str = "") -> bool:
    """Chờ tới khi không còn tiến trình train nào sống (queue dùng trước mỗi stage)."""
    start = time.time()
    while True:
        pids = own_train_pids()
        if not pids:
            return True
        if time.time() - start > timeout_s:
            _log(f"⏱ Chờ GPU quá {timeout_s / 3600:.0f}h — bỏ qua {note}")
            return False
        _log(f"⏳ GPU bận (train PID {sorted(pids)}) — chờ {poll_s:.0f}s nữa{' — ' + note if note else ''}")
        time.sleep(poll_s)


# ---------------------------------------------------------------- các stage

def cmd_scan(_args) -> int:
    from .matrix import scan
    doc = scan()
    for key, e in doc["matrix"].items():
        it = e["iters"]
        tgt = e["family"]["target_iters"]
        print(f"{key:<12} {e['status']:<9} {it:>6}/{tgt:<6} onnx={'✓' if e['onnx'] else '–'} twin={'✓' if e['twin'] else '–'}")
    return 0


def cmd_smoke(args) -> int:
    fam = FAMILIES[args.key]
    if own_train_pids():
        print("[smoke] Đã có tiến trình train đang chạy — smoke sẽ chia sẻ GPU (64 envs chỉ ~1 GB, chấp nhận được)")
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TRAIN_LOG_DIR / f"smoke_{fam.key}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = ["uv", "run", "train", fam.task_id,
           "--env.scene.num-envs", "64", "--agent.max_iterations", "5"]
    _log(f"🧪 SMOKE {fam.key}: {' '.join(cmd)} → {log_path}")
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, env=_base_env(), stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    _log(f"🧪 SMOKE {fam.key}: rc={rc} {'✓' if rc == 0 else '✗ THẤT BẠI — xem ' + str(log_path)}")
    return rc


def cmd_train(args) -> int:
    fam = FAMILIES[args.key]
    if args.wait_gpu:
        wait_for_gpu_free(note=f"train {fam.key}")
    elif own_train_pids():
        print("[train] GPU đang bận (tiến trình train khác đang sống). Dùng --wait-gpu để xếp hàng.")
        return 2

    iters = args.iters or fam.target_iters
    cmd = ["uv", "run", "train", fam.task_id,
           "--env.scene.num-envs", str(args.num_envs),
           "--agent.max_iterations", str(iters)]
    if args.resume:
        got = best_checkpoint(fam)
        if got is None:
            print(f"[train] --resume nhưng {fam.exp_dir} chưa có checkpoint; train mới.")
        else:
            ckpt, _ = got
            cmd += ["--agent.resume", "True",
                    f"--agent.load-checkpoint={ckpt}"]
            _log(f"↻ train {fam.key} resume từ {ckpt.name} tới {iters}")

    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TRAIN_LOG_DIR / f"train_{fam.key}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _log(f"🚂 TRAIN {fam.key} ({iters} iters, {args.num_envs} envs) → {log_path}")
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, env=_base_env(), stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    _log(f"🚂 TRAIN {fam.key}: rc={rc}")

    if rc == 0 and args.chain:
        cmd_export(argparse.Namespace(key=fam.key))
        cmd_demo(argparse.Namespace(key=fam.key, onnx=None, duration=None))
    return rc


def cmd_export(args) -> int:
    fam = FAMILIES[args.key]
    got = best_checkpoint(fam)
    if got is None:
        print(f"[export] {fam.key}: chưa có checkpoint")
        return 1
    ckpt, run_dir = got
    out = run_dir / f"{fam.key}_baked.onnx"
    _log(f"📦 EXPORT {fam.key}: {ckpt.name} → {out.name}")
    cmd = ["uv", "run", "scripts/export.py", fam.task_id,
           "--checkpoint-file", str(ckpt), "--onnx-file", str(out)]
    env = _base_env()
    rc = subprocess.run(cmd, env=env).returncode
    _log(f"📦 EXPORT {fam.key}: rc={rc} {'✓' if rc == 0 else '✗'}")
    return rc


def cmd_demo(args) -> int:
    from .digital_twin import run_twin
    from .registry import TWIN_DIR
    fam = FAMILIES[args.key]
    got = best_checkpoint(fam)
    onnx = Path(args.onnx) if args.onnx else None
    if onnx is None and fam.preferred_onnx and Path(fam.preferred_onnx).exists():
        onnx = Path(fam.preferred_onnx)
    if onnx is None and got is not None:
        run_dir = got[1]
        cand = sorted(run_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if cand:
            onnx = cand[-1]
    if onnx is None:
        print(f"[demo] {fam.key}: chưa có ONNX")
        return 1
    m = run_twin(fam.key, onnx, TWIN_DIR / fam.key,
                 duration_s=args.duration)
    ok = "✓" if m["success"] else "✗"
    _log(f"🎬 TWIN {fam.key}: {ok} {m['verdict']} → {m['gif']}")
    from .matrix import scan
    scan()
    return 0 if m["success"] else 1


def cmd_queue(args) -> int:
    RTX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    keys = args.keys or ESSENTIAL_KEYS
    plan = []
    for k in keys:
        if k not in FAMILIES:
            print(f"[queue] Không biết family '{k}'")
            return 1
        fam = FAMILIES[k]
        # Chỉ train những model chưa ready (hoặc --force)
        got = best_checkpoint(fam)
        mi = got[0].stem.split("_")[1] if got else -1
        try:
            mi = int(mi)
        except ValueError:
            mi = -1
        if mi >= 0.9 * fam.target_iters and not args.force:
            _log(f"[queue] {k}: đã đạt {mi}/{fam.target_iters} — bỏ qua (dùng --force để train lại)")
            continue
        plan.append(k)

    state = {"started": datetime.now().isoformat(timespec="seconds"),
             "plan": plan, "current": None, "done": [], "failed": [], "status": "running"}
    QUEUE_STATE.write_text(json.dumps(state, indent=2))

    rc_all = 0
    ns_train = argparse.Namespace
    for k in plan:
        state["current"] = k
        state["status"] = f"training:{k}"
        QUEUE_STATE.write_text(json.dumps(state, indent=2))
        wait_for_gpu_free(note=f"queue → train {k}")
        rc = cmd_train(ns_train(key=k, wait_gpu=False, num_envs=args.num_envs,
                                iters=None, resume=False, chain=True))
        (state["done"] if rc == 0 else state["failed"]).append(k)
        rc_all |= rc
    state["current"] = None
    state["status"] = "finished" if rc_all == 0 else "finished_with_errors"
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    QUEUE_STATE.write_text(json.dumps(state, indent=2))
    _log(f"✅ Queue hoàn tất: done={state['done']} failed={state['failed']}")
    from .matrix import scan
    scan()
    return rc_all


# ------------------------------------------------------------- live status

CKPT_RE = re.compile(r"model_(\d+)\.pt$")


def live_status() -> dict:
    """Trạng thái realtime: tiến trình train sống, iteration từ checkpoint,
    tốc độ + ETA từ nhịp sinh checkpoint (mỗi 250 iters một file)."""
    out = {"ts": datetime.now().isoformat(timespec="seconds"),
           "training": [], "gpu": None, "queue": None}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        if smi:
            name, util, memu, memt, temp = [x.strip() for x in smi.split(",")]
            out["gpu"] = {"name": name, "util_pct": int(util),
                          "mem_used_mb": int(memu), "mem_total_mb": int(memt),
                          "temp_c": int(temp)}
    except Exception:
        pass

    for pid in sorted(own_train_pids()):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
        except OSError:
            continue
        m = re.search(r"train (Mjlab-\S+)", cmdline)
        task_id = m.group(1) if m else "?"
        fam_key = family_of_task(task_id)
        mi_re = re.search(r"max.iterations[= ](\d+)", cmdline)
        target = int(mi_re.group(1)) if mi_re else None
        info = {"pid": pid, "task_id": task_id, "family": fam_key,
                "iters": None, "target": target, "iters_per_min": None,
                "eta_h": None}
        if fam_key:
            # Run ĐANG GHI là run có mtime mới nhất (khác run tốt nhất lịch sử).
            runs = find_runs(FAMILIES[fam_key])
            if runs:
                newest = max(runs, key=lambda t: t[1].stat().st_mtime)
                _, run = newest
                ckpts = sorted(run.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    mi = max(int(CKPT_RE.search(p.name).group(1)) for p in ckpts)
                    info["iters"] = mi
                    info["run_dir"] = str(run)
                    if len(ckpts) >= 2:
                        a, b = ckpts[-2], ckpts[-1]
                        ia = int(CKPT_RE.search(a.name).group(1))
                        ib = int(CKPT_RE.search(b.name).group(1))
                        dt_min = max((b.stat().st_mtime - a.stat().st_mtime) / 60.0, 1e-6)
                        rate = (ib - ia) / dt_min
                        info["iters_per_min"] = round(rate, 1)
                        if target:
                            eta_h = (target - ib) / rate / 60.0
                            info["eta_h"] = round(max(eta_h, 0.0), 2)
        out["training"].append(info)

    if QUEUE_STATE.exists():
        try:
            out["queue"] = json.loads(QUEUE_STATE.read_text())
        except json.JSONDecodeError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(prog="rtx-pipeline",
                                 description="Pipeline multi-mini-model cho Microduck RTX")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="Quét matrix model → logs/rtx/matrix.json").set_defaults(fn=cmd_scan)

    p = sub.add_parser("smoke", help="Smoke test 64 envs × 5 iters")
    p.add_argument("key", choices=sorted(FAMILIES))
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("train", help="Train đầy đủ (mặc định 1024 envs trên RTX 4060)")
    p.add_argument("key", choices=sorted(FAMILIES))
    p.add_argument("--iters", type=int, default=None, help="Mặc định: target_iters của family")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--resume", action="store_true", help="Resume từ checkpoint tốt nhất")
    p.add_argument("--wait-gpu", action="store_true", help="Chờ GPU rảnh trước khi train")
    p.add_argument("--chain", action="store_true", help="Sau train: tự export ONNX + digital twin")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("export", help="Bake ONNX (normalizer) từ checkpoint tốt nhất")
    p.add_argument("key", choices=sorted(FAMILIES))
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("demo", help="Digital twin: GIF + chỉ số")
    p.add_argument("key", choices=sorted(FAMILIES))
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("queue", help="Train tuần tự các model còn thiếu (--wait-gpu mặc định)")
    p.add_argument("keys", nargs="*", help="Bỏ trống = toàn bộ model thiết yếu")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--force", action="store_true", help="Train lại cả model đã ready")
    p.set_defaults(fn=cmd_queue)

    sub.add_parser("live", help="JSON trạng thái realtime (dùng bởi dashboard)").set_defaults(
        fn=lambda a: (print(json.dumps(live_status(), indent=2)), 0)[1])

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
