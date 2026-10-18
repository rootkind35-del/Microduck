#!/usr/bin/env python3
"""Automated sequential training pipeline for the 4 Rough Terrain specialized tasks.

Tasks in sequence:
  1. Mjlab-VelStand-Rough-MicroDuck (warm-start from walk expert, MICRODUCK_WARM_START=1)
  2. Mjlab-StandUp-Rough-MicroDuck  (stand recovery on rough terrain)
  3. Mjlab-SitStand-Rough-MicroDuck  (crouch/stand stabilization on rough terrain)
  4. Mjlab-GroundPick-Rough-MicroDuck (beak reach down & return on rough terrain)

Usage:
  uv run scripts/train_rough_sequential.py --num-envs 1024
"""

import argparse
import datetime
import os
import subprocess
import time
from pathlib import Path

TASKS = [
    {
        "id": "Mjlab-VelStand-Rough-MicroDuck",
        "name": "VelStand-Rough",
        "exp_dir": "velstand",
        "warm_start": True,
        "checkpoint": "logs/rsl_rl/expert_cache/walk_expert.pt",
        "env_vars": {"MICRODUCK_WARM_START": "1"},
        "max_iterations": 6000,
    },
    {
        "id": "Mjlab-StandUp-Rough-MicroDuck",
        "name": "StandUp-Rough",
        "exp_dir": "microduck_stand",
        "warm_start": False,
        "env_vars": {},
        "max_iterations": 5000,
    },
    {
        "id": "Mjlab-SitStand-Rough-MicroDuck",
        "name": "SitStand-Rough",
        "exp_dir": "microduck_sitstand",
        "warm_start": False,
        "env_vars": {},
        "max_iterations": 4000,
    },
    {
        "id": "Mjlab-GroundPick-Rough-MicroDuck",
        "name": "GroundPick-Rough",
        "exp_dir": "ground_pick",
        "warm_start": False,
        "env_vars": {},
        "max_iterations": 3000,
    },
]


def find_latest_run_dir(exp_dir_name: str) -> Path | None:
    base_dir = Path("logs/rsl_rl") / exp_dir_name
    if not base_dir.exists():
        return None
    runs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def find_highest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("model_*.pt"))
    if not ckpts:
        return None

    def ckpt_num(p: Path):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    ckpts.sort(key=ckpt_num, reverse=True)
    return ckpts[0]


def export_baked_onnx(task_id: str, ckpt_path: Path, output_onnx: Path) -> bool:
    print(f"[*] Exporting baked ONNX for {task_id} from {ckpt_path} -> {output_onnx}...")
    cmd = [
        "uv", "run", "scripts/export.py",
        task_id,
        "--checkpoint-file", str(ckpt_path),
        "--onnx-file", str(output_onnx),
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONPATH"] = ""
    res = subprocess.run(cmd, env=env)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Sequential training for 4 rough tasks")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments (default: 1024)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 iterations per model for rapid smoke test")
    parser.add_argument("--start-idx", type=int, default=1, help="Start from task 1-4 (default: 1)")
    args = parser.parse_args()

    pipeline_log = Path("logs/rough_pipeline.log")
    pipeline_log.parent.mkdir(parents=True, exist_ok=True)

    summary = []

    for idx, task_info in enumerate(TASKS, start=1):
        if idx < args.start_idx:
            continue

        task_id = task_info["id"]
        task_name = task_info["name"]
        t_start = time.time()
        start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_envs = 64 if args.smoke_test else args.num_envs
        cmd = ["uv", "run", "train", task_id, f"--env.scene.num-envs={num_envs}"]

        if args.smoke_test:
            cmd.append("--agent.max_iterations=5")
        elif task_info.get("max_iterations"):
            cmd.append(f"--agent.max_iterations={task_info['max_iterations']}")

        if task_info["warm_start"] and task_info.get("checkpoint"):
            ckpt = Path(task_info["checkpoint"])
            if ckpt.exists():
                cmd.extend(["--agent.resume=True", f"--agent.load-checkpoint={ckpt}"])

        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"
        env["PYTHONPATH"] = ""
        for k, v in task_info.get("env_vars", {}).items():
            env[k] = v

        with open(pipeline_log, "a") as plog:
            plog.write(f"\n--- Starting {task_id} at {start_str} ---\nCmd: {' '.join(cmd)}\n")

        p = subprocess.Popen(cmd, env=env)
        rc = p.wait()

        elapsed = time.time() - t_start
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

        if rc != 0:
            summary.append({"task": task_name, "status": f"FAILED (rc={rc})", "duration": elapsed_str})
            continue

        run_dir = find_latest_run_dir(task_info["exp_dir"])
        ckpt_path = find_highest_checkpoint(run_dir) if run_dir else None
        baked_onnx = None

        if run_dir and ckpt_path:
            baked_onnx = run_dir / f"{task_name.lower()}_baked.onnx"
            export_baked_onnx(task_id, ckpt_path, baked_onnx)

        summary.append({
            "task": task_name,
            "status": "SUCCESS",
            "duration": elapsed_str,
            "ckpt": str(ckpt_path) if ckpt_path else "N/A",
            "onnx": str(baked_onnx) if baked_onnx else "N/A",
        })

    print("\n================ PIPELINE SUMMARY ================")
    for s in summary:
        print(f"Task: {s['task']} | Status: {s['status']} | Duration: {s['duration']}")
        if "ckpt" in s:
            print(f"  Checkpoint: {s['ckpt']}")
            print(f"  ONNX:       {s['onnx']}")
    print("==================================================\n")


if __name__ == "__main__":
    main()

