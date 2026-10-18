#!/usr/bin/env python3
"""Automated Model Exporter & Downloader for Google Colab.

Automatically locates the newest checkpoints for:
  1. Mjlab-StairsClimb-MicroDuck -> stairs_latest.onnx
  2. Mjlab-Velocity-Rough-MicroDuck -> rough_latest.onnx

Bakes empirical observation normalizers into ONNX graphs and triggers
automatic browser downloads in Google Colab.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(task_subdir: str) -> str | None:
    pattern = os.path.join("logs", "rsl_rl", task_subdir, "*", "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def export_task(task_id: str, ckpt_path: str, out_onnx: str) -> bool:
    print(f"\n=======================================================")
    print(f"[*] Exporting: {task_id}")
    print(f"    Checkpoint : {ckpt_path}")
    print(f"    Output ONNX: {out_onnx}")
    print(f"=======================================================")
    
    cmd = [
        "uv", "run", "scripts/export.py",
        task_id,
        "--checkpoint-file", ckpt_path,
        "--onnx-file", out_onnx,
    ]
    ret = subprocess.run(cmd)
    if ret.returncode == 0 and os.path.exists(out_onnx):
        size_kb = os.path.getsize(out_onnx) / 1024.0
        print(f"[✓] Successfully exported {out_onnx} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[✗] Failed to export {out_onnx}")
        return False


def main():
    print("🦆 Microduck Colab Multi-Skill Automatic Exporter")
    
    exported_files = []
    
    # 1. Stairs Climb Policy
    stairs_ckpt = find_latest_checkpoint("stairs_climb")
    if stairs_ckpt:
        out_stairs = "stairs_latest.onnx"
        if export_task("Mjlab-StairsClimb-MicroDuck", stairs_ckpt, out_stairs):
            exported_files.append(out_stairs)
    else:
        print("[!] No stairs_climb checkpoint found in logs/rsl_rl/stairs_climb/")

    # 2. Velocity Rough Policy
    rough_ckpt = find_latest_checkpoint("velocity")
    if rough_ckpt:
        out_rough = "rough_latest.onnx"
        if export_task("Mjlab-Velocity-Rough-MicroDuck", rough_ckpt, out_rough):
            exported_files.append(out_rough)
    else:
        print("[!] No velocity checkpoint found in logs/rsl_rl/velocity/")

    if not exported_files:
        print("\n[!] No models were exported.")
        return

    print("\n[✓] Export complete for:", exported_files)

    # Check if running in Google Colab and trigger download
    try:
        import google.colab
        from google.colab import files
        print("\n[Google Colab Detected] Triggering automatic download of exported ONNX models...")
        for f in exported_files:
            print(f"  -> Downloading {f}...")
            files.download(f)
    except ImportError:
        print("\n[Local Environment] Files saved in current directory:")
        for f in exported_files:
            print(f"  -> {os.path.abspath(f)}")


if __name__ == "__main__":
    main()
