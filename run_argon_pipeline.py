#!/usr/bin/env python3
"""Central Orchestrator for the Argon Single-Model Sim-to-Real Pipeline (Phases 1-4).

Executes individual stages or the complete end-to-end toolchain:
  - Stage A (`01_freecad_cad`): Parametric CAD & Terrain Mesh Generation (`.STEP`, `.STL`, `.OBJ`)
  - Stage B (`02_blender_omniverse_usd`): Blender Mesh Normalization + MJCF -> URDF & OpenUSD (`.usda`)
  - Stage C (`03_newton_isaaclab_rl`): Newton / Isaac Lab Spec + MuJoCo Warp (`mjlab`) & Neko Core Verification

Usage:
  uv run python run_argon_pipeline.py --stage all
  uv run python run_argon_pipeline.py --stage cad
  uv run python run_argon_pipeline.py --stage usd
  uv run python run_argon_pipeline.py --stage rl
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_AND_LOGS_DIR = PROJECT_ROOT / "docs_and_logs"


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_host_environment() -> dict[str, Any]:
    """Inspect installed binaries and Python libraries on the current machine."""
    env_info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "binaries": {
            "uv": shutil.which("uv"),
            "freecadcmd": shutil.which("freecadcmd"),
            "blender": shutil.which("blender"),
        },
        "packages": {},
    }
    for pkg in ("torch", "warp", "mujoco", "mjlab", "trimesh", "scipy", "onnx", "onnxruntime", "pxr", "isaaclab"):
        try:
            mod = __import__(pkg)
            env_info["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            env_info["packages"][pkg] = None
    return env_info


def run_pipeline(stage: str = "all") -> dict[str, Any]:
    DOCS_AND_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DOCS_AND_LOGS_DIR / "pipeline_execution.log"
    t0 = time.perf_counter()

    log_lines: list[str] = [
        f"=== ARGON SIM-TO-REAL PIPELINE EXECUTION ({datetime.datetime.now(datetime.timezone.utc).isoformat()}) ===",
        f"Requested stage: {stage}",
    ]

    env_report = check_host_environment()
    results: dict[str, Any] = {
        "executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "requested_stage": stage,
        "environment": env_report,
        "stages": {},
    }

    if stage in ("all", "cad", "a"):
        cad_mod = _load_module(
            "parametric_cad_generator",
            PROJECT_ROOT / "01_freecad_cad" / "parametric_cad_generator.py",
        )
        cad_manifest = cad_mod.generate_all_cad_models()
        results["stages"]["stage_a_cad"] = cad_manifest
        msg = (
            f"[STAGE A: CAD] Completed via {cad_manifest['backend']['active_engine']} | "
            f"Models exported (.STEP, .STL, .OBJ): {list(cad_manifest['models'].keys())}"
        )
        print(msg)
        log_lines.append(msg)

    if stage in ("all", "usd", "b"):
        mesh_mod = _load_module(
            "blender_mesh_processor",
            PROJECT_ROOT / "02_blender_omniverse_usd" / "blender_mesh_processor.py",
        )
        usd_mod = _load_module(
            "mjcf_to_urdf_usd_bridge",
            PROJECT_ROOT / "02_blender_omniverse_usd" / "mjcf_to_urdf_usd_bridge.py",
        )
        mesh_summary = mesh_mod.process_all_meshes()
        usd_summary = usd_mod.build_urdf_and_openusd_assets()
        results["stages"]["stage_b_mesh"] = {
            "backend": mesh_summary["backend"],
            "robot_mesh_count": mesh_summary["robot_mesh_count"],
            "terrain_mesh_count": mesh_summary["terrain_mesh_count"],
            "all_scales_verified_meters": mesh_summary["all_scales_verified_meters"],
        }
        results["stages"]["stage_b_usd"] = usd_summary
        msg = (
            f"[STAGE B: BLENDER & OPENUSD] Completed | "
            f"Robot STLs={mesh_summary['robot_mesh_count']}, Terrain STLs={mesh_summary['terrain_mesh_count']}, "
            f"Links={usd_summary['total_links']}, 14-Servos Match={usd_summary['canonical_14_servo_match']} | "
            f"Generated microduck.urdf, microduck_simready.usda, terrain_course.usda"
        )
        print(msg)
        log_lines.append(msg)

    if stage in ("all", "rl", "c"):
        rl_mod = _load_module(
            "verify_rl_pipeline",
            PROJECT_ROOT / "03_newton_isaaclab_rl" / "verify_rl_pipeline.py",
        )
        rl_summary = rl_mod.verify_rl_configs_and_neko_core(run_gpu_rollout=True)
        results["stages"]["stage_c_rl"] = rl_summary
        mj = rl_summary["mjlab_warp_verification"]
        cr = rl_summary["closed_loop_cad_terrain_rollout"]
        gr = rl_summary.get("live_warp_gpu_rollout_and_export", {})
        msg = (
            f"[STAGE C: NEWTON / ISAAC LAB / WARP RL] Completed | "
            f"61D Obs={mj['actor_61d_contract_verified']}, "
            f"CAD Rollout (5.0s): dx={cr['forward_distance_dx_m']:+.3f}m, dz={cr['elevation_gain_dz_mm']:+.1f}mm, "
            f"Survived={cr['survived_without_fall']} | "
            f"GPU Rollout (16 envs): survival={gr.get('survival_rate_pct', 0.0):.1f}%, "
            f"Exported ONNX={gr.get('exported_onnx_path', '')}"
        )
        print(msg)
        log_lines.append(msg)

    elapsed_s = round(time.perf_counter() - t0, 3)
    results["elapsed_seconds"] = elapsed_s
    log_lines.append(f"=== PIPELINE FINISHED IN {elapsed_s}s ===")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n\n")

    validation_path = DOCS_AND_LOGS_DIR / "asset_validation_report.json"
    validation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Argon End-to-End Sim-to-Real Pipeline (Stage A -> Stage B -> Stage C)."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "cad", "usd", "rl", "a", "b", "c"),
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    args = parser.parse_args()
    run_pipeline(stage=args.stage)


if __name__ == "__main__":
    main()

