"""Tests for the Argon Single-Model End-to-End Sim-to-Real Pipeline (Stages A, B, C)."""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_mod(name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stage_a_parametric_cad_exports() -> None:
    """Stage A must generate valid .STEP (ISO-10303-21), .STL, and .OBJ files."""
    cad_mod = _load_mod(
        "parametric_cad_generator",
        "01_freecad_cad/parametric_cad_generator.py",
    )
    manifest = cad_mod.generate_all_cad_models()
    assert "nc_high_stairs_course" in manifest["models"]
    assert "step_over_hurdles" in manifest["models"]
    assert "microduck_beak_bumper_bracket" in manifest["models"]

    for model_name, info in manifest["models"].items():
        assert info["is_watertight"] is True, f"{model_name} mesh is not watertight"
        assert info["num_vertices"] > 0
        assert info["num_faces"] > 0
        step_path = Path(info["files"]["step"])
        stl_path = Path(info["files"]["stl"])
        obj_path = Path(info["files"]["obj"])
        assert step_path.exists() and step_path.stat().st_size > 100
        assert stl_path.exists() and stl_path.stat().st_size > 100
        assert obj_path.exists() and obj_path.stat().st_size > 100

        step_text = step_path.read_text(encoding="utf-8")
        assert step_text.startswith("ISO-10303-21;")
        assert "END-ISO-10303-21;" in step_text


def test_stage_b_urdf_and_openusd_simready() -> None:
    """Stage B must normalize all robot meshes and export valid URDF & OpenUSD (.usda) files."""
    mesh_mod = _load_mod(
        "blender_mesh_processor",
        "02_blender_omniverse_usd/blender_mesh_processor.py",
    )
    usd_mod = _load_mod(
        "mjcf_to_urdf_usd_bridge",
        "02_blender_omniverse_usd/mjcf_to_urdf_usd_bridge.py",
    )

    mesh_report = mesh_mod.process_all_meshes()
    assert mesh_report["robot_mesh_count"] >= 40
    assert mesh_report["all_scales_verified_meters"] is True

    usd_report = usd_mod.build_urdf_and_openusd_assets()
    assert usd_report["actuated_servo_count"] == 14
    assert usd_report["canonical_14_servo_match"] is True
    assert usd_report["all_joint_limits_valid"] is True
    assert 0.50 <= usd_report["total_robot_mass_kg"] <= 1.20

    # Verify URDF XML structure
    urdf_path = Path(usd_report["outputs"]["urdf"])
    assert urdf_path.exists()
    urdf_root = ET.parse(urdf_path).getroot()
    assert urdf_root.tag == "robot"
    assert len(urdf_root.findall("joint")) == 14

    # Verify OpenUSD SimReady schemas
    robot_usda = Path(usd_report["outputs"]["robot_simready_usda"]).read_text(encoding="utf-8")
    assert "#usda 1.0" in robot_usda
    assert 'upAxis = "Z"' in robot_usda
    assert "metersPerUnit = 1.0" in robot_usda
    assert "PhysicsArticulationRootAPI" in robot_usda
    assert "PhysicsRigidBodyAPI" in robot_usda
    assert "PhysicsMassAPI" in robot_usda
    assert "PhysicsCollisionAPI" in robot_usda
    assert "PhysicsRevoluteJoint" in robot_usda
    assert "PhysicsDriveAPI:angular" in robot_usda

    terrain_usda = Path(usd_report["outputs"]["terrain_course_usda"]).read_text(encoding="utf-8")
    assert "#usda 1.0" in terrain_usda
    assert "HighStairsCourse" in terrain_usda
    assert "StepOverHurdles" in terrain_usda


def test_stage_c_rl_and_physics_bridge() -> None:
    """Stage C must verify 61D actor contract, BAM M6, Neko Core, and real CAD terrain rollout."""
    rl_mod = _load_mod(
        "verify_rl_pipeline",
        "03_newton_isaaclab_rl/verify_rl_pipeline.py",
    )
    report = rl_mod.verify_rl_configs_and_neko_core(run_gpu_rollout=False)
    mj = report["mjlab_warp_verification"]
    assert mj["actor_61d_contract_verified"] is True
    assert mj["bam_startup_event_registered"] is True
    assert mj["curriculum_aligned"] is True

    sc = report["cad_to_physics_scene_bridge"]
    assert sc["nu"] == 14
    assert sc["simulation_step_nan_free"] is True

    cr = report["closed_loop_cad_terrain_rollout"]
    assert cr["total_control_steps"] >= 250
    assert cr["survived_without_fall"] is True
    assert Path(cr["trajectory_csv"]).exists()
    assert Path(cr["rollout_gif"]).exists()

    nc = report["neko_core_verification"]
    assert nc["gait_phase_latch_double_support_fired"] is True
    assert nc["neko_output_shape"] == [14]


def test_hermes_see_and_walk_and_astra_package() -> None:
    """Verify Hermes 'See, Traverse Rough Terrain & Climb Stairs' outputs and Astra package structure."""
    analysis_path = PROJECT_ROOT / "docs_and_logs" / "hermes_see_and_walk_analysis.json"
    assert analysis_path.exists()
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    exec_info = data["see_and_walk_execution"]
    assert exec_info["opengl_backend"] == "egl"
    assert exec_info["survived_without_fall"] is True
    assert exec_info["goal_reached_and_neko_transitioned"] is True
    assert exec_info["rough_visual_detections_frames"] > 0
    assert exec_info["stair_visual_detections_frames"] > 0
    assert exec_info["goal_visual_detections_frames"] > 0
    assert Path(exec_info["dual_view_gif"]).exists()
    assert Path(exec_info["trajectory_csv"]).exists()

    rough_info = data["dedicated_rough_terrain_rollout"]
    assert rough_info["survived_without_fall"] is True
    assert rough_info["final_x_m"] >= 1.50
    assert Path(rough_info["rollout_gif"]).exists()

    stairs_info = data["dedicated_high_stairs_rollout"]
    assert stairs_info["survived_without_fall"] is True
    assert stairs_info["final_x_m"] >= 2.10
    assert stairs_info["elevation_gain_dz_mm"] >= 100.0
    assert Path(stairs_info["rollout_gif"]).exists()

    astra_root = PROJECT_ROOT / "astra_implementation_package"
    assert (astra_root / "README_ASTRA_HANDOFF.md").exists()
    assert (astra_root / "00_hermes_plan_and_analysis" / "hermes_workflow_plan.json").exists()
    assert (astra_root / "01_asset_modeling" / "vision_see_and_walk_course.step").exists()
    assert (astra_root / "02_scene_assembly_and_rigging" / "vision_obstacle_course.usda").exists()
    assert (astra_root / "03_isaac_sim_and_training" / "isaac_sim_see_and_walk_env_cfg.py").exists()
    assert (astra_root / "04_simulation_results" / "microduck_see_and_walk_rollout.gif").exists()
    assert (astra_root / "04_simulation_results" / "microduck_rough_terrain_rollout.gif").exists()
    assert (astra_root / "04_simulation_results" / "microduck_stairs_climb_rollout.gif").exists()


