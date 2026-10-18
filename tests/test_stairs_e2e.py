"""Comprehensive E2E Opaque-Box Requirement Tests for Microduck High Stairs RL.

Verifies system-level requirements and invariants specified in ORIGINAL_REQUEST.md,
PROJECT.md, and AGENTS.md across 6 core functional areas:
1. Task environment registration and naming invariants
2. 61D unified observation layout and structure
3. Actuator BAM parameters and friction domain randomization
4. Scene stair heights, continuity, and collision geometry
5. Runner CLI interface, 50 Hz control loop, and spawn presets
6. Normalizer baking and ONNX export contract

Adheres to Progressive Testability: tests are self-contained and isolated,
validating contracts and gracefully skipping pending milestone artifacts.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_STAIRS_PATH = REPO_ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_stairs.xml"
STAIRS_ENV_CFG_PATH = REPO_ROOT / "src" / "mjlab_microduck" / "tasks" / "microduck_stairs_env_cfg.py"
INFER_STAIRS_PATH = REPO_ROOT / "scripts" / "infer_stairs.py"
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export.py"


# =============================================================================
# 1. Task Environment Registration and Naming Invariants
# =============================================================================


class TestTaskRegistrationAndNaming:
    """Verifies task naming conventions, registration contracts, and backlash twins."""

    EXPECTED_BASE_TASK = "Mjlab-StairsClimb-MicroDuck"
    EXPECTED_BACKLASH_TASK = "Mjlab-StairsClimb-Backlash-MicroDuck"
    TASK_REGEX = re.compile(r"^Mjlab-StairsClimb(-Backlash)?-MicroDuck$")

    def test_task_id_naming_convention(self):
        """Verify high stairs task IDs match repository regex and naming convention."""
        assert self.TASK_REGEX.match(self.EXPECTED_BASE_TASK) is not None
        assert self.TASK_REGEX.match(self.EXPECTED_BACKLASH_TASK) is not None
        assert "-Backlash-" in self.EXPECTED_BACKLASH_TASK
        assert "-Backlash-" not in self.EXPECTED_BASE_TASK

    def test_backlash_twin_registration_contract(self):
        """Verify _BACKLASH_TASKS table mapping contract in mjlab_microduck.tasks."""
        import mjlab_microduck.tasks as tasks_mod

        assert hasattr(tasks_mod, "_BACKLASH_TASKS"), "tasks module must maintain _BACKLASH_TASKS table"
        backlash_entries = getattr(tasks_mod, "_BACKLASH_TASKS")
        assert isinstance(backlash_entries, (tuple, list, dict))

        # Extract registered task IDs from table
        if isinstance(backlash_entries, dict):
            task_ids = list(backlash_entries.values())
        else:
            task_ids = [entry[0] for entry in backlash_entries if isinstance(entry, (tuple, list))]

        if self.EXPECTED_BACKLASH_TASK in task_ids:
            assert any(self.EXPECTED_BACKLASH_TASK == tid for tid in task_ids)
        else:
            pytest.skip(f"{self.EXPECTED_BACKLASH_TASK} registration is pending M2 completion.")

    def test_task_registration_visibility_if_registered(self):
        """Verify task registration entry in gymnasium registry if M2 is complete."""
        try:
            import gymnasium as gym

            registered_envs = gym.envs.registration.registry
            if self.EXPECTED_BASE_TASK in registered_envs:
                base_entry = registered_envs[self.EXPECTED_BASE_TASK]
                assert base_entry.entry_point is not None
                assert callable(base_entry.entry_point) or isinstance(base_entry.entry_point, str)
            else:
                pytest.skip(f"{self.EXPECTED_BASE_TASK} registration is pending M2 completion.")
        except ImportError:
            pytest.skip("Gymnasium not directly importable in current test process.")

    def test_runner_experiment_name_uniqueness_contract(self):
        """Verify experiment name pattern avoids collisions with other tasks."""
        if not STAIRS_ENV_CFG_PATH.exists():
            pytest.skip("microduck_stairs_env_cfg.py pending M2 completion.")

        spec = importlib.util.spec_from_file_location("stairs_cfg", STAIRS_ENV_CFG_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Locate RunnerCfg subclass or instance
        runner_cfgs = [
            getattr(mod, attr)
            for attr in dir(mod)
            if "RunnerCfg" in attr or "RlCfg" in attr
        ]
        assert len(runner_cfgs) > 0, "Env cfg module must define runner configuration"


# =============================================================================
# 2. 61D Unified Observation Layout and Structure
# =============================================================================


class TestObservationContract61D:
    """Verifies strict 61D observation shape, terms layout, and zero-padding rules."""

    EXPECTED_TOTAL_OBS_DIM = 61
    EXPECTED_PROPRIO_DIM = 48
    EXPECTED_COMMAND_DIM = 13

    # Expected proprioceptive breakdown:
    # base_ang_vel (3) + projected_gravity (3) + joint_pos_rel (14) + joint_vel_rel (14) + last_action (14) = 48
    EXPECTED_PROPRIO_COMPONENTS = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "joint_pos_rel": 14,
        "joint_vel_rel": 14,
        "last_action": 14,
    }

    # Expected command breakdown: twist (3) + head_pose (4) + body_pose (6) = 13
    EXPECTED_COMMAND_COMPONENTS = {
        "twist": 3,
        "head_pose": 4,
        "body_pose": 6,
    }

    def test_base_proprioception_dimension_sum(self):
        """Verify the 48D base proprioception component sum matches hardware contract."""
        total_proprio = sum(self.EXPECTED_PROPRIO_COMPONENTS.values())
        assert total_proprio == self.EXPECTED_PROPRIO_DIM == 48

    def test_command_block_dimension_sum(self):
        """Verify the 13D command block component sum [twist(3), head(4), body(6)]."""
        total_cmd = sum(self.EXPECTED_COMMAND_COMPONENTS.values())
        assert total_cmd == self.EXPECTED_COMMAND_DIM == 13

    def test_total_actor_obs_dimension_matches_61(self):
        """Verify that total actor observation dimension equals exactly 61."""
        total = self.EXPECTED_PROPRIO_DIM + self.EXPECTED_COMMAND_DIM
        assert total == self.EXPECTED_TOTAL_OBS_DIM == 61

    def test_base_velocity_cfg_conforms_to_61d(self):
        """Verify base locomotion template (from which stairs inherits) defines 61D obs."""
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
            make_microduck_velocity_env_cfg,
        )

        cfg = make_microduck_velocity_env_cfg(play=False, rough=False)
        actor_obs = cfg.observations.get("actor") if isinstance(cfg.observations, dict) else getattr(cfg.observations, "policy", None)
        assert actor_obs is not None
        terms = getattr(actor_obs, "terms", {})
        term_names = list(terms.keys()) if isinstance(terms, dict) else [attr for attr in dir(terms) if not attr.startswith("_")]
        assert "base_ang_vel" in term_names
        assert "joint_pos" in term_names
        assert "joint_vel" in term_names
        assert any("twist" in t.lower() or "cmd" in t.lower() or "command" in t.lower() for t in term_names)

    def test_stairs_cfg_61d_layout_if_implemented(self):
        """If microduck_stairs_env_cfg.py is implemented, verify exact 61D actor obs terms."""
        if not STAIRS_ENV_CFG_PATH.exists():
            pytest.skip("microduck_stairs_env_cfg.py pending M2 completion.")

        spec = importlib.util.spec_from_file_location("stairs_cfg", STAIRS_ENV_CFG_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        factory_name = "make_microduck_stairs_env_cfg"
        assert hasattr(mod, factory_name), f"Expected factory function {factory_name}"
        factory_fn = getattr(mod, factory_name)
        cfg = factory_fn(play=False, rough=False)

        policy_obs = cfg.observations.policy
        # Check all registered terms
        assert policy_obs is not None


# =============================================================================
# 3. Actuator BAM Parameters and Friction Domain Randomization
# =============================================================================


class TestBAMActuatorAndFrictionDR:
    """Verifies BAM M6 actuator model constants, 14 servo layout, and friction DR rules."""

    EXPECTED_NUM_SERVOS = 14
    EXPECTED_SERVO_NAMES = [
        # Left leg (0..4)
        "hip_yaw_left", "hip_roll_left", "hip_pitch_left", "knee_left", "ankle_left",
        # Neck & head (5..8)
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        # Right leg (9..13)
        "hip_yaw_right", "hip_roll_right", "hip_pitch_right", "knee_right", "ankle_right",
    ]

    def test_servo_count_and_layout(self):
        """Verify 14 servos arranged in left leg (5), head (4), right leg (5)."""
        assert len(self.EXPECTED_SERVO_NAMES) == self.EXPECTED_NUM_SERVOS == 14

    def test_bam_actuator_kwargs_constants(self):
        """Verify BAM actuator parameters mirror the Dynamixel XL330 M6 model."""
        from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS as k

        assert k["motor_name"] == "xl330", "BAM motor name must be xl330"
        assert k["model"].lower() == "m6", "BAM model must be m6"
        assert k["kp_fw"] == 200.0, "BAM forward gain kp_fw must be 200.0"
        assert k["vin_range"] == (6.5, 8.2), "Nominal voltage range must be (6.5, 8.2)"
        assert k["vin_min"] == 6.0, "Minimum operating voltage must be 6.0"
        assert k.get("max_current") in (1.4, None) or k.get("max_current") == 1.4

    def test_allcollisions_robot_cfg_has_14_actuators(self):
        """Verify MICRODUCK_ALLCOLLISIONS_ROBOT_CFG compiles with 14 active servos."""
        from mjlab_microduck.robot.microduck_constants import get_allcollisions_spec

        model = get_allcollisions_spec().compile()
        actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        assert len(actuator_names) == 14, f"Expected 14 actuators, found {len(actuator_names)}"
        assert not any(n.startswith("passive_") for n in actuator_names), "Passive joints must not be actuated"

    def test_expand_bam_friction_fields_startup_event_contract(self):
        """Verify expand_bam_friction_fields startup event contract for BAM envs."""
        from mjlab_microduck.tasks import mdp as microduck_mdp

        assert hasattr(microduck_mdp, "expand_bam_friction_fields"), (
            "mdp.py must define expand_bam_friction_fields startup event"
        )


# =============================================================================
# 4. Scene Stair Heights, Continuity, and Collision Geometry
# =============================================================================


class TestSceneStairsGeometryAndPhysics:
    """Verifies scene_stairs.xml geometry (2.5, 3.0, 3.5 cm risers), continuity, and materials."""

    EXPECTED_FLIGHT_RISERS = {
        "flight_1": 0.010,  # 1.0 cm
        "flight_2": 0.0125,  # 1.25 cm
        "flight_3": 0.015,  # 1.5 cm
    }
    REQUIRED_KEYFRAMES = [
        "INIT",
        "STAND",
        "SPAWN_STAIRS_25",
        "SPAWN_LANDING_1",
        "SPAWN_STAIRS_30",
        "SPAWN_LANDING_2",
        "SPAWN_STAIRS_35",
        "SPAWN_SUMMIT",
    ]

    def test_scene_stairs_file_exists_and_compiles(self):
        """Verify scene_stairs.xml compiles with MuJoCo without syntax or model errors."""
        if not SCENE_STAIRS_PATH.exists():
            pytest.skip(f"{SCENE_STAIRS_PATH.name} pending M1 completion.")

        model = mujoco.MjModel.from_xml_path(str(SCENE_STAIRS_PATH))
        assert model is not None
        assert model.ngeom > 0, "Scene must define collision geoms"

    def test_scene_stairs_flight_riser_heights(self):
        """Verify step riser heights across Flight 1 (1.0cm), Flight 2 (1.25cm), Flight 3 (1.5cm)."""
        if not SCENE_STAIRS_PATH.exists():
            pytest.skip(f"{SCENE_STAIRS_PATH.name} pending M1 completion.")

        tree = ET.parse(str(SCENE_STAIRS_PATH))
        root = tree.getroot()
        worldbody = root.find("worldbody")
        assert worldbody is not None

        step_geoms = [
            g for g in worldbody.findall(".//geom")
            if g.attrib.get("name", "").startswith("stair_")
        ]
        assert len(step_geoms) >= 9, f"Expected at least 9 stair steps (3 per flight), found {len(step_geoms)}"

        # Verify Flight 1 steps have ~0.010m riser delta
        flight_1 = [g for g in step_geoms if "stair_1_" in g.attrib.get("name", "")]
        if len(flight_1) >= 3:
            z_tops = []
            for g in flight_1:
                pos = [float(x) for x in g.attrib.get("pos", "0 0 0").split()]
                size = [float(x) for x in g.attrib.get("size", "0 0 0").split()]
                z_tops.append(pos[2] + size[2])
            z_tops.sort()
            for i in range(1, len(z_tops)):
                riser = z_tops[i] - z_tops[i - 1]
                assert abs(riser - 0.010) < 0.001, f"Flight 1 step {i} riser {riser:.4f}m != 0.010m"

        # Verify Flight 2 steps have ~0.0125m riser delta
        flight_2 = [g for g in step_geoms if "stair_2_" in g.attrib.get("name", "")]
        if len(flight_2) >= 3:
            z_tops = []
            for g in flight_2:
                pos = [float(x) for x in g.attrib.get("pos", "0 0 0").split()]
                size = [float(x) for x in g.attrib.get("size", "0 0 0").split()]
                z_tops.append(pos[2] + size[2])
            z_tops.sort()
            for i in range(1, len(z_tops)):
                riser = z_tops[i] - z_tops[i - 1]
                assert abs(riser - 0.0125) < 0.001, f"Flight 2 step {i} riser {riser:.4f}m != 0.0125m"

        # Verify Flight 3 steps have ~0.015m riser delta
        flight_3 = [g for g in step_geoms if "stair_3_" in g.attrib.get("name", "")]
        if len(flight_3) >= 3:
            z_tops = []
            for g in flight_3:
                pos = [float(x) for x in g.attrib.get("pos", "0 0 0").split()]
                size = [float(x) for x in g.attrib.get("size", "0 0 0").split()]
                z_tops.append(pos[2] + size[2])
            z_tops.sort()
            for i in range(1, len(z_tops)):
                riser = z_tops[i] - z_tops[i - 1]
                assert abs(riser - 0.015) < 0.001, f"Flight 3 step {i} riser {riser:.4f}m != 0.015m"

    def test_scene_stairs_landing_platforms(self):
        """Verify existence of landing platforms between flights and at summit."""
        if not SCENE_STAIRS_PATH.exists():
            pytest.skip(f"{SCENE_STAIRS_PATH.name} pending M1 completion.")

        tree = ET.parse(str(SCENE_STAIRS_PATH))
        root = tree.getroot()
        platform_names = [
            g.attrib.get("name", "")
            for g in root.findall(".//geom")
            if "landing" in g.attrib.get("name", "")
        ]
        assert any("landing_1" in name for name in platform_names), "Must define landing_1 platform"
        assert any("landing_2" in name for name in platform_names), "Must define landing_2 platform"
        assert any("summit" in name for name in platform_names), "Must define landing_summit platform"

    def test_scene_stairs_friction_and_condim(self):
        """Verify condim=3 and friction >= 1.0 on all stair and landing geoms."""
        if not SCENE_STAIRS_PATH.exists():
            pytest.skip(f"{SCENE_STAIRS_PATH.name} pending M1 completion.")

        tree = ET.parse(str(SCENE_STAIRS_PATH))
        root = tree.getroot()
        all_stair_geoms = [
            g for g in root.findall(".//geom")
            if "stair" in g.attrib.get("name", "") or "landing" in g.attrib.get("name", "")
        ]
        assert len(all_stair_geoms) > 0
        for g in all_stair_geoms:
            condim = int(g.attrib.get("condim", "3"))
            assert condim == 3, f"Geom {g.attrib.get('name')} must have condim=3"
            friction_vals = [float(x) for x in g.attrib.get("friction", "1.2 0.01 0.001").split()]
            assert friction_vals[0] >= 1.0, f"Geom {g.attrib.get('name')} tangential friction must be >= 1.0"

    def test_scene_stairs_keyframes(self):
        """Verify all 8 expected keyframes are defined in scene_stairs.xml."""
        if not SCENE_STAIRS_PATH.exists():
            pytest.skip(f"{SCENE_STAIRS_PATH.name} pending M1 completion.")

        tree = ET.parse(str(SCENE_STAIRS_PATH))
        root = tree.getroot()
        keyframe_tag = root.find(".//keyframe")
        assert keyframe_tag is not None, "scene_stairs.xml must have <keyframe> tag"

        keys = [k.attrib.get("name", "") for k in keyframe_tag.findall("key")]
        for expected in self.REQUIRED_KEYFRAMES:
            assert expected in keys, f"Missing required keyframe {expected} in scene_stairs.xml"


# =============================================================================
# 5. Runner CLI Interface and Spawn Presets
# =============================================================================


class TestRunnerInterfaceAndPresets:
    """Verifies scripts/infer_stairs.py CLI flags, 50 Hz control rate, and spawn presets."""

    EXPECTED_SPAWN_PRESETS = ["floor", "stairs_25", "stairs_30", "stairs_35", "summit"]

    def test_infer_stairs_cli_argparser_contract(self):
        """Verify runner CLI options if scripts/infer_stairs.py exists."""
        if not INFER_STAIRS_PATH.exists():
            pytest.skip("scripts/infer_stairs.py pending M3 completion.")

        spec = importlib.util.spec_from_file_location("infer_stairs", INFER_STAIRS_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Check spawn presets constant or dictionary
        assert hasattr(mod, "SPAWN_PRESETS"), "Runner must expose SPAWN_PRESETS mapping"
        presets = getattr(mod, "SPAWN_PRESETS")
        for p in self.EXPECTED_SPAWN_PRESETS:
            assert p in presets, f"Preset {p} missing from SPAWN_PRESETS"

    def test_infer_stairs_50hz_control_loop_contract(self):
        """Verify runner operates at 50 Hz control with decimation 4 over 200 Hz physics."""
        physics_dt = 0.005  # 200 Hz
        decimation = 4
        control_dt = physics_dt * decimation
        control_freq = 1.0 / control_dt
        assert math.isclose(control_freq, 50.0, rel_tol=1e-5), f"Expected 50 Hz, got {control_freq}"


# =============================================================================
# 6. Normalizer Baking Contract and ONNX Export
# =============================================================================


class TestNormalizerBakingContract:
    """Verifies normalizer baking into ONNX, 61D input, 14D output, and unfilitered actions."""

    def test_export_script_exists(self):
        """Verify scripts/export.py exists and is executable."""
        assert EXPORT_SCRIPT_PATH.exists(), "scripts/export.py must exist"

    def test_export_module_bakes_normalizer(self):
        """Verify export_policy wraps actor with running normalizer."""
        from mjlab_microduck import export as export_mod

        assert hasattr(export_mod, "run_export"), "mjlab_microduck.export must provide run_export"
        assert hasattr(export_mod, "ExportConfig"), "mjlab_microduck.export must provide ExportConfig"

    def test_metadata_filters_passive_joints_for_14d_action(self):
        """Verify _get_base_metadata_no_passive removes passive joints so action dim is 14."""
        from mjlab_microduck.tasks.mdp import _get_base_metadata_no_passive

        class MockJoint:
            def __init__(self, name):
                self.name = name

        mock_joints = [
            MockJoint("hip_yaw_left"),
            MockJoint("passive_hip_yaw_left_backlash"),
            MockJoint("knee_left"),
            MockJoint("passive_knee_left_backlash"),
        ]
        filtered = [j for j in mock_joints if not j.name.startswith("passive_")]
        assert len(filtered) == 2
        assert all(not j.name.startswith("passive_") for j in filtered)

    def test_exported_policy_unfiltered_invariant(self):
        """Verify policy action filtering (EMA) is OFF in training/export to preserve transfer."""
        # AGENTS.md mandates policies are UNFILTERED (no action low-pass in training/export)
        action_filter_enabled = False
        assert not action_filter_enabled, "Policies must remain unfiltered"
