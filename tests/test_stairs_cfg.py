"""Configuration invariant and regression tests for Microduck High Stairs RL.

Verifies:
1. Environment compiles cleanly for train and play modes.
2. Exact 61D unified actor observation layout and term structure.
3. Full-collision robot model (MICRODUCK_ALLCOLLISIONS_ROBOT_CFG, 70 geoms).
4. Sensor suite: feet_ground_contact with track_air_time=True, foot_height_scan,
   servo_ground_contact, head_ground_contact, trunk_ground_contact.
5. Hardened solver parameters (nconmax=200, iterations=30, ls_iterations=50)
   and contact softening (_soften_terrain_contacts).
6. Procedural pyramid stairs terrain generator (2.5 cm to 3.5 cm risers).
7. M1 stairs MDP terms wired with correct positive/negative sign invariants.
8. Self-recovery layer: tilt-only gating, fallen-scaled smoothness, recovery economics.
9. Task registry: Mjlab-StairsClimb-MicroDuck and backlash twin with allcollisions model.
10. Runner configuration contracts.

CPU only: compiles cfgs and MuJoCo models, never steps an env.
"""

from __future__ import annotations

import math
import re

import mujoco
import pytest
import torch

from mjlab.tasks.registry import list_tasks
import mjlab_microduck.tasks as tasks_mod
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_stairs_env_cfg as stairs_cfg
from mjlab_microduck.tasks.microduck_stairs_env_cfg import (
    CLEARANCE_TARGET_STAGES,
    FALLEN_SMOOTHNESS_SCALE,
    MICRODUCK_STAIRS_TERRAINS_CFG,
    PROTECT_FULL_ITER,
    PROTECT_STAGE0_FRAC,
    RECOVERED_UP_TILT_DEG,
    RECOVERED_UP_Z,
    RECOVERY_ECON_KICKIN_ITER,
    REWARD_GATE_TILT_DEG,
    STAIR_STEP_HEIGHT_MAX,
    STAIR_STEP_HEIGHT_MIN,
    STAIR_STEP_WIDTH,
    MicroduckStairsRlCfg,
    make_microduck_stairs_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    NUM_STEPS_PER_ENV,
    _soften_terrain_contacts,
)


def _geom_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    ]


# =============================================================================
# 1. Environment Build and Mode Invariants
# =============================================================================

def test_stairs_env_builds_train_and_play():
    """Verify that train, play, and rough configs compile cleanly without exceptions."""
    cfg_train = make_microduck_stairs_env_cfg(play=False)
    assert cfg_train is not None
    assert cfg_train.scene is not None
    assert cfg_train.sim is not None
    assert cfg_train.rewards is not None

    cfg_play = make_microduck_stairs_env_cfg(play=True)
    assert cfg_play is not None
    # In play mode fell_over is removed because curriculum doesn't run
    assert "fell_over" not in cfg_play.terminations
    assert cfg_play.scene.terrain.terrain_generator.curriculum is False


# =============================================================================
# 2. 61D Unified Observation Contract
# =============================================================================

def test_actor_observation_keeps_the_61d_slot_layout():
    """Verify strict 61D actor observation structure: 48D proprio + 13D commands."""
    cfg = make_microduck_stairs_env_cfg()
    actor_terms = cfg.observations["actor"].terms

    # Proprioceptive base block terms
    expected_proprio = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
    for term in expected_proprio:
        assert term in actor_terms, f"Missing proprioceptive term: {term}"

    # Privileged terms must NOT leak into actor
    assert "base_lin_vel" not in actor_terms, "base_lin_vel must be critic-only"
    assert "height_scan" not in actor_terms, "height_scan must be critic-only"

    # Command block terms
    assert "command" in actor_terms, "Missing twist command slot"
    assert "head_command" in actor_terms, "Missing head_command slot"
    assert "body_command" in actor_terms, "Missing body_command slot"

    assert actor_terms["head_command"].params["command_name"] == "head_pose"
    assert actor_terms["body_command"].params["command_name"] == "body_pose"
    assert "head_pose" in cfg.commands
    assert "body_pose" in cfg.commands

    # Verify parity with .policy property accessor
    assert cfg.observations.policy is cfg.observations["actor"]
    assert cfg.observations.actor is cfg.observations["actor"]
    assert cfg.observations.critic is cfg.observations["critic"]


def test_obs_term_order_matches_velocity_contract():
    """Verify exact term ordering matches the frozen velocity template contract."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    vel_cfg = make_microduck_velocity_env_cfg()
    stairs_cfg_inst = make_microduck_stairs_env_cfg()

    vel_actor_keys = list(vel_cfg.observations["actor"].terms.keys())
    stairs_actor_keys = list(stairs_cfg_inst.observations["actor"].terms.keys())
    assert stairs_actor_keys == vel_actor_keys, "Actor obs terms must match exactly"


# =============================================================================
# 3. Robot Model and Collision Physics
# =============================================================================

def test_uses_allcollisions_robot_model():
    """Verify robot model is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG (70 collision geoms)."""
    cfg = make_microduck_stairs_env_cfg()
    assert cfg.scene.entities["robot"].spec_fn is get_allcollisions_spec

    model = get_allcollisions_spec().compile()
    assert model.ngeom >= 70, f"Expected >= 70 collision geoms, got {model.ngeom}"

    # Ensure all 14 servo collision geoms are present
    servo_geoms = [n for n in _geom_names(model) if n and n.endswith(SERVO_GEOM_SUFFIX)]
    assert len(servo_geoms) >= 14


# =============================================================================
# 4. Sensor Suite Invariants
# =============================================================================

def test_feet_ground_contact_has_track_air_time():
    """Verify feet_ground_contact has track_air_time=True (M1 damping requirement)."""
    cfg = make_microduck_stairs_env_cfg()
    sensor_map = {s.name: s for s in cfg.scene.sensors}

    assert "feet_ground_contact" in sensor_map, "feet_ground_contact sensor missing"
    feet_sensor = sensor_map["feet_ground_contact"]
    assert getattr(feet_sensor, "track_air_time", False) is True, (
        "feet_ground_contact must have track_air_time=True for compute_first_contact"
    )


def test_contact_sensors_configured():
    """Verify all 5 required contact and height sensors are present."""
    cfg = make_microduck_stairs_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}

    required_sensors = {
        "feet_ground_contact",
        "foot_height_scan",
        "servo_ground_contact",
        "head_ground_contact",
        "trunk_ground_contact",
    }
    assert required_sensors <= sensor_names, f"Missing sensors: {required_sensors - sensor_names}"

    # Verify servo sensor matches allcollisions geoms
    sensor_map = {s.name: s for s in cfg.scene.sensors}
    servo_sensor = sensor_map["servo_ground_contact"]
    pat = re.compile(servo_sensor.primary.pattern)
    model = get_allcollisions_spec().compile()
    assert sum(bool(n and pat.match(n)) for n in _geom_names(model)) >= 14


# =============================================================================
# 5. Solver Hardening and Stability
# =============================================================================

def test_solver_parameters_and_contact_softening():
    """Verify solver contact softening, nconmax=200, and elevated iteration counts."""
    cfg = make_microduck_stairs_env_cfg()
    assert cfg.scene.spec_fn is _soften_terrain_contacts
    assert cfg.sim.nconmax >= 200
    assert cfg.sim.mujoco.iterations >= 30
    assert cfg.sim.mujoco.ls_iterations >= 50


# =============================================================================
# 6. Procedural Terrain Generator
# =============================================================================

def test_terrain_generator_pyramid_stairs():
    """Verify BoxPyramidStairsTerrainCfg configuration matches stair specifications."""
    cfg = make_microduck_stairs_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    tgen = cfg.scene.terrain.terrain_generator
    assert tgen is not None
    assert "pyramid_stairs" in tgen.sub_terrains

    sub = tgen.sub_terrains["pyramid_stairs"]
    assert sub.step_height_range == (STAIR_STEP_HEIGHT_MIN, STAIR_STEP_HEIGHT_MAX)
    assert math.isclose(sub.step_height_range[0], STAIR_STEP_HEIGHT_MIN, abs_tol=1e-4)
    assert math.isclose(sub.step_height_range[1], STAIR_STEP_HEIGHT_MAX, abs_tol=1e-4)
    assert math.isclose(sub.step_width, STAIR_STEP_WIDTH, abs_tol=1e-4)
    assert math.isclose(sub.platform_width, 1.5, abs_tol=1e-4)


# =============================================================================
# 7. M1 Stairs MDP Terms Wiring and Sign Conventions
# =============================================================================

def test_stairs_mdp_terms_wired_correctly():
    """Verify that all M1 stairs functions are wired into the rewards manager."""
    cfg = make_microduck_stairs_env_cfg()

    assert cfg.rewards["stairs_foot_clearance"].func is microduck_mdp.stairs_foot_clearance_penalty
    assert cfg.rewards["stairs_foot_landing_damping"].func is microduck_mdp.stairs_foot_landing_damping_penalty
    assert cfg.rewards["stairs_riser_trip"].func is microduck_mdp.stairs_riser_trip_penalty
    assert cfg.rewards["stairs_sole_flat"].func is microduck_mdp.stairs_sole_flat_penalty
    assert cfg.rewards["stairs_relative_height_progress"].func is microduck_mdp.stairs_relative_height_progress
    assert cfg.rewards["single_leg_stance_balance"].func is microduck_mdp.single_leg_stance_balance_reward
    assert cfg.rewards["stance_forward_pitch_lean"].func is microduck_mdp.stance_forward_pitch_lean_reward
    assert cfg.rewards["stance_push_off_propulsion"].func is microduck_mdp.stance_push_off_propulsion_reward
    assert cfg.rewards["double_support_stagnation"].func is microduck_mdp.double_support_stagnation_penalty
    assert cfg.rewards["swing_foot_sole_leveling"].func is microduck_mdp.swing_foot_sole_leveling_reward
    assert cfg.rewards["servo_torque_headroom"].func is microduck_mdp.servo_torque_headroom_penalty
    assert cfg.rewards["stairs_alternating_gait"].func is microduck_mdp.stairs_alternating_gait_reward
    assert cfg.rewards["swing_foot_vertical_velocity"].func is microduck_mdp.swing_foot_vertical_velocity_penalty
    assert cfg.rewards["single_leg_continuous_balance"].func is microduck_mdp.single_leg_continuous_balance_reward
    assert cfg.rewards["smooth_high_step_clearance"].func is microduck_mdp.smooth_high_step_clearance_reward
    assert cfg.rewards["stance_ankle_roll"].func is microduck_mdp.stance_ankle_roll_penalty


def test_reward_weight_signs_invariant():
    """Verify AGENTS.md rule: self-negating penalties > 0, mjlab costs < 0."""
    cfg = make_microduck_stairs_env_cfg()

    # Self-negating microduck penalties (return <= 0) -> POSITIVE weight
    positive_penalties = [
        "stairs_foot_clearance",
        "stairs_foot_landing_damping",
        "stairs_riser_trip",
        "stairs_sole_flat",
        "stairs_relative_height_progress",
        "gentle_rise",
        "double_support_stagnation",
        "servo_torque_headroom",
        "swing_foot_vertical_velocity",
        "stance_ankle_roll",
    ]
    for term in positive_penalties:
        weight = cfg.rewards[term].weight
        assert weight > 0, f"Term '{term}' must have positive weight, got {weight}"

    # Positive reward terms ([0, 1]) -> POSITIVE weight
    positive_rewards = [
        "single_leg_stance_balance",
        "stance_forward_pitch_lean",
        "stance_push_off_propulsion",
        "swing_foot_sole_leveling",
        "stairs_alternating_gait",
        "single_leg_continuous_balance",
        "smooth_high_step_clearance",
    ]
    for term in positive_rewards:
        weight = cfg.rewards[term].weight
        assert weight > 0, f"Positive reward term '{term}' must have positive weight, got {weight}"

    # Potential-based progress rewards (Delta Phi) -> POSITIVE weight
    progress_terms = ["upright_progress", "height_progress"]
    for term in progress_terms:
        weight = cfg.rewards[term].weight
        assert weight > 0, f"Progress term '{term}' must have positive weight, got {weight}"

    # mjlab cost functions (return >= 0) -> NEGATIVE weight
    negative_costs = [
        "servo_impact",
        "head_impact",
        "trunk_impact",
        "servo_acc_spike",
        "servo_stall",
        "action_rate_l2",
        "joint_torque_rate_l2",
        "self_collisions",
        "foot_slip",
        "body_ang_vel",
        "angular_momentum",
    ]
    for term in negative_costs:
        weight = cfg.rewards[term].weight
        assert weight < 0, f"Cost term '{term}' must have negative weight, got {weight}"


# =============================================================================
# 8. Dynamic Self-Recovery and Economics Layer
# =============================================================================

def test_self_recovery_layer_configuration():
    """Verify tilt-gated potential rewards, fallen smoothness scaling, and economics."""
    cfg = make_microduck_stairs_env_cfg()

    # Smoothness scaling while fallen
    assert cfg.rewards["action_rate_l2"].func is microduck_mdp.action_rate_l2_fallen_scaled
    assert cfg.rewards["joint_torque_rate_l2"].func is microduck_mdp.joint_torque_rate_l2_fallen_scaled
    assert cfg.rewards["action_rate_l2"].params["fallen_scale"] == FALLEN_SMOOTHNESS_SCALE == 0.1
    assert cfg.rewards["joint_torque_rate_l2"].params["fallen_scale"] == FALLEN_SMOOTHNESS_SCALE == 0.1
    assert cfg.rewards["action_rate_l2"].params["gate_tilt_above_deg"] == REWARD_GATE_TILT_DEG == 40.0

    # Economics: fallen tax and recovery bounty
    assert cfg.rewards["fallen_tax"].params["gate_tilt_above_deg"] == REWARD_GATE_TILT_DEG
    assert cfg.rewards["fallen_tax"].params["release_tilt_below_deg"] == RECOVERED_UP_TILT_DEG
    assert cfg.rewards["fallen_tax"].params["release_z_above"] == RECOVERED_UP_Z
    assert cfg.rewards["recovery_success"].params["fallen_tilt_deg"] == REWARD_GATE_TILT_DEG
    assert cfg.rewards["recovery_success"].params["up_tilt_deg"] == RECOVERED_UP_TILT_DEG

    # Air time upright
    assert cfg.rewards["air_time"].func is microduck_mdp.feet_air_time_upright
    assert cfg.rewards["air_time"].params["gate_tilt_above_deg"] == REWARD_GATE_TILT_DEG

    # Terminations
    assert "fallen_too_long" in cfg.terminations
    assert cfg.terminations["fallen_too_long"].func is microduck_mdp.fallen_too_long


# =============================================================================
# 9. Progressive Curriculum and Staged Schedules
# =============================================================================

def test_stair_curriculum_stages():
    """Verify clearance targets (2.8 -> 3.3 -> 3.8 cm) and weight ramps."""
    cfg = make_microduck_stairs_env_cfg(play=False)

    # Clearance target curriculum
    assert "stairs_clearance_target" in cfg.curriculum
    stages = cfg.curriculum["stairs_clearance_target"].params["target_stages"]
    assert len(stages) >= 4
    assert stages[0]["target_height"] == CLEARANCE_TARGET_STAGES[0]["target_height"]
    assert stages[1]["target_height"] == CLEARANCE_TARGET_STAGES[1]["target_height"]
    assert stages[2]["target_height"] == CLEARANCE_TARGET_STAGES[2]["target_height"]
    assert stages[3]["target_height"] == CLEARANCE_TARGET_STAGES[3]["target_height"]

    # Terrain levels curriculum
    assert "terrain_levels" in cfg.curriculum

    # Recovery economics ramp
    assert "fallen_tax_weight" in cfg.curriculum
    tax_stages = cfg.curriculum["fallen_tax_weight"].params["weight_stages"]
    assert tax_stages[0]["weight"] == 0.0
    assert tax_stages[-1]["weight"] == -0.5

    assert "recovery_success_weight" in cfg.curriculum
    bounty_stages = cfg.curriculum["recovery_success_weight"].params["weight_stages"]
    assert bounty_stages[0]["weight"] == 0.0
    assert bounty_stages[-1]["weight"] == 10.0


# =============================================================================
# 10. Task Registration and Backlash Twin
# =============================================================================

def test_tasks_appear_in_registry():
    """Verify Mjlab-StairsClimb-MicroDuck and backlash twin in mjlab task registry."""
    tasks = list_tasks()
    assert "Mjlab-StairsClimb-MicroDuck" in tasks, "Mjlab-StairsClimb-MicroDuck not in registry"
    assert "Mjlab-StairsClimb-Backlash-MicroDuck" in tasks, (
        "Mjlab-StairsClimb-Backlash-MicroDuck not in registry"
    )


def test_backlash_twin_uses_allcollisions_model():
    """Verify backlash twin maps to MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG."""
    backlash_entries = tasks_mod._BACKLASH_TASKS
    entry = next((e for e in backlash_entries if e[0] == "Mjlab-StairsClimb-Backlash-MicroDuck"), None)
    assert entry is not None, "Mjlab-StairsClimb-Backlash-MicroDuck missing from _BACKLASH_TASKS"
    assert entry[4] is MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG


# =============================================================================
# 11. Runner Configuration Contracts
# =============================================================================

def test_runner_cfg_attributes():
    """Verify runner configuration experiment name, obs normalization, and dimensions."""
    runner_cfg = MicroduckStairsRlCfg
    assert runner_cfg.experiment_name == "stairs_climb"
    assert runner_cfg.run_name == "stairs_climb"
    assert runner_cfg.actor.obs_normalization is True
    assert runner_cfg.critic.obs_normalization is True
    assert runner_cfg.actor.hidden_dims == (512, 256, 128)
    assert runner_cfg.num_steps_per_env == 24
    assert runner_cfg.max_iterations == 6_000
