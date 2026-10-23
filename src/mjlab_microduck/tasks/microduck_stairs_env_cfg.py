"""High stairs reinforcement learning environment for Microduck.

Learns robust high stairs navigation (2.5 cm -> 3.0 cm -> 3.5 cm risers, up to
~15% of the ~25 cm robot height), dynamic posture stabilization against topple
perturbations, and automatic self-recovery / standing up when tripped or fallen.

Invariants preserved:
- Strict 61D unified actor observation layout (48D proprio + 13D commands).
- 14 Dynamixel XL330 servos with BAM M6 actuator dynamics and friction DR.
- Full-collision robot model (MICRODUCK_ALLCOLLISIONS_ROBOT_CFG, 70 geoms) for
  realistic shin/knee stair interaction and ground self-recovery.
- Contact softening (_soften_terrain_contacts) and hardened solver settings to
  prevent normal force spikes on step edges.
- Running observation normalizer baked directly into exported ONNX models.
"""

from __future__ import annotations

from copy import deepcopy
import math

import mujoco as _mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import primitive_terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
)
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    NUM_STEPS_PER_ENV,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


# ── Constants & Hyperparameters ──────────────────────────────────────────────

# Stair terrain geometry parameters (matching physical robot foot clearance: ~1.2 - 1.5 cm)
STAIR_STEP_HEIGHT_MIN = 0.010   # 1.0 cm minimum step riser
STAIR_STEP_HEIGHT_MAX = 0.035   # 1.5 cm maximum step riser (capped at 1.5cm per microduck_velocity_env_cfg line 126)
STAIR_STEP_WIDTH = 0.18         # 18 cm tread depth
STAIR_PLATFORM_WIDTH = 1.5      # 1.5 m top platform
STAIR_TILE_SIZE = (10.0, 10.0)  # 10 m x 10 m terrain patch
STAIR_BORDER_WIDTH = 1.0        # 1.0 m border around staircase

# Progressive curriculum stage boundaries (iterations)
STAGE0_ITER = 0       # Stage 0: bootstrap flat walking & initial clearance (3.5 cm)
STAGE1_ITER = 1500    # Stage 1: bootstrap stair gait & low steps (4.5 cm)
STAGE2_ITER = 3000    # Stage 2: consolidate higher lift & mid steps (5.0 cm)
STAGE3_ITER = 4500    # Stage 3: maximum step riser clearance (5.5 cm)

# Clearance target curriculum stages (matching riser height progression across 6000 iters)
CLEARANCE_TARGET_STAGES = [
    {"step": STAGE0_ITER * NUM_STEPS_PER_ENV, "target_height": 0.035},  # bootstrap clearance (3.5 cm)
    {"step": STAGE1_ITER * NUM_STEPS_PER_ENV, "target_height": 0.045},  # stage 1 clearance (4.5 cm)
    {"step": STAGE2_ITER * NUM_STEPS_PER_ENV, "target_height": 0.050},  # stage 2 clearance (5.0 cm)
    {"step": STAGE3_ITER * NUM_STEPS_PER_ENV, "target_height": 0.055},  # stage 3 target high lift (5.5 cm)
]

# Action rate smoothing curriculum: enforce slow, smooth, deliberate leg motions
ACTION_RATE_STAGES = [
    {"step": 0, "weight": -0.05},
    {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.10},
    {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.15},
    {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.20},
]

# Self-recovery parameters (proven velstand lessons: tilt-only gating, hysteresis)
REWARD_GATE_TILT_DEG = 40.0
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09
FALLEN_SMOOTHNESS_SCALE = 0.1
FELL_OVER_DISABLE_ITER = 500
RECOVERY_ECON_KICKIN_ITER = 1500
FALLEN_TIMEOUT_S = 5.0
TERM_GATE_TILT_DEG = 50.0

# Servo protection costs (scaled down early, full at PROTECT_FULL_ITER)
PROTECT_STAGE0_FRAC = 0.2
PROTECT_FULL_ITER = 2000
SERVO_IMPACT_WEIGHT = -0.02
SERVO_IMPACT_THRESH_N = 2.0
HEAD_IMPACT_WEIGHT = -0.01
HEAD_IMPACT_THRESH_N = 15.0
TRUNK_IMPACT_WEIGHT = -0.01
TRUNK_IMPACT_THRESH_N = 20.0
SERVO_ACC_SPIKE_WEIGHT = -0.001
SERVO_ACC_THRESH = 300.0
SERVO_STALL_WEIGHT = -0.05
SERVO_STALL_TORQUE = 0.65
SERVO_STALL_VEL = 0.5
GENTLE_RISE_WEIGHT = 0.005  # positive weight for self-negating trunk_vertical_accel_penalty

# Spawns: 100% upright standing on feet to dedicate full capacity to climbing stairs
PRONE_SIDE_PROB = 0.0
PRONE_RAMP_STAGES = [
    {
        "step": 0,
        "params": {
            "prone_prob": 0.0,
            "face_down_prob": 0.0,
            "side_prob": 0.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,
        },
    },
]

TOPPLE_PUSH_INTERVAL_S = (8.0, 15.0)
TOPPLE_PUSH_STAGES = [
    {"step": 0, "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.0, 0.0)}},
]


# ── Procedural Stair Terrain Generator ───────────────────────────────────────

MICRODUCK_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=STAIR_TILE_SIZE,
    border_width=20.0,
    num_rows=6,
    num_cols=5,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # 1. Flat ground apron: baseline flat walking, posture stability, and approach
        "flat": terrain_gen.BoxFlatTerrainCfg(
            proportion=0.25,
        ),
        # 2. Gentle obstacles / low steps: progressive stepping over (0.5 cm -> 2.0 cm)
        "gentle_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.005, 0.020),
            step_width=0.20,
            platform_width=1.8,
            border_width=1.5,
            holes=False,
        ),
        # 3. High stairs: full stair riser clearance (2.5 cm -> 3.5 cm)
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.40,
            step_height_range=(STAIR_STEP_HEIGHT_MIN, STAIR_STEP_HEIGHT_MAX),
            step_width=STAIR_STEP_WIDTH,
            platform_width=STAIR_PLATFORM_WIDTH,
            border_width=STAIR_BORDER_WIDTH,
            holes=False,
        ),
    },
    add_lights=False,
)


# ── Curriculum Helper Functions ─────────────────────────────────────────────

def stairs_clearance_target_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    target_stages: list[dict],
) -> torch.Tensor:
    """Step-staged clearance target curriculum."""
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    current_target = target_stages[0]["target_height"]
    for stage in target_stages:
        if env.common_step_counter >= stage["step"]:
            current_target = stage["target_height"]
    term_cfg.params["target_height"] = current_target
    return torch.tensor([current_target], device=env.device)


def terrain_levels_stairs(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Curriculum for stairs terrain level progression based on traversal distance."""
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    if terrain is None or not hasattr(terrain, "cfg") or terrain.cfg.terrain_generator is None:
        return torch.zeros(1, device=env.device)

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # Promote if robot traverses onto the stair platform (> 2.5 m)
    move_up = distance > 2.5
    # Demote if robot falls or covers less than 0.8 m
    move_down = (distance < 0.8) & (~move_up)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


class _ObservationsDict(dict):
    """Dict subclass providing .policy property for test harness parity."""

    @property
    def policy(self):
        return self.get("actor") or self.get("policy")

    @property
    def actor(self):
        return self.get("actor")

    @property
    def critic(self):
        return self.get("critic")


# ── Environment Factory ──────────────────────────────────────────────────────

def make_microduck_stairs_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the high stairs climbing RL environment configuration.

    Builds on make_microduck_velocity_env_cfg to inherit domain randomization,
    sensor noise, delay buffers, BAM startup events, and NaN guards.
    """
    # Locomotion foundation: base velocity recipe
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.observations = _ObservationsDict(cfg.observations)

    # Active deliberate command ranges: ensure robot breaks out of freezing
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.10  # 10% stationary hold (reduced from 20%)
    command.ranges.lin_vel_x = (0.05, 0.15)  # Natural stepping speed: 5 to 15 cm/s
    command.ranges.lin_vel_y = (-0.01, 0.01)  # Narrow lateral drift
    command.ranges.ang_vel_z = (-0.05, 0.05)  # Straight stair climbing alignment

    # In play mode the curriculum doesn't run, so fell_over is removed
    if play:
        cfg.terminations.pop("fell_over", None)

    # Focus spawns facing forward towards the stairs (+X) with narrow yaw and offset
    cfg.events["reset_base"].params["pose_range"]["x"] = (-0.1, 0.1)
    cfg.events["reset_base"].params["pose_range"]["y"] = (-0.1, 0.1)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.1, 0.1)

    # Full-collision robot model (70 geoms) for full-body contact physics
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}

    # Solver hardening & contact softening for step edges
    cfg.scene.spec_fn = _soften_terrain_contacts
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 300)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # Procedural stair terrain generator
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = deepcopy(MICRODUCK_STAIRS_TERRAINS_CFG)
    if play:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5

    # Contact sensors for servo, head, and trunk ground interaction
    servo_ground_cfg = ContactSensorCfg(
        name="servo_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^.*{SERVO_GEOM_SUFFIX}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        servo_ground_cfg,
        head_ground_cfg,
        trunk_ground_cfg,
    )

    # Gated head pose bias to avoid taxing droop during recovery
    if "head_pose_bias" in cfg.rewards:
        cfg.rewards["head_pose_bias"].params.update({
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG,
        })

    # Replace two-sided stock foot clearance with one-sided stairs foot clearance
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_swing_height", None)

    # Relax pose penalty on knee and hip pitch joints so robot can flex legs to clear high steps
    std_walking = cfg.rewards["pose"].params["std_walking"].copy()
    std_walking[".*knee.*"] = 1.2
    std_walking[".*hip_pitch.*"] = 1.0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking

    # Set forward velocity tracking weight to 3.5 with tight std (0.12) to actively propel robot up the stairs
    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 3.5
        cfg.rewards["track_linear_velocity"].params["std"] = 0.12


    # Heading lock: lock base heading straight ahead to prevent lateral spin and yaw instability
    if "heading_hold" in cfg.rewards:
        cfg.rewards["heading_hold"].weight = 3.0
        cfg.rewards["heading_hold"].params.update({"command_name": "twist", "std": 0.20})
    else:
        cfg.rewards["heading_hold"] = RewardTermCfg(
            func=microduck_mdp.heading_hold_reward,
            weight=3.0,
            params={"command_name": "twist", "std": 0.20},
        )

    # Endogenous heading tracking potential (Rudin et al., Science Robotics 2022)
    cfg.rewards["track_heading"] = RewardTermCfg(
        func=microduck_mdp.track_heading_potential,
        weight=2.0,
        params={"command_name": "twist", "std": 0.20},
    )

    # Torso forward pitch lean reward: induce continuous anterior pitch inclination (APA)
    cfg.rewards["torso_forward_pitch"] = RewardTermCfg(
        func=microduck_mdp.torso_forward_pitch_reward,
        weight=2.5,
        params={
            "target_forward_lean": 0.10,
            "lean_std": 0.06,
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # ── Custom Stairs MDP Terms ───────────────────────────────────────────────

    # 1. One-sided stairs foot clearance penalty (<= 0 -> positive weight)
    cfg.rewards["stairs_foot_clearance"] = RewardTermCfg(
        func=microduck_mdp.stairs_foot_clearance_penalty,
        weight=2.0,
        params={
            "target_height": CLEARANCE_TARGET_STAGES[0]["target_height"],
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.01,
            "vel_threshold": 0.02,
        },
    )

    # 2. Foot touchdown impact damping & normal velocity penalty (<= 0 -> positive weight)
    cfg.rewards["stairs_foot_landing_damping"] = RewardTermCfg(
        func=microduck_mdp.stairs_foot_landing_damping_penalty,
        weight=0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "impact_force_thresh": 8.0,
            "impact_vel_thresh": 0.08,
            "vel_weight": 10.0,
        },
    )

    # 3. Vertical stair riser trip / shear penalty (<= 0 -> positive weight)
    cfg.rewards["stairs_riser_trip"] = RewardTermCfg(
        func=microduck_mdp.stairs_riser_trip_penalty,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_friction_ratio": 0.85,
        },
    )

    # 4. Foot sole flat stance contact penalty (<= 0 -> positive weight)
    cfg.rewards["stairs_sole_flat"] = RewardTermCfg(
        func=microduck_mdp.stairs_sole_flat_penalty,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "tilt_deadband_deg": 12.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )

    # 5. Potential-based relative height progress (elevation-invariant Delta z_rel)
    cfg.rewards["stairs_relative_height_progress"] = RewardTermCfg(
        func=microduck_mdp.stairs_relative_height_progress,
        weight=3.0,
        params={
            "ceiling": 0.115,
        },
    )

    # 6. Stairs Elevation Gain Progress: pays positive reward every time robot climbs up to higher stair steps
    cfg.rewards["stair_elevation_gain_progress"] = RewardTermCfg(
        func=microduck_mdp.stair_elevation_gain_progress,
        weight=5.0,
        params={
            "max_elevation": 0.50,
        },
    )

    # 7. Single-leg stance balance reward (stable upright balance on one leg while stepping up)
    cfg.rewards["single_leg_stance_balance"] = RewardTermCfg(
        func=microduck_mdp.single_leg_stance_balance_reward,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # 8. Swing leg digitigrade knee flexion backward & forward hip reach
    cfg.rewards["swing_knee_flexion"] = RewardTermCfg(
        func=microduck_mdp.swing_knee_flexion_reward,
        weight=2.5,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "target_knee_flex": 0.55,
            "knee_std": 0.25,
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # 9. Stance leg forward pitch lean (shifting CoM forward over upper tread)
    cfg.rewards["stance_forward_pitch_lean"] = RewardTermCfg(
        func=microduck_mdp.stance_forward_pitch_lean_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "target_forward_lean": 0.12,
            "lean_std": 0.08,
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # 10. Stance push-off propulsion (forward extension and positive work lifting body)
    cfg.rewards["stance_push_off_propulsion"] = RewardTermCfg(
        func=microduck_mdp.stance_push_off_propulsion_reward,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "min_fwd_vel": 0.03,
        },
    )

    # 11. Double-support stagnation penalty (anti-stalling against stair risers)
    cfg.rewards["double_support_stagnation"] = RewardTermCfg(
        func=microduck_mdp.double_support_stagnation_penalty,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "command_name": "twist",
            "cmd_threshold": 0.03,
            "max_double_time_s": 0.65,  # 0.65s (increased from 0.40s) allows full coronal pelvic sway transfer before leg lift
        },
    )

    # 12. Swing foot sole leveling (maintaining horizontal sole pitch 0 deg on swing foot)
    cfg.rewards["swing_foot_sole_leveling"] = RewardTermCfg(
        func=microduck_mdp.swing_foot_sole_leveling_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # 13. Servo torque headroom penalty (protect XL330 plastic gears past 75% stall torque limit)
    cfg.rewards["servo_torque_headroom"] = RewardTermCfg(
        func=microduck_mdp.servo_torque_headroom_penalty,
        weight=1.0,
        params={
            "safe_torque_limit": 0.72,
        },
    )

    # 14. Alternating bipedal stepping reward (enforces left-right-left sequencing, prevents hopping)
    cfg.rewards["stairs_alternating_gait"] = RewardTermCfg(
        func=microduck_mdp.stairs_alternating_gait_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "contact_force_thresh": 2.0,
            "command_name": "twist",
            "cmd_threshold": 0.03,
        },
    )

    # 15. Swing foot vertical velocity cap (penalize upward jerking vz > 0.16 m/s for slow gentle rise)
    cfg.rewards["swing_foot_vertical_velocity"] = RewardTermCfg(
        func=microduck_mdp.swing_foot_vertical_velocity_penalty,
        weight=2.5,
        params={
            "sensor_name": "feet_ground_contact",
            "max_safe_vz": 0.16,
            "swing_force_thresh": 1.5,
        },
    )

    # 16. Single-leg continuous dynamic balance reward (damp roll/pitch ang vel while leg is elevated)
    cfg.rewards["single_leg_continuous_balance"] = RewardTermCfg(
        func=microduck_mdp.single_leg_continuous_balance_reward,
        weight=4.0,  # Maximized weight for single-leg dynamic balance
        params={
            "sensor_name": "feet_ground_contact",
            "stance_force_thresh": 3.5,
            "swing_force_thresh": 1.5,
            "max_tilt_deg": 18.0,
            "min_trunk_z": 0.10,
            "roll_pitch_rate_std": 0.25,
            "command_name": "twist",
            "cmd_threshold": 0.02,
        },
    )

    # 17. Smooth high-step clearance reward (Gaussian elevation tracking for 5.5 cm high clearance)
    cfg.rewards["smooth_high_step_clearance"] = RewardTermCfg(
        func=microduck_mdp.smooth_high_step_clearance_reward,
        weight=2.5,
        params={
            "height_sensor_name": "foot_height_scan",
            "sensor_name": "feet_ground_contact",
            "target_clearance": 0.055,  # 5.5 cm target elevation (higher than previous 4.1 - 4.6 cm)
            "clearance_std": 0.016,
            "swing_force_thresh": 1.5,
            "command_name": "twist",
            "cmd_threshold": 0.02,
        },
    )

    # 18. Stance ankle roll stability penalty (catches extreme hyperextension > 16 deg)
    cfg.rewards["stance_ankle_roll"] = RewardTermCfg(
        func=microduck_mdp.stance_ankle_roll_penalty,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "stance_force_thresh": 3.5,
            "max_roll_dev_deg": 16.0,
        },
    )

    # 19. Coronal Pelvic Sway (active lateral weight shift over stance foot for 3D LIPM balance)
    cfg.rewards["coronal_pelvic_sway"] = RewardTermCfg(
        func=microduck_mdp.coronal_pelvic_sway_reward,
        weight=3.5,  # Increased to 3.5 to promote lateral weight transfer before swing initiation
        params={
            "sensor_name": "feet_ground_contact",
            "stance_force_thresh": 3.5,
            "swing_force_thresh": 1.5,
            "target_sway_deg": 8.5,
            "sway_std_deg": 4.0,
            "command_name": "twist",
            "cmd_threshold": 0.02,
        },
    )

    # ── Self-Recovery Layer ───────────────────────────────────────────────────
    # Potential-based upright progress (Delta cos tilt)
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Standard trunk height progress companion
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": 0.115,
        },
    )

    # Upward trunk velocity incentive when fallen (0.0 weight)
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )

    # Smoothness taxes scaled down by FALLEN_SMOOTHNESS_SCALE while fallen
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2_fallen_scaled,
        weight=-2e-3,
        params={
            "fallen_scale": FALLEN_SMOOTHNESS_SCALE,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    ar = cfg.rewards["action_rate_l2"]
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.action_rate_l2_fallen_scaled,
        weight=ar.weight,
        params={
            "fallen_scale": FALLEN_SMOOTHNESS_SCALE,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )

    # Servo-protection costs (body_impact_cost >= 0 -> negative weight)
    cfg.rewards["servo_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=SERVO_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": servo_ground_cfg.name, "threshold": SERVO_IMPACT_THRESH_N},
    )
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=HEAD_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=TRUNK_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": TRUNK_IMPACT_THRESH_N},
    )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=SERVO_ACC_SPIKE_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"acc_thresh": SERVO_ACC_THRESH},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=SERVO_STALL_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"torque_thresh": SERVO_STALL_TORQUE, "vel_thresh": SERVO_STALL_VEL},
    )
    # Self-negating trunk vertical accel penalty (returns -|a_z| <= 0 -> positive weight)
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=GENTLE_RISE_WEIGHT * PROTECT_STAGE0_FRAC,
    )

    # Recovery economics: zero air time while fallen, tax lying still, bounty on completed stand
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=at.weight,
        params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    # Phạt liên tục nếu nằm lì hoặc không chịu tự đứng dậy sau khi bị ngã/xô
    cfg.rewards["fallen_persistence"] = RewardTermCfg(
        func=microduck_mdp.fallen_persistence_penalty,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            "grace_period_s": 0.8,
            "max_penalty_scale": 5.0,
        },
    )
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": RECOVERED_UP_TILT_DEG,
            "up_z": RECOVERED_UP_Z,
        },
    )

    # ── Events: Prone inits & Topple Pushes ───────────────────────────────────

    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,
            "face_down_prob": 1.0,
            "side_prob": PRONE_SIDE_PROB,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,
        },
    )
    cfg.events["topple_push"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 2.0) if play else TOPPLE_PUSH_INTERVAL_S,
        params={
            "velocity_range": TOPPLE_PUSH_STAGES[-1 if play else 0]["velocity_range"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── Terminations ──────────────────────────────────────────────────────────

    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )

    # ── Curricula ─────────────────────────────────────────────────────────────

    if not play:
        # fell_over termination remains STRICTLY ACTIVE throughout training (limit_angle=65°).
        # Falling over immediately truncates future rewards, making flopping completely unviable.
        cfg.curriculum["topple_push_range"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "topple_push", "push_stages": TOPPLE_PUSH_STAGES},
        )

    # Prone spawn ramp
    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": PRONE_RAMP_STAGES,
        },
    )

    # Recovery economics ramp: tax + bounty active after walk bootstraps
    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )

    # Servo-protection ramp
    for name, full in (
        ("servo_impact", SERVO_IMPACT_WEIGHT),
        ("head_impact", HEAD_IMPACT_WEIGHT),
        ("trunk_impact", TRUNK_IMPACT_WEIGHT),
        ("servo_acc_spike", SERVO_ACC_SPIKE_WEIGHT),
        ("servo_stall", SERVO_STALL_WEIGHT),
        ("gentle_rise", GENTLE_RISE_WEIGHT),
    ):
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": name,
                "weight_stages": [
                    {"step": 0, "weight": full * PROTECT_STAGE0_FRAC},
                    {"step": PROTECT_FULL_ITER * NUM_STEPS_PER_ENV, "weight": full},
                ],
            },
        )

    # Action rate smoothing ramp
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": ACTION_RATE_STAGES,
        },
    )

    # Stairs clearance target progression (2.5 cm -> 3.0 cm -> 3.5 cm)
    cfg.curriculum["stairs_clearance_target"] = CurriculumTermCfg(
        func=stairs_clearance_target_curriculum,
        params={
            "reward_name": "stairs_foot_clearance",
            "target_stages": CLEARANCE_TARGET_STAGES,
        },
    )

    # Terrain difficulty progression based on distance covered
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=terrain_levels_stairs,
    )

    return cfg


# ── Runner Configuration ─────────────────────────────────────────────────────

MicroduckStairsRlCfg = RslRlOnPolicyRunnerCfg(
    clip_actions=1.0,
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,  # Symmetry OFF for stairs
    ),
    wandb_project="mjlab_microduck",
    experiment_name="stairs_climb",
    run_name="stairs_climb",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
