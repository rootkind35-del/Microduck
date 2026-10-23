#!/usr/bin/env python3
"""Stage C — Bridge 3 (Part 1): NVIDIA Isaac Lab & Newton Physics RL Environment Specification.

Defines the cross-simulator `MicroduckIsaacLabNewtonSpec` that binds the OpenUSD
assets produced in Stage B (`02_blender_omniverse_usd/usd_assets/microduck_simready.usda`
and `terrain_course.usda`) to both:
  1. **NVIDIA Isaac Lab / Newton Physics Engine** (`ManagerBasedRLEnvCfg` / `ArticulationCfg`
     referencing the SimReady `.usda` stage)
  2. **MuJoCo Warp (`mjlab` 1.3.0)** running on NVIDIA Warp GPU kernels (`warp-lang 1.12.0`),
     preserving 100% parity with the Microduck Sim-to-Real contract:
       - 61D Actor Observation Space (`48D proprioception + 13D command [twist(3), head_pose(4), body_pose(6)]`)
       - 14D Action Space (Dynamixel XL330 BAM M6 actuator dynamics)
       - 50 Hz control frequency (`decimation = 4`, `sim.dt = 0.005s`)
       - Neko Core Hierarchical Controller (`QuinticSkillBlender` + `GaitPhaseLatch`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STAGE_C_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_C_DIR.parent
STAGE_B_USD_DIR = PROJECT_ROOT / "02_blender_omniverse_usd" / "usd_assets"


@dataclass
class IsaacLabNewtonArticulationSpec:
    """Articulation configuration for Microduck in Isaac Lab / Newton Physics."""

    prim_path: str = "{ENV_REGEX_NS}/Microduck"
    usd_path: str = str(STAGE_B_USD_DIR / "microduck_simready.usda")
    urdf_path: str = str(STAGE_B_USD_DIR / "microduck.urdf")
    terrain_usd_path: str = str(STAGE_B_USD_DIR / "terrain_course.usda")
    actuated_joint_names: tuple[str, ...] = (
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    )
    passive_joint_regex: str = r"^passive_.*"
    actuator_model: str = "BAM_M6_XL330_VoltageControl"
    kp_firmware: float = 200.0
    vin_range_v: tuple[float, float] = (6.5, 8.2)
    max_effort_nm: float = 0.963
    max_velocity_rad_s: float = 9.95
    init_pos_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.125)
    init_joint_pos_rad: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "left_hip_pitch": -0.4579,
            "left_knee": -0.0049,
            "left_ankle": 0.4530,
            "neck_pitch": 0.3491,
            "head_pitch": 0.3491,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0873,
            "right_hip_pitch": 0.4579,
            "right_knee": 0.0049,
            "right_ankle": -0.4530,
        }
    )


@dataclass
class MicroduckIsaacLabNewtonEnvSpec:
    """Unified RL Environment Specification for Newton / Isaac Lab & MuJoCo Warp."""

    task_id: str = "Mjlab-StairsClimb-MicroDuck"
    physics_engine: str = "NVIDIA_Warp_Newton_MJWarp"
    sim_dt_s: float = 0.005
    decimation: int = 4
    control_freq_hz: float = 50.0
    episode_length_s: float = 20.0
    num_envs_default: int = 4096
    actor_obs_dim: int = 61
    action_dim: int = 14
    actor_obs_layout: tuple[str, ...] = (
        "base_ang_vel(3)",
        "projected_gravity(3)",
        "joint_pos_rel_14(14)",
        "joint_vel_14(14)",
        "actions_14(14)",
        "command_twist_vx_vy_wz(3)",
        "head_command_4d(4)",
        "body_command_6d(6)",
    )
    curriculum_stages_iters: tuple[int, int, int, int] = (0, 1500, 3000, 4500)
    swing_clearance_stages_m: tuple[float, float, float, float] = (0.035, 0.045, 0.050, 0.055)
    hierarchical_coordinator: str = "NekoCore(QuinticSkillBlender_C2 + GaitPhaseLatch_DoubleSupport)"
    articulation: IsaacLabNewtonArticulationSpec = field(
        default_factory=IsaacLabNewtonArticulationSpec
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_isaaclab_newton_spec(task_id: str = "Mjlab-StairsClimb-MicroDuck") -> MicroduckIsaacLabNewtonEnvSpec:
    """Return the cross-simulator RL environment specification."""
    return MicroduckIsaacLabNewtonEnvSpec(task_id=task_id)

