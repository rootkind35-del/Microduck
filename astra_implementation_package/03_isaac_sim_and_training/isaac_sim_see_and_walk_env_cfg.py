#!/usr/bin/env python3
"""Astra Implementation Module — Isaac Sim / Isaac Lab CUDA RTX Vision-Guided Rough Terrain & Stairs Config.

Configures Microduck's 61D Actor Observation + Onboard RayCaster / TiledCameraCfg (OpenGL / RTX)
and BAM M6 voltage-controlled actuators for parallel GPU training on the 4-Zone Rough Terrain & Staircase Course.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IsaacSimVisionCameraCfg:
    """RTX / OpenGL Egocentric Head Camera & ToF Sensor Specification."""

    prim_path: str = "{ENV_REGEX_NS}/Robot/Links/head/OnboardPerceptionSensors/head_camera"
    update_period_s: float = 0.04  # 25 Hz visual perception update
    height: int = 120
    width: int = 160
    data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane")
    focal_length_mm: float = 18.0
    clipping_range_m: tuple[float, float] = (0.02, 10.0)
    tof_prim_path: str = "{ENV_REGEX_NS}/Robot/Links/head/OnboardPerceptionSensors/tof_rangefinder"


@dataclass
class AstraSeeAndWalkEnvSpec:
    """Complete Isaac Lab / Newton / MuJoCo Warp Rough Terrain & Stair Climbing Specification."""

    task_id: str = "Isaac-SeeAndWalk-RoughStairs-Microduck-v0"
    mjlab_rough_task_id: str = "Mjlab-Velocity-Rough-MicroDuck"
    mjlab_stairs_task_id: str = "Mjlab-StairsClimb-MicroDuck"
    usd_stage_path: str = "02_blender_omniverse_usd/usd_assets/vision_obstacle_course.usda"
    urdf_path: str = "02_blender_omniverse_usd/usd_assets/microduck.urdf"
    num_envs: int = 4096
    control_freq_hz: int = 50
    physics_freq_hz: int = 200
    actor_obs_dim: int = 61
    critic_obs_dim: int = 76
    num_actuated_servos: int = 14
    camera_cfg: IsaacSimVisionCameraCfg = field(default_factory=IsaacSimVisionCameraCfg)
    neko_core_transition_s: float = 0.25
    neko_dsp_force_threshold_n: float = 1.5
