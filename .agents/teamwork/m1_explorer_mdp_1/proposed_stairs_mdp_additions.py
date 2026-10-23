"""Stairs Climbing MDP Functions for src/mjlab_microduck/tasks/mdp.py

Validated and hardened by M1 MDP Explorer.
Includes:
- _FEET_SITE_CFG definition isolating the 2 foot sites ('left_foot', 'right_foot')
  from the 7-site Microduck model.
- NaN sanitization across all sensor reads and velocity/quaternion inputs.
- Robust contact gate fallbacks.
- Strictly self-negating penalties (<= 0.0 under all inputs).
- Potential-based relative height progress with episode reset masking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

# Dedicated site config for feet sites on Microduck.
# Prevents SceneEntityCfg("robot") default from selecting all 7 sites (imu, camera, tof, etc.)
# which would trigger RuntimeError dimension mismatch against 2-primary sensors.
_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def stairs_foot_clearance_penalty(
    env: ManagerBasedRlEnv,
    target_height: float = 0.035,
    height_sensor_name: str = "foot_height_scan",
    command_name: str | None = "twist",
    command_threshold: float = 0.01,
    vel_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _FEET_SITE_CFG,
) -> torch.Tensor:
    """Penalize foot clearance shortfall during forward swing on stairs.

    Unlike mjlab's feet_clearance (which penalizes lifting higher than target),
    this is ONE-SIDED: lifting higher than target_height is completely free
    (0 penalty). Only penalizes clearance deficits (target_height - h > 0)
    when the foot is moving forward horizontally with velocity > vel_threshold.

    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    foot_height = torch.nan_to_num(height_sensor.data.heights, nan=0.0)
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_norm = torch.nan_to_num(torch.linalg.norm(foot_vel_xy, dim=-1), nan=0.0)

    # One-sided clearance deficit: positive only when clearance is insufficient
    clearance_deficit = torch.clamp(target_height - foot_height, min=0.0)
    moving = torch.clamp(vel_norm - vel_threshold, min=0.0)

    cost = torch.sum(clearance_deficit * moving, dim=1)

    if command_name is not None and hasattr(env, "command_manager"):
        cmd = env.command_manager.get_command(command_name)
        if cmd is not None:
            cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
            cmd_active = (cmd_norm > command_threshold).float()
            cost = cost * cmd_active

    return -cost


def stairs_foot_landing_damping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    impact_force_thresh: float = 8.0,
    impact_vel_thresh: float = 0.08,
    vel_weight: float = 10.0,
    asset_cfg: SceneEntityCfg = _FEET_SITE_CFG,
) -> torch.Tensor:
    """Penalize excessive impact force and downward normal velocity at foot landing.

    Evaluated exclusively at touchdown (first_contact).
    - Forces below impact_force_thresh (~1x body weight) are free.
    - Downward normal velocity (-v_z) above impact_vel_thresh is penalized quadratically.

    Protects XL330 gear trains and damps footfall bounce on rigid stair treads.
    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)

    forces = torch.nan_to_num(contact_sensor.data.force, nan=0.0)
    force_mag = torch.linalg.norm(forces, dim=-1)
    excess_force = torch.clamp(force_mag - impact_force_thresh, min=0.0)

    asset: Entity = env.scene[asset_cfg.name]
    foot_vz = torch.nan_to_num(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2], nan=0.0)
    down_vel = torch.clamp(-foot_vz - impact_vel_thresh, min=0.0)

    touchdown_cost = (excess_force + vel_weight * torch.square(down_vel)) * first_contact.float()
    return -torch.sum(touchdown_cost, dim=1)


def stairs_riser_trip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_friction_ratio: float = 0.5,
    asset_cfg: SceneEntityCfg = _FEET_SITE_CFG,
) -> torch.Tensor:
    """Penalize foot collision with vertical stair risers (stubbing/tripping).

    On horizontal treads, contact normal is vertical: F_z dominates and |F_xy| <= mu * F_z.
    On a vertical riser, contact normal is horizontal: F_xy dominates while F_z is near zero.
    Penalizes horizontal shear forces in excess of max_friction_ratio * F_z.

    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces = torch.nan_to_num(contact_sensor.data.force, nan=0.0)
    f_xy = torch.linalg.norm(forces[:, :, :2], dim=-1)
    f_z = torch.clamp(forces[:, :, 2], min=0.0)

    found = contact_sensor.data.found
    if found.dim() == 1:
        in_contact = (found > 0).float().unsqueeze(1)
    else:
        in_contact = (found > 0).float()

    riser_excess = torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)
    trip_cost = torch.sum(riser_excess * in_contact, dim=1)
    return -trip_cost


def stairs_sole_flat_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _FEET_SITE_CFG,
) -> torch.Tensor:
    """Penalize foot sole tilt during stance on stair treads.

    Projects gravity into the foot site frame. When flat on a tread, g_site_xy = 0.
    Gated by contact so swing feet are free to articulate.
    Returns <= 0 (self-negating microduck penalty) -> weight with positive sign.
    """
    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)
    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]

    num_feet = foot_quats.shape[1]
    per_foot = torch.zeros(env.num_envs, num_feet, device=env.device, dtype=torch.float32)
    for i in range(num_feet):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)

    contact_sensor: ContactSensor = env.scene[sensor_name]
    contact_time = contact_sensor.data.current_contact_time
    if contact_time is not None:
        per_foot = per_foot * (contact_time > 0.0).float()
    else:
        found = contact_sensor.data.found
        per_foot = per_foot * (found > 0).float()

    return -per_foot.sum(dim=1)


def stairs_relative_height_progress(
    env: ManagerBasedRlEnv,
    ceiling: float = 0.115,
    asset_cfg: SceneEntityCfg = _FEET_SITE_CFG,
) -> torch.Tensor:
    """Potential-based relative height shaping on stairs: Delta min(z_trunk_rel, ceiling).

    CRITICAL FOR STAIRS: Standard height_progress uses (z_trunk - z_origin), which
    saturates above step 1 because step risers add elevation.
    This function uses trunk height RELATIVE to the lowest foot site:
        z_rel = z_trunk - min(z_left_foot, z_right_foot)

    Properties:
    - Invariant to which stair step the robot is standing or lying on.
    - Climbing stairs leaves z_rel constant (~0.117 m), so climbing cannot farm recovery!
    - Rising out of a crouch or prone state pays Delta z_rel > 0.
    - Holding any pose pays exactly 0.0.

    Returns Delta Phi in [-0.05, 0.05] -> weight with positive sign.
    """
    asset: Entity = env.scene[asset_cfg.name]
    trunk_z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2], nan=0.0)
    foot_z = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], nan=0.0)
    min_foot_z = torch.min(foot_z, dim=1).values

    z_rel = torch.nan_to_num(trunk_z - min_foot_z, nan=0.0)
    pot = torch.clamp(z_rel, max=ceiling)

    if not hasattr(env, "_stairs_rel_height_pot_prev"):
        env._stairs_rel_height_pot_prev = pot.clone()

    fresh = env.episode_length_buf <= 1
    env._stairs_rel_height_pot_prev[fresh] = pot[fresh]

    delta = pot - env._stairs_rel_height_pot_prev
    env._stairs_rel_height_pot_prev = pot.clone()
    return torch.nan_to_num(delta, nan=0.0)
