"""Unit tests for Microduck high stairs MDP functions (tests/test_stairs_mdp.py).

Verifies:
1. Output shape (env.num_envs,), dtype float32, and device consistency.
2. Self-negating penalties return values <= 0.0 under all valid and boundary inputs.
3. stairs_relative_height_progress handles initial resets (episode_length_buf <= 1),
   asynchronous resets, elevation invariance on stairs, and NaN safety.
4. Correct site_ids selection via _FEET_SITE_CFG preventing 7-site mismatch.
"""

from __future__ import annotations

import math
import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


# -----------------------------------------------------------------------------
# Standalone Mock Infrastructure for Fast CPU Unit Testing
# -----------------------------------------------------------------------------

class MockAssetData:
    def __init__(self, n: int, num_sites: int = 7):
        self.num_sites = num_sites
        self.site_lin_vel_w = torch.zeros(n, num_sites, 3)
        self.site_pos_w = torch.zeros(n, num_sites, 3)
        self.site_quat_w = torch.zeros(n, num_sites, 4)
        self.site_quat_w[:, :, 0] = 1.0  # identity quaternion [w, x, y, z]
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0  # [w, x, y, z]
        self.root_link_ang_vel_b = torch.zeros(n, 3)
        self.gravity_vec_w = torch.tensor([[0.0, 0.0, -9.81]]).repeat(n, 1)
        self.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)
        self.joint_pos = torch.zeros(n, 14)
        self.joint_vel = torch.zeros(n, 14)
        self.actuator_force = torch.zeros(n, 14)
        self.applied_torque = self.actuator_force


class MockAsset:
    def __init__(self, n: int, num_sites: int = 7):
        self.data = MockAssetData(n, num_sites)
        self.joint_names = [f"servo_{i}" for i in range(14)]

    def find_joints(self, pattern: str):
        return list(range(14)), self.joint_names


class MockSensorData:
    def __init__(self, n: int, num_primaries: int = 2):
        self.heights = torch.zeros(n, num_primaries)
        self.force = torch.zeros(n, num_primaries, 3)
        self.found = torch.zeros(n, num_primaries, dtype=torch.int32)
        self.current_contact_time = torch.zeros(n, num_primaries)


class MockContactSensor:
    def __init__(self, n: int, num_primaries: int = 2):
        self.data = MockSensorData(n, num_primaries)
        self._air_time_state = self.data

    def compute_first_contact(self, dt: float = 0.02) -> torch.Tensor:
        is_in = self.data.current_contact_time > 0.0
        within = self.data.current_contact_time < (dt + 1e-8)
        return is_in & within


class MockHeightSensor:
    def __init__(self, n: int, num_primaries: int = 2):
        self.data = MockSensorData(n, num_primaries)


class MockCommandManager:
    def __init__(self, n: int):
        self.cmd = torch.zeros(n, 3)

    def get_command(self, name: str) -> torch.Tensor:
        return self.cmd


class MockTerrain:
    def __init__(self, n: int):
        self.env_origins = torch.zeros(n, 3)


class MockScene:
    def __init__(self, n: int):
        self.robot = MockAsset(n, num_sites=7)
        self.foot_height_scan = MockHeightSensor(n, num_primaries=2)
        self.feet_ground_contact = MockContactSensor(n, num_primaries=2)
        self.terrain = MockTerrain(n)

    def __getitem__(self, name: str):
        return getattr(self, name)


class MockEnv:
    def __init__(self, n: int = 4):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.scene = MockScene(n)
        self.command_manager = MockCommandManager(n)


class MockFeetSiteCfg:
    def __init__(self):
        self.name = "robot"
        self.site_ids = [1, 6]  # left_foot (1) and right_foot (6)


class MockAllSitesCfg:
    def __init__(self):
        self.name = "robot"
        self.site_ids = slice(None)  # selects all 7 sites


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("b", [1, 4, 32, 64])
def test_tensor_dimensions_and_dtype(b):
    env = MockEnv(b)
    for fn in [
        microduck_mdp.stairs_foot_clearance_penalty,
        microduck_mdp.stairs_foot_landing_damping_penalty,
        microduck_mdp.stairs_riser_trip_penalty,
        microduck_mdp.stairs_sole_flat_penalty,
        microduck_mdp.stairs_relative_height_progress,
        microduck_mdp.swing_foot_sole_leveling_reward,
        microduck_mdp.servo_torque_headroom_penalty,
        microduck_mdp.stairs_alternating_gait_reward,
        microduck_mdp.single_leg_stance_balance_reward,
        microduck_mdp.double_support_stagnation_penalty,
        microduck_mdp.swing_foot_vertical_velocity_penalty,
        microduck_mdp.single_leg_continuous_balance_reward,
        microduck_mdp.smooth_high_step_clearance_reward,
        microduck_mdp.stance_ankle_roll_penalty,
    ]:
        out = fn(env)
        assert out.shape == (b,), f"{fn.__name__} returned shape {out.shape}, expected ({b},)"
        assert out.dtype == torch.float32, f"{fn.__name__} returned {out.dtype}"
        assert not torch.isnan(out).any(), f"{fn.__name__} produced NaNs"


def test_foot_clearance_self_negating_and_bounds():
    env = MockEnv(3)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])  # active walk command
    env.scene.robot.data.site_lin_vel_w[:, [1, 6], :2] = 0.2    # moving forward fast

    # Case A: Low foot (deficit = 0.035 - 0.01 = 0.025m)
    env.scene.foot_height_scan.data.heights[:] = 0.01
    out_low = microduck_mdp.stairs_foot_clearance_penalty(env, target_height=0.035, vel_threshold=0.02)
    assert (out_low < 0.0).all(), "Low foot should produce negative penalty"

    # Case B: Sufficient foot clearance (0.04m > 0.035m)
    env.scene.foot_height_scan.data.heights[:] = 0.04
    out_high = microduck_mdp.stairs_foot_clearance_penalty(env, target_height=0.035, vel_threshold=0.02)
    assert (out_high == 0.0).all(), "High foot clearance must be 0.0 (one-sided, not penalized)"

    # Case C: Zero velocity (stationary foot)
    env.scene.foot_height_scan.data.heights[:] = 0.01
    env.scene.robot.data.site_lin_vel_w[:, [1, 6], :2] = 0.005  # below vel_threshold
    out_stat = microduck_mdp.stairs_foot_clearance_penalty(env, target_height=0.035, vel_threshold=0.02)
    assert (out_stat == 0.0).all(), "Stationary foot below clearance must not be penalized"


def test_landing_damping_self_negating_and_bounds():
    env = MockEnv(2)
    # Touchdown trigger
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.01  # established within dt

    # Case A: Gentle landing (F = 5N < 8N, vz = -0.05m/s > -0.08m/s)
    env.scene.feet_ground_contact.data.force[:] = torch.tensor([0.0, 0.0, 5.0])
    env.scene.robot.data.site_lin_vel_w[:, [1, 6], 2] = -0.05
    out_gentle = microduck_mdp.stairs_foot_landing_damping_penalty(env)
    assert (out_gentle == 0.0).all(), "Gentle touchdown should be 0.0"

    # Case B: Severe impact (F = 25N > 8N, vz = -0.5m/s)
    env.scene.feet_ground_contact.data.force[:] = torch.tensor([0.0, 0.0, 25.0])
    env.scene.robot.data.site_lin_vel_w[:, [1, 6], 2] = -0.5
    out_hard = microduck_mdp.stairs_foot_landing_damping_penalty(env)
    assert (out_hard < 0.0).all(), "Severe impact touchdown should be negative penalty"

    # Case C: Airborne foot (no contact)
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.0
    out_air = microduck_mdp.stairs_foot_landing_damping_penalty(env)
    assert (out_air == 0.0).all(), "Airborne foot must produce 0.0"


def test_riser_trip_self_negating_and_bounds():
    env = MockEnv(2)
    env.scene.feet_ground_contact.data.found[:] = 1

    # Case A: Normal horizontal tread contact (Fz = 12N, Fxy = 2N; ratio 2/12 = 0.16 < 0.5)
    env.scene.feet_ground_contact.data.force[:] = torch.tensor([2.0, 0.0, 12.0])
    out_tread = microduck_mdp.stairs_riser_trip_penalty(env, max_friction_ratio=0.5)
    assert (out_tread == 0.0).all(), "Normal tread contact must produce 0.0"

    # Case B: Vertical riser face stubbing (Fxy = 15N, Fz = 1N; ratio 15/1 = 15 >> 0.5)
    env.scene.feet_ground_contact.data.force[:] = torch.tensor([15.0, 0.0, 1.0])
    out_riser = microduck_mdp.stairs_riser_trip_penalty(env, max_friction_ratio=0.5)
    assert (out_riser < 0.0).all(), "Riser stubbing must produce negative penalty"

    # Case C: No contact
    env.scene.feet_ground_contact.data.found[:] = 0
    out_none = microduck_mdp.stairs_riser_trip_penalty(env)
    assert (out_none == 0.0).all(), "No contact must produce 0.0"


def test_sole_flat_self_negating_and_bounds():
    env = MockEnv(2)
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.1  # in contact

    # Case A: Perfectly flat foot (quat = [1, 0, 0, 0])
    env.scene.robot.data.site_quat_w[:, [1, 6], :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out_flat = microduck_mdp.stairs_sole_flat_penalty(env)
    assert (out_flat == 0.0).all(), "Flat foot on tread must produce 0.0"

    # Case B: Tilted foot (pitched 30 deg)
    pitch_rad = math.radians(30)
    q_pitch = torch.tensor([math.cos(pitch_rad / 2), 0.0, math.sin(pitch_rad / 2), 0.0])
    env.scene.robot.data.site_quat_w[:, [1, 6], :] = q_pitch
    out_tilted = microduck_mdp.stairs_sole_flat_penalty(env)
    assert (out_tilted < 0.0).all(), "Tilted stance foot must produce negative penalty"

    # Case C: Airborne foot (contact_time = 0.0)
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.0
    out_air = microduck_mdp.stairs_sole_flat_penalty(env)
    assert (out_air == 0.0).all(), "Airborne swinging foot must produce 0.0"


def test_relative_height_progress_lifecycle():
    env = MockEnv(2)
    # Step 0: Spawn / Reset
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.05
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.02
    env.episode_length_buf[:] = 0
    d0 = microduck_mdp.stairs_relative_height_progress(env)
    assert (d0 == 0.0).all(), "Step 0 must produce exactly 0.0"

    # Step 1: Settling transient (must still be zeroed)
    env.episode_length_buf[:] = 1
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.04
    d1 = microduck_mdp.stairs_relative_height_progress(env)
    assert (d1 == 0.0).all(), "Step 1 settling must produce exactly 0.0"

    # Step 2: Physical rise (+0.04m)
    env.episode_length_buf[:] = 2
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.08
    d2 = microduck_mdp.stairs_relative_height_progress(env)
    assert torch.allclose(d2, torch.tensor([0.04, 0.04]), atol=1e-5), f"Step 2 rise: {d2}"

    # Step 3: Stand reached (z_rel = 0.12m > ceiling 0.115m)
    env.episode_length_buf[:] = 3
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.14
    d3 = microduck_mdp.stairs_relative_height_progress(env)
    assert torch.allclose(d3, torch.tensor([0.055, 0.055]), atol=1e-5), f"Step 3 cap: {d3}"

    # Step 4: Stance hold
    env.episode_length_buf[:] = 4
    d4 = microduck_mdp.stairs_relative_height_progress(env)
    assert (d4 == 0.0).all(), "Holding stance must produce exactly 0.0"

    # Step 5: Asynchronous reset (env 0 resets, env 1 stays standing)
    env.episode_length_buf[0] = 0
    env.scene.robot.data.root_link_pos_w[0, 2] = 0.05
    env.episode_length_buf[1] = 5
    d5 = microduck_mdp.stairs_relative_height_progress(env)
    assert d5[0] == 0.0, "Reset env must yield 0.0"
    assert d5[1] == 0.0, "Unchanged env must yield 0.0"


def test_relative_height_progress_elevation_invariance():
    env = MockEnv(2)
    # Standing on step 0
    env.episode_length_buf[:] = 10
    env.scene.robot.data.root_link_pos_w[:] = 0.117
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.000
    microduck_mdp.stairs_relative_height_progress(env)  # sync

    # Step up to step 2 (both trunk and feet rise by 0.06m)
    env.episode_length_buf[:] = 11
    env.scene.robot.data.root_link_pos_w[:] = 0.177
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.060
    d_step = microduck_mdp.stairs_relative_height_progress(env)
    assert (d_step == 0.0).all(), "Climbing stairs with constant relative height must produce 0.0"


def test_site_mismatch_protection():
    """Verify that MockFeetSiteCfg correctly isolates 2 feet from 7-geom model."""
    env = MockEnv(2)
    # Default uses _FEET_SITE_CFG (matching 2 feet), so shape matches foot_height_scan (2 primaries)
    out = microduck_mdp.stairs_foot_clearance_penalty(env)
    assert out.shape == (2,)


def test_fallen_persistence_penalty_dynamics():
    """Verify fallen_persistence_penalty is self-negating (<= 0), respects grace period, and escalates."""
    env = MockEnv(2)
    asset_cfg = microduck_mdp.SceneEntityCfg("robot")

    # Env 0: Upright (quat = [1, 0, 0, 0] -> cos_tilt = 1.0 -> tilt = 0 deg)
    # Env 1: Fallen on side/back (quat = [0.7071, 0.7071, 0, 0] -> cos_tilt = 0.0 -> tilt = 90 deg)
    env.scene.robot.data.root_link_quat_w[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.scene.robot.data.root_link_quat_w[1] = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0])

    # First call at t = 0
    env.episode_length_buf[:] = 1
    p1 = microduck_mdp.fallen_persistence_penalty(env, asset_cfg, gate_tilt_above_deg=40.0, grace_period_s=0.8)
    assert p1.shape == (2,)
    assert p1.dtype == torch.float32
    assert (p1 <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p1[0] == 0.0, "Upright env must have 0.0 penalty"
    assert p1[1] == 0.0, "Within grace period (t=0.02s < 0.8s), penalty must be 0.0"

    # Advance time beyond grace period (50 steps * 0.02s = 1.0s > 0.8s)
    for step in range(2, 51):
        env.episode_length_buf[:] = step
        p2 = microduck_mdp.fallen_persistence_penalty(env, asset_cfg, gate_tilt_above_deg=40.0, grace_period_s=0.8)
    assert p2[0] == 0.0, "Upright env must remain 0.0 penalty"
    assert p2[1] < 0.0, "Persistently fallen env beyond grace period must receive negative penalty"

    # Upright recovery clears penalty immediately
    env.scene.robot.data.root_link_quat_w[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.episode_length_buf[:] = 51
    p3 = microduck_mdp.fallen_persistence_penalty(env, asset_cfg, gate_tilt_above_deg=40.0, grace_period_s=0.8)
    assert (p3 == 0.0).all(), "Once upright, penalty must immediately clear to 0.0"


def test_standstill_or_sluggish_penalty():
    """Verify standstill_or_sluggish_penalty heavily taxes frozen/sluggish robots and ignores idle."""
    env = MockEnv(4)
    asset_cfg = microduck_mdp.SceneEntityCfg("robot")

    # Command: env 0, 1, 3 have vx = 0.20 m/s; env 2 has vx = 0.0 m/s (idle)
    env.command_manager.cmd = torch.tensor([
        [0.20, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.00, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ])

    # Base lin_vel_b:
    # env 0: frozen (vx = 0.002 m/s)
    # env 1: sluggish (vx = 0.05 m/s)
    # env 2: idle (vx = 0.00 m/s)
    # env 3: moving well (vx = 0.18 m/s >= target 0.12 m/s)
    if not hasattr(env.scene.robot.data, "root_link_lin_vel_b"):
        env.scene.robot.data.root_link_lin_vel_b = torch.zeros(4, 3)

    env.scene.robot.data.root_link_lin_vel_b[0, 0] = 0.002
    env.scene.robot.data.root_link_lin_vel_b[1, 0] = 0.050
    env.scene.robot.data.root_link_lin_vel_b[2, 0] = 0.000
    env.scene.robot.data.root_link_lin_vel_b[3, 0] = 0.180

    p = microduck_mdp.standstill_or_sluggish_penalty(
        env, command_name="twist", asset_cfg=asset_cfg, cmd_threshold=0.05, min_vel_ratio=0.60
    )

    assert p.shape == (4,)
    assert p.dtype == torch.float32
    assert (p <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p[0] < -1.5, "Frozen robot when moving commanded must receive heavy penalty (< -1.5)"
    assert p[1] < -1.0, "Sluggish robot creeping must receive penalty"
    assert p[2] == 0.0, "Idle commanded robot must NOT be penalized (cost=0.0)"
    assert p[3] == 0.0, "Fast moving robot meeting target speed must NOT be penalized (cost=0.0)"


def test_feet_slide_penalty():
    """Verify feet_slide_penalty heavily penalizes foot sliding/skating while in stance contact."""
    env = MockEnv(3)
    asset_cfg = MockFeetSiteCfg()

    # Site linear velocity:
    # Env 0: left foot sliding at 0.15 m/s, right foot air
    # Env 1: both feet planted firmly (< 0.02 m/s)
    # Env 2: feet swinging fast in air (Fn = 0.0) -> must be free (no slide cost)
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[0, 0, 2] = 5.0  # left foot in contact (5N)
    env.scene.feet_ground_contact.data.force[1, 0, 2] = 6.0  # left foot in contact (6N)
    env.scene.feet_ground_contact.data.force[1, 1, 2] = 6.0  # right foot in contact (6N)
    env.scene.feet_ground_contact.data.force[2, :, 2] = 0.0  # both airborne (0N)

    # Foot horizontal velocities
    env.scene.robot.data.site_lin_vel_w = torch.zeros(3, 7, 3)
    # Env 0: left foot (site 1) sliding at 0.15 m/s
    env.scene.robot.data.site_lin_vel_w[0, 1, 0] = 0.15
    # Env 1: both feet stationary (0.01 m/s)
    env.scene.robot.data.site_lin_vel_w[1, 1, 0] = 0.01
    env.scene.robot.data.site_lin_vel_w[1, 6, 0] = 0.01
    # Env 2: swinging in air at 0.30 m/s
    env.scene.robot.data.site_lin_vel_w[2, 1, 0] = 0.30
    env.scene.robot.data.site_lin_vel_w[2, 6, 0] = 0.30

    p = microduck_mdp.feet_slide_penalty(
        env, sensor_name="feet_ground_contact", asset_cfg=asset_cfg, contact_force_thresh=1.0, slide_vel_thresh=0.03
    )

    assert p.shape == (3,)
    assert p.dtype == torch.float32
    assert (p <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p[0] < 0.0, "Sliding/skating in stance contact must be penalized"
    assert p[1] == 0.0, "Planted feet below slide threshold must NOT be penalized"
    assert p[2] == 0.0, "Airborne swinging feet must NOT be penalized (free to move in air)"


def test_swing_foot_sole_leveling_reward():
    """Verify swing_foot_sole_leveling_reward gates tilt/height and rewards horizontal sole dorsiflexion."""
    env = MockEnv(3)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])  # active walk command
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.15  # upright height > 0.095m

    # Env 0: Valid single stance - Left airborne (0.5N), Right firmly planted (5.0N)
    # Left ankle at target dorsiflexion: 0.4530 + 0.10 = 0.5530
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[0, 0, 2] = 0.5   # Left foot swing (< 1.5N)
    env.scene.feet_ground_contact.data.force[0, 1, 2] = 5.0   # Right foot stance (> 3.5N)
    env.scene.robot.data.joint_pos[0, 4] = 0.5530             # Left ankle tuned dorsiflexion

    # Env 1: Double support (both 5.0N) -> must produce 0.0
    env.scene.feet_ground_contact.data.force[1, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[1, 1, 2] = 5.0

    # Env 2: Fallen robot (trunk z = 0.05m < 0.095m) -> must produce 0.0
    env.scene.feet_ground_contact.data.force[2, 0, 2] = 0.5
    env.scene.feet_ground_contact.data.force[2, 1, 2] = 5.0
    env.scene.robot.data.root_link_pos_w[2, 2] = 0.05

    r = microduck_mdp.swing_foot_sole_leveling_reward(env)

    assert r.shape == (3,)
    assert (r >= 0.0).all(), "Reward must be non-negative"
    assert (r <= 1.0).all(), "Reward must be bounded by 1.0"
    assert r[0] > 0.8, f"Env 0 well-leveled swing foot should receive high reward, got {r[0]}"
    assert r[1] == 0.0, "Double support should receive 0.0"
    assert r[2] == 0.0, "Fallen robot should receive 0.0"


def test_servo_torque_headroom_penalty():
    """Verify servo_torque_headroom_penalty only penalizes torques > 0.72 Nm (75% stall)."""
    env = MockEnv(3)
    # Env 0: Safe torques <= 0.60 Nm -> zero penalty (0.0)
    env.scene.robot.data.actuator_force[0, :] = 0.55

    # Env 1: One servo exceeding safe limit (0.85 Nm > 0.72 Nm) -> negative penalty (< 0.0)
    env.scene.robot.data.actuator_force[1, :] = 0.50
    env.scene.robot.data.actuator_force[1, 3] = 0.85  # knee motor under heavy load

    # Env 2: Multiple servos stalled near 0.96 Nm -> heavy negative penalty
    env.scene.robot.data.actuator_force[2, :] = 0.95

    p = microduck_mdp.servo_torque_headroom_penalty(env, safe_torque_limit=0.72)

    assert p.shape == (3,)
    assert (p <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p[0] == 0.0, f"Safe torque must have 0 penalty, got {p[0]}"
    assert p[1] < 0.0, "Single overloaded servo must receive penalty"
    assert p[2] < p[1], "Multiple overloaded servos must receive heavier penalty"


def test_stairs_alternating_gait_reward():
    """Verify stairs_alternating_gait_reward enforces alternating bipedal phases and blocks bunny-hopping."""
    env = MockEnv(4)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])  # active walk command

    env.scene.feet_ground_contact.data.force = torch.zeros(4, 2, 3)
    # Env 0: Left stance (4N), Right swing (0.5N) -> alternating step
    env.scene.feet_ground_contact.data.force[0, 0, 2] = 4.0
    env.scene.feet_ground_contact.data.force[0, 1, 2] = 0.5

    # Env 1: Right stance (4N), Left swing (0.5N) -> alternating step
    env.scene.feet_ground_contact.data.force[1, 0, 2] = 0.5
    env.scene.feet_ground_contact.data.force[1, 1, 2] = 4.0

    # Env 2: Double support (both 4N) -> stalling
    env.scene.feet_ground_contact.data.force[2, 0, 2] = 4.0
    env.scene.feet_ground_contact.data.force[2, 1, 2] = 4.0

    # Env 3: Both airborne (0.2N) -> bunny-hopping flight phase
    env.scene.feet_ground_contact.data.force[3, 0, 2] = 0.2
    env.scene.feet_ground_contact.data.force[3, 1, 2] = 0.2

    r = microduck_mdp.stairs_alternating_gait_reward(env, contact_force_thresh=2.0)

    assert r.shape == (4,)
    assert (r >= 0.0).all() and (r <= 1.0).all()
    assert r[0] == 1.0, "Left stance single support must receive 1.0"
    assert r[1] == 1.0, "Right stance single support must receive 1.0"
    assert r[2] == 0.0, "Double support must receive 0.0"
    assert r[3] == 0.0, "Bunny hopping (both airborne) must receive 0.0"


def test_single_leg_stance_balance_reward():
    """Verify single_leg_stance_balance_reward rewards upright stable single-foot support."""
    env = MockEnv(3)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.15

    # Env 0: Upright and stable single leg stance
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[0, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[0, 1, 2] = 0.5

    # Env 1: Double support
    env.scene.feet_ground_contact.data.force[1, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[1, 1, 2] = 5.0

    # Env 2: Fallen robot
    env.scene.feet_ground_contact.data.force[2, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[2, 1, 2] = 0.5
    env.scene.robot.data.root_link_pos_w[2, 2] = 0.05

    r = microduck_mdp.single_leg_stance_balance_reward(env)
    assert r.shape == (3,)
    assert r[0] > 0.8
    assert r[1] == 0.0
    assert r[2] == 0.0


def test_double_support_stagnation_penalty():
    """Verify double_support_stagnation_penalty penalizes lingering with both feet on the ground."""
    env = MockEnv(2)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])
    env.episode_length_buf[:] = 50  # non-fresh episode (> 1)

    # Pre-seed internal timer:
    # Env 0: Brief contact (0.10s < max 0.40s)
    # Env 1: Lingering prolonged double support (0.80s > 0.40s)
    env._stair_double_support_timer = torch.tensor([0.10, 0.80])

    env.scene.feet_ground_contact.data.force = torch.zeros(2, 2, 3)
    env.scene.feet_ground_contact.data.force[:, :, 2] = 4.0

    p = microduck_mdp.double_support_stagnation_penalty(env, max_double_time_s=0.40)
    assert p.shape == (2,)
    assert (p <= 0.0).all()
    assert p[0] == 0.0, "Brief double support within budget must NOT be penalized"
    assert p[1] < 0.0, "Lingering double support must be penalized"


def test_swing_foot_vertical_velocity_penalty():
    """Verify swing_foot_vertical_velocity_penalty penalizes upward jerking vz > max_safe_vz."""
    env = MockEnv(3)
    asset_cfg = MockFeetSiteCfg()

    # Feet ground forces: left swing (0.5N), right stance (5.0N)
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[:, 0, 2] = 0.5   # Left foot swing (< 1.5N)
    env.scene.feet_ground_contact.data.force[:, 1, 2] = 5.0   # Right foot stance (> 1.5N)

    # Foot vertical velocities:
    env.scene.robot.data.site_lin_vel_w = torch.zeros(3, 7, 3)
    # Env 0: gentle upward lift (0.12 m/s <= 0.20 m/s) -> 0.0 penalty
    env.scene.robot.data.site_lin_vel_w[0, 1, 2] = 0.12
    # Env 1: violent upward jerk (0.60 m/s > 0.20 m/s) -> negative penalty (< 0.0)
    env.scene.robot.data.site_lin_vel_w[1, 1, 2] = 0.60
    # Env 2: stance foot moving at 0.60 m/s (not in swing) -> must NOT be penalized
    env.scene.robot.data.site_lin_vel_w[2, 6, 2] = 0.60

    p = microduck_mdp.swing_foot_vertical_velocity_penalty(
        env, sensor_name="feet_ground_contact", asset_cfg=asset_cfg, max_safe_vz=0.20
    )

    assert p.shape == (3,)
    assert (p <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p[0] == 0.0, f"Gentle lift must not be penalized, got {p[0]}"
    assert p[1] < -0.1, f"Violent upward jerk must be penalized, got {p[1]}"
    assert p[2] == 0.0, "Stance foot must not trigger swing velocity penalty"


def test_single_leg_continuous_balance_reward():
    """Verify single_leg_continuous_balance_reward requires upright posture and low roll/pitch angular rates."""
    env = MockEnv(3)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.15

    # Single stance: left foot stance (5N), right foot swing (0.5N)
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[:, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[:, 1, 2] = 0.5

    # Env 0: stable single stance (ang vel ~ 0.0)
    env.scene.robot.data.root_link_ang_vel_b[0, :] = 0.0
    # Env 1: wobbling violently (roll rate = 1.8 rad/s)
    env.scene.robot.data.root_link_ang_vel_b[1, 0] = 1.8
    # Env 2: fallen (trunk z = 0.05m < 0.10m)
    env.scene.robot.data.root_link_pos_w[2, 2] = 0.05

    r = microduck_mdp.single_leg_continuous_balance_reward(env)
    assert r.shape == (3,)
    assert (r >= 0.0).all() and (r <= 1.0).all()
    assert r[0] > 0.8, f"Stable single stance must receive high reward, got {r[0]}"
    assert r[1] < 0.1, f"Wobbling robot must receive minimal reward, got {r[1]}"
    assert r[2] == 0.0, "Fallen robot must receive 0.0"


def test_smooth_high_step_clearance_reward():
    """Verify smooth_high_step_clearance_reward smoothly peaks at target clearance."""
    env = MockEnv(3)
    env.command_manager.cmd[:] = torch.tensor([0.2, 0.0, 0.0])

    # Left foot in swing (< 1.5N), right in stance (5N)
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[:, 0, 2] = 0.5
    env.scene.feet_ground_contact.data.force[:, 1, 2] = 5.0

    # Foot heights:
    env.scene.foot_height_scan.data.heights = torch.zeros(3, 2)
    # Env 0: perfectly at target clearance (0.035m)
    env.scene.foot_height_scan.data.heights[0, 0] = 0.035
    # Env 1: far below target (0.005m)
    env.scene.foot_height_scan.data.heights[1, 0] = 0.005
    # Env 2: both feet in stance (5N) -> swing score 0
    env.scene.feet_ground_contact.data.force[2, 0, 2] = 5.0

    r = microduck_mdp.smooth_high_step_clearance_reward(env, target_clearance=0.035, clearance_std=0.015)
    assert r.shape == (3,)
    assert (r >= 0.0).all() and (r <= 1.0).all()
    assert r[0] > 0.95, f"Perfect clearance match should score ~1.0, got {r[0]}"
    assert r[1] < 0.20, f"Far from target should score low, got {r[1]}"
    assert r[2] == 0.0, "Double stance should produce 0.0"


def test_stance_ankle_roll_penalty():
    """Verify stance_ankle_roll_penalty penalizes edge-tipping only on the stance leg."""
    env = MockEnv(3)
    # Left foot stance (5N), right foot swing (0.5N)
    env.scene.feet_ground_contact.data.force = torch.zeros(3, 2, 3)
    env.scene.feet_ground_contact.data.force[:, 0, 2] = 5.0
    env.scene.feet_ground_contact.data.force[:, 1, 2] = 0.5

    # Hip roll joints: left roll = joint 1, right roll = joint 10
    # Env 0: roll angle within safe limit (2 deg <= 4 deg)
    env.scene.robot.data.joint_pos[0, 1] = math.radians(2.0)
    # Env 1: stance foot tilted/edge-tipped (10 deg > 4 deg)
    env.scene.robot.data.joint_pos[1, 1] = math.radians(10.0)
    # Env 2: swing foot rolled (10 deg) -> must NOT be penalized (free in air)
    env.scene.robot.data.joint_pos[2, 10] = math.radians(10.0)

    p = microduck_mdp.stance_ankle_roll_penalty(env, max_roll_dev_deg=4.0)
    assert p.shape == (3,)
    assert (p <= 0.0).all(), "Penalty must be self-negating (<= 0.0)"
    assert p[0] == 0.0, f"Within safe roll limit must have 0.0 penalty, got {p[0]}"
    assert p[1] < 0.0, f"Edge-tipped stance foot must be penalized, got {p[1]}"
    assert p[2] == 0.0, "Airborne swing leg roll must NOT trigger stance roll penalty"


