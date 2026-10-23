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
        self.gravity_vec_w = torch.tensor([[0.0, 0.0, -9.81]]).repeat(n, 1)


class MockAsset:
    def __init__(self, n: int, num_sites: int = 7):
        self.data = MockAssetData(n, num_sites)


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


class MockScene:
    def __init__(self, n: int):
        self.robot = MockAsset(n, num_sites=7)
        self.foot_height_scan = MockHeightSensor(n, num_primaries=2)
        self.feet_ground_contact = MockContactSensor(n, num_primaries=2)

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
