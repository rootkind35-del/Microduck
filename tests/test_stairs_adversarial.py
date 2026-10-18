"""Adversarial stress-testing suite for Microduck stairs MDP functions.

Covers:
1. Sign oracle: penalties must be <= 0.0 under all random, extreme, and edge inputs.
2. NaN/Inf immunity: extreme inputs, NaNs in sensors, zero vectors, zero gravity.
3. Supersonic velocities and massive forces (10,000 N, -10,000 N).
4. Inverted gravity, zero gravity, arbitrary rotations.
5. Relative height progress: reset lifecycle, asynchronous resets, telescopic sum,
   elevation invariance, and upside-down posture handling.
6. Exact threshold boundary conditions.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from mjlab_microduck.tasks import mdp as microduck_mdp


# -----------------------------------------------------------------------------
# Test Harness & Mocks
# -----------------------------------------------------------------------------

class MockAssetData:
    def __init__(self, n: int, num_sites: int = 7):
        self.num_sites = num_sites
        self.site_lin_vel_w = torch.zeros(n, num_sites, 3)
        self.site_pos_w = torch.zeros(n, num_sites, 3)
        self.site_quat_w = torch.zeros(n, num_sites, 4)
        self.site_quat_w[:, :, 0] = 1.0  # identity quaternion
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


# -----------------------------------------------------------------------------
# 1. Sign Oracle: Randomized Stress (10,000 samples)
# -----------------------------------------------------------------------------

def test_sign_oracle_monte_carlo():
    """Verify that all penalties are strictly <= 0.0 across 10,000 diverse random states."""
    torch.manual_seed(42)
    n = 2000
    for _ in range(5):
        env = MockEnv(n)

        # Randomize velocities [-100, 100] m/s
        env.scene.robot.data.site_lin_vel_w = (torch.rand(n, 7, 3) - 0.5) * 200.0

        # Randomize commands [-2, 2] m/s
        env.command_manager.cmd = (torch.rand(n, 3) - 0.5) * 4.0

        # Randomize foot heights [-0.5, 0.5] m
        env.scene.foot_height_scan.data.heights = (torch.rand(n, 2) - 0.5)

        # Randomize contact forces [-100, 500] N
        env.scene.feet_ground_contact.data.force = (torch.rand(n, 2, 3) - 0.2) * 500.0
        env.scene.feet_ground_contact.data.found = torch.randint(0, 2, (n, 2), dtype=torch.int32)
        env.scene.feet_ground_contact.data.current_contact_time = torch.rand(n, 2) * 0.1

        # Randomize orientations (normalized unit quaternions)
        q = torch.randn(n, 7, 4)
        env.scene.robot.data.site_quat_w = F.normalize(q, dim=-1)

        # Randomize gravity
        env.scene.robot.data.gravity_vec_w = torch.randn(n, 3)

        # Run all penalty functions
        p_clearance = microduck_mdp.stairs_foot_clearance_penalty(env)
        p_damping = microduck_mdp.stairs_foot_landing_damping_penalty(env)
        p_trip = microduck_mdp.stairs_riser_trip_penalty(env)
        p_sole = microduck_mdp.stairs_sole_flat_penalty(env)

        assert (p_clearance <= 0.0).all(), f"Clearance penalty violated <= 0: max={p_clearance.max()}"
        assert (p_damping <= 0.0).all(), f"Damping penalty violated <= 0: max={p_damping.max()}"
        assert (p_trip <= 0.0).all(), f"Trip penalty violated <= 0: max={p_trip.max()}"
        assert (p_sole <= 0.0).all(), f"Sole flat penalty violated <= 0: max={p_sole.max()}"

        assert not torch.isnan(p_clearance).any()
        assert not torch.isnan(p_damping).any()
        assert not torch.isnan(p_trip).any()
        assert not torch.isnan(p_sole).any()


# -----------------------------------------------------------------------------
# 2. Extreme Velocity Stress (Supersonic Mach 30, Zero, Massive Downward)
# -----------------------------------------------------------------------------

def test_extreme_velocity_behavior():
    env = MockEnv(5)
    env.command_manager.cmd[:] = torch.tensor([1.0, 0.0, 0.0])

    # Env 0: Zero velocity
    env.scene.robot.data.site_lin_vel_w[0] = 0.0
    # Env 1: Supersonic velocity (10,000 m/s horizontal)
    env.scene.robot.data.site_lin_vel_w[1, [1, 6], 0] = 10000.0
    # Env 2: Supersonic downward velocity (-10,000 m/s vertical)
    env.scene.robot.data.site_lin_vel_w[2, [1, 6], 2] = -10000.0
    # Env 3: Supersonic upward velocity (+10,000 m/s vertical)
    env.scene.robot.data.site_lin_vel_w[3, [1, 6], 2] = 10000.0
    # Env 4: Exactly threshold velocity (vx = 0.02, vy = 0.0 -> norm = 0.02)
    env.scene.robot.data.site_lin_vel_w[4, [1, 6], 0] = 0.02
    env.scene.robot.data.site_lin_vel_w[4, [1, 6], 1:] = 0.0

    # Foot height low (deficit = 0.035 - 0.0 = 0.035)
    env.scene.foot_height_scan.data.heights[:] = 0.0

    p_clear = microduck_mdp.stairs_foot_clearance_penalty(env, target_height=0.035, vel_threshold=0.02)

    # Env 0: Zero velocity -> 0 penalty
    assert p_clear[0] == 0.0
    # Env 1: Supersonic horizontal -> heavily penalized, negative
    assert p_clear[1] < -100.0
    # Env 4: Exactly threshold -> 0 penalty
    assert p_clear[4] == 0.0
    assert (p_clear <= 0.0).all()

    # Touchdown damping test
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.01  # first contact active
    p_damp = microduck_mdp.stairs_foot_landing_damping_penalty(env, impact_vel_thresh=0.08, vel_weight=10.0)

    # Env 2: Downward supersonic -> heavily penalized
    assert p_damp[2] < -1e6
    # Env 3: Upward supersonic -> NOT penalized for downward velocity
    assert p_damp[3] == 0.0
    assert (p_damp <= 0.0).all()


# -----------------------------------------------------------------------------
# 3. Extreme Contact Forces (Massive Impact, Shear, Negative Normal)
# -----------------------------------------------------------------------------

def test_extreme_contact_forces():
    env = MockEnv(6)
    env.scene.feet_ground_contact.data.found[:] = 1
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.01  # touchdown

    # Env 0: Massive normal force (100,000 N)
    env.scene.feet_ground_contact.data.force[0, :, :] = torch.tensor([0.0, 0.0, 100000.0])
    # Env 1: Massive horizontal shear (10,000 N) with zero normal force (vertical riser crash)
    env.scene.feet_ground_contact.data.force[1, :, :] = torch.tensor([10000.0, 0.0, 0.0])
    # Env 2: Negative normal force (-1,000 N) - adhesive / sensor artifact
    env.scene.feet_ground_contact.data.force[2, :, :] = torch.tensor([50.0, 0.0, -1000.0])
    # Env 3: Balanced normal force (10 N) and shear (5 N) at exactly max_friction_ratio=0.5
    env.scene.feet_ground_contact.data.force[3, :, :] = torch.tensor([5.0, 0.0, 10.0])
    # Env 4: Normal force 10 N, shear 4.9 N (below ratio)
    env.scene.feet_ground_contact.data.force[4, :, :] = torch.tensor([4.9, 0.0, 10.0])
    # Env 5: Zero force
    env.scene.feet_ground_contact.data.force[5, :, :] = 0.0

    p_trip = microduck_mdp.stairs_riser_trip_penalty(env, max_friction_ratio=0.5)

    # Env 0: Normal force dominates -> trip penalty 0.0
    assert p_trip[0] == 0.0
    # Env 1: Pure horizontal crash -> heavily penalized
    assert p_trip[1] == -20000.0  # 10000 N * 2 feet
    # Env 2: Negative normal force clamped to 0 -> horizontal shear 50 N penalized
    assert p_trip[2] == -100.0    # 50 N * 2 feet
    # Env 3: Exactly on friction cone limit -> 0 penalty
    assert p_trip[3] == 0.0
    # Env 4: Below friction cone limit -> 0 penalty
    assert p_trip[4] == 0.0
    # Env 5: Zero force -> 0 penalty
    assert p_trip[5] == 0.0

    assert (p_trip <= 0.0).all()

    # Damping penalty under massive crush force
    p_damp = microduck_mdp.stairs_foot_landing_damping_penalty(env, impact_force_thresh=8.0)
    assert p_damp[0] < -100000.0
    assert (p_damp <= 0.0).all()


# -----------------------------------------------------------------------------
# 4. Inverted Gravity, Zero Gravity, and Orientation Stress
# -----------------------------------------------------------------------------

def test_gravity_and_orientation_stress():
    env = MockEnv(6)
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.1  # in stance contact

    # Env 0: Standard Earth gravity [0, 0, -9.81], flat feet
    env.scene.robot.data.gravity_vec_w[0] = torch.tensor([0.0, 0.0, -9.81])
    env.scene.robot.data.site_quat_w[0, [1, 6], :] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # Env 1: Inverted gravity [0, 0, +9.81], flat feet
    env.scene.robot.data.gravity_vec_w[1] = torch.tensor([0.0, 0.0, 9.81])
    env.scene.robot.data.site_quat_w[1, [1, 6], :] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # Env 2: Zero gravity [0, 0, 0] (checking F.normalize stability / zero divide)
    env.scene.robot.data.gravity_vec_w[2] = torch.tensor([0.0, 0.0, 0.0])

    # Env 3: Wall gravity [9.81, 0, 0] (horizontal gravity)
    env.scene.robot.data.gravity_vec_w[3] = torch.tensor([9.81, 0.0, 0.0])

    # Env 4: 90 degree pitched foot (vertical foot on toe/heel)
    q_pitch90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    env.scene.robot.data.site_quat_w[4, [1, 6], :] = q_pitch90

    # Env 5: Massive gravity [0, 0, -10000.0]
    env.scene.robot.data.gravity_vec_w[5] = torch.tensor([0.0, 0.0, -10000.0])

    p_sole = microduck_mdp.stairs_sole_flat_penalty(env)

    # Flat feet under standard gravity -> 0 penalty
    assert p_sole[0] == 0.0
    # Inverted gravity with aligned z -> projection on xy is 0 -> 0 penalty
    assert p_sole[1] == 0.0
    # Zero gravity -> no NaNs or Infs!
    assert not torch.isnan(p_sole[2])
    assert not torch.isinf(p_sole[2])
    # 90 deg pitched foot -> sin^2(90) = 1.0 per foot -> sum = 2.0 -> penalty = -2.0
    assert torch.isclose(p_sole[4], torch.tensor(-2.0), atol=1e-5)
    # Massive gravity normalized -> 0 penalty for flat feet
    assert p_sole[5] == 0.0

    assert (p_sole <= 0.0).all()


# -----------------------------------------------------------------------------
# 5. NaN and Inf Injection (Sensor Faults / Out-of-Bounds Raycasts)
# -----------------------------------------------------------------------------

def test_nan_and_inf_robustness():
    env = MockEnv(4)

    # Inject NaNs into heights (e.g. raycast miss)
    env.scene.foot_height_scan.data.heights[:] = float("nan")
    p_clear = microduck_mdp.stairs_foot_clearance_penalty(env)
    assert not torch.isnan(p_clear).any()
    assert (p_clear <= 0.0).all()

    # Inject NaNs into contact forces
    env.scene.feet_ground_contact.data.force[:] = float("nan")
    env.scene.feet_ground_contact.data.found[:] = 1
    p_trip = microduck_mdp.stairs_riser_trip_penalty(env)
    assert not torch.isnan(p_trip).any()
    assert (p_trip <= 0.0).all()

    # Inject NaNs into site linear velocities
    env.scene.robot.data.site_lin_vel_w[:] = float("nan")
    env.scene.feet_ground_contact.data.current_contact_time[:] = 0.01
    p_damp = microduck_mdp.stairs_foot_landing_damping_penalty(env)
    assert not torch.isnan(p_damp).any()
    assert (p_damp <= 0.0).all()

    # Inject NaNs into root link and site positions for relative height progress
    env.scene.robot.data.root_link_pos_w[:] = float("nan")
    env.scene.robot.data.site_pos_w[:] = float("nan")
    p_rel = microduck_mdp.stairs_relative_height_progress(env)
    assert not torch.isnan(p_rel).any()


# -----------------------------------------------------------------------------
# 6. Relative Height Progress: Dynamics, Telescoping, and Elevation Invariance
# -----------------------------------------------------------------------------

def test_relative_height_progress_telescopic_and_invariance():
    """Verify telescoping property sum(delta_t) = pot_T - pot_0 and stair elevation invariance."""
    env = MockEnv(2)

    # Spawn lying down: trunk_z = 0.03, foot_z = 0.03 -> z_rel = 0.0
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.03
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.03
    env.episode_length_buf[:] = 0

    d0 = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
    assert (d0 == 0.0).all(), "Reset step 0 must be 0"

    env.episode_length_buf[:] = 1
    d1 = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
    assert (d1 == 0.0).all(), "Reset step 1 transient must be 0"

    # Multi-step rising trajectory
    deltas = []
    trajectory_z_trunk = [0.05, 0.08, 0.11, 0.145]  # final z_rel = 0.145 - 0.03 = 0.115 (ceiling)
    for step_idx, z in enumerate(trajectory_z_trunk, start=2):
        env.episode_length_buf[:] = step_idx
        env.scene.robot.data.root_link_pos_w[:, 2] = z
        d = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
        deltas.append(d)

    total_delta = torch.stack(deltas).sum(dim=0)
    # pot_0 = clamp(0.03 - 0.03, max=0.115) = 0.0
    # pot_T = clamp(0.145 - 0.03, max=0.115) = 0.115
    expected_total = torch.tensor([0.115, 0.115])
    assert torch.allclose(total_delta, expected_total, atol=1e-5), f"Telescopic sum mismatch: {total_delta} != {expected_total}"

    # Now step up to step 1 (+0.03m), step 2 (+0.06m), step 3 (+0.09m) maintaining standing posture
    for step_idx, elevation in enumerate([0.03, 0.06, 0.09], start=len(trajectory_z_trunk) + 2):
        env.episode_length_buf[:] = step_idx
        env.scene.robot.data.root_link_pos_w[:, 2] = 0.145 + elevation
        env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.03 + elevation
        d_stair = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
        assert torch.allclose(d_stair, torch.zeros(2), atol=1e-5), f"Stair climbing gave non-zero reward: {d_stair}"


def test_relative_height_inverted_posture():
    """Verify behavior when robot is inverted (trunk below feet)."""
    env = MockEnv(2)
    # Inverted: trunk_z = 0.02, feet up in air foot_z = 0.20 -> z_rel = -0.18
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.02
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.20
    env.episode_length_buf[:] = 0
    d0 = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
    assert (d0 == 0.0).all()

    # Step 2: Still inverted -> delta is 0
    env.episode_length_buf[:] = 2
    d1 = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
    assert (d1 == 0.0).all()

    # Step 3: Recovers to upright stand (trunk_z = 0.14, foot_z = 0.02 -> z_rel = 0.12 > ceiling)
    env.episode_length_buf[:] = 3
    env.scene.robot.data.root_link_pos_w[:, 2] = 0.14
    env.scene.robot.data.site_pos_w[:, [1, 6], 2] = 0.02
    d2 = microduck_mdp.stairs_relative_height_progress(env, ceiling=0.115)
    # Recovery gives positive progress: ceiling - (-0.18) = 0.115 + 0.18 = 0.295
    assert (d2 > 0.0).all()
    assert torch.allclose(d2, torch.tensor([0.295, 0.295]), atol=1e-5)


# -----------------------------------------------------------------------------
# 7. Real-Scale Batch (B=4096) and CUDA Compatibility
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_real_scale_batch_and_device(device):
    """Stress test with full training batch size (B=4096) on CPU and CUDA."""
    n = 4096
    env = MockEnv(n)
    env.device = device

    # Move all tensors to device
    env.episode_length_buf = env.episode_length_buf.to(device)
    env.command_manager.cmd = env.command_manager.cmd.to(device)
    env.scene.robot.data.site_lin_vel_w = torch.randn(n, 7, 3, device=device) * 5.0
    env.scene.robot.data.site_pos_w = torch.randn(n, 7, 3, device=device)
    env.scene.robot.data.site_quat_w = F.normalize(torch.randn(n, 7, 4, device=device), dim=-1)
    env.scene.robot.data.root_link_pos_w = torch.randn(n, 3, device=device)
    env.scene.robot.data.gravity_vec_w = torch.tensor([[0.0, 0.0, -9.81]], device=device).repeat(n, 1)

    env.scene.foot_height_scan.data.heights = torch.rand(n, 2, device=device) * 0.1
    env.scene.feet_ground_contact.data.force = torch.randn(n, 2, 3, device=device) * 20.0
    env.scene.feet_ground_contact.data.found = torch.randint(0, 2, (n, 2), dtype=torch.int32, device=device)
    env.scene.feet_ground_contact.data.current_contact_time = torch.rand(n, 2, device=device) * 0.05

    # Evaluate all 5 functions
    p_clear = microduck_mdp.stairs_foot_clearance_penalty(env)
    p_damp = microduck_mdp.stairs_foot_landing_damping_penalty(env)
    p_trip = microduck_mdp.stairs_riser_trip_penalty(env)
    p_sole = microduck_mdp.stairs_sole_flat_penalty(env)
    p_rel = microduck_mdp.stairs_relative_height_progress(env)

    for p, name in [
        (p_clear, "stairs_foot_clearance_penalty"),
        (p_damp, "stairs_foot_landing_damping_penalty"),
        (p_trip, "stairs_riser_trip_penalty"),
        (p_sole, "stairs_sole_flat_penalty"),
    ]:
        assert p.shape == (n,), f"{name} bad shape {p.shape}"
        assert p.device.type == device, f"{name} bad device {p.device.type}"
        assert (p <= 0.0).all(), f"{name} produced positive value: max={p.max()}"
        assert not torch.isnan(p).any(), f"{name} produced NaN"
        assert not torch.isinf(p).any(), f"{name} produced Inf"

    assert p_rel.shape == (n,)
    assert p_rel.device.type == device
    assert not torch.isnan(p_rel).any()
    assert not torch.isinf(p_rel).any()


# -----------------------------------------------------------------------------
# 8. Sensor Degradation & Fallback Modes
# -----------------------------------------------------------------------------

def test_missing_contact_time_fallback():
    """Verify sole flat penalty gracefully falls back to sensor.data.found when current_contact_time is None."""
    env = MockEnv(3)
    env.scene.feet_ground_contact.data.current_contact_time = None
    env.scene.feet_ground_contact.data.found[:] = 1  # in contact via found flag

    # Tilt foot
    env.scene.robot.data.site_quat_w[:, [1, 6], :] = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
    p_sole = microduck_mdp.stairs_sole_flat_penalty(env)
    assert (p_sole < 0.0).all(), "Tilted foot with found=1 must be penalized even when contact_time is None"

    # No contact via found flag
    env.scene.feet_ground_contact.data.found[:] = 0
    p_sole_nocontact = microduck_mdp.stairs_sole_flat_penalty(env)
    assert (p_sole_nocontact == 0.0).all(), "Foot with found=0 must produce 0 penalty"


def test_command_manager_absent():
    """Verify clearance penalty functions when env lacks command_manager (eval/standalone scenes)."""
    env = MockEnv(2)
    delattr(env, "command_manager")
    env.scene.robot.data.site_lin_vel_w[:, [1, 6], 0] = 0.5
    env.scene.foot_height_scan.data.heights[:] = 0.01

    p_clear = microduck_mdp.stairs_foot_clearance_penalty(env)
    assert (p_clear < 0.0).all(), "Clearance penalty must function even without command_manager"
    assert (p_clear <= 0.0).all()

