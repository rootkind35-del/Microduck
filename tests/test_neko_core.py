"""Tests for hierarchical_controller.py — Neko Core.

Validates the mathematical properties of the Quintic S-Curve blending
weight and the GaitPhaseLatch safety mechanism.  All tests run on CPU
with no GPU or simulation dependency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mjlab_microduck.tasks.hierarchical_controller import (
    GaitPhaseLatch,
    LatchState,
    NekoCore,
    QuinticSkillBlender,
    SkillBlender,
    quintic_s_curve,
    quintic_s_curve_derivative,
    quintic_s_curve_second_derivative,
)


# ── Quintic S-Curve Mathematical Properties ──────────────────────────────────

class TestQuinticSCurve:
    """Verify the 6 boundary conditions of the quintic polynomial."""

    def test_boundary_position_zero(self):
        """s(0) = 0."""
        assert quintic_s_curve(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_boundary_position_one(self):
        """s(1) = 1."""
        assert quintic_s_curve(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_boundary_velocity_zero_at_start(self):
        """ṡ(0) = 0 — no velocity jump at transition start."""
        assert quintic_s_curve_derivative(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_boundary_velocity_zero_at_end(self):
        """ṡ(1) = 0 — no velocity jump at transition end."""
        assert quintic_s_curve_derivative(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_boundary_acceleration_zero_at_start(self):
        """s̈(0) = 0 — no acceleration/jerk spike at transition start."""
        assert quintic_s_curve_second_derivative(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_boundary_acceleration_zero_at_end(self):
        """s̈(1) = 0 — no acceleration/jerk spike at transition end."""
        assert quintic_s_curve_second_derivative(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_monotonically_increasing(self):
        """s(τ) must be strictly monotonically increasing on (0, 1)."""
        taus = np.linspace(0.0, 1.0, 1001)
        values = [quintic_s_curve(t) for t in taus]
        diffs = np.diff(values)
        assert np.all(diffs >= -1e-12), "S-curve is not monotonically increasing"

    def test_derivative_non_negative(self):
        """ṡ(τ) ≥ 0 for all τ ∈ [0, 1] (ensures monotonicity)."""
        taus = np.linspace(0.0, 1.0, 1001)
        derivs = [quintic_s_curve_derivative(t) for t in taus]
        assert np.all(np.array(derivs) >= -1e-12)

    def test_derivative_peaks_at_midpoint(self):
        """ṡ(0.5) is the maximum velocity — the hump of the S-curve."""
        d_mid = quintic_s_curve_derivative(0.5)
        # Analytical: ṡ(0.5) = 30 * 0.25 * 0.25 = 1.875
        assert d_mid == pytest.approx(1.875, rel=1e-6)

    def test_midpoint_value(self):
        """s(0.5) = 0.5 — the S-curve passes through the center."""
        assert quintic_s_curve(0.5) == pytest.approx(0.5, rel=1e-6)

    def test_clamping_below_zero(self):
        """s(τ < 0) is clamped to s(0) = 0."""
        assert quintic_s_curve(-0.5) == pytest.approx(0.0, abs=1e-12)

    def test_clamping_above_one(self):
        """s(τ > 1) is clamped to s(1) = 1."""
        assert quintic_s_curve(1.5) == pytest.approx(1.0, abs=1e-12)

    def test_numerical_derivative_matches_analytical(self):
        """Finite-difference approximation matches analytical ṡ(τ)."""
        h = 1e-7
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            numerical = (quintic_s_curve(tau + h) - quintic_s_curve(tau - h)) / (2 * h)
            analytical = quintic_s_curve_derivative(tau)
            assert numerical == pytest.approx(analytical, rel=1e-4)

    def test_numerical_second_derivative_matches_analytical(self):
        """Finite-difference approximation matches analytical s̈(τ)."""
        h = 1e-4
        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            numerical = (
                quintic_s_curve(tau + h) - 2 * quintic_s_curve(tau) + quintic_s_curve(tau - h)
            ) / (h * h)
            analytical = quintic_s_curve_second_derivative(tau)
            assert numerical == pytest.approx(analytical, abs=0.5)


# ── QuinticSkillBlender ──────────────────────────────────────────────────────

class TestQuinticSkillBlender:

    def test_no_transition_returns_target(self):
        """Without start_transition(), target is passed through."""
        blender = QuinticSkillBlender(n_actions=4)
        target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = blender.step(target)
        np.testing.assert_allclose(result, target)

    def test_transition_starts_at_source(self):
        """At τ=0, the output equals the source action."""
        blender = QuinticSkillBlender(n_actions=4, transition_duration_s=0.10, control_dt=0.02)
        source = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        blender.start_transition(source)
        # First step: τ = 0.02/0.10 = 0.2 (not exactly 0, but close to source)
        result = blender.step(target)
        s_val = quintic_s_curve(0.2)
        expected = (1.0 - s_val) * source + s_val * target
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_transition_ends_at_target(self):
        """After enough steps, the output converges to the target."""
        dt = 0.02
        dur = 0.10
        blender = QuinticSkillBlender(n_actions=4, transition_duration_s=dur, control_dt=dt)
        source = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        blender.start_transition(source)
        n_steps = int(math.ceil(dur / dt)) + 2
        for _ in range(n_steps):
            result = blender.step(target)

        np.testing.assert_allclose(result, target, atol=1e-6)
        assert not blender.is_transitioning

    def test_c2_smoothness_lower_peak_than_linear(self):
        """Verify quintic peak acceleration is significantly lower than linear.

        The mathematical C²-continuity is proven by the analytical tests
        (boundary conditions s̈(0)=s̈(1)=0).  This test verifies the
        PRACTICAL consequence: quintic produces much lower discrete-time
        acceleration peaks than the C⁰-linear blender, which is the
        metric that directly correlates with actuator current spikes.
        """
        dt = 0.0005  # Very fine timestep
        dur = 0.10

        def measure_peak_accel(blender_cls):
            blender = blender_cls(n_actions=1, transition_duration_s=dur, control_dt=dt)
            source = np.array([0.0], dtype=np.float32)
            target = np.array([1.0], dtype=np.float32)
            blender.start_transition(source)
            actions = []
            for _ in range(int(dur / dt) + 10):
                a = blender.step(target)
                actions.append(a[0])
            actions = np.array(actions)
            velocities = np.diff(actions) / dt
            accelerations = np.diff(velocities) / dt
            return float(np.max(np.abs(accelerations)))

        peak_linear = measure_peak_accel(SkillBlender)
        peak_quintic = measure_peak_accel(QuinticSkillBlender)

        # Quintic should be at least 30% smoother
        assert peak_quintic < peak_linear * 0.7, (
            f"Quintic ({peak_quintic:.1f}) not sufficiently smoother than "
            f"Linear ({peak_linear:.1f})"
        )

    def test_reset_clears_state(self):
        """reset() returns blender to idle state."""
        blender = QuinticSkillBlender(n_actions=4)
        blender.start_transition(np.ones(4, dtype=np.float32))
        assert blender.is_transitioning
        blender.reset()
        assert not blender.is_transitioning
        np.testing.assert_allclose(blender.current_action, np.zeros(4))


# ── Legacy SkillBlender (backward compatibility) ─────────────────────────────

class TestLegacySkillBlender:

    def test_linear_blend_midpoint(self):
        """Legacy linear blender should produce midpoint at 50% progress."""
        dt = 0.02
        dur = 0.10
        blender = SkillBlender(n_actions=2, transition_duration_s=dur, control_dt=dt)
        source = np.array([0.0, 0.0], dtype=np.float32)
        target = np.array([1.0, 1.0], dtype=np.float32)

        blender.start_transition(source)
        # Step to ~50% (2.5 steps = γ ≈ 0.4 at step 2)
        blender.step(target)  # γ = 0.2
        blender.step(target)  # γ = 0.4
        result = blender.step(target)  # γ = 0.6
        # With linear blending: result = 0.6 * target = [0.6, 0.6]
        np.testing.assert_allclose(result, [0.6, 0.6], atol=1e-6)


# ── GaitPhaseLatch ───────────────────────────────────────────────────────────

class TestGaitPhaseLatch:

    def test_idle_returns_false(self):
        """No transition requested → update always returns False."""
        latch = GaitPhaseLatch()
        assert not latch.update(10.0, 10.0)
        assert latch.state == LatchState.IDLE

    def test_opens_on_double_support(self):
        """Transition request + both feet grounded → opens immediately."""
        latch = GaitPhaseLatch(contact_force_threshold=2.0)
        latch.request_transition()
        assert latch.state == LatchState.WAITING_DSP

        opened = latch.update(5.0, 5.0)  # Both feet > 2.0 N
        assert opened is True
        assert latch.state == LatchState.BLENDING

    def test_waits_during_single_support(self):
        """If only one foot is grounded, the latch waits."""
        latch = GaitPhaseLatch(contact_force_threshold=2.0, control_dt=0.02)
        latch.request_transition()

        # Left foot grounded, right foot airborne
        opened = latch.update(5.0, 0.5)
        assert opened is False
        assert latch.state == LatchState.WAITING_DSP

    def test_timeout_forces_transition(self):
        """After max_wait_s, transition is forced even without DSP."""
        dt = 0.02
        max_wait = 0.10
        latch = GaitPhaseLatch(
            contact_force_threshold=2.0,
            max_wait_s=max_wait,
            control_dt=dt,
        )
        latch.request_transition()

        # Simulate single support for max_wait_s
        n_steps = int(math.ceil(max_wait / dt))
        opened = False
        for _ in range(n_steps):
            opened = latch.update(0.0, 0.0)  # Both feet airborne
            if opened:
                break

        assert opened is True, "Timeout should have forced the transition"
        assert latch.state == LatchState.BLENDING

    def test_on_blend_complete_resets(self):
        """After blending completes, latch returns to IDLE."""
        latch = GaitPhaseLatch()
        latch.request_transition()
        latch.update(10.0, 10.0)  # Opens
        assert latch.state == LatchState.BLENDING

        latch.on_blend_complete()
        assert latch.state == LatchState.IDLE

    def test_does_not_fire_twice(self):
        """Once opened, subsequent updates return False until reset."""
        latch = GaitPhaseLatch()
        latch.request_transition()
        first = latch.update(10.0, 10.0)
        assert first is True

        second = latch.update(10.0, 10.0)
        assert second is False


# ── NekoCore Integration ─────────────────────────────────────────────────────

class TestNekoCore:

    def test_passthrough_without_transition(self):
        """Without requesting a transition, target passes through."""
        neko = NekoCore(n_actions=4)
        target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = neko.step(target, left_foot_force=5.0, right_foot_force=5.0)
        np.testing.assert_allclose(result, target)

    def test_holds_source_during_single_support(self):
        """While waiting for DSP, the old policy action is held."""
        neko = NekoCore(n_actions=4, contact_force_threshold=2.0)
        source = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        neko.request_transition(source)

        # Single support: left foot only
        result = neko.step(target, left_foot_force=5.0, right_foot_force=0.5)
        np.testing.assert_allclose(result, source, atol=1e-6)

    def test_blends_after_dsp_detected(self):
        """Once DSP is detected, blending begins and result differs from source."""
        neko = NekoCore(n_actions=4, transition_duration_s=0.10, control_dt=0.02)
        source = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        neko.request_transition(source)

        # First step: DSP detected → blend starts
        result = neko.step(target, left_foot_force=5.0, right_foot_force=5.0)
        # Should be between source and target (not exactly either)
        assert not np.allclose(result, source, atol=1e-3)
        assert not np.allclose(result, target, atol=1e-3)

    def test_full_transition_converges(self):
        """After enough steps with DSP, output converges to target."""
        dt = 0.02
        dur = 0.10
        neko = NekoCore(n_actions=4, transition_duration_s=dur, control_dt=dt)
        source = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        neko.request_transition(source)

        n_steps = int(math.ceil(dur / dt)) + 5
        for _ in range(n_steps):
            result = neko.step(target, left_foot_force=5.0, right_foot_force=5.0)

        np.testing.assert_allclose(result, target, atol=1e-6)
        assert not neko.is_transitioning

    def test_timeout_forces_blend_start(self):
        """If DSP never comes, timeout starts the blend anyway."""
        dt = 0.02
        max_wait = 0.08  # 4 steps
        dur = 0.10
        neko = NekoCore(
            n_actions=2,
            transition_duration_s=dur,
            control_dt=dt,
            max_dsp_wait_s=max_wait,
        )
        source = np.array([1.0, 1.0], dtype=np.float32)
        target = np.array([0.0, 0.0], dtype=np.float32)

        neko.request_transition(source)

        # All steps with airborne feet
        results = []
        for _ in range(20):
            r = neko.step(target, left_foot_force=0.0, right_foot_force=0.0)
            results.append(r.copy())

        # After timeout + blend duration, should be at target
        np.testing.assert_allclose(results[-1], target, atol=1e-6)

    def test_reset_clears_everything(self):
        """reset() returns NekoCore to passthrough mode."""
        neko = NekoCore(n_actions=4)
        neko.request_transition(np.ones(4, dtype=np.float32))
        assert neko.is_transitioning

        neko.reset()
        assert not neko.is_transitioning

        target = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        result = neko.step(target)
        np.testing.assert_allclose(result, target)


# ── Comparative: Quintic vs Linear smoothness ────────────────────────────────

class TestQuinticVsLinearSmoothness:
    """Ablation-ready: prove Quintic is smoother than Linear."""

    def _measure_peak_acceleration(self, blender_cls, **kwargs):
        """Run a transition and return peak |acceleration| of the blended action."""
        dt = 0.001
        dur = 0.10
        blender = blender_cls(n_actions=1, transition_duration_s=dur, control_dt=dt, **kwargs)
        source = np.array([0.0], dtype=np.float32)
        target = np.array([1.0], dtype=np.float32)

        blender.start_transition(source)

        actions = []
        for _ in range(int(dur / dt) + 10):
            a = blender.step(target)
            actions.append(a[0])

        actions = np.array(actions)
        velocities = np.diff(actions) / dt
        accelerations = np.diff(velocities) / dt
        return float(np.max(np.abs(accelerations)))

    def test_quintic_has_lower_peak_acceleration(self):
        """QuinticSkillBlender should produce lower peak acceleration than SkillBlender."""
        peak_linear = self._measure_peak_acceleration(SkillBlender)
        peak_quintic = self._measure_peak_acceleration(QuinticSkillBlender)

        assert peak_quintic < peak_linear, (
            f"Quintic peak accel ({peak_quintic:.1f}) should be less than "
            f"Linear peak accel ({peak_linear:.1f})"
        )
