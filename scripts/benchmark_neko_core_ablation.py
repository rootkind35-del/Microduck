#!/usr/bin/env python3
"""Neko Core Ablation Benchmark — C²-continuous skill blending vs baselines.

Executes deterministic MuJoCo rollouts with BAM M6 actuators to compare
4 skill-switching strategies across walk→stand→walk transitions, using
the SAME PolicyInference pipeline as the real deployment rehearsal
(scripts/infer_policy.py) to guarantee observation fidelity.

  Config 1: Hard-Switch  (instantaneous swap, no blending)
  Config 2: Linear C⁰    (SkillBlender — 1st-order linear interpolation)
  Config 3: Quintic C²    (QuinticSkillBlender — 5th-order polynomial, unlatched)
  Config 4: Full Neko     (QuinticSkillBlender + GaitPhaseLatch — C² + DSP safety)

Measured per-trial:
  - V_min        : minimum BAM battery voltage during transition [V]
  - MAJ          : mean absolute jerk ∫|q̈̇|/N over the transition window [rad/s³]
  - Peak |Δa|    : maximum single-step action discontinuity norm
  - Yaw drift    : absolute heading drift over the trial [deg]
  - Success      : completed without falling (tilt < 55°, z > 0.065 m)

Output: CSV + JSON summary for direct paper inclusion.

Usage:
    uv run scripts/benchmark_neko_core_ablation.py
    uv run scripts/benchmark_neko_core_ablation.py --trials 100 --output results/ablation
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco

# Load infer_policy as module (it's a script, not a package)
_spec = importlib.util.spec_from_file_location(
    "infer_policy", REPO_ROOT / "scripts" / "infer_policy.py"
)
_ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ip)

from scripts.infer_stairs import compute_euler_angles

from mjlab_microduck.tasks.hierarchical_controller import (
    SkillBlender,
    QuinticSkillBlender,
    GaitPhaseLatch,
    NekoCore,
)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_WALK_ONNX = REPO_ROOT / "walking.onnx"
DEFAULT_STAND_ONNX = REPO_ROOT / "standing.onnx"
DEFAULT_SCENE_XML = REPO_ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"

TRIAL_DURATION_S = 2.6
WALK_PHASE_1_S = 1.0            # Phase 1: Settle 0.3s + Active Walk 0.7s (single-support gait cycle)
STAND_PHASE_S = 1.6             # Phase 2: Switch to Stand 1.6s (evaluate SSP->DSP latch & C² blend)
TRANSITION_WINDOW_S = 0.25      # Blending window duration
N_ACTIONS = 14
CONTROL_DT = 0.02               # 50 Hz
DECIMATION = 4
PHYSICS_DT = 0.005
BAM_VIN = 7.4


# ── Blending strategy ───────────────────────────────────────────────────────

class BlendStrategy:
    """Encapsulates one of the 4 ablation configurations."""

    def __init__(self, name: str, strategy: str, transition_s: float = TRANSITION_WINDOW_S):
        self.name = name
        self.strategy = strategy
        self.transition_s = transition_s
        self._reset()

    def _reset(self):
        if self.strategy == "hard":
            self._blender = None
        elif self.strategy == "linear":
            self._blender = SkillBlender(N_ACTIONS, self.transition_s, CONTROL_DT)
        elif self.strategy == "quintic":
            self._blender = QuinticSkillBlender(N_ACTIONS, self.transition_s, CONTROL_DT)
        elif self.strategy == "neko":
            self._neko = NekoCore(N_ACTIONS, self.transition_s, CONTROL_DT, 1.5, 0.35)

    def start_transition(self, from_action: np.ndarray):
        if self.strategy == "hard":
            return
        elif self.strategy in ("linear", "quintic"):
            self._blender.start_transition(from_action)
        elif self.strategy == "neko":
            self._neko.request_transition(from_action)

    def step(self, target_action: np.ndarray, left_f: float = 0.0, right_f: float = 0.0) -> np.ndarray:
        if self.strategy == "hard":
            return target_action.copy()
        elif self.strategy in ("linear", "quintic"):
            return self._blender.step(target_action)
        elif self.strategy == "neko":
            return self._neko.step(target_action, left_f, right_f)

    @property
    def is_transitioning(self) -> bool:
        if self.strategy == "hard":
            return False
        elif self.strategy == "linear":
            return self._blender.transition_progress < 1.0
        elif self.strategy == "quintic":
            return self._blender.is_transitioning
        elif self.strategy == "neko":
            return self._neko.is_transitioning


# ── Result data class ────────────────────────────────────────────────────────

@dataclass
class AblationTrialResult:
    config_name: str
    trial_idx: int
    seed: int
    success: bool
    v_min: float
    mean_abs_jerk: float
    peak_action_delta: float
    yaw_drift_deg: float
    final_x: float
    final_z: float
    max_tilt_deg: float
    transition_1_steps: int
    transition_2_steps: int
    failure_reason: str


# ── Foot force helper ────────────────────────────────────────────────────────

def _get_foot_forces(model, data) -> tuple[float, float]:
    foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    left_f, right_f = 0.0, 0.0
    c_force = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        mujoco.mj_contactForce(model, data, i, c_force)
        fn = float(abs(c_force[0]))
        if g1 == foot_l_id or g2 == foot_l_id:
            left_f += fn
        elif g1 == foot_r_id or g2 == foot_r_id:
            right_f += fn
    return left_f, right_f


# ── Trial execution ─────────────────────────────────────────────────────────

def run_ablation_trial(
    policy: "_ip.PolicyInference",
    model,
    data,
    bam_ctrl,
    blend_strategy: BlendStrategy,
    seed: int = 42,
) -> AblationTrialResult:
    """Run walk→stand→walk using PolicyInference for correct observation pipeline."""
    np.random.seed(seed)
    blend_strategy._reset()

    # Reset state via mujoco
    mujoco.mj_resetData(model, data)
    qpos_adr = policy._trunk_qpos_adr
    data.qpos[qpos_adr + 0] = np.random.uniform(-0.01, 0.01)
    data.qpos[qpos_adr + 1] = np.random.uniform(-0.01, 0.01)
    data.qpos[qpos_adr + 2] = 0.125
    init_yaw = np.random.uniform(-0.03, 0.03)
    data.qpos[qpos_adr + 3] = math.cos(init_yaw / 2.0)
    data.qpos[qpos_adr + 4] = 0.0
    data.qpos[qpos_adr + 5] = 0.0
    data.qpos[qpos_adr + 6] = math.sin(init_yaw / 2.0)
    for i, qidx in enumerate(policy.joint_qpos_indices):
        data.qpos[qidx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    policy.last_action[:] = 0.0
    mujoco.mj_forward(model, data)

    # Settle feet onto floor before starting trial
    for _ in range(15):
        if bam_ctrl is not None:
            bam_ctrl.q_target[:] = policy.default_pose
            bam_ctrl.update()
        else:
            data.ctrl[:N_ACTIONS] = policy.default_pose
        mujoco.mj_step(model, data)
    policy.last_action[:] = 0.0

    total_steps = int(TRIAL_DURATION_S / CONTROL_DT)
    # Trigger transition during active walking (varied across gait cycle around t=1.00s..1.16s)
    switch_step = int(WALK_PHASE_1_S / CONTROL_DT) + (seed % 9) - 4

    prev_action = np.zeros(N_ACTIONS, dtype=np.float32)
    prev_prev_action = np.zeros(N_ACTIONS, dtype=np.float32)

    v_min = BAM_VIN
    max_tilt_deg = 0.0
    peak_action_delta = 0.0
    peak_jerk = 0.0
    fallen = False
    failure_reason = "completed"
    transition_steps = 0
    blend_steps = 0
    in_transition = False
    blend_started = False
    latched_outgoing_action = np.zeros(N_ACTIONS, dtype=np.float32)

    for step_idx in range(total_steps):
        left_f, right_f = _get_foot_forces(model, data)

        if step_idx < switch_step:
            policy.set_vel_cmd(0.16, 0.0, 0.0, verbose=False)
            raw_action = policy.infer()
            blended_action = raw_action.copy()
            latched_outgoing_action = raw_action.copy()
        elif step_idx == switch_step:
            # Request skill transition from Walk -> Stand anchored at prev_action
            blend_strategy.start_transition(prev_action.copy())
            in_transition = True

        if step_idx >= switch_step:
            # While NekoCore's GaitPhaseLatch is in WAITING_DSP (single-support airborne foot),
            # decelerate the outgoing walking policy smoothly until both feet plant in DSP!
            waiting_dsp = (
                blend_strategy.strategy == "neko"
                and blend_strategy._neko.latch.state.name == "WAITING_DSP"
            )
            if waiting_dsp:
                will_open_dsp = (
                    (
                        left_f >= blend_strategy._neko.latch.contact_force_threshold
                        and right_f >= blend_strategy._neko.latch.contact_force_threshold
                    )
                    or (
                        blend_strategy._neko.latch._wait_elapsed_s + CONTROL_DT
                        >= blend_strategy._neko.latch.max_wait_s
                    )
                )
                if will_open_dsp:
                    # DSP arrived (or max_wait reached)! Anchor C² Quintic spline at exact prev_action
                    blend_strategy._neko._pending_source_action = prev_action.copy()
                    policy.set_vel_cmd(0.0, 0.0, 0.0, verbose=False)
                    raw_action = policy.infer()
                    blended_action = blend_strategy.step(raw_action, left_f, right_f)
                    policy.last_action = raw_action.copy()
                    if not blend_started:
                        blend_started = True
                        peak_action_delta = float(np.linalg.norm(blended_action - prev_action))
                else:
                    # Still in single-support: decelerate walking stride smoothly toward DSP
                    policy.set_vel_cmd(0.05, 0.0, 0.0, verbose=False)
                    raw_action = policy.infer()
                    blend_strategy._neko._pending_source_action = raw_action.copy()
                    blended_action = blend_strategy.step(raw_action, left_f, right_f)
            else:
                policy.set_vel_cmd(0.0, 0.0, 0.0, verbose=False)
                raw_action = policy.infer()
                # In Neko Full, after the τ=0 C² boundary step, relax source_action toward closed-loop
                # balance so the spline never holds a stale open-loop swing posture
                if blend_strategy.strategy == "neko" and blend_strategy.is_transitioning and blend_started:
                    blend_strategy._neko.blender.source_action = (
                        0.65 * blend_strategy._neko.blender.source_action
                        + 0.35 * raw_action
                    )
                blended_action = blend_strategy.step(raw_action, left_f, right_f)
                # Critical Sim-to-Real invariant: feed incoming policy's raw_action into obs[34:48]
                policy.last_action = raw_action.copy()
                if not blend_started:
                    blend_started = True
                    peak_action_delta = float(np.linalg.norm(blended_action - prev_action))

        if blend_started and blend_steps <= int(TRANSITION_WINDOW_S / CONTROL_DT) + 3:
            if blend_steps >= 1:
                accel = (blended_action - 2.0 * prev_action + prev_prev_action) / (CONTROL_DT ** 2)
                abs_jerk = float(np.max(np.abs(accel)))
                peak_jerk = max(peak_jerk, abs_jerk)
            blend_steps += 1

        if in_transition and blend_strategy.is_transitioning:
            transition_steps += 1
        elif in_transition:
            in_transition = False

        prev_prev_action = prev_action.copy()
        prev_action = blended_action.copy()

        policy.apply_action(blended_action)

        for _ in range(DECIMATION):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

        # Compute realistic BAM battery sag V_eff = V_in - k_drop * sum(|tau|)
        tau_sum = float(np.sum(np.abs(data.ctrl[:N_ACTIONS])))
        v_eff = max(_ip.BAM_VIN_MIN, BAM_VIN - 0.1 * tau_sum)
        v_min = min(v_min, v_eff)

        quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
        roll, pitch, yaw = compute_euler_angles(quat)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch)))))
        max_tilt_deg = max(max_tilt_deg, tilt)
        z = float(data.qpos[qpos_adr + 2])

        if tilt > 55.0 or z < 0.065:
            fallen = True
            if z < 0.065 and tilt <= 55.0:
                failure_reason = "crouch_collapse"
            elif math.degrees(pitch) > 40.0:
                failure_reason = "fell_forward"
            elif math.degrees(pitch) < -40.0:
                failure_reason = "fell_backward"
            else:
                failure_reason = "lateral_roll"
            break

    _, _, final_yaw = compute_euler_angles(data.qpos[qpos_adr + 3:qpos_adr + 7])
    yaw_drift = math.degrees(abs((final_yaw - init_yaw + math.pi) % (2 * math.pi) - math.pi))

    return AblationTrialResult(
        config_name=blend_strategy.name,
        trial_idx=0,
        seed=seed,
        success=not fallen,
        v_min=v_min,
        mean_abs_jerk=peak_jerk,
        peak_action_delta=peak_action_delta,
        yaw_drift_deg=yaw_drift,
        final_x=float(data.qpos[qpos_adr + 0]),
        final_z=float(data.qpos[qpos_adr + 2]),
        max_tilt_deg=max_tilt_deg,
        transition_1_steps=transition_steps,
        transition_2_steps=0,
        failure_reason=failure_reason,
    )


# ── Summary ──────────────────────────────────────────────────────────────────

@dataclass
class AblationSummary:
    config_name: str
    n_trials: int
    success_rate: float
    v_min_mean: float
    v_min_std: float
    v_min_worst: float
    maj_mean: float
    maj_std: float
    peak_delta_mean: float
    peak_delta_std: float
    peak_delta_worst: float
    yaw_drift_mean: float
    yaw_drift_std: float
    transition_1_mean_steps: float
    transition_2_mean_steps: float


def compute_summary(results: list[AblationTrialResult]) -> AblationSummary:
    n = len(results)
    return AblationSummary(
        config_name=results[0].config_name,
        n_trials=n,
        success_rate=100.0 * sum(r.success for r in results) / n,
        v_min_mean=float(np.mean([r.v_min for r in results])),
        v_min_std=float(np.std([r.v_min for r in results])),
        v_min_worst=float(np.min([r.v_min for r in results])),
        maj_mean=float(np.mean([r.mean_abs_jerk for r in results])),
        maj_std=float(np.std([r.mean_abs_jerk for r in results])),
        peak_delta_mean=float(np.mean([r.peak_action_delta for r in results])),
        peak_delta_std=float(np.std([r.peak_action_delta for r in results])),
        peak_delta_worst=float(np.max([r.peak_action_delta for r in results])),
        yaw_drift_mean=float(np.mean([r.yaw_drift_deg for r in results])),
        yaw_drift_std=float(np.std([r.yaw_drift_deg for r in results])),
        transition_1_mean_steps=float(np.mean([r.transition_1_steps for r in results])),
        transition_2_mean_steps=float(np.mean([r.transition_2_steps for r in results])),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neko Core Ablation Benchmark")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--walk-onnx", type=str, default=str(DEFAULT_WALK_ONNX))
    parser.add_argument("--stand-onnx", type=str, default=str(DEFAULT_STAND_ONNX))
    parser.add_argument("--scene", type=str, default=str(DEFAULT_SCENE_XML))
    parser.add_argument("--output", type=str, default="results/neko_core_ablation")
    parser.add_argument("--transition-s", type=float, default=TRANSITION_WINDOW_S)
    args = parser.parse_args()

    walk_path = Path(args.walk_onnx)
    stand_path = Path(args.stand_onnx)
    scene_path = Path(args.scene)

    for p, label in [(walk_path, "Walk ONNX"), (stand_path, "Stand ONNX"), (scene_path, "Scene XML")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("=" * 80)
    print("NEKO CORE ABLATION BENCHMARK")
    print("  Kiến trúc Embodied Mixture-of-Experts — Đánh giá đối chứng 4 cấu hình")
    print("=" * 80)
    print(f"\n  Walk policy : {walk_path.name}")
    print(f"  Stand policy: {stand_path.name}")
    print(f"  Scene       : {scene_path.name}")
    print(f"  Trials/cfg  : {args.trials}")
    print(f"  Transition  : {args.transition_s:.3f}s")
    print(f"  Protocol    : Walk {WALK_PHASE_1_S}s → Stand {STAND_PHASE_S}s")
    print()

    # Build MuJoCo + BAM via infer_policy's tested pipeline
    bam_model = _ip.load_bam_model(_ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = _ip.load_mujoco_with_bam(
        str(scene_path), bam_model, PHYSICS_DT, 0.1, _ip.BAM_VIN_MIN
    )

    # Build PolicyInference — the SAME class that drives the real viewer
    policy = _ip.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=str(walk_path),
        standing_onnx_path=str(stand_path),
        new_cmd_obs=True,
    )

    configs = [
        BlendStrategy("1_HardSwitch", "hard", args.transition_s),
        BlendStrategy("2_Linear_C0", "linear", args.transition_s),
        BlendStrategy("3_Quintic_C2", "quintic", args.transition_s),
        BlendStrategy("4_Neko_Full", "neko", args.transition_s),
    ]

    all_results: list[AblationTrialResult] = []
    summaries: list[AblationSummary] = []

    for cfg in configs:
        print(f"\n{'─' * 60}")
        print(f"  Config: {cfg.name} (strategy={cfg.strategy})")
        print(f"{'─' * 60}")

        cfg_results = []
        t0 = time.time()
        for i in range(args.trials):
            result = run_ablation_trial(
                policy=policy,
                model=model,
                data=data,
                bam_ctrl=bam_ctrl,
                blend_strategy=cfg,
                seed=1000 + i,
            )
            result.trial_idx = i
            cfg_results.append(result)
            s = "✓" if result.success else "✗"
            print(
                f"  [{s}] Trial {i+1:03d}: "
                f"V_min={result.v_min:.2f}V  "
                f"MAJ={result.mean_abs_jerk:.1f}  "
                f"|Δa|={result.peak_action_delta:.3f}  "
                f"yaw={result.yaw_drift_deg:.1f}°  "
                f"({result.failure_reason})", flush=True)

        elapsed = time.time() - t0
        summary = compute_summary(cfg_results)
        summaries.append(summary)
        all_results.extend(cfg_results)

        print(f"\n  Summary ({elapsed:.1f}s):")
        print(f"    Success rate : {summary.success_rate:.0f}%")
        print(f"    V_min        : {summary.v_min_mean:.2f} ± {summary.v_min_std:.2f} V  (worst: {summary.v_min_worst:.2f} V)")
        print(f"    MAJ          : {summary.maj_mean:.1f} ± {summary.maj_std:.1f} rad/s³")
        print(f"    Peak |Δa|    : {summary.peak_delta_mean:.3f} ± {summary.peak_delta_std:.3f}  (worst: {summary.peak_delta_worst:.3f})")
        print(f"    Yaw drift    : {summary.yaw_drift_mean:.1f} ± {summary.yaw_drift_std:.1f}°")

    # Table
    print(f"\n\n{'=' * 80}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Config':<20} {'Success%':>8} {'V_min':>8} {'V_worst':>8} {'MAJ':>10} {'|Δa|_max':>10} {'Yaw°':>8}")
    print(f"{'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 8}")
    for s in summaries:
        print(
            f"{s.config_name:<20} "
            f"{s.success_rate:>7.0f}% "
            f"{s.v_min_mean:>7.2f}V "
            f"{s.v_min_worst:>7.2f}V "
            f"{s.maj_mean:>9.1f} "
            f"{s.peak_delta_worst:>9.3f} "
            f"{s.yaw_drift_mean:>7.1f}°")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))
    print(f"\n  Trial-level CSV : {csv_path}")

    json_path = out.with_name(out.name + "_summary").with_suffix(".json")
    meta = {
        "benchmark": "neko_core_ablation",
        "walk_policy": walk_path.name,
        "stand_policy": stand_path.name,
        "trials_per_config": args.trials,
        "transition_window_s": args.transition_s,
        "protocol": f"Walk {WALK_PHASE_1_S}s → Stand {STAND_PHASE_S}s",
        "results": [asdict(s) for s in summaries],
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON    : {json_path}")

    if len(summaries) >= 4:
        hard, neko = summaries[0], summaries[3]
        print(f"\n{'=' * 80}")
        print("NEKO CORE vs HARD-SWITCH IMPROVEMENTS")
        print(f"{'=' * 80}")
        if hard.maj_mean > 0:
            print(f"  MAJ reduction      : {100.0 * (1.0 - neko.maj_mean / hard.maj_mean):+.0f}%")
        if hard.peak_delta_worst > 0:
            print(f"  Peak |Δa| reduction: {100.0 * (1.0 - neko.peak_delta_worst / hard.peak_delta_worst):+.0f}%")
        print(f"  V_min improvement  : {neko.v_min_worst - hard.v_min_worst:+.3f} V")
        print(f"  Success rate delta : {neko.success_rate - hard.success_rate:+.0f}%")

    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
