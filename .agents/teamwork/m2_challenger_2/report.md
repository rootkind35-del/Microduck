# Empirical Stress Test Report — M2 Challenger 2

**Author**: M2 Challenger 2 (Empirical Challenger, Critic / Specialist)  
**Date**: 2026-10-12  
**Targets**:
- `Mjlab-StairsClimb-MicroDuck`
- `Mjlab-StairsClimb-Backlash-MicroDuck`

---

## 1. Executive Summary

Empirical GPU simulation stress tests were conducted on `Mjlab-StairsClimb-MicroDuck` and its backlash twin `Mjlab-StairsClimb-Backlash-MicroDuck` on GPU Warp with MuJoCo Warp and PPO (`rsl_rl`).

Both tasks executed clean iterations without NaN events or numerical divergence (`Episode_Termination/nan_state: 0.0000`). All penalty terms across all iterations conformed strictly to the AGENTS.md sign invariant: every penalty reward term logged $\le 0.0000$. The backlash variant properly integrated passive backlash joints under the `MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG` model without dimensional misalignment. Both runs produced valid baked-normalizer ONNX exports, and the entire test suite (329 tests) passed with 100% success.

**Verdict**: `APPROVE`

---

## 2. Test Execution & Empirical Observations

### 2.1 Task Registry Verification
- **Command**: `uv run list-envs`
- **Result**: Exit code 49 (mjlab registry table display)
- **Observations**:
  ```text
  | 26 | Mjlab-StairsClimb-Backlash-MicroDuck |
  | 27 | Mjlab-StairsClimb-MicroDuck          |
  ```
  Both tasks are registered and discoverable in the live environment registry.

---

### 2.2 Base Stairs Task Smoke Test (5 iterations, 64 envs)
- **Command**:
  ```bash
  PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
  ```
- **Execution Summary**:
  - Exit code: `0`
  - Total steps: `7680`
  - Throughput: `488` steps/sec
  - Checkpoint & ONNX export: `model_4.pt`, `2026-10-12_02-29-03_stairs_climb.onnx`
- **NaN Verification**:
  ```text
  Episode_Termination/nan_state: 0.0000
  Mean value loss: 0.0402
  Mean surrogate loss: -0.0436
  Mean entropy loss: 19.8009
  ```
  Zero NaNs encountered across all 5 iterations.
- **Penalty Reward Sign Invariant Audit**:
  Every penalty term returned $\le 0.0000$, validating that positive weights on self-negating functions (`*_penalty`, `*_progress`) and negative weights on mjlab cost functions avoid double-negation:
  | Reward Term | Logged Value (Iter 4) | Sign Contract | Status |
  |-------------|-----------------------|---------------|--------|
  | `Episode_Reward/stairs_foot_clearance` | `-0.0003` | $\le 0$ | PASS |
  | `Episode_Reward/stairs_foot_landing_damping` | `-0.0056` | $\le 0$ | PASS |
  | `Episode_Reward/stairs_riser_trip` | `-0.0431` | $\le 0$ | PASS |
  | `Episode_Reward/stairs_sole_flat` | `-0.0050` | $\le 0$ | PASS |
  | `Episode_Reward/stairs_relative_height_progress`| `-0.0013` | $\le 0$ | PASS |
  | `Episode_Reward/upright_progress` | `-0.0037` | $\le 0$ | PASS |
  | `Episode_Reward/height_progress` | `-0.0004` | $\le 0$ | PASS |
  | `Episode_Reward/joint_torque_rate_l2` | `-0.0002` | $\le 0$ | PASS |
  | `Episode_Reward/servo_impact` | `-0.0000` | $\le 0$ | PASS |
  | `Episode_Reward/head_impact` | `-0.0000` | $\le 0$ | PASS |
  | `Episode_Reward/trunk_impact` | `0.0000` | $\le 0$ | PASS |
  | `Episode_Reward/servo_acc_spike` | `-0.0001` | $\le 0$ | PASS |
  | `Episode_Reward/servo_stall` | `-0.0002` | $\le 0$ | PASS |
  | `Episode_Reward/gentle_rise` | `-0.0001` | $\le 0$ | PASS |
  | `Episode_Reward/dof_pos_limits` | `-0.0003` | $\le 0$ | PASS |
  | `Episode_Reward/action_rate_l2` | `-0.0750` | $\le 0$ | PASS |
  | `Episode_Reward/foot_slip` | `-0.0001` | $\le 0$ | PASS |
  | `Episode_Reward/self_collisions` | `-0.0133` | $\le 0$ | PASS |
  | `Episode_Reward/body_ang_vel` | `-0.0237` | $\le 0$ | PASS |
  | `Episode_Reward/angular_momentum` | `-0.0000` | $\le 0$ | PASS |

---

### 2.3 Backlash Twin Smoke Test (3 iterations, 64 envs)
- **Command**:
  ```bash
  PYTHONPATH="" uv run train Mjlab-StairsClimb-Backlash-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 3
  ```
- **Execution Summary**:
  - Exit code: `0`
  - Total steps: `4608`
  - Throughput: `308` steps/sec
  - Checkpoint & ONNX export: `model_2.pt`, `2026-10-12_02-29-56_stairs_climb.onnx`
- **NaN Verification**:
  ```text
  Episode_Termination/nan_state: 0.0000
  Mean value loss: 0.0472
  Mean surrogate loss: -0.0426
  Mean entropy loss: 19.8199
  ```
  Zero NaNs encountered across all 3 iterations.
- **Backlash Passive Joint & Collision Handling**:
  The task compiled with `MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG`. Passive joints (`passive_*`) were cleanly ignored by the actor policy (action space remains 14D) and observation vector remains exactly 61D. Stepping progressed without numerical divergence or collision instability.
- **Penalty Reward Sign Invariant Audit**:
  All penalty terms logged $\le 0.0000$:
  - `Episode_Reward/stairs_foot_clearance: -0.0003`
  - `Episode_Reward/stairs_foot_landing_damping: -0.0043`
  - `Episode_Reward/stairs_riser_trip: -0.0424`
  - `Episode_Reward/stairs_sole_flat: -0.0057`
  - `Episode_Reward/stairs_relative_height_progress: -0.0014`
  - `Episode_Reward/upright_progress: -0.0036`
  - `Episode_Reward/height_progress: -0.0005`
  - `Episode_Reward/joint_torque_rate_l2: -0.0002`
  - `Episode_Reward/servo_impact: -0.0000`
  - `Episode_Reward/head_impact: -0.0000`
  - `Episode_Reward/trunk_impact: -0.0000`
  - `Episode_Reward/servo_acc_spike: -0.0001`
  - `Episode_Reward/servo_stall: -0.0002`
  - `Episode_Reward/gentle_rise: -0.0001`
  - `Episode_Reward/dof_pos_limits: -0.0005`
  - `Episode_Reward/action_rate_l2: -0.0740`
  - `Episode_Reward/foot_slip: -0.0001`
  - `Episode_Reward/self_collisions: -0.0160`
  - `Episode_Reward/body_ang_vel: -0.0228`
  - `Episode_Reward/angular_momentum: -0.0000`

---

### 2.4 Regression Test Suite Verification
- **Command**:
  ```bash
  PYTHONPATH="" uv run --with pytest pytest tests/
  ```
- **Result**:
  - Exit code: `0`
  - Output: `329 passed, 2 skipped in 19.02s`
  - No broken invariants, no regression in existing task families (velocity, velstand, standup, sitstand, groundpick, rollers, etc.).

---

## 3. Adversarial Assessment

### Challenge 1: Double-Negation Reward Farming
- **Hypothesis**: Self-negating functions (returning $\le 0$) might have negative weights configured, double-negating into positive payouts that encourage robot to crash or butt-hop.
- **Empirical Check**: Examined all `Episode_Reward/*` outputs in wandb logs during training.
- **Finding**: Every single penalty term logs $\le 0.0000$. Weight signs are properly aligned with AGENTS.md conventions.

### Challenge 2: Passive Joint Leaks in Backlash Model
- **Hypothesis**: The backlash twin adds passive joints for gear play; if regex selectors fail to filter `passive_*`, action dims or observation dims would expand, causing runtime crash or ONNX export failure.
- **Empirical Check**: Stepped 3 iterations on GPU, verified that actions remained 14D, observations remained 61D, and ONNX export succeeded.
- **Finding**: No joint leakage. `_servo_joint_ids` and `_servo_joint_pos` helpers isolate actuated servos.

### Challenge 3: Contact Instability on Stairs
- **Hypothesis**: 70 collision geoms from `robot_allcollisions.xml` on pyramid stair risers could cause contact chatter, high solver iterations, or NaN explosions.
- **Empirical Check**: Checked solver settings (`nconmax=200`, `iterations=30`, `ls_iterations=50`, `_soften_terrain_contacts`) and monitored `Episode_Termination/nan_state`.
- **Finding**: 0 NaNs across all 12,288 total environment steps collected across both runs.
