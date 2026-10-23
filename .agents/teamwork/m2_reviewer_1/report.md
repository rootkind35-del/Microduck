# Milestone M2 Code Review & Adversarial Challenge Report

**Date**: 2026-10-11T19:32:00Z  
**Reviewer**: M2 Reviewer 1 (Archetype: `reviewer_critic`)  
**Target Milestone**: M2 — High Stairs RL Environment Config, Curriculum & Task Registration  
**Verdict**: **APPROVE**

---

## 1. Executive Summary

Milestone M2 delivers the core RL environment configuration and task registration for Microduck climbing high stairs (2.5 cm -> 3.0 cm -> 3.5 cm risers) with dynamic balance stabilization and automatic ground self-recovery.

The implementation strictly satisfies all architectural constraints in `PROJECT.md`, `ORIGINAL_REQUEST.md`, and `AGENTS.md`:
1. **Unified 61D Observation Layout**: Exactly 48D base proprioception + 13D command block `[twist(3), head_pose(4), body_pose(6)]` in identical order to the base velocity recipe.
2. **Reward Sign Convention Invariant**: All self-negating penalties ($\le 0$) have positive weights; all cost functions ($\ge 0$) have negative weights. Wandb/console logs from live training confirm that every `Episode_Reward/<penalty>` evaluates strictly $\le 0.0$.
3. **Full-Body Contact Physics**: Uses `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms) with contact softening (`_soften_terrain_contacts`) and hardened solver settings (`nconmax >= 300`, `iterations = 30`, `ls_iterations = 50`).
4. **Task Registration & Backlash Twin**: Registered as `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` in `src/mjlab_microduck/tasks/__init__.py`, with the backlash twin faithfully mapped to `MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG`.
5. **Empirical Verification**:
   - `uv run list-envs | grep StairsClimb`: Both environments registered and discoverable.
   - `uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`: 38 passed, 1 skipped.
   - `uv run --with pytest pytest tests/`: 329 passed, 2 skipped (100% pass rate across the full repo suite).
   - Live 5-iteration GPU smoke test at 64 envs: 0 NaNs (`Episode_Termination/nan_state: 0.0`), valid loss convergence, and non-positive penalty metrics.

---

## 2. Integrity Audit

In accordance with system integrity guidelines, the codebase was inspected for shortcuts, mock implementations, and verification cheating:
- **No Hardcoded Test Bypasses**: All tests execute real physics computations against compiled MuJoCo models and PyTorch tensors.
- **No Facade Implementations**: `microduck_stairs_env_cfg.py` configures authentic `mjlab` simulation objects, real contact sensors (`servo_ground_contact`, `head_ground_contact`, `trunk_ground_contact`), genuine BoxPyramidStairs terrain generator, and authentic MDP reward functions from `mdp.py`.
- **Zero Fabrication**: All test and training execution outputs documented in this review were generated live through shell tools during this session.
- **Integrity Status**: **CLEAN — NO INTEGRITY VIOLATIONS DETECTED**.

---

## 3. Detailed Review Dimensions

### 3.1 Observation Contract (61D Invariant)
The actor observation shape must strictly preserve the 61D layout so that policies are hot-swappable across locomotion, recovery, and manipulation tasks without altering runtime buffers.

In `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
- `make_microduck_stairs_env_cfg` derives from `make_microduck_velocity_env_cfg(play=play, rough=rough)`.
- Base Proprioception (48D):
  - `base_ang_vel` (3D)
  - `projected_gravity` (3D)
  - `joint_pos` (14D)
  - `joint_vel` (14D)
  - `actions` (14D)
  - Subtotal: 48D
- Command Block (13D):
  - `command` (twist: 3D)
  - `head_command` (head_pose: 4D)
  - `body_command` (body_pose: 6D)
  - Subtotal: 13D
- Total Actor Dimension: $48 + 13 = 61\text{D}$.
- Term ordering: Verified by `test_obs_term_order_matches_velocity_contract` to match the frozen `microduck_velocity_env_cfg.py` contract index-for-index.

### 3.2 Reward Sign Invariant & Reward Mass
A notorious RL failure mode in bipedal locomotion is double-negating penalties (assigning negative weights to functions returning $\le 0$), causing the agent to farm violations (butt-hopping, edge-crashing).

Audit of all reward terms in `microduck_stairs_env_cfg.py`:
| Term | Function | Function Return | Config Weight | Sign Audit |
|---|---|---|---|---|
| `stairs_foot_clearance` | `stairs_foot_clearance_penalty` | $\le 0$ | $+2.0$ | PASS |
| `stairs_foot_landing_damping` | `stairs_foot_landing_damping_penalty` | $\le 0$ | $+0.25$ | PASS |
| `stairs_riser_trip` | `stairs_riser_trip_penalty` | $\le 0$ | $+0.5$ | PASS |
| `stairs_sole_flat` | `stairs_sole_flat_penalty` | $\le 0$ | $+1.0$ | PASS |
| `stairs_relative_height_progress` | `stairs_relative_height_progress` | $\Delta \Phi \ge 0$ | $+20.0$ | PASS |
| `upright_progress` | `upright_progress` | $\Delta \cos(\theta)$ | $+5.0$ | PASS |
| `height_progress` | `height_progress` | $\Delta z$ | $+10.0$ | PASS |
| `com_upward_velocity` | `com_upward_velocity` | $\ge 0$ | $0.0 \rightarrow +2.0$ | PASS |
| `gentle_rise` | `trunk_vertical_accel_penalty` | $-|a_z| \le 0$ | $+0.005 \times 0.2$ | PASS |
| `recovery_success` | `recovery_success` | $\ge 0$ (bounty) | $0.0 \rightarrow +10.0$ | PASS |
| `fallen_tax` | `fallen_state_penalty` | $\ge 0$ (1.0 while fallen) | $0.0 \rightarrow -0.5$ | PASS |
| `action_rate_l2` | `action_rate_l2_fallen_scaled` | $\ge 0$ | $-0.1 \rightarrow -1.0$ | PASS |
| `joint_torque_rate_l2` | `joint_torque_rate_l2_fallen_scaled` | $\ge 0$ | $-2\times 10^{-3}$ | PASS |
| `servo_impact` | `body_impact_cost` | $\ge 0$ | $-0.02 \times 0.2$ | PASS |
| `head_impact` | `body_impact_cost` | $\ge 0$ | $-0.01 \times 0.2$ | PASS |
| `trunk_impact` | `body_impact_cost` | $\ge 0$ | $-0.01 \times 0.2$ | PASS |
| `servo_acc_spike` | `servo_acc_spike_penalty` | $\ge 0$ | $-0.001 \times 0.2$ | PASS |
| `servo_stall` | `servo_stall_penalty` | $\ge 0$ | $-0.05 \times 0.2$ | PASS |

Live training output from iteration 4:
- `Episode_Reward/stairs_foot_clearance`: `-0.0003` $\le 0$
- `Episode_Reward/stairs_foot_landing_damping`: `-0.0050` $\le 0$
- `Episode_Reward/stairs_riser_trip`: `-0.0465` $\le 0$
- `Episode_Reward/stairs_sole_flat`: `-0.0066` $\le 0$
- `Episode_Reward/action_rate_l2`: `-0.0814` $\le 0$
- `Episode_Reward/servo_stall`: `-0.0002` $\le 0$
- `Episode_Reward/gentle_rise`: `-0.0001` $\le 0$
All penalties are strictly non-positive.

### 3.3 Progressive Curriculum & Recovery Architecture
1. **Clearance Scheduling**:
   - `CLEARANCE_TARGET_STAGES`: `[0.028m (0 iters), 0.033m (1500 iters), 0.038m (3000 iters)]`.
   - Incorporates a calibrated 3 mm safety clearance above the respective stair risers (2.5 cm, 3.0 cm, 3.5 cm), preventing toe drag on sharp riser corners.
2. **Terrain Level Progression**:
   - `terrain_levels_stairs`: Evaluates distance from origin; promotes envs reaching the platform ($> 2.5\text{ m}$), demotes envs covering $< 0.8\text{ m}$.
3. **Attempt Tax Freeze Defense**:
   - Uses `FALLEN_SMOOTHNESS_SCALE = 0.1` on `action_rate_l2` and `joint_torque_rate_l2` when body tilt exceeds 40°. This prevents the policy from parking on the ground to save action-rate penalty costs.
   - Delays `fallen_tax` (-0.5) and `recovery_success` (10.0) until iteration 1,500 (`RECOVERY_ECON_KICKIN_ITER`) to allow the base stair walking gait to bootstrap cleanly before demanding multi-state self-recovery.
4. **Pacing Alignment**:
   - `fell_over` termination is progressively relaxed at iteration 500 (`FELL_OVER_DISABLE_ITER`), allowing the policy to experience fallen orientations once the forward stepping reflex is partially grounded.

### 3.4 Task Registration & Backlash Twin Parity
- Task registry entries in `src/mjlab_microduck/tasks/__init__.py`:
  - `Mjlab-StairsClimb-MicroDuck` registered with `make_microduck_stairs_env_cfg()` and `MicroduckStairsRlCfg`.
  - `Mjlab-StairsClimb-Backlash-MicroDuck` registered via `_BACKLASH_TASKS` with `MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG`.
- Both entries appear in `uv run list-envs`.
- Backlash twin maintains strict collision parity (70 geoms) with the base task.

---

## 4. Adversarial Challenge & Stress-Testing

### 4.1 Stress Scenarios & Edge Cases
1. **Extreme Impact & Collision Forces**:
   - Adversarial tests in `tests/test_stairs_adversarial.py` subjected the MDP terms to $100,000\text{ N}$ crush forces, $-1,000\text{ N}$ negative forces (sensor anomalies), and supersonic $10,000\text{ m/s}$ velocities.
   - Result: All terms remained bounded, strictly negative, and completely free of NaNs/Infs.
2. **Elevation Invariance**:
   - Potential reward `stairs_relative_height_progress` was tested over multi-flight ascents ($+0.03\text{m}, +0.06\text{m}, +0.09\text{m}$).
   - While ascending stairs in steady-state posture, `stairs_relative_height_progress` correctly returned $0.0\text{ reward}$, confirming elevation invariance (no infinite elevation-farming on infinite stairs).
3. **Inverted Gravitational & Orientational Perturbations**:
   - Evaluated under zero gravity, inverted gravity ($+9.81\text{ m/s}^2$), and 90-degree pitched postures. `stairs_sole_flat_penalty` properly penalizes toe/heel contact without numerical instability.

### 4.2 Minor Observations (Non-Blocking)
- **Play Mode Terrain Generation**: In `make_microduck_stairs_env_cfg(play=True)`, `curriculum = False` is set, creating a 5x5 fixed terrain grid. The user should note that interactive play uses the baseline difficulty distribution rather than dynamically tracking progression.
- **Batch Vectorization**: Contact reduction on `servo_ground_contact` uses `reduce="netforce"` across 14 geoms with `num_slots=1`. This computes the global body contact penalty efficiently in parallel.

---

## 5. Verified Claims Matrix

| Claim | Verification Method | Result | Status |
|---|---|---|---|
| `Mjlab-StairsClimb-MicroDuck` registered | `PYTHONPATH="" uv run list-envs \| grep StairsClimb` | Task present in registry output | PASS |
| `Mjlab-StairsClimb-Backlash-MicroDuck` registered | `PYTHONPATH="" uv run list-envs \| grep StairsClimb` | Task present in registry output | PASS |
| Unified 61D observation shape | `test_actor_observation_keeps_the_61d_slot_layout` in `tests/test_stairs_cfg.py` | 48D proprio + 13D commands confirmed | PASS |
| All collision geoms $\ge 70$ | `test_uses_allcollisions_robot_model` in `tests/test_stairs_cfg.py` | 70 geoms compiled with 14 servos | PASS |
| Solver hardening active | `test_solver_parameters_and_contact_softening` | `nconmax=300`, `iters=30`, `ls_iters=50`, softened contacts | PASS |
| Reward weight signs correct | `test_reward_weight_signs_invariant` & live smoke test wandb output | Self-negating $> 0$, costs $< 0$, Episode_Rewards $\le 0$ | PASS |
| 5-iter GPU smoke test NaN-free | `WANDB_MODE=disabled uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` | Exited 0, nan_state = 0.0, 5/5 iters complete | PASS |
| Staged clearance curriculum | `test_stair_curriculum_stages` in `tests/test_stairs_cfg.py` | 0.028m -> 0.033m -> 0.038m scheduled | PASS |
| Full repository test suite | `PYTHONPATH="" uv run --with pytest pytest tests/` | 329 passed, 2 skipped, 0 failed | PASS |

---

## 6. Review Verdict

**Verdict**: **APPROVE**

Milestone M2 is fully complete, logically sound, thoroughly tested, and ready for Milestone M3 (Standalone Rehearsal Runner & Interactive Telemetry).
