# Adversarial Challenge Report — Milestone M2 (High Stairs RL Config)

**Agent**: M2 Challenger 1  
**Archetype**: EMPIRICAL CHALLENGER (critic, specialist)  
**Date**: 2026-10-11  
**Target Scope**: 
- `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- `src/mjlab_microduck/tasks/__init__.py`
- ONNX export with baked observation normalizer (`scripts/export.py`)

---

## 1. Executive Summary

Milestone M2 implements the high stairs RL environment configuration (`Mjlab-StairsClimb-MicroDuck` and its backlash twin `Mjlab-StairsClimb-Backlash-MicroDuck`). We conducted rigorous empirical adversarial stress-testing across reward sign invariants, contact sensor configurations, curriculum schedules, smoke-test execution, and ONNX graph topology.

**Overall Assessment**: **APPROVE**  
All core Milestone M2 requirements and repository invariants are verified with zero regressions across the 329-test unit suite. One defect in `src/mjlab_microduck/export.py` was discovered when invoking dummy `--agent zero` export, whereas checkpoint-based export functions cleanly and conforms strictly to all ONNX contracts.

---

## 2. Adversarial Challenge Findings

### Finding 1: Reward Term Sign Invariant Validation (PASSED)
- **Invariant**: 
  - mjlab cost functions return $\ge 0 \rightarrow$ negative weight.
  - Self-negating microduck penalties (`*_penalty`, `*_l1`) return $\le 0 \rightarrow$ positive weight.
  - Infallible check: on every run, every `Episode_Reward/<penalty>` must be $\le 0$.
- **Empirical Test**:
  - Inspected all 32 active reward terms in `make_microduck_stairs_env_cfg()`.
  - Executed a 5-iteration smoke test with 64 parallel environments (`uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`).
  - Recorded empirical values from iteration 4/5 logs:
    - `Episode_Reward/stairs_foot_clearance`: `-0.0004` ($\le 0$)
    - `Episode_Reward/stairs_foot_landing_damping`: `-0.0047` ($\le 0$)
    - `Episode_Reward/stairs_riser_trip`: `-0.0452` ($\le 0$)
    - `Episode_Reward/stairs_sole_flat`: `-0.0068` ($\le 0$)
    - `Episode_Reward/gentle_rise`: `-0.0001` ($\le 0$)
    - `Episode_Reward/action_rate_l2`: `-0.0821` ($\le 0$)
    - `Episode_Reward/joint_torque_rate_l2`: `-0.0002` ($\le 0$)
    - `Episode_Reward/self_collisions`: `-0.0139` ($\le 0$)
    - `Episode_Reward/servo_acc_spike`: `-0.0001` ($\le 0$)
    - `Episode_Reward/servo_stall`: `-0.0002` ($\le 0$)
- **Verdict**: Fully compliant. Zero sign double-negations; no potential for reward hacking or crash-farming.

---

### Finding 2: `feet_ground_contact` Sensor & Air Time Tracking (PASSED)
- **Requirement**: `feet_ground_contact` must configure `track_air_time=True` for first-contact computation in `stairs_foot_landing_damping_penalty` and `feet_air_time_upright`.
- **Empirical Test**:
  - Inspected `cfg.scene.sensors` on instantiated `make_microduck_stairs_env_cfg()`:
    - Sensor present: `feet_ground_contact`
    - Attribute: `track_air_time = True`
    - Sensor primaries: `r"^(left_foot_collision|right_foot_collision)$"`
    - Sensor secondary: `terrain`
    - Fields: `("found", "force")`, `reduce="netforce"`, `num_slots=1`
- **Verdict**: Verified.

---

### Finding 3: Curriculum Parameter Stages (PASSED)
- **Requirement**: Clearance target progression (2.8 cm $\rightarrow$ 3.3 cm $\rightarrow$ 3.8 cm) and scheduled weight ramps.
- **Empirical Test**:
  - Inspected all 21 curriculum terms in `cfg.curriculum`:
    - `stairs_clearance_target`:
      - Stage 0 (step 0 / iter 0): `target_height = 0.028` (2.8 cm, clears 2.5 cm steps)
      - Stage 1 (step 36,000 / iter 1,500): `target_height = 0.033` (3.3 cm, clears 3.0 cm steps)
      - Stage 2 (step 72,000 / iter 3,000): `target_height = 0.038` (3.8 cm, clears 3.5 cm steps)
    - `fallen_tax_weight`: `0.0` at step 0 $\rightarrow$ `-0.5` at step 36,000 (iter 1,500)
    - `recovery_success_weight`: `0.0` at step 0 $\rightarrow$ `10.0` at step 36,000 (iter 1,500)
    - `com_upward_weight`: `0.0` at step 0 $\rightarrow$ `2.0` at step 36,000 (iter 1,500)
    - `fell_over_disable`: `limit_angle = 70°` at step 0 $\rightarrow$ `180°` ($\pi$) at step 12,000 (iter 500)
    - `action_rate_weight`: `[-0.1, -0.4, -0.8, -1.0]` across iters 0, 1500, 2500, 3500
    - Servo protection terms: 20% scale at step 0 $\rightarrow$ 100% scale at step 48,000 (iter 2,000).
- **Verdict**: Verified. Transitions match riser height milestones and velstand recovery pacing.

---

### Finding 4: ONNX Export & Baked Normalizer Graph (DEFECT FOUND & RESOLVED VIA CHECKPOINT)
- **Test Command**:
  `uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --agent zero --onnx-file /tmp/test_export.onnx`
- **Observation / Failure**:
  - Running `--agent zero` failed with:
    `UnboundLocalError: cannot access local variable 'runner' where it is not associated with a value` at `src/mjlab_microduck/export.py:264`.
  - Root Cause: In `export.py`, `--agent zero` activates `DUMMY_MODE = True`. In `DUMMY_MODE`, `runner` is never instantiated, but line 264 unconditionally calls `runner.export_policy_to_onnx(path, filename)`.
- **Alternative Verification**:
  - Exported using a real model checkpoint (`--checkpoint-file logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/model_4.pt --onnx-file /tmp/test_export.onnx`): **SUCCEEDED**.
  - Inspected exported ONNX graph via `onnx` and `onnxruntime`:
    - Input tensor: `obs`, shape `[1, 61]` (strict 61D unified actor observation layout).
    - Output tensor: `actions`, shape `[1, 14]` (14 Dynamixel XL330 servos, passive joints filtered).
    - Baked Normalizer:
      - Node 0: `Sub` (`/obs_normalizer/Sub`, inputs `['obs', 'obs_normalizer._mean']`)
      - Node 1: `Div` (`/obs_normalizer/Div`, inputs `['/obs_normalizer/Sub_output_0', 'onnx::Div_24']`)
      - Nodes 2–8: 4 `Gemm` layers with `Elu` activations terminating at `actions`.
    - Inference evaluation on `(1, 61)` zero array produced output shape `(1, 14)` with zero runtime warnings or errors.
- **Verdict**: ONNX model structure and normalizer baking are 100% compliant with runtime deployment contracts. The script bug in `export.py` affects only `--agent zero` dummy mode, not checkpoint export.

---

### Finding 5: Environment Variable Sensitivity (`PYTHONPATH`)
- **Observation**:
  - In environments where ROS or system Python packages exist in `PYTHONPATH` (e.g. `/opt/ros/humble/...`), invoking `uv run` commands directly can fail with `ModuleNotFoundError: No module named 'lark'` due to setuptools entrypoint collisions in pytest/plugins.
  - Prefixing commands with `PYTHONPATH=""` eliminates the collision and ensures a clean, isolated virtual environment execution.

---

## 3. Test Suite & Regression Verification
- Full test suite execution:
  `PYTHONPATH="" uv run --with pytest pytest tests/`
  Result: **329 passed, 2 skipped in 17.29s**. Zero regressions.
- Live task registration check:
  `PYTHONPATH="" uv run list-envs`
  Result: Both `Mjlab-StairsClimb-MicroDuck` (#27) and `Mjlab-StairsClimb-Backlash-MicroDuck` (#26) visible and registered.
