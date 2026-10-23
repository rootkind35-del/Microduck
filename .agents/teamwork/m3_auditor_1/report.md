# Milestone M3 Forensic Audit Report

**Work Product**: `scripts/infer_stairs.py`, `tests/test_infer_stairs.py`  
**Profile**: General Project  
**Integrity Mode**: Demo Mode (from `ORIGINAL_REQUEST.md`)  
**Verdict**: **CLEAN**  

---

## Executive Summary

Milestone M3 implements the dedicated standalone rehearsal runner `scripts/infer_stairs.py` and its comprehensive regression test suite `tests/test_infer_stairs.py`. The forensic audit performed independent source inspection and empirical execution checks to detect any shortcuts, facade implementations, hardcoded test results, or bypasses.

Every check passed with 100% genuine implementation. The verdict is **CLEAN**.

---

## Phase 1: Source Code Inspection

### 1.1 Genuine Logic vs. Hardcoded Test Mocks / Facades
- **Target**: `scripts/infer_stairs.py`
- **Findings**:
  - No mock tables, no hardcoded returns, no dummy constant stubs, and no environment sniffing (e.g., checking `sys.argv`, `PYTEST_CURRENT_TEST`, or bypassing execution during tests).
  - The script executes genuine 50 Hz control decimation (4 physics substeps at 200 Hz = 50 Hz control loop).
  - Terminal interactive input uses non-blocking `cbreak` terminal handling via `termios` and background thread reading with GLFW fallback in GUI viewer mode.
  - Headless CI execution cleanly terminates at `--duration-s` or defaults to 2.0s in `--headless --debug` mode with zero hanging threads.

### 1.2 BAM M6 Motor Conversion & Stiff Friction Solver
- **Target**: `scripts/infer_stairs.py:load_mujoco_with_bam`, `load_bam_model`
- **Findings**:
  - Genuinely imports and initializes `bam.model.load_model(motor_name="xl330", model="m6")` with `kp_fw=200.0`, `vin=7.4` (matching `_BAM_ACTUATOR_KWARGS` in `src/mjlab_microduck/robot/microduck_constants.py`).
  - Iterates over XML actuators, explicitly filtering unactuated joints (`passive_*`).
  - Converts every non-passive actuator to torque motor mode:
    - `act.set_to_motor()` (`gaintype=mjGAIN_FIXED`, `biastype=mjBIAS_NONE`, `gainprm=[1.0, ...]`).
    - `act.forcelimited = True` with `forcerange = (-force_limit, force_limit)` where `force_limit = vin * kt / R`.
    - Damping is zeroed and joint frictionloss is set to 0.0 because BAM computes internal friction.
    - Stiff friction solver parameters are set directly on joints:
      - `joint.solref_friction = (-5.0e4, -2.0e2)` (matching `BamActuator._STIFF_SOLREF_FRICTION`).
      - `joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)` (matching `BamActuator._STIFF_SOLIMP_FRICTION`).
  - Instantiates `bam.mujoco.MujocoController` wrapping the compiled model and data.
  - In each simulation step, `bam_ctrl.q_target[:] = target_pos` and `bam_ctrl.update()` are invoked across all 4 decimation substeps.

### 1.3 61D Observation Layout Computation
- **Target**: `scripts/infer_stairs.py:build_61d_observation`
- **Findings**:
  - Genuinely computes each of the 6 components:
    1. Base angular velocity (3D): reads directly from `data.sensordata` via sensor `imu_ang_vel` (with fallback to `data.qvel[3:6]`).
    2. Projected gravity (3D): reads orientation quaternion of `trunk_base` body (`data.xquat`), normalizes it, and computes $R^T [0, 0, -1]^T$ via `compute_projected_gravity`.
    3. Joint positions relative to default (14D): computed as `data.qpos[7:21] - DEFAULT_JOINT_POS`.
    4. Joint velocities (14D): computed as `data.qvel[6:20]`.
    5. Last actions (14D): stores previous control actions.
    6. Command block (13D): `[cmd_twist(3), cmd_head(4), cmd_body(6)]`.
  - Command slots are live and wired to interactive keyboard shortcuts (`H` toggles head gaze, `B` + arrows trims body pitch, `U` sets standup recovery elevation). Unused slots are zero-padded as mandated by project invariants.
  - Final assertion strictly verifies `obs.shape == (61,)` and absence of NaNs.

### 1.4 Balance Telemetry & Contact Force Resolution
- **Target**: `scripts/infer_stairs.py:identify_step_contacts_and_forces`, `compute_balance_telemetry`
- **Findings**:
  - Resolves foot collision geoms `left_foot_collision` and `right_foot_collision`.
  - Iterates over all active contacts (`data.ncon`) and genuinely calls the C-API function `mujoco.mj_contactForce(model, data, i, c_force)`.
  - Extracts the normal force `abs(c_force[0])` and maps the opposing geom ID via `mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id)` directly to geoms defined in `scene_stairs.xml` (`stair_1_1..3`, `landing_1`, `stair_2_1..3`, `landing_2`, `stair_3_1..3`, `landing_summit`, `floor`, or `"air"` when airborne).
  - Computes knee and ankle servo torque loads relative to the 0.96 Nm Dynamixel XL330 stall torque limit (`BAM_STALL_TORQUE`).
  - Dynamically classifies robot balance status into `BALANCED`, `UNSTABLE`, or `FALLEN` based on Euler roll/pitch tilt angles.

### 1.5 Test Suite Genuineness
- **Target**: `tests/test_infer_stairs.py`
- **Findings**:
  - 10 distinct, rigorous test cases covering invariants, BAM constants, actuator conversion, 100-step simulation loops, 5 spawn presets, balance telemetry, 61D observation extraction, and headless execution.
  - No `assert True` tautologies, no mocked passes, and no test skips (`0` skipped).
  - All tests exercise actual physics compilation, stepping, and numerical verification.

---

## Phase 2: Independent Execution Verification

### Check 2.1: CLI Help Command
```bash
PYTHONPATH="" uv run scripts/infer_stairs.py --help
```
- **Exit Code**: `0`
- **Result**: Output confirmed valid CLI interface with `--policy`, `--scene`, `--spawn`, `--lin-vel-x`, `--vin`, `--headless`, `--debug`, `--telemetry-hz`, `--duration-s`.

### Check 2.2: Headless Rehearsal with Debug Telemetry
```bash
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug
```
- **Exit Code**: `0`
- **Result**: Simulated 100 steps (2.0s) at 50 Hz control rate with 0 NaNs and clean exit.
- **Sample Telemetry Output**:
  ```
  [ 0.20s] pos:[0.00, -0.00, 0.12] | tilt: P: +2.7° R: +0.0° [BALANCED] | feet: L:floor( 3.6N) R:floor( 3.6N) | knee:  5% ank:  0% | Floor (Apron)
  [ 0.40s] pos:[0.01, -0.00, 0.11] | tilt: P: +6.2° R: +0.0° [BALANCED] | feet: L:floor( 3.7N) R:floor( 3.5N) | knee:  6% ank:  1% | Floor (Apron)
  ...
  [infer_stairs] Target duration 2.0s reached (100 steps). Exiting cleanly.
  [infer_stairs] Finished 100 steps with 0 NaNs and clean exit.
  ```

### Check 2.3: Targeted Unit & E2E Pytest Suite
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py
```
- **Exit Code**: `0`
- **Result**: 33 passed, 1 skipped (task registry check pending M4 environment activation). 100% of test_infer_stairs.py passed (10/10).

### Check 2.4: Full Repository Regression Suite
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
- **Exit Code**: `0`
- **Result**: 334 passed, 2 skipped, 0 failed in 17.72s. Zero regressions introduced.

---

## Phase 3: Stress-Testing & Adversarial Verification

We conducted dynamic parameter and spawn preset stress tests across all 5 keyframe stair presets:
- `bottom` $\rightarrow$ Positioned on Floor Apron ($z \approx 0.12m$, zone: `Floor (Apron)`)
- `flight1` $\rightarrow$ Positioned on Flight 1 Step 2 ($z \approx 0.17m$, zone: `Flight 1 (Step 1_2, 2.5cm)`)
- `flight2` $\rightarrow$ Positioned on Flight 2 Step 2 ($z \approx 0.26m$, zone: `Flight 2 (Step 2_2, 3.0cm)`)
- `flight3` $\rightarrow$ Positioned on Flight 3 Step 2 ($z \approx 0.36m$, zone: `Flight 3 (Step 3_2, 3.5cm)`)
- `top` $\rightarrow$ Positioned on Summit Platform ($z \approx 0.39m$, zone: `Summit (27.0cm elev)`)

All presets stepped cleanly without NaN occurrences and correctly reported elevation and contact profiles.

---

## Conclusion
The Milestone M3 work product adheres strictly to all requirements in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md`. No integrity violations, shortcuts, or facades were identified.

**Final Verdict**: **CLEAN**
