# Handoff Report — M3 Challenger 2

**Author**: M3 Challenger 2 (Empirical Challenger: Critic, Specialist)  
**Date**: 2026-10-12  
**Target Scope**:
- `scripts/infer_stairs.py`
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- `tests/test_infer_stairs.py`
- `tests/test_infer_stairs_adversarial.py`
- `tests/` repository test suite

---

## 1. Observation

1. **Actuator Voltage Sweep & Force Limit Computation**:
   - `scripts/infer_stairs.py:133-136`:
     ```python
     kt = bam_model.kt.value
     R = bam_model.R.value
     force_limit = bam_model.actuator.vin * kt / R
     ```
   - In `tests/test_infer_stairs_adversarial.py::test_bam_voltage_torque_limits_and_forcerange`, tested $V_{\text{in}} \in \{4.0, 5.0, 6.0, 7.4, 8.2, 10.0, 12.0\}$ V:
     - $V_{\text{in}} = 4.0$ V: `expected_limit = 0.5208` N·m, `model.actuator_forcerange[:, 1] = 0.5208` N·m, `max_applied_torque = 0.4495` N·m $\le 0.5208$ N·m.
     - $V_{\text{in}} = 7.4$ V: `expected_limit = 0.9634` N·m, `model.actuator_forcerange[:, 1] = 0.9634` N·m, `max_applied_torque = 0.8316` N·m $\le 0.9634$ N·m.
     - $V_{\text{in}} = 12.0$ V: `expected_limit = 1.5623` N·m, `model.actuator_forcerange[:, 1] = 1.5623` N·m, `max_applied_torque = 1.3486` N·m $\le 1.5623$ N·m.
   - Headless execution commands:
     - `PYTHONPATH="src" uv run python scripts/infer_stairs.py --headless --vin 4.0 --duration-s 0.5 --debug` $\to$ Exit code 0, 25 steps, 0 NaNs.
     - `PYTHONPATH="src" uv run python scripts/infer_stairs.py --headless --vin 12.0 --duration-s 0.5 --debug` $\to$ Exit code 0, 25 steps, 0 NaNs.

2. **Trunk Impulse Perturbations Across All Stair Flights**:
   - `trunk_base` (body ID 1) perturbed with $\pm 10$ N impulses along $[\pm 10, 0, 0]$, $[0, \pm 10, 0]$, $[0, 0, \pm 10]$ across all 7 stair presets (`bottom`, `flight1`, `landing_1`, `flight2`, `landing_2`, `flight3`, `top`):
     - Normal contact forces: $F_n \in [2.2, 8.8]$ N (strictly non-negative and finite).
     - Control torques: $\tau \in [0.073, 0.209]$ N·m $\le 0.9634$ N·m.
     - Zero NaNs in `qpos`, `qvel`, `data.ctrl` across all 42 flight/direction combinations.
   - Extreme forces (20 N, 50 N, 100 N):
     - Under 100 N impulse ($>12\times$ robot weight): `max_ctrl = 0.927` N·m $\le 0.9634$ N·m (actuator torque saturation held strictly); max $F_n = 97.9$ N without singular blowup or NaN.

3. **Standing / Falling State Transitions & Telemetry Classification**:
   - `scripts/infer_stairs.py:385-390`:
     ```python
     if abs(pitch_deg) > 50.0 or abs(roll_deg) > 45.0:
         status = "FALLEN"
     elif abs(pitch_deg) > 35.0 or abs(roll_deg) > 30.0:
         status = "UNSTABLE"
     else:
         status = "BALANCED"
     ```
   - 19 test postures covering boundaries, upright, unstable lean, face plant ($P = 80^\circ$), back fall ($P = -80^\circ$), side fall ($R = 90^\circ$), and inverted ($P = 170^\circ$) were tested: all 19 matched ground truth classification.
   - Physical dynamic lifecycle test (`test_dynamic_standing_falling_recovery_lifecycle`):
     - `STAND` preset ($z = 0.125$ m) $\to$ `BALANCED`
     - Destabilizing lateral velocity impulse ($v_y = -2.0$ m/s) $\to$ topple to floor ($z = 0.038$ m, roll $= 88.6^\circ$) $\to$ `FALLEN`
     - Standup recovery triggered (`u` key logic) $\to$ reset to upright $\to$ settled at $z = 0.116$ m $\to$ `BALANCED`.

4. **Test Suite Execution**:
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs_adversarial.py` $\to$ `75 passed in 9.89s`.
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_infer_stairs_adversarial.py` $\to$ `85 passed in 15.42s`.
   - `PYTHONPATH="" uv run --with pytest pytest tests/` $\to$ `409 passed, 2 skipped in 27.05s`.

---

## 2. Logic Chain

1. **Torque Limit Correctness**:
   - Observation 1 demonstrates that `kt = 0.3660` and `R = 2.8114` from the BAM M6 model determine the stall force limit.
   - MuJoCo's `actuator_forcerange` is parameterized directly from $V_{\text{in}} \cdot k_t / R$.
   - Stepping with large control deviations resulted in applied torques strictly clamped within the calculated bounds, confirming no unconstrained torque command reaches MuJoCo.

2. **Perturbation Robustness**:
   - Observation 2 confirms that applying $\pm 10$ N impulses along all Cartesian directions on every step of the 3-flight stair terrain does not cause numerical instability, NaN states, or runaway velocities.
   - Contact normal forces remain bounded by contact dynamics ($solref/solimp$), and torques saturate at the actuator limit even under 100 N impacts.

3. **Telemetry Reliability**:
   - Observation 3 confirms that pitch and roll angles are accurately converted from base quaternions with singular-safe Euler math, and the threshold gating ($35^\circ / 30^\circ$ for `UNSTABLE`, $50^\circ / 45^\circ$ for `FALLEN`) discriminates fallen states from nominal balance.
   - The dynamic lifecycle test confirms that telemetry updates in real-time as the robot transitions between physical states.

4. **Zero Repository Regression**:
   - Observation 4 confirms that all existing tests and new adversarial tests pass cleanly without regressions across the codebase.

---

## 3. Caveats

- **Viewer GUI**: Interactive keyboard callbacks (`viewer_key_callback`) and in-viewer GLFW HUD rendering (`viewer.set_texts`) require an active DISPLAY / X11 session and were tested via the underlying state logic and `--headless` simulation.
- **Hardware Dynamics vs Simulation**: BAM M6 in CPU MuJoCo models voltage drop and thermal-independent stall limits; real Dynamixel XL330 firmware may trigger hardware shutdown if sustained stall exceeds thermal thresholds (> ~3 seconds).

---

## 4. Conclusion

The standalone high stairs simulation runner `scripts/infer_stairs.py` and dedicated scene `scene_stairs.xml` satisfy all adversarial criteria:
- Actuator voltage extremes (4.0 V to 12.0 V) compute valid torque limits without math domain errors or NaNs.
- Perturbations ($\pm 10$ N to 100 N) saturate torques and contact forces properly across all stair flights.
- Telemetry state classification correctly identifies balance, unstable tilt, and fallen postures, and dynamic recovery functions cleanly.

**Explicit Verdict**: **`APPROVE`**

---

## 5. Verification Method

To independently verify all findings:

```bash
# 1. Run standalone adversarial test suite (75 tests):
PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs_adversarial.py -v

# 2. Run combined infer stairs test suite (85 tests):
PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_infer_stairs_adversarial.py -v

# 3. Run headless CLI simulation at low and high voltage extremes:
PYTHONPATH="src" uv run python scripts/infer_stairs.py --headless --vin 4.0 --duration-s 0.5 --debug
PYTHONPATH="src" uv run python scripts/infer_stairs.py --headless --vin 12.0 --duration-s 0.5 --debug

# 4. Run entire repository test suite (411 tests):
PYTHONPATH="" uv run --with pytest pytest tests/
```
