# Adversarial Stress Testing Report — Milestone M3 Standalone Rehearsal Runner

**Author**: M3 Challenger 1 (`critic`, `specialist`)  
**Target Module**: `scripts/infer_stairs.py`  
**Test Suite**: `tests/test_infer_stairs_cli_adversarial.py` (15 tests) + `tests/test_infer_stairs.py` (10 tests) + `tests/test_infer_stairs_adversarial.py` (75 tests)  
**Date**: 2026-10-11  

---

## 1. Executive Summary

Milestone M3 delivers `scripts/infer_stairs.py`, the standalone CPU rehearsal simulator for Microduck high stairs climbing under BAM M6 voltage-controlled actuator dynamics at 50 Hz control / 200 Hz physics decimation.

M3 Challenger 1 subjected `scripts/infer_stairs.py` to empirical adversarial stress testing across 5 critical dimensions:
1. **CLI Boundary & Error Trapping**: Non-existent scene path, non-existent policy, invalid preset names, negative/zero telemetry frequency, extreme actuator voltage limits.
2. **Multi-Preset Headless Rollouts (100+ Steps)**: Long-horizon execution across all 5 primary presets (`bottom`, `flight1`, `flight2`, `flight3`, `top`) and all 13 preset aliases without crashing or emitting NaNs.
3. **Extreme Velocity Commands**: Command inputs at $\pm 2.0\text{ m/s}$ (far exceeding physical capacity) tested over 100+ steps.
4. **Keyboard Command State Transitions**: Full programmatic and end-to-end state machine verification for all keys (`up`, `down`, `left`, `right`, `a`, `e`, `space`, `b`, `h`, `c`, `p`, `u`, `t`, `q`, `1`..`5`).
5. **Full Test Suite Integrity**: Zero regressions across the full project test suite (424 tests passing).

**Empirical Result**: 100% of tests passed. All failure conditions are trapped with clean exceptions, fallback mechanisms operate robustly, and zero NaNs or Infs were observed across any rollout.

---

## 2. Detailed Findings & Challenge Dimensions

### Challenge 1: CLI Boundary & Edge Parameter Validation

| Parameter Under Test | Tested Value | Expected Behavior | Observed Behavior | Status |
|---|---|---|---|---|
| `--scene` | `nonexistent_scene_xyz.xml` | Trapped, `FileNotFoundError`, non-zero exit | `FileNotFoundError: Scene XML not found: ...`, return code 1 | **PASS** |
| `--policy` | `nonexistent_policy.onnx` | Trapped, `FileNotFoundError`, non-zero exit | `FileNotFoundError: Policy file not found: ...`, return code 1 | **PASS** |
| `--spawn` | `totally_invalid_preset_name` | Fallback to `"bottom"` (`STAND` keyframe), 0 NaNs | Falls back to `bottom`, logs `Spawn preset: bottom`, clean exit code 0 | **PASS** |
| `--telemetry-hz` | `-50.0`, `-1.0`, `0.0` | Clamped to $\ge 0.5\text{ Hz}$, no `ZeroDivisionError` | Clamped by `max(0.5, args.telemetry_hz)`, clean 2.0s period, exit code 0 | **PASS** |
| `--telemetry-hz` | `1000.0` | High-frequency logging without crash | Executed cleanly at 50 Hz control rate, exit code 0 | **PASS** |
| `--vin` | `5.0\text{V}`, `10.0\text{V}` | Low and high voltage bounds within BAM limits | Actuator limits scaled correctly, exit code 0 | **PASS** |
| `--lin-vel-x` | `+2.0\text{ m/s}`, `-2.0\text{ m/s}` | Rollout 100+ steps without crash or NaNs | 100 steps executed cleanly with 0 NaNs, exit code 0 | **PASS** |

### Challenge 2: Multi-Preset 100+ Steps Headless Verification

Each of the 5 primary spawn presets was tested in headless mode for 125 control steps ($2.5\text{s}$ at 50 Hz control rate = 500 physics steps at 200 Hz):
- **`bottom`**: Apron floor initial position ($x=0.00\text{m}, z=0.125\text{m}$), 125 steps $\rightarrow$ 0 NaNs.
- **`flight1`**: 2.5cm step 1_2 initial position ($x=0.48\text{m}, z=0.150\text{m}$), 125 steps $\rightarrow$ 0 NaNs.
- **`flight2`**: 3.0cm step 2_2 initial position ($x=1.20\text{m}, z=0.235\text{m}$), 125 steps $\rightarrow$ 0 NaNs.
- **`flight3`**: 3.5cm step 3_2 initial position ($x=1.92\text{m}, z=0.320\text{m}$), 125 steps $\rightarrow$ 0 NaNs.
- **`top`**: Summit platform initial position ($x=2.34\text{m}, z=0.395\text{m}$), 125 steps $\rightarrow$ 0 NaNs.

Furthermore, all 13 preset aliases (`bottom`, `flight1`, `flight2`, `flight3`, `top`, `floor`, `stairs_25`, `landing_1`, `stairs_30`, `landing_2`, `stairs_35`, `summit`, `init`) were verified for 100 steps each under deep sensor inspection (`data.qpos`, `data.qvel`, `data.ctrl`, `data.sensordata`, and unified 61D `obs`). Result: **Zero NaNs, zero Infs across all 1,300 evaluated steps**.

CLI subprocess invocation verified for all 5 presets:
```bash
python scripts/infer_stairs.py --headless --spawn <preset> --duration-s 2.1 --lin-vel-x 0.05
# Exited with code 0: "Finished 105 steps with 0 NaNs and clean exit."
```

### Challenge 3: Keyboard Command Mappings & State Machine Verification

The interactive keyboard control state machine was programmatically stress-tested across all available key actions:
1. **Linear & Angular Velocities**:
   - `up`: Increment $v_x$ by $+0.05\text{ m/s}$, strictly saturates at $+0.35\text{ m/s}$.
   - `down`: Decrement $v_x$ by $-0.05\text{ m/s}$, strictly saturates at $-0.25\text{ m/s}$.
   - `left` / `right`: Lateral velocity $v_y$ increment/decrement by $\pm 0.02\text{ m/s}$, saturates at $\pm 0.20\text{ m/s}$.
   - `a` / `e`: Turn rate $w_z$ increment/decrement by $\pm 0.10\text{ rad/s}$, saturates at $\pm 0.80\text{ rad/s}$.
   - ` ` (space): Emergency stop, instantly zeroes `cmd_twist = [0.0, 0.0, 0.0]`.
2. **Body Pitch Trim Mode (`b`)**:
   - Toggles mode on/off.
   - When active, `up`/`down` diverts from $v_x$ to trim `cmd_body[4]` (pitch angle) by $\pm 2^\circ$ ($\pm 0.0349\text{ rad}$).
   - Saturates at $[-30^\circ, +30^\circ]$ ($[-0.524, +0.524]\text{ rad}$).
   - Reverting mode re-enables $v_x$ velocity trim without altering the stored pitch angle.
3. **Head Gaze Mode (`h`)**:
   - Toggles head pitch gaze between $0^\circ$ (forward) and $-20^\circ$ (downward beak tilt at stairs).
4. **Heading Lock Mode (`c`)**:
   - Captures current robot yaw at moment of engagement.
   - Regulates yaw rate in closed loop: $w_z = \text{clip}(0.8 \times \text{yaw\_err}, -0.4, 0.4)$ when forward velocity $|v_x| > 0.01\text{ m/s}$.
5. **Perturbation Push (`p`)**:
   - Directly injects random impulse into `data.qvel[:3]`.
6. **Standup Recovery Trigger (`u`)**:
   - Issues recovery posture command (`cmd_body[2] = 0.02`, `cmd_body[4] = -10^\circ`).
   - If robot is fallen ($z < 0.08\text{m}$), automatically resets height to $0.125\text{m}$ and zeroes velocity.
7. **Policy Inference Toggle (`t`)**:
   - Pauses neural network action computation and holds default joint targets. Resuming restores ONNX inference.
8. **Preset Hotkeys (`1`..`5`)**:
   - Hotkeys 1 through 5 jump instantly to `bottom`, `flight1`, `flight2`, `flight3`, and `top` presets via `reset_to_preset`.
9. **Clean Exit (`q`)**:
   - Sets `quit_requested = True`, breaks simulation loop cleanly, and returns code 0.

### Challenge 4: End-to-End Simulation with Dynamic Key Stream

Simulated execution of `run_simulation` with mocked `TerminalInput` injecting a sequence of 10 key batches (`["up", "up"]`, `["left"]`, `["b"]`, `["up"]`, `["h"]`, `["c"]`, `["p"]`, `["u"]`, `["3"]`, `["q"]`):
- Simulation processed all keyboard inputs sequentially during active physics stepping.
- Switched modes dynamically between velocity trim and pitch trim.
- Successfully applied perturbation and recovery triggers.
- Jumped preset to `flight2` on hotkey `3`.
- Exited cleanly upon receiving `q` with return code 0 and 0 NaNs.

---

## 3. Test Suite Implementation

The full adversarial test suite is implemented in:
`tests/test_infer_stairs_cli_adversarial.py` (15 tests)

```python
# Key test structure summary:
# 1. test_cli_invalid_scene_path
# 2. test_cli_invalid_policy_path
# 3. test_cli_invalid_preset_fallback
# 4. test_cli_telemetry_hz_clamping
# 5. test_cli_extreme_commanded_velocities
# 6. test_all_five_presets_125_steps_headless (parametrized for 5 presets)
# 7. test_cli_subprocess_all_five_presets
# 8. test_all_13_presets_100_steps_nan_guard
# 9. test_terminal_input_unit
# 10. test_keyboard_command_state_transitions
# 11. test_run_simulation_with_mocked_keystream
```

Verification command:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs_cli_adversarial.py
# Result: 15 passed in 8.83s
```

Full combined infer stairs tests:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_infer_stairs_adversarial.py tests/test_infer_stairs_cli_adversarial.py
# Result: 100 passed in 19.5s
```

---

## 4. Verdict

**Final Verdict**: `APPROVE`

The rehearsal simulation script `scripts/infer_stairs.py` satisfies all robustness, boundary-trapping, preset rollout, NaN-immunity, and keyboard state machine requirements of Milestone M3.
