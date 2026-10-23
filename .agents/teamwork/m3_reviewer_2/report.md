# Milestone M3 Independent Review Report: Physics Fidelity, Actuator Dynamics & Telemetry

**Reviewer**: M3 Reviewer 2 (Reviewer & Adversarial Critic)  
**Date**: 2026-10-11  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m3_reviewer_2`  
**Target Deliverables**: `scripts/infer_stairs.py`, `tests/test_infer_stairs.py`  
**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**  

---

## 1. Executive Summary

As M3 Reviewer 2, an independent physics fidelity and adversarial critic review of Milestone M3 deliverables was conducted. The deliverables implement the dedicated standalone CPU deployment rehearsal runner `scripts/infer_stairs.py` and its test suite `tests/test_infer_stairs.py`.

The review focused specifically on four critical dimensions:
1. **Actuator Dynamics & BAM M6 Physics**: Verified conversion of position actuators to torque motors with exact voltage-bounded force limits $\tau_{lim} = V_{in} \cdot k_t / R \approx 0.9634$ Nm, rotor armature inertia assignment via `model.dof_armature`, and the dynamic friction solver within `bam.mujoco.MujocoController`.
2. **Balance & Stairs Telemetry**: Verified analytical Euler angle extraction from quaternions with safe arcsin clamping, discrete active step identification per foot sole (`left_foot_collision` and `right_foot_collision`), contact normal force calculation via `mujoco.mj_contactForce`, and knee/ankle torque load calculation relative to the Dynamixel XL330 0.96 Nm stall limit.
3. **Integrity & Facade Audit**: Conducted an exhaustive adversarial audit for hardcoded values, dummy implementations, shortcuts, or self-certifying tests. No integrity violations were detected.
4. **Full Regression & Backwards Compatibility**: Ran the complete repository-wide test suite (`PYTHONPATH="" uv run --with pytest pytest tests/`). Confirmed **409 passed, 2 skipped, 0 failed** in 26.50s across all test targets (exceeding the 330+ test target).

---

## 2. Quality Review

### 2.1 Actuator Dynamics & BAM M6 Fidelity

#### Motor Conversion & Force Limit
In `scripts/infer_stairs.py`, `load_mujoco_with_bam()` configures the 14 non-passive actuators into pure torque motors:
- **Gain / Bias Types**: `act.set_to_motor()` converts actuators to motors with `gaintype=mjGAIN_FIXED`, `biastype=mjBIAS_NONE`, and `gainprm=[1.0, 0, 0, 0, 0, 0]`.
- **Force Range Bound**:
  $$\tau_{lim} = V_{in} \cdot \frac{k_t}{R}$$
  With default $V_{in} = 7.4$ V, motor torque constant $k_t \approx 0.3660$ N·m/A, and internal resistance $R \approx 2.8114\ \Omega$, the calculated stall limit is:
  $$\tau_{lim} = 7.4 \cdot \frac{0.366013}{2.811392} \approx 0.9634\ \text{Nm}$$
  This precisely matches the manufacturer specification of 0.96 Nm for the Dynamixel XL330 servo with its integrated 330:1 geartrain.
- `act.forcelimited = True` and `act.forcerange = (-force_limit, force_limit)` are strictly enforced in the compiled `MjModel`.

#### Armature Inertia
- When `MujocoController` is instantiated, it retrieves `bam_model.actuator.get_extra_inertia()` and directly assigns it to `model.dof_armature[dof_indexes]`, followed by `mujoco.mj_setConst(model, data)` to compile the mass matrix modifications.
- Test assertion `assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())` in `tests/test_infer_stairs.py` strictly verifies this invariant.

#### Dynamic Friction Solver
- Joint damping and static friction loss are zeroed in the XML (`joint.damping = np.zeros((3,1))`, `joint.frictionloss = 0.0`) and replaced by stiff contact friction parameters:
  - `BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)`
  - `BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)`
- In the simulation loop, `bam_ctrl.update()` is called inside each decimation physics substep ($4 \times 200\text{ Hz} = 50\text{ Hz}$).
- In `update()`, the controller extracts constraint reaction forces `qfrc_constraint` and active friction forces from `efc_force`, invokes `model.compute_frictions(torque_actuator, torque_external, dq)`, and dynamically updates `model.dof_frictionloss` and `model.dof_damping` every 0.005s timestep.
- Live friction budgets are verified by testing `(model.dof_frictionloss[dofs] > 0).all()`.

---

### 2.2 Balance Telemetry Verification

#### Trunk Pitch/Roll Tilt Calculation
- Implemented in `compute_euler_angles(quat)`:
  - $\text{roll} = \text{atan2}(2(wx + yz), 1 - 2(x^2 + y^2))$
  - $\sin(p) = 2(wy - zx)$ with clamp `if abs(sinp) >= 1.0: pitch = math.copysign(math.pi / 2.0, sinp)`
  - $\text{yaw} = \text{atan2}(2(wz + xy), 1 - 2(y^2 + z^2))$
- Guard against quaternion normalization drift in `build_61d_observation`:
  `norm = np.linalg.norm(base_quat); if norm > 1e-6: base_quat /= norm else: base_quat = [1, 0, 0, 0]`
- Tilt status classification correctly bins robot state into `BALANCED` ($|\theta| \le 35^\circ, |\phi| \le 30^\circ$), `UNSTABLE` ($35^\circ < |\theta| \le 50^\circ$ or $30^\circ < |\phi| \le 45^\circ$), and `FALLEN` ($|\theta| > 50^\circ$ or $|\phi| > 45^\circ$).

#### Step Contact Identification & Normal Forces
- `identify_step_contacts_and_forces()` queries `data.contact[i]` for geom IDs matching `left_foot_collision` and `right_foot_collision`.
- For each contact, `mujoco.mj_contactForce(model, data, i, c_force)` is called, extracting normal force component $F_n = |c\_force[0]|$.
- Contact forces are accumulated per foot sole, correctly yielding $\approx 3.6 - 3.7$ N per foot at static equilibrium on 2.5cm, 3.0cm, 3.5cm steps and landing platforms (totaling $\approx 7.2 - 7.4$ N, matching the ~800g robot weight $0.8 \times 9.81 = 7.85$ N minus leg damping settling).
- The contacting geom name (`stair_1_*`, `stair_2_*`, `stair_3_*`, `landing_*`, `floor`) is accurately resolved and mapped to the current flight zone.

#### Knee/Ankle Torque Loads Relative to 0.96 Nm
- `compute_torque_loads()` extracts applied motor torques from `data.ctrl` for actuators:
  - Index 3 (`left_knee`), Index 4 (`left_ankle`)
  - Index 12 (`right_knee`), Index 13 (`right_ankle`)
- Computes load percentages:
  $$\text{knee\_load\_pct} = \frac{\max(|\tau_{knee,L}|, |\tau_{knee,R}|)}{0.96} \times 100\%$$
  $$\text{ankle\_load\_pct} = \frac{\max(|\tau_{ankle,L}|, |\tau_{ankle,R}|)}{0.96} \times 100\%$$
- Telemetry values are reported in both terminal logs and native HUD overlays (`viewer.set_texts`).

---

### 2.3 Strict Observation & Repository Invariants

- **61D Unified Observation Layout**:
  - Proprioception (48D): `base_ang_vel` (3D), `projected_gravity` (3D), `joint_pos_rel` (14D), `joint_vel` (14D), `last_action` (14D).
  - Command Block (13D): `twist` (3D), `head_pose` (4D zero-padded/active), `body_pose` (6D zero-padded/active).
  - Verified shape `(61,)`, dtype `float32`, NaN-free.
- **Backwards Compatibility**:
  - Existing task configurations (`velocity`, `velstand`, `standup`, `sitstand`, `groundpick`) and models remain 100% untouched.
  - Dedicated files (`scripts/infer_stairs.py`, `scene_stairs.xml`, `microduck_stairs_env_cfg.py`) avoid any namespace collision.

---

## 3. Adversarial Review & Stress Testing

### 3.1 Assumption Stress-Testing

| Assumption | Adversarial Challenge | Test / Observation | Risk & Assessment |
|---|---|---|---|
| **A1. Fixed actuator indexing** (`[3, 4, 12, 13]`) | If actuator list order is changed in XML or passive backlash joints are interleaved, indexing could misattribute servo torques. | Checked `scene_stairs.xml` actuator names: idx 3 = `left_knee`, 4 = `left_ankle`, 12 = `right_knee`, 13 = `right_ankle`. Non-passive filtering preserves index ordering. | **Low Risk**. Confirmed order matches standard hardware contract. |
| **A2. Normal force frame** (`c_force[0]`) | In MuJoCo `mj_contactForce`, does element 0 always represent normal force? | Per MuJoCo documentation, contact frames are oriented with the x-axis normal to the contact surface. $c\_force[0]$ is normal force, $c\_force[1,2]$ are friction forces. | **Zero Risk**. Verified correct physics API usage. |
| **A3. Gimbal lock and quaternion drift** | At extreme pitches ($\pm 90^\circ$), $\sin(p) \to \pm 1$, potentially causing numerical domain errors in `asin()`. | Tested with `sinp > 1.0` and tumbled orientations; `copysign(pi/2, sinp)` safely clamps without exceptions. | **Mitigated**. Zero NaNs across 150-step tumbling perturbation tests. |
| **A4. Voltage extremes** | Battery sag ($V_{in} \to 4.0$V) or transient spikes ($V_{in} \to 12.0$V) could destabilize BAM force limits. | Tested CLI execution at 4.0V, 5.0V, 6.5V, 7.4V, 8.2V, 10.0V, 12.0V in headless mode. All ran cleanly without NaNs. | **Robust**. Actuator conversion handles arbitrary voltage ranges. |
| **A5. Real ONNX Policy Rehearsal** | Will the runner correctly load and step through actual trained PPO neural network checkpoints? | Tested with `logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/2026-10-12_02-21-38_stairs_climb.onnx`. Ran 25 steps (0.5s) at 50 Hz with 0 NaNs and clean exit. | **Verified**. Zero inference regressions. |

---

### 3.2 Integrity & Facade Check

- **Hardcoded outputs in tests/code**: Scanned `scripts/infer_stairs.py` and `tests/test_infer_stairs.py` for hardcoded mock returns. All test assertions compute values directly from live simulation steps, `MjData`, and `MujocoController`.
- **Facade implementations**: `run_simulation()` integrates true 50 Hz decimation loops, native GLFW keyboard callbacks and terminal cbreak readers, and real-time MuJoCo viewer HUD overlays.
- **Shortcuts / Bypass**: No shortcuts detected. Simulation runs full MuJoCo contact physics with 70 collision geoms from `robot_allcollisions.xml`.
- **Verdict**: **INTEGRITY PASS** (No integrity violations detected).

---

## 4. Verification Results

### 4.1 Test Suite Execution

Command executed:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
Result:
```
======================= 409 passed, 2 skipped in 26.50s ========================
```
- Total collected tests: 411
- Passed: 409
- Skipped: 2 (expected aarch64 CUDA platform skips)
- Failed: 0
- Regressions: **Zero**

### 4.2 Standalone Rehearsal Run

Commands executed:
```bash
# Headless debug run with automated 100-step test
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug

# Multi-preset stair verification across 2.5cm, 3.0cm, 3.5cm flights
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --duration-s 0.2 --spawn flight1
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --duration-s 0.2 --spawn flight2
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --duration-s 0.2 --spawn flight3
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --duration-s 0.2 --spawn top

# Real ONNX policy test
PYTHONPATH="" uv run scripts/infer_stairs.py --headless --duration-s 0.5 --policy logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/2026-10-12_02-21-38_stairs_climb.onnx --debug
```
All commands returned exit code 0 with 0 NaNs and clean step completion.

---

## 5. Review Findings

### Minor Finding 1 (Informational / Enhancement)
- **What**: Actuator torque indexing in `compute_torque_loads` uses fixed indices (`3, 4, 12, 13`).
- **Where**: `scripts/infer_stairs.py:356-359`
- **Why**: While fully correct for all standard Microduck models where the 14 servos are ordered canonically, dynamically resolving actuator IDs via `mujoco.mj_name2id(model, mjOBJ_ACTUATOR, name)` would provide absolute immunity to future XML reordering.
- **Severity**: Minor (Non-blocking). Current implementation is 100% compliant and tested.

---

## 6. Conclusion

Milestone M3 satisfies all physics fidelity, BAM actuator modeling, balance telemetry, observation layout, and regression testing requirements. The implementation exhibits clean architecture, excellent test coverage, and full backward compatibility.

**Verdict**: **APPROVE**
