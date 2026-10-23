# Forensic Audit Report — Milestone M1

**Work Product**:
- `src/mjlab_microduck/tasks/mdp.py` (Stairs Climbing MDP terms)
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml` (Standalone MJCF stairs simulation scene)
- `tests/test_scene_stairs.py` (Scene geometry and stability unit tests)
- `tests/test_stairs_mdp.py` (MDP numerical and physics unit tests)
**Profile**: General Project
**Integrity Mode**: Demo Mode (per `ORIGINAL_REQUEST.md`)
**Auditor**: M1 Forensic Auditor (`m1_auditor_1`)
**Timestamp**: 2026-10-11T15:24:00Z
**Verdict**: **CLEAN**

---

## Executive Summary

A forensic integrity audit was conducted on all source code and test artifacts delivered under Milestone M1 of the Microduck RL project. Every check mandated by the Integrity Forensics protocol was executed empirically. The audit verified:
1. Static code analysis confirmed zero mock bypassing, zero hardcoded return values, and zero dummy/facade implementations.
2. All 5 newly added MDP functions in `mdp.py` implement genuine multi-body and contact physics (one-sided clearance deficits, impact impulse and velocity damping, horizontal shear friction thresholds for riser collisions, quaternion gravity projections for sole flatness, and elevation-invariant potential progress).
3. All 12 stair obstacles in `scene_stairs.xml` strictly enforce the required 2.5cm, 3.0cm, and 3.5cm riser height sequence, feature unbroken gapless alignment from $x=0.30$m to $x=2.58$m, and are solidly grounded to $z=0.0$ to eliminate tunneling.
4. Independent test verification confirmed 100% pass rate:
   - Target M1 test suite: 21 of 21 tests passed (`tests/test_scene_stairs.py`, `tests/test_stairs_mdp.py`).
   - Adversarial stress suites: 30 of 30 tests passed (`tests/test_scene_stairs_adversarial.py`, `tests/test_stairs_adversarial.py`).
   - Full repository regression suite: 301 passed, 6 skipped, 0 failed.

Verdict is **CLEAN**. Milestone M1 is cleared for progression to Milestone M2.

---

## Phase 1: Mode-Agnostic Static Analysis & Anti-Cheat Checks

### Check 1.1: Hardcoded Test Return Detection
- **Target**: `src/mjlab_microduck/tasks/mdp.py` (lines 7690–7875)
- **Methodology**: Static inspection of AST and code lines for static returns, lookup dictionaries of test outputs, or bypass logic based on test flags.
- **Findings**:
  - `stairs_foot_clearance_penalty`: computes `torch.linalg.norm(foot_vel_xy, dim=-1)`, `target_height - foot_height`, `torch.clamp`, and command activation masking. Returns `-cost`.
  - `stairs_foot_landing_damping_penalty`: dynamically calls `compute_first_contact(dt=env.step_dt)`, computes Euclidean force magnitude over `impact_force_thresh`, downward velocity over `impact_vel_thresh`, quadratically penalizes impact, and gates by contact. Returns `-torch.sum(...)`.
  - `stairs_riser_trip_penalty`: evaluates horizontal vs vertical contact force ratio `torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)`. Returns `-trip_cost`.
  - `stairs_sole_flat_penalty`: normalizes gravity vector, rotates gravity into foot site frame using `quat_apply_inverse`, penalizes horizontal misalignment, and gates by contact time. Returns `-per_foot.sum(dim=1)`.
  - `stairs_relative_height_progress`: measures trunk $z$ relative to lowest foot $z$, clamps to `ceiling`, maintains state across episode boundaries, and outputs $\Delta \Phi$.
- **Result**: **PASS** (Zero hardcoded returns or shortcuts).

### Check 1.2: Dummy / Facade Implementation Detection
- **Target**: `src/mjlab_microduck/tasks/mdp.py`
- **Methodology**: Detection of trivial constant returns (`return 0.0`), unhandled stubs (`raise NotImplementedError`), or hollow passthroughs.
- **Findings**: Every function performs vectorized tensor computations using PyTorch and MuJoCo simulation state. No dummy routines exist.
- **Result**: **PASS**.

### Check 1.3: Pre-Populated Result Artifact Detection
- **Target**: Workspace directory tree
- **Methodology**: Executed `find . -name '*.log' -o -name '*result*' -o -name '*output*'`.
- **Findings**: No pre-generated test logs, mock outputs, or fabricated verification tokens detected in project directories.
- **Result**: **PASS**.

### Check 1.4: Self-Certifying / Tautological Test Detection
- **Target**: `tests/test_scene_stairs.py`, `tests/test_stairs_mdp.py`
- **Methodology**: Audited test logic to ensure assertions evaluate external physical truths rather than duplicating implementation constants.
- **Findings**:
  - `test_scene_stairs.py`: parses raw XML via MuJoCo C bindings, checks exact geom positions and extents against the specification, evaluates continuous contact dynamics and stability under 50-step simulation rollouts.
  - `test_stairs_mdp.py`: tests physical scenarios (low vs high clearance, gentle vs violent touchdown, tread vs riser contact, flat vs tilted foot sole, initial reset vs dynamic rise vs elevation climbing).
- **Result**: **PASS**.

### Check 1.5: Dependency & Execution Delegation Audit (Demo Mode)
- **Target**: Codebase imports and dependencies
- **Methodology**: Checked whether core MDP logic was delegated to external pre-built RL packages.
- **Findings**: Implemented natively in PyTorch utilizing standard repository helpers (`quat_apply_inverse`). No forbidden external library delegation.
- **Result**: **PASS**.

---

## Phase 2: Physics & Geometry Mathematical Verification

### Check 2.1: Mathematical Rigor of MDP Penalties
1. **Stair Clearance Penalty (`stairs_foot_clearance_penalty`)**:
   $$\text{deficit} = \max(0, h_{\text{target}} - h_{\text{foot}})$$
   $$\text{moving} = \max(0, \|\mathbf{v}_{xy}\| - v_{\text{thresh}})$$
   $$\text{penalty} = - \sum \left(\text{deficit} \cdot \text{moving}\right) \cdot \mathbf{1}(\|\mathbf{cmd}_{xy}\| > cmd_{\text{thresh}})$$
   - *Physical validity*: One-sided design ensures the robot is never penalized for lifting high over steps. Stationary stance feet are shielded by $v_{\text{thresh}}$.
   - *Sign convention*: Always $\le 0.0$. Correctly paired with positive reward weight in cfg.

2. **Landing Damping Penalty (`stairs_foot_landing_damping_penalty`)**:
   $$\text{penalty} = - \sum \left[ \max(0, \|\mathbf{F}\| - F_{\text{thresh}}) + w_v \cdot \max(0, -v_z - v_{\text{thresh}})^2 \right] \cdot \mathbf{1}(\text{first\_contact})$$
   - *Physical validity*: Evaluated exclusively at touchdown. Forces below 8 N (~1x body weight) and velocities below 0.08 m/s are free. Quadratic penalty protects XL330 gear trains against violent stair impact.
   - *Sign convention*: Always $\le 0.0$.

3. **Stair Riser Collision Penalty (`stairs_riser_trip_penalty`)**:
   $$\text{penalty} = - \sum \max\left(0, \|\mathbf{F}_{xy}\| - \mu_{\max} F_z\right) \cdot \mathbf{1}(\text{in\_contact})$$
   - *Physical validity*: On horizontal treads, normal force is vertical ($F_z$ dominates, $\|\mathbf{F}_{xy}\| \le \mu F_z$). On vertical risers, contact normal is horizontal ($\|\mathbf{F}_{xy}\|$ dominates while $F_z \approx 0$). Accurately detects riser stubbing without false-flagging tread stance.
   - *Sign convention*: Always $\le 0.0$.

4. **Stance Sole Flatness Penalty (`stairs_sole_flat_penalty`)**:
   $$\mathbf{g}_{\text{site}} = \mathbf{q}_{\text{foot}}^{-1} \otimes \hat{\mathbf{g}}_w \otimes \mathbf{q}_{\text{foot}}$$
   $$\text{penalty} = - \sum \left( g_{\text{site}, x}^2 + g_{\text{site}, y}^2 \right) \cdot \mathbf{1}(\text{contact\_time} > 0)$$
   - *Physical validity*: Projects gravity into foot coordinate frame; flat sole yields zero horizontal gravity projection. Free during swing phase.
   - *Sign convention*: Always $\le 0.0$.

5. **Elevation-Invariant Recovery Potential (`stairs_relative_height_progress`)**:
   $$z_{\text{rel}} = z_{\text{trunk}} - \min(z_{\text{left\_foot}}, z_{\text{right\_foot}})$$
   $$\Phi(t) = \min(z_{\text{rel}}, \text{ceiling})$$
   $$\text{reward} = \Phi(t) - \Phi(t-1)$$
   - *Physical validity*: Standard global height reward saturates or creates farming loops on stairs as elevation increases. Relative height remains constant during steady stair climbing ($z_{\text{rel}} \approx 0.117$m $\rightarrow \Delta \Phi = 0.0$), but rewards standing up from crouched/prone states. Properly handles episode resets via `episode_length_buf <= 1`.

6. **Site Dimension Guard (`_FEET_SITE_CFG`)**:
   - `_FEET_SITE_CFG.site_ids = [1, 6]` specifically isolates the 2 feet sites from the 7 robot sites, preventing runtime dimension mismatches against 2-primary sensors (`foot_height_scan`, `feet_ground_contact`).

### Check 2.2: XML Geometry and Contact Parameters (`scene_stairs.xml`)
Every obstacle box was verified against the ground-truth specification:

| Geom Name | Type | $x$ Range (m) | $\Delta x$ (m) | $z$ Center (m) | $z$ Half-Size (m) | Top $z$ (m) | Bottom $z$ (m) | Riser $\Delta z$ |
|-----------|------|---------------|----------------|----------------|-------------------|-------------|----------------|------------------|
| `stair_1_1` | Box | [0.30, 0.42] | 0.12 | 0.0125 | 0.0125 | 0.0250 | 0.0000 | +0.025m (2.5cm) |
| `stair_1_2` | Box | [0.42, 0.54] | 0.12 | 0.0250 | 0.0250 | 0.0500 | 0.0000 | +0.025m (2.5cm) |
| `stair_1_3` | Box | [0.54, 0.66] | 0.12 | 0.0375 | 0.0375 | 0.0750 | 0.0000 | +0.025m (2.5cm) |
| `landing_1` | Box | [0.66, 1.02] | 0.36 | 0.0375 | 0.0375 | 0.0750 | 0.0000 | 0.000m (Platform) |
| `stair_2_1` | Box | [1.02, 1.14] | 0.12 | 0.0525 | 0.0525 | 0.1050 | 0.0000 | +0.030m (3.0cm) |
| `stair_2_2` | Box | [1.14, 1.26] | 0.12 | 0.0675 | 0.0675 | 0.1350 | 0.0000 | +0.030m (3.0cm) |
| `stair_2_3` | Box | [1.26, 1.38] | 0.12 | 0.0825 | 0.0825 | 0.1650 | 0.0000 | +0.030m (3.0cm) |
| `landing_2` | Box | [1.38, 1.74] | 0.36 | 0.0825 | 0.0825 | 0.1650 | 0.0000 | 0.000m (Platform) |
| `stair_3_1` | Box | [1.74, 1.86] | 0.12 | 0.1000 | 0.1000 | 0.2000 | 0.0000 | +0.035m (3.5cm) |
| `stair_3_2` | Box | [1.86, 1.98] | 0.12 | 0.1175 | 0.1175 | 0.2350 | 0.0000 | +0.035m (3.5cm) |
| `stair_3_3` | Box | [1.98, 2.10] | 0.12 | 0.1350 | 0.1350 | 0.2700 | 0.0000 | +0.035m (3.5cm) |
| `landing_summit` | Box | [2.10, 2.58] | 0.48 | 0.1350 | 0.1350 | 0.2700 | 0.0000 | 0.000m (Summit) |

- **Grounding**: For all 12 geoms, $\text{bottom\_z} = \text{pos\_z} - \text{size\_z} = 0.0000$ m. Zero hollow voids; zero risk of sub-step penetration.
- **Continuity**: Every step $x_{\min}$ matches the previous step $x_{\max}$ to $< 10^{-6}$ m. Unbroken traversal profile.
- **Width**: All size $y = 0.25$ m (0.50 m total width) centered at $y = 0.0$ m.
- **Friction**: All boxes configure `condim="3"`, `friction="1.2 0.01 0.001"`, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001 0.5 2"`.
- **Keyframes**: All 8 keyframes (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`) have valid 21D qpos and 14D ctrl vectors with identity orientation and dynamically stable settling.

---

## Phase 3: Independent Empirical Test Execution

### 1. Mandatory Milestone Verification Command
```bash
PYTHONPATH="" uv run --with pytest pytest -v tests/test_scene_stairs.py tests/test_stairs_mdp.py
```
**Raw Result**:
```
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/dhp01/microduck_rl
configfile: pyproject.toml
plugins: typeguard-4.4.4, anyio-4.12.1
collected 21 items

tests/test_scene_stairs.py::test_scene_stairs_compiles PASSED            [  4%]
tests/test_scene_stairs.py::test_scene_stairs_robot_model PASSED         [  9%]
tests/test_scene_stairs.py::test_all_stair_geometries_exist PASSED       [ 14%]
tests/test_scene_stairs.py::test_stair_riser_heights PASSED              [ 19%]
tests/test_scene_stairs.py::test_stair_gapless_continuity PASSED         [ 23%]
tests/test_scene_stairs.py::test_stair_solid_grounding PASSED            [ 28%]
tests/test_scene_stairs.py::test_stair_contact_friction_parameters PASSED [ 33%]
tests/test_scene_stairs.py::test_stair_width PASSED                      [ 38%]
tests/test_scene_stairs.py::test_all_keyframes_exist_and_valid PASSED    [ 42%]
tests/test_scene_stairs.py::test_keyframe_dynamic_settle_stability PASSED [ 47%]
tests/test_stairs_mdp.py::test_tensor_dimensions_and_dtype[1] PASSED     [ 52%]
tests/test_stairs_mdp.py::test_tensor_dimensions_and_dtype[4] PASSED     [ 57%]
tests/test_stairs_mdp.py::test_tensor_dimensions_and_dtype[32] PASSED    [ 61%]
tests/test_stairs_mdp.py::test_tensor_dimensions_and_dtype[64] PASSED    [ 66%]
tests/test_stairs_mdp.py::test_foot_clearance_self_negating_and_bounds PASSED [ 71%]
tests/test_stairs_mdp.py::test_landing_damping_self_negating_and_bounds PASSED [ 76%]
tests/test_stairs_mdp.py::test_riser_trip_self_negating_and_bounds PASSED [ 80%]
tests/test_stairs_mdp.py::test_sole_flat_self_negating_and_bounds PASSED [ 85%]
tests/test_stairs_mdp.py::test_relative_height_progress_lifecycle PASSED [ 90%]
tests/test_stairs_mdp.py::test_relative_height_progress_elevation_invariance PASSED [ 95%]
tests/test_stairs_mdp.py::test_site_mismatch_protection PASSED           [100%]

============================= 21 passed in 10.29s ==============================
```

### 2. Full Regression Suite Execution
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
**Raw Result**:
```
======================= 301 passed, 6 skipped in 17.24s ========================
```
- **Regressions**: 0
- **Pass rate**: 100% of runnable tests.

---

## Final Forensic Verdict

**VERDICT**: **CLEAN**

All Milestone M1 requirements and integrity conditions have been verified empirically and found to be genuine, mathematically sound, and fully compliant with project invariants.
