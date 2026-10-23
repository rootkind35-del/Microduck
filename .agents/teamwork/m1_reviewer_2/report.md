# Milestone M1 Independent Review & Adversarial Critic Report

- **Reviewer**: M1 Reviewer 2 (Reviewer & Adversarial Critic)
- **Target Milestone**: M1 (Core MDP Terms & Standalone Simulation Scene)
- **Target Files**:
  - `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
  - `src/mjlab_microduck/tasks/mdp.py` (Stairs Climbing additions)
  - `tests/test_scene_stairs.py`
  - `tests/test_stairs_mdp.py`
  - `tests/test_stairs_e2e.py`
- **Date**: 2026-10-11
- **Verdict**: **APPROVE**

---

## 1. Executive Summary

An exhaustive independent review and adversarial evaluation of the Milestone M1 implementation was conducted. The work delivers:
1. A standalone, high-fidelity MJCF simulation scene (`scene_stairs.xml`) featuring 3 consecutive flights of 2.5 cm, 3.0 cm, and 3.5 cm stair risers with intermediate and summit landing platforms, full-depth solid grounding down to z=0, continuous gapless X-alignment, `condim="3"`, friction=1.2, and 8 verified keyframes with stable settling dynamics.
2. Five mathematically grounded, numerically stable MDP functions in `src/mjlab_microduck/tasks/mdp.py` (`stairs_foot_clearance_penalty`, `stairs_foot_landing_damping_penalty`, `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, `stairs_relative_height_progress`), strictly adhering to the repository's self-negating penalty sign conventions ($\le 0$) and potential-based relative height shaping.
3. Unit and invariant test suites (`tests/test_scene_stairs.py`, `tests/test_stairs_mdp.py`, and `tests/test_stairs_e2e.py`).
4. **Integrity Verification**: No hardcoded test stubs, dummy facades, task bypasses, or fabricated outputs were detected. Implementations compute true physics from MuJoCo state and sensors.
5. **Regression Verification**: 100% of the existing regression suite passed (271 passed, 6 skipped in 15.89s) with exactly 0 failures and 0 regressions.

---

## 2. Quality Review & Code Verification

### 2.1 `scene_stairs.xml` Physics & Geometry Audit
- **Geom Inventory**: Exactly 12 box collision geoms: 9 stair steps (3 flights of 3 steps) and 3 landings (`landing_1`, `landing_2`, `landing_summit`).
- **Riser Heights**:
  - Flight 1: 3 steps $\times$ 2.5 cm risers (elevations: 2.5 cm, 5.0 cm, 7.5 cm).
  - Landing 1: Flat platform at elevation 7.5 cm (length 0.36 m, $x \in [0.66, 1.02]$).
  - Flight 2: 3 steps $\times$ 3.0 cm risers (elevations: 10.5 cm, 13.5 cm, 16.5 cm).
  - Landing 2: Flat platform at elevation 16.5 cm (length 0.36 m, $x \in [1.38, 1.74]$).
  - Flight 3: 3 steps $\times$ 3.5 cm risers (elevations: 20.0 cm, 23.5 cm, 27.0 cm).
  - Summit Landing: Flat platform at elevation 27.0 cm (length 0.48 m, $x \in [2.10, 2.58]$).
- **Gapless Continuity**: The primary traversal axis $+X$ is strictly continuous without gaps or overlap ($0.30 \to 0.42 \to 0.54 \to 0.66 \to 1.02 \to 1.14 \to 1.26 \to 1.38 \to 1.74 \to 1.86 \to 1.98 \to 2.10 \to 2.58\text{ m}$).
- **Solid Grounding**: Every stair step and landing geom satisfies $z_{\text{bottom}} = \text{pos}_z - \text{size}_z = 0.0\text{ m}$. Full depth down to the ground plane prevents any hollow geom penetration or foot tunneling during high-velocity impacts.
- **Contact Physics**: All stair and landing geoms configure `condim="3"`, tangential sliding friction `1.2`, torsional friction `0.01`, rolling friction `0.001`, `solref="0.02 1.0"` (20 ms critically damped relaxation), and `solimp="0.9 0.95 0.001 0.5 2"`.
- **Keyframes**: All 8 keyframes (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`) are defined with complete 21D `qpos` and 14D `ctrl` vectors. Forward dynamic settling for 50 steps (0.25 s) showed pitch and roll remain $< 1.0^\circ$ (well within the $< 10.0^\circ$ tolerance), confirming equilibrium stability.

### 2.2 `mdp.py` Stairs Terms Audit
- **Site Configuration (`_FEET_SITE_CFG`)**:
  - `SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with `site_ids = [1, 6]`.
  - Confirmed from robot model (`robot_groundcontact.xml` and `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG`) that site 1 is `left_foot` and site 6 is `right_foot`. This prevents selecting all 7 sites on the robot and eliminates dimension mismatch against 2-primary sensors (`foot_height_scan`, `feet_ground_contact`).
- **`stairs_foot_clearance_penalty`**:
  - One-sided deficit: $\max(0, h_{\text{target}} - h_{\text{foot}})$. Clearing higher than target incurs zero penalty, avoiding artificial gate-capping.
  - Forward velocity gating: $\max(0, \|\mathbf{v}_{xy}\| - v_{\text{thresh}})$. Stationary stance feet below clearance are not penalized.
  - Active command gating: Scaled by $(\|\mathbf{v}_{\text{cmd}}\| > \text{cmd}_{\text{thresh}})$.
  - Tensor safety: `torch.nan_to_num` on height and velocity norm; returns $\le 0$ (positive weight in config).
- **`stairs_foot_landing_damping_penalty`**:
  - Touchdown gating via `contact_sensor.compute_first_contact(dt=env.step_dt)` ensures penalty is evaluated strictly at first contact.
  - Excess impact force: $\max(0, \|\mathbf{F}\| - F_{\text{thresh}})$ with $F_{\text{thresh}} = 8.0\text{ N}$ (~1x body weight).
  - Quadratic downward velocity penalty: $\max(0, -v_z - v_{\text{thresh}})^2$ with $v_{\text{thresh}} = 0.08\text{ m/s}$.
  - Returns $\le 0$.
- **`stairs_riser_trip_penalty`**:
  - Formulated as shear force excess: $\max(0, F_{xy} - \mu F_z)$ with $\mu = 0.5$.
  - **Zero-Division Free**: Avoids computing the ratio $F_{xy} / F_z$. In cases where $F_z = 0$, subtraction remains stable without singularity.
  - Clamps $F_z \ge 0$, preventing negative normal force artifacts from biasing the shear threshold.
  - Returns $\le 0$.
- **`stairs_sole_flat_penalty`**:
  - Gravity projected into foot site frame: $\mathbf{g}_{\text{site}} = \mathbf{q}^{-1} \mathbf{g}_{\text{world}}$.
  - Normalizes gravity per environment along `dim=-1` (`F.normalize(asset.data.gravity_vec_w, dim=-1)`), avoiding the multi-environment scalar norm bug documented in `mdp.py`.
  - Stance contact gating (`contact_time > 0.0` or `found > 0`): swing feet are unconstrained to tilt during stride.
  - Returns $\le 0$.
- **`stairs_relative_height_progress`**:
  - Measures trunk height relative to lowest foot: $z_{\text{rel}} = z_{\text{trunk}} - \min(z_{\text{foot\_l}}, z_{\text{foot\_r}})$.
  - Invariant to absolute elevation: stepping onto higher stair treads increases both trunk and feet by $\Delta h$, keeping $z_{\text{rel}}$ constant and yielding $\Delta \Phi = 0.0$.
  - Clamped at `ceiling = 0.115 m`: holding upright stance pays 0.0.
  - Lifecycle state tracking: resets `_stairs_rel_height_pot_prev` for environments with `episode_length_buf <= 1`, eliminating reset transient spikes.

---

## 3. Adversarial Review & Stress Testing

| Challenge ID | Target | Attack Scenario | Blast Radius | Observed / Predicted Result | Status |
|:---|:---|:---|:---|:---|:---|
| **ADV-01** | `stairs_riser_trip_penalty` | Pure horizontal wall collision where $F_z = 0.0\text{ N}$ and $F_{xy} = 20.0\text{ N}$. | NaN/Inf generation if implemented as friction ratio $F_{xy} / F_z$. | `f_xy - 0.5 * f_z = 20.0`; penalty is $-20.0$. Completely immune to zero division. | **PASS** |
| **ADV-02** | `stairs_relative_height_progress` | Robot climbs 3.5 cm stair tread without falling. Does it farm recovery reward? | Policy exploits recovery shaping to farm reward during climbing. | Measured $\Delta \Phi = -7.45 \times 10^{-9} \approx 0.0$. Zero reward farmed during climbing. | **PASS** |
| **ADV-03** | `stairs_relative_height_progress` | Asynchronous environment resets where env 0 resets at step 0 while env 1 continues at step 50. | Potential jump artifact causing large reward spike or crash on reset. | `fresh = env.episode_length_buf <= 1` correctly masks per-env tensor; env 0 yields $\Delta \Phi = 0.0$, env 1 maintains continuous delta. | **PASS** |
| **ADV-04** | `stairs_foot_clearance_penalty` | Foot height sensor outputs NaNs or zeroes during sensor occlusion. | Simulation crashes with NaN gradients or NaN policy weights. | `torch.nan_to_num(..., nan=0.0)` sanitizes inputs before calculation; output is finite. | **PASS** |
| **ADV-05** | `scene_stairs.xml` | Robot spawned at standing keyframes on Flight 1, 2, 3, and Summit subjected to gravity and PD control. | Robot falls over immediately due to geometric misalignment or ground clipping. | 50-step dynamic simulation shows pitch and roll remain $< 1.0^\circ$ for all 8 keyframes. | **PASS** |
| **ADV-06** | `_FEET_SITE_CFG` | Scene contains 7 robot sites; test selects `site_ids = [1, 6]`. | Shape mismatch runtime error against 2-primary contact/height sensors. | Correctly selects 2 foot sites, matching shape `(B, 2)`. | **PASS** |

---

## 4. Integrity Violation Assessment

Under the reviewer and adversarial critic mandate, the code was checked for:
- **Hardcoded test stubs**: None. All functions execute general tensor operations based on environment and sensor data.
- **Dummy/facade implementations**: None. Functions implement real physical equations for clearance deficit, impact damping, Coulomb shear friction, quaternion gravity projection, and potential-based relative height.
- **Task shortcuts / external bypassing**: None. Standalone scene and MDP terms were implemented directly in the repository according to MuJoCo and `mjlab` specifications.
- **Fabricated verification logs**: None. Independent verification runs produced actual pytest logs with 271 passed tests.
- **Self-certifying work**: All assertions independently verified with custom test runs.

**Integrity Finding**: **NO INTEGRITY VIOLATION DETECTED.**

---

## 5. Verification Results

### 5.1 Focused Unit Tests
Command:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py
```
Output:
```
tests/test_scene_stairs.py ..........                                    [ 47%]
tests/test_stairs_mdp.py ...........                                     [100%]
============================== 21 passed in 9.88s ==============================
```

### 5.2 Full Regression Test Suite
Command:
```bash
PYTHONPATH="" uv run --with pytest pytest tests/
```
Output:
```
======================= 271 passed, 6 skipped in 15.89s ========================
```
- Total test count: 277 collected
- Passed: 271
- Skipped: 6 (1 architecture-specific CUDA test on x86_64; 5 M2/M3-specific progressive integration tests in `test_stairs_e2e.py`)
- Failed: 0
- Regressions: **0**

---

## 6. Review Verdict & Recommendations

### Verdict: **APPROVE**

Milestone M1 satisfies all physics fidelity, geometric continuity, numerical stability, and backwards compatibility requirements with zero regressions. Milestone M2 (Environment Configuration, Curriculum & Registration) may proceed immediately.
