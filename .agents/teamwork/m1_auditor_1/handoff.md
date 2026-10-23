# Handoff Report — Milestone M1 Forensic Audit

**Agent**: M1 Forensic Auditor (`m1_auditor_1`)
**Date**: 2026-10-11T15:25:00Z
**Target**: Milestone M1 (MDP terms & Standalone Stairs Scene)
**Verdict**: **CLEAN**

---

### 1. Observation

- **Source Code Additions in `src/mjlab_microduck/tasks/mdp.py`**:
  - Lines 7690–7875 implement 5 core stairs MDP terms under the group header `# Stairs Climbing Task — Clearance, Landing Damping, Contact & Recovery Terms`:
    1. `stairs_foot_clearance_penalty` (lines 7704–7741): Computes one-sided clearance shortfall `torch.clamp(target_height - foot_height, min=0.0)` multiplied by horizontal motion above threshold `torch.clamp(vel_norm - vel_threshold, min=0.0)` and gated by active planar command (`cmd_norm > command_threshold`). Returns `-cost` ($\le 0.0$).
    2. `stairs_foot_landing_damping_penalty` (lines 7744–7774): Directly inspects `contact_sensor.compute_first_contact(dt=env.step_dt)`, computes normal impact force excess above `impact_force_thresh=8.0` N, and downward velocity excess above `impact_vel_thresh=0.08` m/s quadratically weighted (`vel_weight=10.0`). Returns `-torch.sum(...)` ($\le 0.0$).
    3. `stairs_riser_trip_penalty` (lines 7776–7804): Computes contact shear ratio excess `torch.clamp(f_xy - max_friction_ratio * f_z, min=0.0)` gated by `contact_sensor.data.found > 0`. Returns `-trip_cost` ($\le 0.0$).
    4. `stairs_sole_flat_penalty` (lines 7806–7836): Normalizes gravity vector, applies quaternion inverse rotation `quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)` to evaluate horizontal gravity projection in the foot site frame, gated by stance contact duration. Returns `-per_foot.sum(dim=1)` ($\le 0.0$).
    5. `stairs_relative_height_progress` (lines 7838–7874): Computes trunk height relative to lowest foot site $z_{\text{rel}} = z_{\text{trunk}} - \min(z_{\text{left\_foot}}, z_{\text{right\_foot}})$, clamped to `ceiling=0.115`, tracking state across resets (`episode_length_buf <= 1`). Returns $\Delta \Phi \in [-0.05, 0.05]$.
  - Lines 7700–7701 define `_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with explicit `site_ids = [1, 6]`, protecting against dimension mismatches when referencing 2-primary sensors.

- **Scene Definition in `src/mjlab_microduck/robot/microduck/scene_stairs.xml`**:
  - Lines 33–73 define 12 sequential boxes and landing platforms:
    - Flight 1 (`stair_1_1` .. `stair_1_3`): 3 steps with riser $\Delta z = 0.025$m (2.5cm) each, reaching $z = 0.075$m.
    - Landing 1 (`landing_1`): Flat platform at $z = 0.075$m from $x=0.66$m to $1.02$m ($\Delta x = 0.36$m).
    - Flight 2 (`stair_2_1` .. `stair_2_3`): 3 steps with riser $\Delta z = 0.030$m (3.0cm) each, reaching $z = 0.165$m.
    - Landing 2 (`landing_2`): Flat platform at $z = 0.165$m from $x=1.38$m to $1.74$m ($\Delta x = 0.36$m).
    - Flight 3 (`stair_3_1` .. `stair_3_3`): 3 steps with riser $\Delta z = 0.035$m (3.5cm) each, reaching $z = 0.270$m.
    - Summit (`landing_summit`): Flat platform at $z = 0.270$m from $x=2.10$m to $2.58$m ($\Delta x = 0.48$m).
  - Solid Grounding: Every box has $z_{\text{pos}} - z_{\text{half\_size}} = 0.000$m, anchoring down to the floor plane at $z=0.0$.
  - Gapless X continuity: $x_{\text{min}}$ of step $k+1$ exactly equals $x_{\text{max}}$ of step $k$ for all $k \in [0, 11]$ across $[0.30, 2.58]$m.
  - Keyframes: 8 keyframes defined (`INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`), all featuring valid 21D qpos and 14D ctrl vectors.

- **Independent Test Execution**:
  - `PYTHONPATH="" uv run --with pytest pytest -v tests/test_scene_stairs.py tests/test_stairs_mdp.py`:
    Output: `21 passed in 10.29s`, exit code 0.
  - `PYTHONPATH="" uv run --with pytest pytest -v tests/test_scene_stairs_adversarial.py tests/test_stairs_adversarial.py`:
    Output: `30 passed in 11.26s`, exit code 0.
  - Full regression test: `PYTHONPATH="" uv run --with pytest pytest tests/`:
    Output: `301 passed, 6 skipped in 17.24s`, exit code 0. Zero regressions across existing task suites.

---

### 2. Logic Chain

1. **Static Anti-Cheat Evaluation**:
   - *Premise*: Under Demo Mode (specified in `ORIGINAL_REQUEST.md`), hardcoded test returns, dummy/facade implementations, pre-populated logs, self-certifying tests, and external framework delegations are strictly prohibited.
   - *Evidence*: Direct code audit of `src/mjlab_microduck/tasks/mdp.py` demonstrates that all 5 MDP functions execute genuine PyTorch tensor operations on live scene data. No constant dummy returns, lookup tables, or shortcuts exist. Search for pre-populated logs returned zero hits in project code.
   - *Conclusion*: Static anti-cheat checks pass with zero violations.

2. **Physical & Mathematical Rigor**:
   - *Premise*: Reward functions must adhere to physical laws, Microduck sign conventions (penalties return $\le 0.0$), and elevation invariance on stairs.
   - *Evidence*:
     - `stairs_foot_clearance_penalty` is one-sided and safely bounded $\le 0.0$.
     - `stairs_foot_landing_damping_penalty` strictly gates on `first_contact` to avoid taxing ongoing stance.
     - `stairs_riser_trip_penalty` correctly exploits the friction cone geometry (normal force direction) to differentiate treads from risers.
     - `stairs_relative_height_progress` utilizes relative foot-to-trunk elevation $z_{\text{trunk}} - \min(z_{\text{feet}})$, ensuring steady stair climbing scores $0.0$ while rising from falls pays positive progress $\Delta \Phi$.
     - `_FEET_SITE_CFG.site_ids = [1, 6]` prevents dimension crashes with 2-primary sensors.
   - *Conclusion*: The mathematical formulation is sound, robust against reward farming, and compliant with Microduck sign rules.

3. **Geometric & Simulation Verification**:
   - *Premise*: `scene_stairs.xml` must implement exact 2.5cm, 3.0cm, and 3.5cm riser heights, continuous gapless layout, solid grounding, and stable spawn states.
   - *Evidence*: Automated verification in `test_scene_stairs.py` and `test_scene_stairs_adversarial.py` proves that all 12 boxes are grounded to $z=0.0$, riser deltas match specifications to within $< 10^{-5}$m, and all standing keyframes dynamically settle within $< 10^\circ$ pitch/roll under 50-step simulation rollouts.
   - *Conclusion*: Scene geometry and contact physics are verified without defects.

4. **Independent Test Execution**:
   - *Premise*: All tests must execute independently and pass cleanly without masking or environment dependence.
   - *Evidence*: `pytest` executed with clean `PYTHONPATH=""` yielded 100% passes across unit tests (21/21), adversarial stress tests (30/30), and full repository regression tests (301/301 runnable).
   - *Conclusion*: All implementation deliverables satisfy the project acceptance criteria.

---

### 3. Caveats

- **No caveats**: All required checks were executed empirically on live files, and both isolated and end-to-end regression tests passed with zero failures.

---

### 4. Conclusion

- **Verdict**: **CLEAN**
- All Milestone M1 work products (`src/mjlab_microduck/tasks/mdp.py`, `src/mjlab_microduck/robot/microduck/scene_stairs.xml`, `tests/test_scene_stairs.py`, `tests/test_stairs_mdp.py`) meet all integrity constraints, physical requirements, and repository invariants.
- **Recommended Action**: Milestone M1 is formally approved. The orchestrator may proceed to Milestone M2 (Environment Configuration, Curricula, and Registration).

---

### 5. Verification Method

To independently verify the audit conclusions:

1. **Verify MDP and Scene Tests**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest -v tests/test_scene_stairs.py tests/test_stairs_mdp.py
   ```
   *Expected outcome*: 21 passed.

2. **Verify Adversarial Stress Tests**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest -v tests/test_scene_stairs_adversarial.py tests/test_stairs_adversarial.py
   ```
   *Expected outcome*: 30 passed.

3. **Verify Full Repository Regression**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   *Expected outcome*: 301 passed, 6 skipped, 0 failed.

4. **Invalidation Conditions**:
   - Any modification to `scene_stairs.xml` that alters box elevation, introduces non-zero bottom height, or creates X gaps $> 10^{-4}$m.
   - Any modification to `mdp.py` stairs penalties that returns positive values under violations or hardcodes return values.
