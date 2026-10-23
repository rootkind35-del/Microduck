# HANDOFF — Milestone M1 Review & Adversarial Stress Test

**From**: M1 Reviewer 1 (`d50f9ec4-8b00-4a58-a1c2-a052385a2661`)  
**To**: Orchestrator (`f3cd8a4f-6d90-4952-bb58-a5be7069a95c`)  
**Date**: 2026-10-11T15:25:00Z  
**Verdict**: **APPROVE**  

---

## 1. Observation

1. **Stairs MDP additions in `src/mjlab_microduck/tasks/mdp.py` (lines 7689–7875)**:
   - `_FEET_SITE_CFG`: `SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` with explicit `site_ids = [1, 6]` (lines 7700–7701).
   - `stairs_foot_clearance_penalty` (lines 7704–7741): computes one-sided clearance deficit clamped to $[0, \infty)$, forward moving gate, active twist command gating, returns `-cost <= 0.0`.
   - `stairs_foot_landing_damping_penalty` (lines 7744–7774): computes touchdown impact forces and negative vertical velocities gated by `first_contact`, returns `-torch.sum(touchdown_cost, dim=1) <= 0.0`.
   - `stairs_riser_trip_penalty` (lines 7776–7804): computes horizontal friction shear excess over $0.5 F_z$, gated by `in_contact`, returns `-trip_cost <= 0.0`.
   - `stairs_sole_flat_penalty` (lines 7806–7836): projects normalized world gravity into foot site frames, evaluates horizontal tilt squared, gated by `contact_time > 0`, returns `-per_foot.sum(dim=1) <= 0.0`.
   - `stairs_relative_height_progress` (lines 7838–7875): computes potential $\Delta \Phi$ where $\Phi = \min(z_{\text{trunk}} - \min(z_{\text{left\_foot}}, z_{\text{right\_foot}}), 0.115)$, resets buffer when `episode_length_buf <= 1`.

2. **Dedicated Scene in `src/mjlab_microduck/robot/microduck/scene_stairs.xml` (lines 1–142)**:
   - Includes `robot_groundcontact.xml` with 14 actuated XL330 servos.
   - 3 consecutive flights: Flight 1 (2.5 cm risers), Flight 2 (3.0 cm risers), Flight 3 (3.5 cm risers) with landing platforms and top summit.
   - Gapless horizontal continuity from $x = 0.30$ m to $x = 2.58$ m with 0.0 gap or overlap.
   - Solid vertical grounding to $z = 0.0$ for all step geoms.
   - 3D contact physics: `condim="3"`, tangential friction `1.2`, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001 0.5 2"`.
   - 8 keyframes: `INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`.

3. **Test Results**:
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`:
     ```
     ============================= 21 passed in 10.71s ==============================
     ```
   - Complete project test suite `PYTHONPATH="" uv run --with pytest pytest tests/`:
     ```
     ======================= 271 passed, 6 skipped in 15.96s ========================
     ```

4. **Integrity Audit**:
   - Zero hardcoded test outputs or return shortcuts found.
   - Zero facade or dummy mock logic in source files.
   - Genuine physics calculations and PyTorch tensor operations.

---

## 2. Logic Chain

1. **From Observation 1**: All four penalty functions return negated non-negative costs (`-cost <= 0.0`, `-torch.sum(touchdown_cost) <= 0.0`, `-trip_cost <= 0.0`, `-per_foot.sum() <= 0.0`). Under the microduck sign convention documented in `AGENTS.md`, self-negating penalties return $\le 0.0$ and receive positive weights in task configs. This guarantees that `Episode_Reward/<term>` in wandb remains $\le 0.0$, eliminating potential reward farming exploits.
2. **From Observation 1**: The relative height potential $\Phi = \min(z_{\text{rel}}, \text{ceiling})$ tracks elevation relative to the lowest foot site. Because ascending stair steps lifts both trunk and feet synchronously, $z_{\text{rel}}$ remains constant at standing height (~0.117 m), keeping $\Phi$ saturated at the 0.115 m ceiling. Thus climbing stairs pays exactly 0.0 recovery potential, ensuring that traversal and recovery rewards do not double-count or cross-exploit.
3. **From Observation 1**: `_FEET_SITE_CFG` explicitly targets site names `("left_foot", "right_foot")` with indices `[1, 6]`. Empirical inspection of all 8 robot model XML files verified that `left_foot` is site index 1 and `right_foot` is site index 6 across all variants, preventing 7-site broadcast dimension mismatches with 2-primary sensors.
4. **From Observation 2**: `scene_stairs.xml` satisfies all physical requirements for stairs rehearsal: solid ground anchoring to $z=0$ prevents tunneling under treads, 0.12 m tread depth and 0.50 m tread width accommodate Microduck's footprint, and friction 1.2 with `condim=3` provides realistic traction.
5. **From Observation 3**: Both the targeted M1 test suite (21 passed) and the full repository test suite (271 passed, 0 failed) execute cleanly with zero regressions.
6. **From Observation 4**: No integrity violations were detected.

---

## 3. Caveats

- **Planar vs purely forward velocity in foot clearance**: `stairs_foot_clearance_penalty` checks horizontal velocity norm $\sqrt{v_x^2 + v_y^2}$ rather than solely forward velocity $v_x$. This is physically desirable to prevent foot scuffing during lateral balance corrections, but docstrings should note that planar swing speed is evaluated.
- **Actuator dynamics in standalone XML**: `scene_stairs.xml` includes `robot_groundcontact.xml` with XML position actuators ($k_p=0.55$) for standalone MuJoCo rendering and keyframe FK. Full BAM M6 voltage-actuation dynamics will be driven by `scripts/infer_stairs.py` in Milestone M3.
- **No caveats that impede progress or invalidate approval.**

---

## 4. Conclusion

Milestone M1 changes in `src/mjlab_microduck/tasks/mdp.py`, `src/mjlab_microduck/robot/microduck/scene_stairs.xml`, and their corresponding test suites `tests/test_scene_stairs.py` and `tests/test_stairs_mdp.py` fully satisfy all functional, structural, and architectural invariants.

**Verdict**: **APPROVE**  
Milestone M2 (Environment Configuration, Curriculum & Registration) and Milestone M3 (Standalone Rehearsal Runner) may proceed immediately.

---

## 5. Verification Method

To independently reproduce verification:
1. Run M1 unit test suite:
   ```bash
   PYTHONPATH="" uv run --with pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py
   ```
   *Expected outcome*: 21 passed.
2. Run full repository regression test suite:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   *Expected outcome*: 271 passed, 6 skipped, 0 failures.
3. Verify sign convention on penalties:
   Inspect `src/mjlab_microduck/tasks/mdp.py` lines 7741, 7773, 7803, 7835 to confirm all return values are $\le 0.0$.
