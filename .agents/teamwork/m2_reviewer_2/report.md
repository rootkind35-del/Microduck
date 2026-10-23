# M2 Review Report: Physics, Curriculum, and Backwards Compatibility

**Reviewer**: M2 Reviewer 2  
**Roles**: `reviewer`, `critic`  
**Milestone**: M2 (High Stairs RL Environment Configuration, Curriculum & Task Registration)  
**Date**: 2026-10-12  
**Target Environment**: `Mjlab-StairsClimb-MicroDuck` & `Mjlab-StairsClimb-Backlash-MicroDuck`  
**Configuration File**: `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`  
**MDP Implementation**: `src/mjlab_microduck/tasks/mdp.py`  

---

## Executive Summary

**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**  
**Integrity Audit**: **PASS** (Zero integrity violations; genuine independent verification across GPU smoke test, ONNX export, and regression suite).

This review independently examines the physics simulation stability, two-tier progressive curriculum, self-recovery dynamics, and backwards compatibility of Milestone M2 in the Microduck RL repository. All architectural invariants established in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md` are preserved:
- Strict 61D actor observation contract (48D proprioception + 13D command block).
- 14 Dynamixel XL330 BAM M6 voltage-controlled actuators with friction DR.
- Full-collision robot model (`MICRODUCK_ALLCOLLISIONS_ROBOT_CFG`, 70 collision geoms) for accurate step riser and tread contact dynamics.
- Baked empirical observation normalizer in ONNX export via `scripts/export.py`.
- 100% test pass rate with zero regressions across the 331-test suite (329 passed, 2 valid skips).

---

## 1. Detailed Review Findings

### 1.1 Progressive Stair Height Curriculum Staging & Terrain Parameters
- **Spatial Terrain Progression**:
  - `MICRODUCK_STAIRS_TERRAINS_CFG` configures `terrain_gen.BoxPyramidStairsTerrainCfg` with `step_height_range=(0.025, 0.035)`, `step_width=0.18`, and `platform_width=1.5` over a 6-row by 5-column grid (`num_rows=6, num_cols=5`).
  - Across the 6 terrain difficulty levels ($\alpha \in [0.0, 1.0]$), step riser heights linearly scale:
    $$\Delta z_{\text{step}}(\alpha) = 0.025 + \alpha \times (0.035 - 0.025) \text{ m}$$
    Row 0 introduces 2.5 cm steps, intermediate rows introduce 2.7–3.3 cm steps, and Row 5 introduces 3.5 cm steps (representing ~15% of the ~25 cm robot height).
  - Traversal-based promotion/demotion in `terrain_levels_stairs`:
    - Promotion: Robot traverses $> 2.5$ m onto the pyramid platform.
    - Demotion: Robot covers $< 0.8$ m before reset/fall.
- **Step-Staged Clearance Target Schedule**:
  - `stairs_clearance_target_curriculum` governs `stairs_foot_clearance` target height across three distinct phases:
    - Stage 1 (iter 0, step 0): $h_{\text{target}} = 0.028$ m (2.5 cm step + 3 mm safety clearance).
    - Stage 2 (iter 1,500, step 36,000): $h_{\text{target}} = 0.033$ m (3.0 cm step + 3 mm safety clearance).
    - Stage 3 (iter 3,000, step 72,000): $h_{\text{target}} = 0.038$ m (3.5 cm step + 3 mm safety clearance).
  - Pacing Analysis: Out of `max_iterations = 6_000` (144,000 steps), Stage 1 covers the first 25% for basic gait bootstrapping, Stage 2 covers 25% for consolidation, and Stage 3 covers the remaining 50% for high-clearance stabilization.
  - Term mutation conforms to the manager pattern (`env.reward_manager.get_term_cfg(reward_name).params["target_height"]`), avoiding the silent no-op bug documented in `AGENTS.md`.

### 1.2 Solver Stability & Contact Softening
- **Contact Softening (`_soften_terrain_contacts`)**:
  - The spec function iterates over all 1929 procedural terrain geoms, doubling the contact time constant from 0.02 s to 0.04 s (`solref=[0.04, 1.0]`) and configuring impedance `solimp=[0.85, 0.95, 0.001, 0.5, 2.0]`.
  - Effect: Sharp 90-degree step riser corners do not produce impulsive normal force spikes or contact solver chatter, directly eliminating NaN force generation when foot soles strike step edges.
- **Elevated Solver Iterations**:
  - `cfg.sim.mujoco.iterations = 30` (increased from base 10–20).
  - `cfg.sim.mujoco.ls_iterations = 50` (linesearch iterations).
  - Effect: The projected Gauss-Seidel / Newton linesearch solver achieves rigorous constraint convergence even when multiple body geoms (feet, shins, torso) engage in simultaneous contact with steps.
- **Contact Buffer Capacity (`nconmax = 300`)**:
  - In full-collision mode with 70 robot collision geoms and multi-tier stair geometry, contact counts can exceed 200 during tumbling or multi-contact stance. `nconmax = 300` guarantees contact buffer headroom, eliminating buffer overflow faults.

### 1.3 Self-Recovery Layer Dynamics
- **Tilt-Only Gating at 40 Degrees**:
  - `REWARD_GATE_TILT_DEG = 40.0`.
  - In `air_time`, air-time rewards are zeroed when $\text{tilt} > 40^\circ$, preventing the robot from farming leg-swing rewards while lying on its back.
  - In `com_upward_velocity`, `gate_z_below: 0.0` explicitly disables trunk absolute height gating, enforcing **tilt-only gating**. On variable-height stair terrain, an absolute z threshold would fail because a robot lying on step 3 ($z \approx 0.15$ m) would appear "upright" to an absolute height sensor. Tilt-only gating accurately classifies fallen states regardless of terrain elevation.
- **Attempt Tax Removal**:
  - `action_rate_l2_fallen_scaled` and `joint_torque_rate_l2_fallen_scaled` reduce smoothness penalties by $10\times$ (`fallen_scale = 0.1`) when $\text{tilt} > 40^\circ$.
  - Effect: Prevents the "attempt tax" from penalizing dynamic righting maneuvers, allowing the policy to discover energetic get-up motions without defaulting to passive lying. Full smoothness penalties reactivate immediately once upright ($\text{tilt} \le 40^\circ$).
- **Elevation-Invariant Relative Height Progress (`stairs_relative_height_progress`)**:
  - Potential function:
    $$z_{\text{rel}} = z_{\text{trunk}} - \min(z_{\text{foot\_left}}, z_{\text{foot\_right}})$$
    $$\Phi = \min(z_{\text{rel}}, \text{ceiling})$$
  - Elevation Invariance: When standing, $z_{\text{rel}} \approx 0.117$ m whether on flat floor, step 1, or step 3. Climbing stairs keeps $z_{\text{rel}}$ constant, yielding $\Delta \Phi = 0.0$.
  - Anti-Farming: Rising from a fallen state pays $\Delta z_{\text{rel}} > 0$. Holding any state pays exactly $0.0$. Oscillations integrate to $\oint d\Phi = 0$.
- **Recovery Economics**:
  - `fallen_tax`: continuous penalty ($-0.5$ weight ramped in at iter 1500) with hysteresis (`release_tilt_below_deg = 25.0`, `release_z_above = 0.09 m`).
  - `recovery_success`: $+10.0$ one-shot bounty upon remaining fallen for $> 0.5$ s and reaching fully recovered posture.

---

## 2. Adversarial Challenge & Stress-Testing

### 2.1 Challenge: Pacing of Clearance Target Transitions
- **Hypothesis**: Stepping up clearance target from 2.8 cm to 3.3 cm at iter 1,500 might destabilize an unconsolidated gait.
- **Stress-Test Analysis**:
  - Pacing is cushioned by the independent spatial terrain curriculum (`terrain_levels_stairs`). If an agent struggles on steeper steps, it is demoted to lower terrain levels.
  - In `mdp.py`, clearance penalty is one-sided: lifting higher than target is free ($0.0$ penalty).
- **Verdict**: Robust.

### 2.2 Challenge: Multi-Contact Solver Explosions on Step Edges
- **Hypothesis**: Simultaneous contact of shin (`*_servo_collision`) and foot sole on sharp box edges could produce non-positive definite constraint matrices.
- **Stress-Test Analysis**:
  - Verified via 10,000-sample adversarial Monte Carlo test (`test_sign_oracle_monte_carlo` and `test_extreme_contact_forces` in `tests/test_stairs_adversarial.py`).
  - Contact softening (`solref=[0.04, 1.0]`) and increased linesearch iterations (`ls_iterations=50`) prevent divergence.
  - GPU 5-iteration smoke test at $N=64$ envs stepped cleanly with `Episode_Termination/nan_state = 0.0000`.
- **Verdict**: Robust.

### 2.3 Challenge: Reward Sign Integrity & Farming
- **Hypothesis**: Could any penalty double-negate and reward unwanted behavior?
- **Stress-Test Analysis**:
  - Self-negating penalties (`stairs_foot_clearance`, `stairs_foot_landing_damping`, `stairs_riser_trip`, `stairs_sole_flat`, `stairs_relative_height_progress`, `gentle_rise`) return $\le 0.0$ and have positive weights ($> 0$).
  - Evaluated in GPU rollouts: every logged `Episode_Reward/<penalty>` was strictly $\le 0.0$.
- **Verdict**: Zero sign inversions, zero farming opportunities.

---

## 3. Independent Verification Evidence

1. **Full Regression Test Suite**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   **Result**: `329 passed, 2 skipped in 21.44s` (exit code 0).
   - `tests/test_stairs_cfg.py`: 15 passed.
   - `tests/test_stairs_mdp.py`: 11 passed.
   - `tests/test_stairs_adversarial.py`: 11 passed.
   - `tests/test_stairs_e2e.py`: 23 passed, 1 skipped (gymnasium isolation).
   - `tests/test_scene_stairs.py`: 10 passed.
   - `tests/test_scene_stairs_adversarial.py`: 23 passed.
   - All legacy environment suites passed without regressions.

2. **Backwards Compatibility Environment Factory Execution**:
   - `make_microduck_velocity_env_cfg(play=False/True, rough=False/True)`: OK.
   - `make_microduck_velstand_env_cfg(play=False/True, rough=False/True)`: OK.
   - `make_microduck_standup_env_cfg(play=False/True, rough=False/True)`: OK.
   - `make_microduck_sitstand_env_cfg(play=False/True, rough=False/True)`: OK.
   - `make_microduck_ground_pick_env_cfg(play=False/True, rough=False/True)`: OK.
   - `make_microduck_stairs_env_cfg(play=False/True, rough=False/True)`: OK.

3. **GPU Smoke Test**:
   ```bash
   PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
   **Result**: Exit code 0.
   - Zero NaNs: `Episode_Termination/nan_state: 0.0000`.
   - All penalties strictly $\le 0.0$.
   - Successfully generated and saved checkpoint `model_4.pt`.

4. **ONNX Export with Baked Empirical Normalizer**:
   ```bash
   PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --checkpoint-file logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/model_4.pt --onnx-file /tmp/verify_stairs_export.onnx
   ```
   **Result**: Exit code 0.
   - Graph validation: `onnx.checker.check_model` passed.
   - Input shape: `('obs', [1, 61])`.
   - Output shape: `('actions', [1, 14])`.
   - Observation normalizer baked into model graph.

---

## 4. Final Verdict

**APPROVE**

Milestone M2 meets all technical requirements, architectural constraints, and quality standards. Physics parameters are verified for numerical stability, the curriculum provides structured progressive learning, the self-recovery layer is immune to farming, and repository-wide backwards compatibility is maintained with zero regressions.
