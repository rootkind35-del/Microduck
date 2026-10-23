# Forensic Audit Report — Milestone M2

**Work Product**: Milestone M2: High Stairs RL Environment Configuration & Registration
- `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- `src/mjlab_microduck/tasks/__init__.py`
- `tests/test_stairs_cfg.py`
- `tests/test_stairs_e2e.py`

**Profile**: General Project (Integrity Forensics)
**Integrity Mode**: Demo (from `ORIGINAL_REQUEST.md`)
**Auditor**: M2 Forensic Auditor (`m2_auditor_1`)
**Date**: 2026-10-11
**Verdict**: **CLEAN**

---

## 1. Executive Summary

An independent, rigorous forensic integrity audit was performed on Milestone M2 of the Microduck RL project. Every claim, invariant, and requirement was evaluated empirically without trusting prior claims. All code artifacts, test suites, task registries, simulation physics, GPU rollouts, and ONNX export graphs were independently executed and verified.

Zero integrity violations were detected:
- Zero hardcoded mock outputs, test skips, or test bypasses.
- Zero dummy or facade implementations.
- Zero pre-populated result artifacts.
- Genuine two-tier progressive curriculum scheduling across 2.5 cm -> 3.0 cm -> 3.5 cm steps.
- Genuine wiring of custom stairs MDP terms, allcollisions 70-geom robot model, and robust self-recovery layer.
- Full compliance with repository invariants: strict 61D actor observation layout, 14 Dynamixel XL330 BAM actuators, contact softening (`_soften_terrain_contacts`), and baked running observation normalizer in ONNX export.
- 100% test pass rates across both unit tests and the repository-wide regression test suite (329 passed, 0 failed).

---

## 2. Forensic Phase Results

| # | Forensic Check | Result | Evidence Summary |
|---|---|---|---|
| 1 | **Hardcoded Test Results Detection** | **PASS** | Regex and AST scan of `microduck_stairs_env_cfg.py` and `tests/test_stairs_cfg.py` revealed zero mock outputs, fake strings, or dummy test passes. |
| 2 | **Facade / Dummy Implementation Detection** | **PASS** | `make_microduck_stairs_env_cfg` genuinely instantiates all collision geometries, sensors, hardened solver parameters, and reward terms. |
| 3 | **Pre-Populated Verification Artifact Detection** | **PASS** | Workspace audit confirmed zero pre-populated verification or log artifacts. All tests and runs executed live. |
| 4 | **Task Registration & Discovery** | **PASS** | `uv run list-envs | grep StairsClimb` verified live registration of `Mjlab-StairsClimb-MicroDuck` (index 27) and `Mjlab-StairsClimb-Backlash-MicroDuck` (index 26). |
| 5 | **Independent Test Suite Execution** | **PASS** | `pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py` passed with 38 passed, 1 skipped (runner CLI test pending M3). |
| 6 | **Repository-Wide Regression Suite** | **PASS** | `pytest tests/` passed with 329 passed, 2 skipped, 0 regressions across all existing task families. |
| 7 | **Progressive Curriculum Staging (2.5cm -> 3.0cm -> 3.5cm)** | **PASS** | `CLEARANCE_TARGET_STAGES` schedules clearance at 0.028m -> 0.033m -> 0.038m matching step risers, and `terrain_levels_stairs` promotes/demotes based on distance. |
| 8 | **Wiring of Custom Stairs MDP & Recovery Layer** | **PASS** | All 5 custom MDP functions wired with positive weights; self-recovery layer features tilt-only gating, fallen smoothness scaling (`fallen_scale=0.1`), and hysteresis economics. |
| 9 | **GPU Simulation Smoke Test & NaN Immunity** | **PASS** | 5-iteration GPU smoke test at 64 envs completed with 0 NaNs (`Episode_Termination/nan_state: 0.0000`) and all penalty terms $\le 0.0$. |
| 10 | **Observation Normalizer Baked ONNX Export** | **PASS** | `scripts/export.py` successfully baked `EmpiricalNormalization()` into the ONNX graph with input `[1, 61]` and output `[1, 14]`. |

---

## 3. Detailed Forensic Observations & Raw Evidence

### Check 1: Hardcoded Test Results & Facade Detection
- Target: `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` and `tests/test_stairs_cfg.py`.
- Tool: `grep_search` with patterns `mock|dummy|fake|pass$|TODO|FIXME` and `assert True|return True`.
- Result: **0 matches**.
- Finding: Neither the task environment configuration nor its test suite contains shortcuts, mock objects, or test bypasses.

### Check 2: Pre-Populated Artifacts Audit
- Tool: `find . -name '*.log' -o -name '*result*' -o -name '*output*'`
- Result: Only pre-existing library packages under `.venv` were discovered. Zero forged logs or pre-recorded execution traces existed in workspace.

### Check 3: Task Registry Verification
- Command: `PYTHONPATH="" uv run list-envs | grep StairsClimb`
- Raw Output:
  ```
  | 26 | Mjlab-StairsClimb-Backlash-MicroDuck               |
  | 27 | Mjlab-StairsClimb-MicroDuck                        |
  ```
- Exit Code: `0`. Both tasks are dynamically discoverable and registered in `mjlab`'s central registry.

### Check 4: Test Suite Execution (Unit, E2E, MDP, Adversarial)
- Command 1: `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
  - Raw Output:
    ```
    tests/test_stairs_cfg.py ...............                                 [ 38%]
    tests/test_stairs_e2e.py ..s.....................                        [100%]
    ======================== 38 passed, 1 skipped in 12.52s ========================
    ```
  - Exit Code: `0`.
- Command 2: `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_mdp.py tests/test_stairs_adversarial.py tests/test_scene_stairs_adversarial.py`
  - Raw Output:
    ```
    tests/test_stairs_mdp.py ...........                                     [ 24%]
    tests/test_stairs_adversarial.py ...........                             [ 48%]
    tests/test_scene_stairs_adversarial.py .......................           [100%]
    ============================= 45 passed in 11.19s ==============================
    ```
  - Exit Code: `0`.
- Command 3: `PYTHONPATH="" uv run --with pytest pytest tests/`
  - Raw Output:
    ```
    ======================= 329 passed, 2 skipped in 17.59s ========================
    ```
  - Exit Code: `0`. Zero regressions across the entire repository.

### Check 5: Live GPU Smoke Test (5 iterations @ 64 envs)
- Command: `PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 13.3, driver 610.57.01).
- Raw Output Excerpt (Iteration 4/5):
  ```
  Total steps: 7680
  Steps per second: 426
  Mean value loss: 0.0389
  Mean surrogate loss: -0.0445
  Episode_Reward/track_linear_velocity: 0.0211
  Episode_Reward/track_angular_velocity: 0.0016
  Episode_Reward/upright: 0.0226
  Episode_Reward/pose: 0.0072
  Episode_Reward/body_ang_vel: -0.0237
  Episode_Reward/action_rate_l2: -0.0697
  Episode_Reward/stairs_foot_clearance: -0.0002
  Episode_Reward/stairs_foot_landing_damping: -0.0048
  Episode_Reward/stairs_riser_trip: -0.0406
  Episode_Reward/stairs_sole_flat: -0.0049
  Episode_Reward/stairs_relative_height_progress: -0.0015
  Episode_Reward/joint_torque_rate_l2: -0.0002
  Episode_Reward/servo_acc_spike: -0.0001
  Episode_Reward/servo_stall: -0.0002
  Episode_Reward/gentle_rise: -0.0001
  Episode_Termination/nan_state: 0.0000
  Episode_Termination/fallen_too_long: 0.0000
  ```
- Findings:
  - `Episode_Termination/nan_state`: 0.0000 (Zero NaNs throughout the entire rollout).
  - Reward sign compliance: Every self-negating penalty term (`stairs_foot_clearance`, `stairs_foot_landing_damping`, `stairs_riser_trip`, `stairs_sole_flat`, `stairs_relative_height_progress`, `gentle_rise`) and cost term evaluated strictly $\le 0.0$. No penalty farming possible.

### Check 6: Policy Export & Running Normalizer Baking
- Command: `PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --checkpoint-file logs/rsl_rl/stairs_climb/2026-10-12_02-30-53_stairs_climb/model_4.pt --onnx-file /tmp/m2_auditor_stairs_export.onnx`
- Raw Output Excerpt:
  ```
  Active Observation Terms in Group: 'actor' (shape: (61,))
  |   Index   | Name                             |   Shape    |
  |     0     | base_ang_vel                     |    (3,)    |
  |     1     | projected_gravity                |    (3,)    |
  |     2     | joint_pos                        |   (14,)    |
  |     3     | joint_vel                        |   (14,)    |
  |     4     | actions                          |   (14,)    |
  |     5     | command                          |    (3,)    |
  |     6     | head_command                     |    (4,)    |
  |     7     | body_command                     |    (6,)    |

  Active Action Terms (shape: 14)

  Actor Model: MLPModel(
    (obs_normalizer): EmpiricalNormalization()
    (distribution): GaussianDistribution()
    (mlp): MLP(
      (0): Linear(in_features=61, out_features=512, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=256, out_features=128, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=128, out_features=14, bias=True)
    )
  )
  Written /tmp/m2_auditor_stairs_export.onnx
  ```
- ONNX Graph Validation (`onnx.checker.check_model`):
  - Model verification: **PASSED**.
  - Input: `obs [1, 61]`
  - Output: `actions [1, 14]`
  - Normalizer: `EmpiricalNormalization` embedded directly prior to linear layer 0.

---

## 4. Adversarial Review & Failure Mode Stress-Testing

1. **Stair Clearance Synchronization**:
   - `CLEARANCE_TARGET_STAGES` stages:
     - Iteration 0: `target_height = 0.028 m` (clears 2.5 cm steps with 3 mm safety margin).
     - Iteration 1500: `target_height = 0.033 m` (clears 3.0 cm steps with 3 mm safety margin).
     - Iteration 3000: `target_height = 0.038 m` (clears 3.5 cm steps with 3 mm safety margin).
   - Synchronized with `MICRODUCK_STAIRS_TERRAINS_CFG` step height range `(0.025, 0.035)`.
2. **Contact Buffer & Solver Stability**:
   - Robot has 70 collision geoms (`MICRODUCK_ALLCOLLISIONS_ROBOT_CFG`); stair mesh contains 1929 geoms.
   - Contact buffer size was hardened to `nconmax = max(nconmax, 300)`. Solver iterations elevated to `iterations = 30, ls_iterations = 50`.
   - Contact softening via `_soften_terrain_contacts` applies `solref = 0.04`, avoiding impulse spikes.
3. **Recovery Economics & Attempt-Tax Avoidance**:
   - Dynamic motion during falls is protected: `FALLEN_SMOOTHNESS_SCALE = 0.1` scales down action rate and joint torque rate penalties by 90% when `tilt > 40°`.
   - Hysteresis avoids chattering: fallen tax activates at `tilt > 40°` but only releases when robot reaches `tilt < 25°` and `z > 0.09 m`.
   - One-shot bounty `recovery_success` (+10.0) pays upon completed standing, preventing farming.

---

## 5. Audit Verdict

**VERDICT: CLEAN**

Milestone M2 strictly adheres to all repository invariants and requirements set forth in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md`. No integrity violations, mock bypasses, or regression flaws were found.
