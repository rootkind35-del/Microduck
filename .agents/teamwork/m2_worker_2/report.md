# Milestone M2 Implementation Report: High Stairs RL Environment Configuration & Registration

**Author**: M2 Worker (Replacement)  
**Date**: 2026-10-11  
**Working Directory**: `/home/dhp01/microduck_rl/.agents/teamwork/m2_worker_2`  
**Status**: COMPLETE  

---

## 1. Executive Summary

Milestone M2 establishes the complete, production-ready Reinforcement Learning training environment configuration (`microduck_stairs_env_cfg.py`) for Microduck high stairs navigation (2.5 cm -> 3.0 cm -> 3.5 cm risers, up to ~15% of the ~25 cm robot height), dynamic posture balance against topple disturbances, and self-recovery when fallen.

All requirements and repository invariants from `AGENTS.md`, `PROJECT.md`, and `ORIGINAL_REQUEST.md` were rigorously preserved:
1. **Inheritance from Base Locomotion**: Built directly on `make_microduck_velocity_env_cfg(play=play, rough=rough)` to inherit domain randomization, sensor noise, delay buffers, BAM startup events, and NaN guards without code duplication.
2. **Full-Collision Model**: Reconfigured robot entity to `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (`robot_allcollisions.xml`, 70 collision geoms) to enable realistic physical contacts for shins/knees against stair risers and full-body ground interaction during falls.
3. **Contact Solver Stability & Edge Softening**: Configured `cfg.scene.spec_fn = _soften_terrain_contacts` (`solref=[0.04, 1.0]`), `cfg.sim.nconmax = 300` (hardened against buffer overflow across multi-geom stair pile-ups and single-env ONNX export), `cfg.sim.mujoco.iterations = 30`, and `cfg.sim.mujoco.ls_iterations = 50`.
4. **Sensor Suite**: Retained `feet_ground_contact` with `track_air_time=True` (inherited from base velocity recipe) and configured `foot_height_scan`, `servo_ground_contact`, `head_ground_contact`, and `trunk_ground_contact`.
5. **Procedural Stair Terrain**: Configured `TerrainGeneratorCfg` with `BoxPyramidStairsTerrainCfg` (step_height_range: 2.5 cm to 3.5 cm, step_width: 0.18 m, platform_width: 1.5 m, border_width: 1.0 m) and distance-based promotion curriculum (`terrain_levels_stairs`).
6. **M1 Custom Stairs MDP Terms**: Wired all 5 custom stairs terms (`stairs_foot_clearance_penalty`, `stairs_foot_landing_damping_penalty`, `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, `stairs_relative_height_progress`) with positive weights respecting the Microduck self-negating sign convention ($\le 0 \implies \text{weight} > 0$).
7. **Dynamic Self-Recovery**: Integrated tilt-only gated potential rewards (`upright_progress`, `height_progress`), `FALLEN_SMOOTHNESS_SCALE = 0.1` (`action_rate_l2_fallen_scaled`, `joint_torque_rate_l2_fallen_scaled`), and recovery economics (`fallen_tax` with hysteresis, `recovery_success` bounty).
8. **Registration & Backlash Twin**: Registered `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` (with `_BL_ALLCOLLISIONS`) in `src/mjlab_microduck/tasks/__init__.py`.
9. **Strict 61D Observation Invariant**: Actor observation shape is verified at exactly 61D (48 proprioception + 13 commands) and 76D critic.
10. **Runner Config**: Defined `MicroduckStairsRlCfg` (`experiment_name="stairs_climb"`, `obs_normalization=True`, `hidden_dims=(512, 256, 128)`, `num_steps_per_env=24`, `max_iterations=6_000`).

---

## 2. File Artifacts Created & Modified

### `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- Factory function: `make_microduck_stairs_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
- Runner configuration: `MicroduckStairsRlCfg`
- Constants:
  - `STAIR_STEP_HEIGHT_MIN = 0.025`, `STAIR_STEP_HEIGHT_MAX = 0.035`, `STAIR_STEP_WIDTH = 0.18`, `STAIR_PLATFORM_WIDTH = 1.5`
  - `CLEARANCE_TARGET_STAGES = [{'step': 0, 'target_height': 0.028}, {'step': 36000, 'target_height': 0.033}, {'step': 72000, 'target_height': 0.038}]`
  - `FALLEN_SMOOTHNESS_SCALE = 0.1`, `REWARD_GATE_TILT_DEG = 40.0`, `RECOVERED_UP_TILT_DEG = 25.0`, `RECOVERED_UP_Z = 0.09`
- Procedural terrain generator: `MICRODUCK_STAIRS_TERRAINS_CFG`
- Solver hardening: `nconmax = max(cfg.sim.nconmax or 0, 300)`

### `src/mjlab_microduck/tasks/__init__.py`
- Registered base task:
  ```python
  register_mjlab_task(
      task_id="Mjlab-StairsClimb-MicroDuck",
      env_cfg=make_microduck_stairs_env_cfg(),
      play_env_cfg=make_microduck_stairs_env_cfg(play=True),
      rl_cfg=MicroduckStairsRlCfg,
      runner_cls=MicroduckOnPolicyRunner,
  )
  ```
- Registered backlash twin in `_BACKLASH_TASKS`:
  ```python
  ("Mjlab-StairsClimb-Backlash-MicroDuck", make_microduck_stairs_env_cfg, {}, MicroduckStairsRlCfg, _BL_ALLCOLLISIONS),
  ```

### `tests/test_stairs_cfg.py`
- Comprehensive suite of 15 tests covering:
  - Compilation of train, play, and rough configs
  - Strict 61D actor observation contract and slot ordering matching velocity template
  - Verification of `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` with $\ge 70$ collision geoms and 14 servo housing geoms
  - `track_air_time=True` on `feet_ground_contact`
  - Sensor suite configuration (`feet_ground_contact`, `foot_height_scan`, `servo_ground_contact`, `head_ground_contact`, `trunk_ground_contact`)
  - Solver hardening (`nconmax >= 200`, `iterations >= 30`, `ls_iterations >= 50`, `_soften_terrain_contacts`)
  - Pyramid stairs terrain generator dimensions and parameters
  - M1 MDP term wiring and positive/negative reward sign invariants
  - Self-recovery layer (tilt gating, fallen scale 0.1, hysteresis, economics)
  - Scheduled clearance curriculum stages
  - Task presence in registry and backlash twin allcollisions model mapping
  - Runner configuration attributes

---

## 3. Verification Commands & Results

1. **Task Registry Visibility**:
   ```bash
   PYTHONPATH="" uv run list-envs | grep StairsClimb
   ```
   Output:
   ```
   | 26 | Mjlab-StairsClimb-Backlash-MicroDuck               |
   | 27 | Mjlab-StairsClimb-MicroDuck                        |
   ```
   *Exit code: 0.*

2. **Milestone Cfg & E2E Unit Tests**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py
   ```
   Output:
   ```
   38 passed, 1 skipped in 8.21s
   ```
   *Exit code: 0.*

3. **Full Repository Regression Suite**:
   ```bash
   PYTHONPATH="" uv run --with pytest pytest tests/
   ```
   Output:
   ```
   324 passed, 2 skipped in 14.73s
   ```
   *Exit code: 0. Zero regressions across the entire codebase.*

4. **5-Iteration Smoke Test at 64 Envs**:
   ```bash
   PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
   ```
   Output:
   - Completed all 5 iterations cleanly.
   - `Episode_Termination/nan_state: 0.0000` (Zero NaNs).
   - All penalty terms logged non-positive values:
     - `Episode_Reward/stairs_foot_clearance`: -0.0003
     - `Episode_Reward/stairs_foot_landing_damping`: -0.0049
     - `Episode_Reward/stairs_riser_trip`: -0.0427
     - `Episode_Reward/stairs_sole_flat`: -0.0058
     - `Episode_Reward/stairs_relative_height_progress`: -0.0015
     - `Episode_Reward/servo_impact`: -0.0000
     - `Episode_Reward/head_impact`: -0.0000
     - `Episode_Reward/trunk_impact`: -0.0000
   *Exit code: 0.*

5. **ONNX Export with Baked Normalizer**:
   ```bash
   PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --checkpoint-file logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/model_4.pt --onnx-file /tmp/stairs_smoke.onnx
   ```
   Output:
   - Input shape: 61D actor observation.
   - Output shape: 14D joint action.
   - Running normalizer baked: `(obs_normalizer): EmpiricalNormalization()`.
   - Successfully wrote `/tmp/stairs_smoke.onnx`.
   *Exit code: 0.*
