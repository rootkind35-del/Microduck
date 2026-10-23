# DISPATCH — M2 Worker

**Task**: Implement Milestone M2: High Stairs RL Environment Configuration, Progressive Curriculum & Task Registration.
**Files Owned Exclusively**:
- `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` (create standalone env config module)
- `src/mjlab_microduck/tasks/__init__.py` (register `Mjlab-StairsClimb-MicroDuck` and backlash twin)
- `tests/test_stairs_cfg.py` (create configuration invariant tests)

**Reference Artifacts**:
- `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/TEST_INFRA.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_challenger_1/handoff.md` (Note: `track_air_time=True` on `feet_ground_contact`)

**Key Invariants & Requirements**:
1. Base template: build on `make_microduck_velocity_env_cfg` (preserves DR, sensor noise, delay buffers, BAM startup events, NaN guards).
2. Robot model: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms).
3. Contact softening: `_soften_terrain_contacts` (`solref=[0.04, 1.0]`), `nconmax=200`, `iterations=30`, `ls_iterations=50`.
4. Observation layout: strict 61D unified actor layout (48D proprio + 13D commands).
5. Progressive curriculum:
   - Terrain generator with `BoxPyramidStairsTerrainCfg` (step_height_range: 2.5cm to 3.5cm, step_width: 0.18m).
   - Progressive stair height curriculum staging (2.5cm -> 3.0cm -> 3.5cm) via distance/step promotion.
6. Custom stairs MDP terms: wire the 5 functions implemented in M1 (`stairs_foot_clearance_penalty`, `stairs_foot_landing_damping_penalty`, `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, `stairs_relative_height_progress`).
7. Recovery layer: tilt-only gated potential rewards (`upright_progress`, `height_progress`), `FALLEN_SMOOTHNESS_SCALE = 0.1`, recovery economics (`fallen_tax`, `recovery_success`).
8. Register `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` in `tasks/__init__.py`.
9. Ensure `uv run list-envs` lists `Mjlab-StairsClimb-MicroDuck`.

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

**Verification Command**:
`PYTHONPATH="" uv run list-envs | grep StairsClimb`
`PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
`PYTHONPATH="" uv run --with pytest pytest tests/`


## 2026-10-11T15:26:16Z
You are M2 Worker in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m2_worker_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m2_worker_1/DISPATCH.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You own the following files exclusively:
1. `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
   Create the standalone task config module:
   - Factory function: `make_microduck_stairs_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
   - Runner config: `MicroduckStairsRlCfg` (RslRlOnPolicyRunnerCfg with experiment_name="stairs_climb", obs_normalization=True, hidden_dims=(512, 256, 128), etc.)
   - Base build: build on `make_microduck_velocity_env_cfg(play=play, rough=rough)` to inherit the full DR, obs noise, delay buffers, BAM startup events, and NaN guards.
   - Robot model: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms).
   - Solver & stability: `cfg.scene.spec_fn = _soften_terrain_contacts` (`solref=[0.04, 1.0]`), `cfg.sim.nconmax = 200`, `cfg.sim.mujoco.iterations = 30`, `cfg.sim.mujoco.ls_iterations = 50`.
   - Sensors: `feet_ground_contact` with `track_air_time=True`, `foot_height_scan`, `servo_ground_contact`, `head_ground_contact`, `trunk_ground_contact`.
   - Terrain & progressive curriculum:
     - `TerrainGeneratorCfg` with `BoxPyramidStairsTerrainCfg` (or straight stairs generator), step_height_range=(0.025, 0.035), step_width=0.18, platform_width=1.5.
     - Scheduled clearance targets and reward weights across iterations (2.5cm -> 3.0cm -> 3.5cm stages).
   - Custom stairs MDP terms: wire the 5 terms from M1 (`stairs_foot_clearance_penalty`, `stairs_foot_landing_damping_penalty`, `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, `stairs_relative_height_progress`).
   - Self-recovery layer: tilt-gated potential rewards (`upright_progress`, `height_progress`), `FALLEN_SMOOTHNESS_SCALE = 0.1`, recovery economics (`fallen_tax`, `recovery_success`).
   - Strict 61D observation layout preserved.
2. `src/mjlab_microduck/tasks/__init__.py`:
   - Import `make_microduck_stairs_env_cfg` and `MicroduckStairsRlCfg`.
   - Register `Mjlab-StairsClimb-MicroDuck`.
   - Register `("Mjlab-StairsClimb-Backlash-MicroDuck", make_microduck_stairs_env_cfg, {}, MicroduckStairsRlCfg, _BL_ALLCOLLISIONS)` in `_BACKLASH_TASKS`.
3. `tests/test_stairs_cfg.py`:
   - Test that `make_microduck_stairs_env_cfg(play=False)` and `play=True` compile cleanly.
   - Verify 61D actor observation dimension.
   - Verify that all penalty weights are positive and cost weights are negative.
   - Verify robot model is allcollisions.
   - Verify `feet_ground_contact` has `track_air_time=True`.
   - Verify tasks appear in registry.

Verification:
Run:
`PYTHONPATH="" uv run list-envs | grep StairsClimb`
`PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
`PYTHONPATH="" uv run --with pytest pytest tests/`

Document all commands, results, and file paths in `/home/dhp01/microduck_rl/.agents/teamwork/m2_worker_1/report.md` and `/home/dhp01/microduck_rl/.agents/teamwork/m2_worker_1/handoff.md`.
Notify the orchestrator via `send_message` when done.


## 2026-10-11T15:40:16Z
**Context**: Milestone M2 Implementation
**Content**: Heartbeat check: inspecting progress on `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`, `src/mjlab_microduck/tasks/__init__.py`, and `tests/test_stairs_cfg.py`.
**Action**: Please report current status and test execution progress.
