# Progress — Milestone M2 (High Stairs RL Environment Configuration)

Last visited: 2026-10-11T19:25:30Z

## Status: COMPLETED

### Completed
- [x] Initialized BRIEFING.md and DISPATCH.md
- [x] Read ORIGINAL_REQUEST.md, PROJECT.md, DISPATCH.md, explorer_survey_1/report.md, m1_challenger_1/handoff.md
- [x] Analyzed requirements and invariant constraints
- [x] Implemented `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
  - `make_microduck_stairs_env_cfg(play=False, rough=False)` factory inheriting from `make_microduck_velocity_env_cfg`.
  - `MicroduckStairsRlCfg` (RslRlOnPolicyRunnerCfg with experiment_name="stairs_climb", obs_normalization=True, hidden_dims=(512, 256, 128)).
  - Full-collision robot model `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms).
  - Solver hardening & contact softening: `_soften_terrain_contacts`, `nconmax=300`, `iterations=30`, `ls_iterations=50`.
  - Full contact sensor suite: `feet_ground_contact` with `track_air_time=True`, `foot_height_scan`, `servo_ground_contact`, `head_ground_contact`, `trunk_ground_contact`.
  - Procedural stair terrain generator: `BoxPyramidStairsTerrainCfg` (step_height_range: (0.025, 0.035), step_width: 0.18, platform_width: 1.5).
  - Wired 5 M1 custom stairs MDP terms with correct positive weight sign invariants.
  - Self-recovery layer: tilt-only gated potential rewards (`upright_progress`, `height_progress`), `FALLEN_SMOOTHNESS_SCALE = 0.1`, recovery economics (`fallen_tax`, `recovery_success`).
  - Strict 61D observation layout preserved (48 proprio + 13 commands).
- [x] Registered tasks in `src/mjlab_microduck/tasks/__init__.py`:
  - Registered `Mjlab-StairsClimb-MicroDuck`.
  - Registered `("Mjlab-StairsClimb-Backlash-MicroDuck", make_microduck_stairs_env_cfg, {}, MicroduckStairsRlCfg, _BL_ALLCOLLISIONS)` in `_BACKLASH_TASKS`.
- [x] Implemented `tests/test_stairs_cfg.py` with 15 comprehensive invariant and unit tests.
- [x] Verified `PYTHONPATH="" uv run list-envs | grep StairsClimb`:
  - Displays `Mjlab-StairsClimb-Backlash-MicroDuck` and `Mjlab-StairsClimb-MicroDuck`.
- [x] Verified `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`:
  - 38 passed, 1 skipped (only pending M3 runner skipped).
- [x] Verified `PYTHONPATH="" uv run --with pytest pytest tests/`:
  - 324 passed, 2 skipped (100% pass across repository).
- [x] Verified 5-iteration smoke test at 64 envs:
  - `PYTHONPATH="" uv run train Mjlab-StairsClimb-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5` completed with 0 NaNs and non-positive penalties.
- [x] Verified ONNX export with baked normalizer:
  - `PYTHONPATH="" uv run scripts/export.py Mjlab-StairsClimb-MicroDuck --checkpoint-file logs/rsl_rl/stairs_climb/2026-10-12_02-21-38_stairs_climb/model_4.pt --onnx-file /tmp/stairs_smoke.onnx` generated valid ONNX model.
