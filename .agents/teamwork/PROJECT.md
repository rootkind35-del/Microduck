# Project: Microduck High Stairs RL & Standalone Simulation Rehearsal

## Architecture
- **Framework**: `mjlab` (MuJoCo Warp GPU training at 50 Hz control / 200 Hz physics, decimation 4) with `rsl_rl` PPO runner.
- **Robot Model**: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (`robot_allcollisions.xml`, 70 collision geoms) for full-body contact physics on stairs.
- **Actuation**: 14 Dynamixel XL330 servos with BAM M6 actuator dynamics (voltage control, load-dependent friction, `expand_bam_friction_fields`).
- **Observation Contract**: Strict 61D unified actor observation layout (48D proprioception + 13D command block).
- **Testbed & Standalone Rehearsal**: Dedicated MJCF `scene_stairs.xml` with 3 flights (2.5cm, 3.0cm, 3.5cm) and standalone CPU runner `scripts/infer_stairs.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | MDP Stair Clearance Penalties | `stairs_foot_clearance_penalty`, `stairs_swing_clearance_penalty` in `src/mjlab_microduck/tasks/mdp.py` | M1 | Survey 2 |
| F2 | MDP Foot Landing Damping | `stairs_foot_landing_damping_penalty`, `stairs_foot_impact_penalty`, `stairs_foot_landing_velocity_penalty` in `mdp.py` | M1 | Survey 2 |
| F3 | MDP Contact Distribution & Anti-Tripping | `stairs_riser_trip_penalty`, `stairs_sole_flat_penalty`, `stairs_double_support_distribution_penalty` in `mdp.py` | M1 | Survey 2 |
| F4 | MDP Elevation-Invariant Recovery Potential | `stairs_relative_height_progress`, `stairs_upright_progress`, `stairs_fallen_state_penalty`, `stairs_recovery_success` in `mdp.py` | M1 | Survey 2 |
| F5 | Dedicated Standalone MJCF Scene | `src/mjlab_microduck/robot/microduck/scene_stairs.xml` with 3 flights (2.5cm, 3.0cm, 3.5cm risers), landing platforms, color materials, and keyframe poses | M1 | Survey 3 |
| F6 | High Stairs RL Environment Config | `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py` with `make_microduck_stairs_env_cfg`, `MicroduckStairsRlCfg`, `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG`, `_soften_terrain_contacts`, 61D obs layout, BAM friction DR | M2 | Survey 1 |
| F7 | Progressive Stair Height Curriculum | Two-tier curriculum: spatial terrain progression and step-staged clearance / reward scheduling across 2.5cm -> 3.0cm -> 3.5cm | M2 | Survey 1 |
| F8 | Task Registration & Backlash Twin | `Mjlab-StairsClimb-MicroDuck` and `Mjlab-StairsClimb-Backlash-MicroDuck` in `src/mjlab_microduck/tasks/__init__.py` | M2 | Survey 1 |
| F9 | Dedicated Standalone Rehearsal Runner | `scripts/infer_stairs.py` with 50 Hz CPU loop, BAM M6 dynamics, 5 spawn presets, keyboard velocity/pitch trim, and balance telemetry HUD | M3 | Survey 3 |
| F10 | Invariant Test Suite | Scene tests (`tests/test_scene_stairs.py`), runner tests (`tests/test_infer_stairs.py`), and task config tests (`tests/test_stairs_cfg.py`) | M1, M2, M3 | Survey 3 |
| F11 | Full E2E & Smoke Test Verification & Export | 5-iteration smoke test at 64 envs, baked normalizer ONNX export via `scripts/export.py`, 100% test pass via `PYTHONPATH="" uv run --with pytest pytest tests/` | M4 | Survey 1, 2, 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core MDP Terms & Standalone Simulation Scene | Implement custom stairs MDP functions in `src/mjlab_microduck/tasks/mdp.py` and `scene_stairs.xml` with tests in `tests/test_scene_stairs.py` | none | DONE |
| M2 | RL Environment Config, Curriculum & Registration | Implement `microduck_stairs_env_cfg.py`, register `Mjlab-StairsClimb-MicroDuck` (+ backlash twin) in `tasks/__init__.py`, with tests in `tests/test_stairs_cfg.py` | M1 | DONE |
| M3 | Standalone Rehearsal Runner & Interactive Telemetry | Implement `scripts/infer_stairs.py` with BAM dynamics, spawn presets, keyboard controls, telemetry HUD, with tests in `tests/test_infer_stairs.py` | M1 | DONE |
| M4 | Comprehensive E2E Verification & Export Hardening | Run 5-iteration smoke test at 64 envs, ONNX export with baked normalizer, complete test suite verification, and adversarial hardening | M1, M2, M3 | IN_PROGRESS |

## Interface Contracts
### `src/mjlab_microduck/tasks/mdp.py` ↔ `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- `stairs_foot_clearance_penalty(env, target_height=0.035, height_sensor_name="foot_height_scan", ...)` -> `torch.Tensor` (<= 0)
- `stairs_foot_landing_damping_penalty(env, sensor_name="feet_ground_contact", impact_force_thresh=8.0, ...)` -> `torch.Tensor` (<= 0)
- `stairs_riser_trip_penalty(env, sensor_name="feet_ground_contact", max_friction_ratio=0.5, ...)` -> `torch.Tensor` (<= 0)
- `stairs_sole_flat_penalty(env, sensor_name="feet_ground_contact", ...)` -> `torch.Tensor` (<= 0)
- `stairs_relative_height_progress(env, ceiling=0.115, ...)` -> `torch.Tensor` (delta Phi)

### `src/mjlab_microduck/robot/microduck/scene_stairs.xml` ↔ `scripts/infer_stairs.py`
- Keyframe names: `INIT`, `STAND`, `SPAWN_STAIRS_25`, `SPAWN_LANDING_1`, `SPAWN_STAIRS_30`, `SPAWN_LANDING_2`, `SPAWN_STAIRS_35`, `SPAWN_SUMMIT`.
- Geom naming: `stair_1_1..3`, `landing_1`, `stair_2_1..3`, `landing_2`, `stair_3_1..3`, `landing_summit`.

## Code Layout
- MDP additions: `src/mjlab_microduck/tasks/mdp.py` (under `# Stairs Climbing Task` group)
- Scene file: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- Env config: `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
- Task registration: `src/mjlab_microduck/tasks/__init__.py`
- Inference script: `scripts/infer_stairs.py`
- Unit tests: `tests/test_scene_stairs.py`, `tests/test_stairs_cfg.py`, `tests/test_infer_stairs.py`
