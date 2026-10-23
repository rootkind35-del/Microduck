# BRIEFING — 2026-10-11T14:49:00Z

## Mission
Investigate existing RL environment configs, curricula, and registration patterns to plan `microduck_stairs_env_cfg.py` and task registration in `tasks/__init__.py`.

## 🔒 My Identity
- Archetype: explorer
- Roles: [explorer, investigator, analyst]
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1
- Original parent: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Milestone: survey and plan microduck_stairs_env_cfg.py & task registration

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Zero breaking changes to existing codebase
- Strict adherence to 61D observation layout and BAM actuator invariants
- Output comprehensive report to /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md

## Current Parent
- Conversation ID: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Updated: 2026-10-11T14:48:03Z

## Investigation State
- **Explored paths**:
  - `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
  - `src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`
  - `src/mjlab_microduck/tasks/microduck_standup_env_cfg.py`
  - `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`
  - `src/mjlab_microduck/tasks/slope_terrain.py`
  - `src/mjlab_microduck/tasks/__init__.py`
  - `src/mjlab_microduck/tasks/backlash.py`
  - `src/mjlab_microduck/tasks/mdp.py`
  - `src/mjlab_microduck/robot/microduck/stairs.xml`
  - `mjlab.terrains` (`primitive_terrains.py`, `terrain_generator.py`, `terrain_entity.py`)
  - `scripts/export.py` & `src/mjlab_microduck/export.py`
- **Key findings**:
  - Full architecture designed: build on `make_microduck_velocity_env_cfg` + `velstand` recovery layer.
  - Robot model: `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` (70 collision geoms).
  - Terrain: Procedural `TerrainGeneratorCfg` with `BoxPyramidStairsTerrainCfg` for GPU training; `scene_stairs.xml` for CPU rehearsal.
  - Solver hardening: `_soften_terrain_contacts` (`solref=[0.04, 1.0]`), `nconmax=200`, `iterations=30`, `ls_iterations=50`.
  - Curriculum: Two-tier structure (spatial terrain difficulty progression + temporal step-staged clearance & weight schedules at `iteration * 24`).
  - Recovery: Tilt-only gating (`tilt > 40°`), potential formulation (`upright_progress`, `height_progress`), smoothness tax relief (`FALLEN_SMOOTHNESS_SCALE = 0.1`), recovery economics (`fallen_tax`, `recovery_success`).
  - Task ID: `Mjlab-StairsClimb-MicroDuck` registered in `tasks/__init__.py` with backlash twin and `MicroduckOnPolicyRunner`.
- **Unexplored areas**: None (all survey questions answered in depth).

## Key Decisions Made
- Use `MICRODUCK_ALLCOLLISIONS_ROBOT_CFG` to ensure full physical contact on stair risers and treads during falls and self-recovery.
- Use origin-relative spawn coordinates (`_env_origin_z`) to prevent robots spawning inside elevated stair steps.
- Gated recovery strictly on `tilt > 40°` to prevent sitting/squatting exploitation.

## Artifact Index
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/report.md` — Comprehensive technical survey and architecture plan
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/handoff.md` — 5-component handoff report
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/progress.md` — Liveness and progress tracker
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_1/DISPATCH.md` — Incoming dispatch messages log
