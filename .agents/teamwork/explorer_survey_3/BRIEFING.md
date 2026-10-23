# BRIEFING — 2026-10-11T14:44:00Z

## Mission
Investigate standalone test simulation scenes (`scene_*.xml`), design `scene_stairs.xml` with consecutive stair flights (2.5cm, 3.0cm, 3.5cm), analyze CPU MuJoCo inference scripts (`infer_policy.py`), design `infer_stairs.py`, and inspect tests for validation.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3
- Original parent: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Milestone: survey_3_standalone_scene_and_infer_runner

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Zero breaking changes to existing codebase or scenes
- 61D observation layout invariant
- 14 BAM actuators (Dynamixel XL330)
- Sim2real transfer and 50 Hz control loop consistency

## Current Parent
- Conversation ID: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Updated: 2026-10-11T14:44:00Z

## Investigation State
- **Explored paths**:
  - `src/mjlab_microduck/robot/microduck/scene*.xml`, `stairs.xml`, `robot_walk.xml`, `robot_groundcontact.xml`
  - `src/mjlab_microduck/robot/microduck_constants.py`
  - `scripts/infer_policy.py`, `scripts/infer_interactive.py`
  - `tests/test_infer_policy_bam.py`, `tests/test_velstand_cfg.py`, `tests/test_slope_terrain.py`, `tests/test_fetch_cfg.py`
- **Key findings**:
  - `scene_stairs.xml` must include `robot_groundcontact.xml` to allow collision contact for battery, knees, and head during stair trips and self-recovery.
  - Stair geometries must be solid floor-anchored boxes with gapless boundaries ($sz = pz = H/2$).
  - BAM M6 CPU rehearsal requires pure Python controller without importing `mjlab` (prevents 16s torch/warp startup lag).
  - ROS 2 Humble packages in host environment interfere with pytest; must invoke `PYTHONPATH="" uv run --with pytest pytest tests/`.
- **Unexplored areas**: None (investigation complete).

## Key Decisions Made
- Designed 3 consecutive flights (2.5cm, 3.0cm, 3.5cm) with 3 steps each, 2 intermediate landing platforms, and a summit landing platform ($x=0.30$m to $2.58$m, $z=0$ to $0.27$m).
- Designed interactive runner `scripts/infer_stairs.py` with 5 spawn presets, keyboard controls, and real-time balance telemetry (tilt, contact forces, knee/ankle torque load).
- Defined test suites for `scene_stairs.xml`, `infer_stairs.py`, and `microduck_stairs_env_cfg.py`.

## Artifact Index
- DISPATCH.md — incoming dispatch instructions
- progress.md — liveness and step progress tracking
- report.md — comprehensive final survey report
- handoff.md — 5-component handoff report
- BRIEFING.md — persistent working memory
