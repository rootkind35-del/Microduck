# BRIEFING — 2026-10-11T19:48:00Z

## Mission
Implement Milestone M3: Dedicated Standalone Rehearsal Runner & Interactive Telemetry (`scripts/infer_stairs.py` & `tests/test_infer_stairs.py`).

## 🔒 My Identity
- Archetype: implementer
- Roles: implementer, qa, specialist
- Working directory: /home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1
- Original parent: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Milestone: M3 (Dedicated Standalone Rehearsal Runner & Interactive Telemetry)

## 🔒 Key Constraints
- Exclusive ownership: `scripts/infer_stairs.py` and `tests/test_infer_stairs.py`. Do NOT modify other files.
- 50 Hz control rate (decimation 4 @ 200 Hz physics, dt=0.005s).
- BAM M6 motor dynamics on CPU (`load_mujoco_with_bam`, `MujocoController`).
- Default scene: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.
- Strict 61D unified actor observation layout (48 proprio + 13 commands: twist 3, head_pose 4, body_pose 6).
- Interactive spawn presets (keys 1-5: bottom apron, flight1 2.5cm, flight2 3.0cm, flight3 3.5cm, top summit).
- Real-time keyboard controls: UP/DOWN forward velocity, LEFT/RIGHT strafe, A/E yaw, SPACE zero, B body pitch trim, H head gaze, C heading lock, P perturbation push, U standup recovery, T toggle policy, Q quit.
- Real-time balance telemetry: trunk pitch/roll tilt, step contact identification per foot (`stair_1_*`, `stair_2_*`, `stair_3_*`, `landing_*`), foot normal force, knee/ankle torque load relative to 0.96 Nm stall limit.
- In-viewer HUD overlay via `viewer.set_texts`.
- CLI flags: `--policy`, `--scene`, `--spawn`, `--lin-vel-x`, `--vin`, `--headless`, `--debug`, `--telemetry-hz`.
- Unit tests in `tests/test_infer_stairs.py` covering BAM constants, actuator conversion, 100-step loop (0 NaNs), spawn presets, telemetry.
- Zero breaking changes to existing codebase or test suites. All pytest tests must pass with `PYTHONPATH="" uv run --with pytest pytest tests/`.
- DO NOT CHEAT: Genuine implementations only, no hardcoded results or mock facades.

## Current Parent
- Conversation ID: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
- Updated: 2026-10-11T19:48:00Z

## Task Summary
- **What to build**: `scripts/infer_stairs.py` (interactive CPU rehearsal runner for high stairs climbing and recovery) and `tests/test_infer_stairs.py` (unit tests verifying runner invariants).
- **Success criteria**:
  - `PYTHONPATH="" uv run scripts/infer_stairs.py --help` runs cleanly: PASS
  - `PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug` runs 100+ steps with 0 NaNs: PASS
  - `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py` passes 100%: PASS (33 passed, 1 skipped)
  - `PYTHONPATH="" uv run --with pytest pytest tests/` passes 100%: PASS (334 passed, 2 skipped)
- **Interface contracts**: `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md` § Interface Contracts.
- **Code layout**: `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md` § Code Layout.

## Key Decisions Made
- Implemented `load_bam_model` and `load_mujoco_with_bam` in `scripts/infer_stairs.py` mirroring `infer_policy.py` and `BamActuator` physics (voltage control, torque motors, stiff friction solver, armature inertia).
- Retained `load_bam_controller` as a compatibility wrapper.
- Built comprehensive `SPAWN_PRESETS` covering both named keys ("bottom", "flight1", "flight2", "flight3", "top") and flight keys ("floor", "stairs_25", "stairs_30", "stairs_35", "summit", "landing_1", "landing_2", "init").
- Provided dual keyboard control architecture: asynchronous background `TerminalInput` (cbreak mode) for terminal control + GLFW key callback for viewer window control.
- Designed `compute_balance_telemetry` extracting real-time tilt, contact geom identification (`left_foot_collision` / `right_foot_collision` against step geoms), normal contact force via `mujoco.mj_contactForce`, and knee/ankle torque saturation relative to 0.96 Nm stall capacity.
- Wired native viewer HUD overlay via `viewer.set_texts` and terminal logging at configurable frequency.
- Configured `--headless --debug` to execute a 100-step (2.0s) verification run with zero NaNs when `--duration-s` is not explicitly provided.

## Artifact Index
- `/home/dhp01/microduck_rl/scripts/infer_stairs.py` — Standalone deployment rehearsal runner for stairs.
- `/home/dhp01/microduck_rl/tests/test_infer_stairs.py` — Unit tests for stairs runner.
- `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/report.md` — Detailed implementation and verification report.
- `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/handoff.md` — 5-component handoff report.
- `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/progress.md` — Heartbeat progress tracker.

## Change Tracker
- **Files modified**:
  - `scripts/infer_stairs.py`: Complete implementation of standalone CPU rehearsal runner.
  - `tests/test_infer_stairs.py`: Comprehensive test suite verifying all 5 required invariants plus backwards compatibility.
- **Build status**: PASS (All 334 tests pass cleanly).
- **Pending issues**: None.

## Quality Status
- **Build/test result**: PASS (334 passed, 2 skipped in 17.08s across entire test suite).
- **Lint status**: 0 violations.
- **Tests added/modified**: 10 unit test functions in `tests/test_infer_stairs.py`.

## Loaded Skills
- None required beyond project conventions.
