# DISPATCH — M3 Worker

**Task**: Implement Milestone M3: Dedicated Standalone Rehearsal Runner & Interactive Telemetry (`scripts/infer_stairs.py` & `tests/test_infer_stairs.py`).

**Files Owned Exclusively**:
- `scripts/infer_stairs.py` (create standalone CPU rehearsal runner)
- `tests/test_infer_stairs.py` (create unit & invariant tests for the runner)

**Reference Artifacts**:
- `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/TEST_INFRA.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md` (§4 and §5)
- `/home/dhp01/microduck_rl/scripts/infer_policy.py` (reference for BAM M6 setup and viewer loop)

**Requirements**:
1. 50 Hz CPU loop with BAM M6 dynamics (decimation 4 @ 200 Hz / 0.005s physics).
2. Default scene: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.
3. 61D unified actor observation layout (48 proprio + 13 commands).
4. CLI options: `--policy`, `--scene`, `--spawn`, `--lin-vel-x`, `--vin`, `--headless`, `--debug`, `--telemetry-hz`.
5. 5 interactive spawn presets (hotkeys 1-5: bottom, flight1, flight2, flight3, top summit) matching `scene_stairs.xml` keyframes.
6. Real-time keyboard controls (velocity, pitch trim mode B, perturbation push P, standup trigger U, pause T, quit Q).
7. Real-time balance telemetry (pitch/roll tilt, foot contact identification per step, normal forces, knee/ankle torque load relative to 0.96 Nm stall limit) and in-viewer HUD overlay.
8. Unit tests in `tests/test_infer_stairs.py` verifying BAM constants, actuator conversion, 100-step headless simulation loop (0 NaNs), spawn presets, and telemetry.

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

**Verification Command**:
`PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py`
`PYTHONPATH="" uv run scripts/infer_stairs.py --help`
`PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug`
`PYTHONPATH="" uv run --with pytest pytest tests/`


## 2026-10-11T19:38:46Z
You are M3 Worker in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/DISPATCH.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/explorer_survey_3/report.md` (§4 and §5).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You own the following files exclusively:
1. `scripts/infer_stairs.py`:
   Implement the dedicated standalone deployment rehearsal runner:
   - 50 Hz control rate (decimation 4 @ 200 Hz physics, dt=0.005s).
   - BAM M6 motor dynamics on CPU (`load_mujoco_with_bam`, `MujocoController`).
   - Default scene: `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.
   - 61D unified actor observation layout (48 proprio + 13 commands: twist 3, head_pose 4, body_pose 6).
   - Interactive spawn presets (keys 1-5: bottom apron, flight1 2.5cm, flight2 3.0cm, flight3 3.5cm, top summit).
   - Real-time keyboard controls: UP/DOWN forward velocity, LEFT/RIGHT strafe, A/E yaw, SPACE zero, B body pitch trim, H head gaze, C heading lock, P perturbation push, U standup recovery, T toggle policy, Q quit.
   - Real-time balance telemetry: trunk pitch/roll tilt, step contact identification per foot (`stair_1_*`, `stair_2_*`, `stair_3_*`, `landing_*`), foot normal force, knee/ankle torque load relative to 0.96 Nm stall limit.
   - In-viewer HUD overlay via `viewer.set_texts`.
   - CLI flags: `--policy`, `--scene`, `--spawn`, `--lin-vel-x`, `--vin`, `--headless`, `--debug`, `--telemetry-hz`.
2. `tests/test_infer_stairs.py`:
   Implement unit tests for the runner:
   - `test_infer_stairs_bam_constants`: verify BAM constants match `_BAM_ACTUATOR_KWARGS`.
   - `test_infer_stairs_actuator_conversion`: verify torque motor conversion on `scene_stairs.xml`.
   - `test_infer_stairs_step_loop`: 100-step simulation with dummy policy and verify 0 NaNs.
   - `test_infer_stairs_spawn_presets`: verify all 5 spawn presets position the robot correctly.
   - `test_infer_stairs_telemetry`: verify tilt angle calculation, contact detection, and torque load reporting.

Verification:
Run:
`PYTHONPATH="" uv run scripts/infer_stairs.py --help`
`PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug`
`PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py`
`PYTHONPATH="" uv run --with pytest pytest tests/`

Document all commands, results, and file paths in `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/report.md` and `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
