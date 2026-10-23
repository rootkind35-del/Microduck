# DISPATCH — M1 Worker

**Task**: Implement Milestone M1: Core MDP Terms & Standalone Simulation Scene.
**Files Owned Exclusively**:
- `src/mjlab_microduck/tasks/mdp.py` (append verified stairs MDP functions at end of file)
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml` (place verified standalone scene XML)
- `tests/test_scene_stairs.py` (place scene invariant tests)
- `tests/test_stairs_mdp.py` (place MDP unit tests)

**Reference Artifacts**:
- `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_spec_miner_1/report.md`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_scene_stairs.xml`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_test_scene_stairs.py`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_stairs_mdp_additions.py`
- `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_test_stairs_mdp.py`

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

**Verification Command**:
`PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
`PYTHONPATH="" uv run --with pytest pytest tests/`


## 2026-10-11T15:05:45Z
Sender: f3cd8a4f-6d90-4952-bb58-a5be7069a95c
Priority: MESSAGE_PRIORITY_HIGH
Content:
You are M1 Worker in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_worker_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m1_worker_1/DISPATCH.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You own the following files exclusively:
1. `src/mjlab_microduck/tasks/mdp.py`:
   Append the verified stairs MDP additions from `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_stairs_mdp_additions.py` under the `# Stairs Climbing Task` group banner at the end of the file.
   Ensure `_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))` is used for foot site lookups.
2. `src/mjlab_microduck/robot/microduck/scene_stairs.xml`:
   Create the scene file with the exact verified XML from `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_scene_stairs.xml`.
3. `tests/test_scene_stairs.py`:
   Create the test file with the exact verified tests from `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_scene_1/proposed_test_scene_stairs.py`.
4. `tests/test_stairs_mdp.py`:
   Create the test file with the exact verified tests from `/home/dhp01/microduck_rl/.agents/teamwork/m1_explorer_mdp_1/proposed_test_stairs_mdp.py`.

Verification:
Execute the tests using:
`PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
and verify no regressions:
`PYTHONPATH="" uv run --with pytest pytest tests/`

Document all commands, stdout/stderr, and file paths in `/home/dhp01/microduck_rl/.agents/teamwork/m1_worker_1/report.md` and `/home/dhp01/microduck_rl/.agents/teamwork/m1_worker_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
