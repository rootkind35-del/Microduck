# M3 Forensic Auditor Dispatch

## Target Files
- `scripts/infer_stairs.py`
- `tests/test_infer_stairs.py`

## Mission
Perform a rigorous forensic integrity audit on Milestone M3:
1. Source inspection:
   - Inspect `scripts/infer_stairs.py` for genuine logic vs hardcoded test mocks, shortcuts, or fake returns.
   - Verify that BAM M6 motor conversion (`load_mujoco_with_bam`), stiff friction solver, and `MujocoController` are genuinely configured.
   - Verify that 61D observation extraction genuinely computes each of the 6 components without dummy constants.
   - Verify that balance telemetry genuinely calls `mujoco.mj_contactForce` and maps contacts to actual geom IDs in `scene_stairs.xml`.
   - Inspect `tests/test_infer_stairs.py`: are test cases genuine assertions, or are there tautological tests (`assert True`) or test bypasses?
2. Independent execution verification:
   - `PYTHONPATH="" uv run scripts/infer_stairs.py --help`
   - `PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug`
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py`
3. Conclude with an unambiguous forensic verdict:
   `CLEAN` or `INTEGRITY VIOLATION`.
   Write full evidence chains in `report.md` and verdict in `handoff.md`.


## 2026-10-11T19:49:16Z
You are M3 Forensic Auditor in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m3_auditor_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_auditor_1/DISPATCH.md`.

MANDATORY AUDIT RULES:
Audit Milestone M3 implementation for absolute integrity.
1. Source inspection:
   - Inspect `scripts/infer_stairs.py` for genuine logic vs hardcoded test mocks, shortcuts, or fake returns.
   - Verify that BAM M6 motor conversion (`load_mujoco_with_bam`), stiff friction solver, and `MujocoController` are genuinely configured.
   - Verify that 61D observation extraction genuinely computes each of the 6 components without dummy constants.
   - Verify that balance telemetry genuinely calls `mujoco.mj_contactForce` and maps contacts to actual geom IDs in `scene_stairs.xml`.
   - Inspect `tests/test_infer_stairs.py`: are test cases genuine assertions, or are there tautological tests (`assert True`) or test bypasses?
2. Independent execution verification:
   - `PYTHONPATH="" uv run scripts/infer_stairs.py --help`
   - `PYTHONPATH="" uv run scripts/infer_stairs.py --headless --debug`
   - `PYTHONPATH="" uv run --with pytest pytest tests/test_infer_stairs.py tests/test_stairs_e2e.py`
3. Conclude with an unambiguous forensic verdict:
   `CLEAN` or `INTEGRITY VIOLATION`.
   Write full evidence chains in `report.md` and verdict in `handoff.md`.
Notify the orchestrator via `send_message` when done.
