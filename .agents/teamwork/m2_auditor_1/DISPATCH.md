# DISPATCH — M2 Forensic Auditor

**Task**: Forensic integrity audit of Milestone M2 changes.
**Scope**:
- Examine all files modified or created in Milestone M2:
  - `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
  - `src/mjlab_microduck/tasks/__init__.py`
  - `tests/test_stairs_cfg.py`
- Verify integrity against the Forensic Integrity rules:
  - Zero hardcoding of test outputs or mock shortcuts.
  - Zero dummy or facade implementations.
  - Genuine progressive curriculum scheduling (2.5cm -> 3.0cm -> 3.5cm).
  - Genuine wiring of custom stairs MDP terms and self-recovery layer.
  - Running normalizer compatibility in export.
- Run tests independently:
  `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
- Report verdict: `CLEAN` or `INTEGRITY VIOLATION` in `/home/dhp01/microduck_rl/.agents/teamwork/m2_auditor_1/handoff.md`.


## 2026-10-11T19:27:16Z
You are M2 Forensic Auditor in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m2_auditor_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m2_auditor_1/DISPATCH.md`.

Your objective is to perform a forensic integrity audit on Milestone M2:
1. Audit source files:
   - `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`
   - `src/mjlab_microduck/tasks/__init__.py`
   - `tests/test_stairs_cfg.py`
2. Forensic checks:
   - Zero hardcoded mock outputs or test bypasses.
   - Genuine progressive stair height curriculum staging (2.5cm -> 3.0cm -> 3.5cm).
   - Genuine wiring of custom stairs MDP terms, allcollisions robot model, and recovery layer.
   - Running observation normalizer compatibility in export.
3. Run verification independently:
   `PYTHONPATH="" uv run list-envs | grep StairsClimb`
   `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
4. Report verdict:
   Must conclude with either `CLEAN` or `INTEGRITY VIOLATION`.

Write full report with evidence chains to `/home/dhp01/microduck_rl/.agents/teamwork/m2_auditor_1/report.md` and verdict in `/home/dhp01/microduck_rl/.agents/teamwork/m2_auditor_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
