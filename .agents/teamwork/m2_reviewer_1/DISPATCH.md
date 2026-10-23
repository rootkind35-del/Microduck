# DISPATCH — M2 Reviewer 1

**Task**: Code review and correctness verification for Milestone M2 (High Stairs RL Environment Config & Task Registration).
**Scope**:
- Review `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`.
- Review `src/mjlab_microduck/tasks/__init__.py`.
- Review `tests/test_stairs_cfg.py`.
- Run tests:
  `PYTHONPATH="" uv run list-envs | grep StairsClimb`
  `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`
- Verify that 61D observation layout is strictly preserved and reward sign conventions are obeyed.
- Produce verdict: `APPROVE` or `REQUEST_CHANGES` in `/home/dhp01/microduck_rl/.agents/teamwork/m2_reviewer_1/handoff.md`.

## 2026-10-11T19:27:16Z
You are M2 Reviewer 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m2_reviewer_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m2_reviewer_1/DISPATCH.md`.

Your objective is to conduct a code review of Milestone M2 changes:
1. Examine `src/mjlab_microduck/tasks/microduck_stairs_env_cfg.py`:
   - `make_microduck_stairs_env_cfg`, `MicroduckStairsRlCfg`.
   - 61D observation shape.
   - Reward weights: all self-negating penalties positive, all costs negative.
2. Examine `src/mjlab_microduck/tasks/__init__.py`:
   - Registration of `Mjlab-StairsClimb-MicroDuck` and backlash twin.
3. Examine `tests/test_stairs_cfg.py`.
4. Run tests:
   `PYTHONPATH="" uv run list-envs | grep StairsClimb`
   `PYTHONPATH="" uv run --with pytest pytest tests/test_stairs_cfg.py tests/test_stairs_e2e.py`

Write your review report to `/home/dhp01/microduck_rl/.agents/teamwork/m2_reviewer_1/report.md` and verdict (`APPROVE` or `REQUEST_CHANGES`) in `/home/dhp01/microduck_rl/.agents/teamwork/m2_reviewer_1/handoff.md`.
Notify the orchestrator via `send_message` when done.

