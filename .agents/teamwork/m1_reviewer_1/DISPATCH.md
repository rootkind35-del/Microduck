# DISPATCH — M1 Reviewer 1

**Task**: Code review and correctness verification for Milestone M1 (Core Stairs MDP Terms & Standalone Scene).
**Scope**:
- Examine changes to `src/mjlab_microduck/tasks/mdp.py` and new scene `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.
- Review unit tests in `tests/test_scene_stairs.py` and `tests/test_stairs_mdp.py`.
- Run tests: `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
- Verify that reward sign conventions are strictly adhered to (all penalties return <= 0.0).
- Produce verdict: `APPROVE` or `REQUEST_CHANGES` in `/home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_1/handoff.md`.


## 2026-10-11T15:18:18Z
You are M1 Reviewer 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_1/DISPATCH.md`.

Your objective is to conduct a code review of Milestone M1 changes:
1. Examine `src/mjlab_microduck/tasks/mdp.py` additions (the 5 stairs MDP functions and `_FEET_SITE_CFG`).
2. Examine `src/mjlab_microduck/robot/microduck/scene_stairs.xml`.
3. Examine `tests/test_scene_stairs.py` and `tests/test_stairs_mdp.py`.
4. Run tests: `PYTHONPATH="" uv run --with pytest pytest tests/test_scene_stairs.py tests/test_stairs_mdp.py`
5. Verify that all penalties return <= 0.0, and check interface conformance.

Write your review report to `/home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_1/report.md` and your verdict (`APPROVE` or `REQUEST_CHANGES`) in `/home/dhp01/microduck_rl/.agents/teamwork/m1_reviewer_1/handoff.md`.
Notify the orchestrator via `send_message` when done.
