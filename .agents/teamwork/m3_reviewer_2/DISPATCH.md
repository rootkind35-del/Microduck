# M3 Reviewer 2 Dispatch

## Target Files
- `scripts/infer_stairs.py`
- `tests/test_infer_stairs.py`

## Mission
Independently review physics fidelity, BAM M6 torque motor conversions, balance telemetry, and repository-wide backwards compatibility:
1. Check actuator dynamics: conversion of position motors to torque motors with force limit $\tau_{lim} = V_{in} \cdot k_t / R$, armature inertia setting, and dynamic friction solver in `MujocoController`.
2. Check balance telemetry: trunk pitch/roll tilt calculation, step contact identification per foot (`left_foot_collision` and `right_foot_collision`), normal contact forces via `mujoco.mj_contactForce`, knee/ankle torque load relative to 0.96 Nm.
3. Run full regression test suite:
   `PYTHONPATH="" uv run --with pytest pytest tests/`
4. Verify zero regressions across all 330+ tests.
5. Provide structured review in `report.md` and explicit verdict (`APPROVE` or `REQUEST_CHANGES`) in `handoff.md`.

## 2026-10-11T19:49:16Z
You are M3 Reviewer 2 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m3_reviewer_2

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_reviewer_2/DISPATCH.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_worker_1/report.md`.

Your objective is to independently review physics fidelity, BAM M6 torque motor conversions, balance telemetry, and repository-wide backwards compatibility:
1. Check actuator dynamics: conversion of position motors to torque motors with force limit tau_lim = Vin * kt / R, armature inertia setting, and dynamic friction solver in `MujocoController`.
2. Check balance telemetry: trunk pitch/roll tilt calculation, step contact identification per foot (`left_foot_collision` and `right_foot_collision`), normal contact forces via `mujoco.mj_contactForce`, knee/ankle torque load relative to 0.96 Nm.
3. Run full regression test suite:
   `PYTHONPATH="" uv run --with pytest pytest tests/`
4. Verify zero regressions across all 330+ tests.
5. Provide structured review in `report.md` and explicit verdict (`APPROVE` or `REQUEST_CHANGES`) in `handoff.md`.
Notify the orchestrator via `send_message` when done.

