# M3 Challenger 2 Dispatch

## Target Files
- `scripts/infer_stairs.py`
- `src/mjlab_microduck/robot/microduck/scene_stairs.xml`
- `tests/test_infer_stairs.py`

## Mission
Adversarially challenge BAM M6 actuator dynamics, physics limits, and perturbation response in `scripts/infer_stairs.py`:
1. Actuator voltage extremes:
   - Test low voltage (`--vin 4.0`) and high voltage (`--vin 12.0`). Verify torque limit computation and ensure no NaNs or math domain errors.
2. Perturbations and stability:
   - Programmatically apply impulse forces/perturbations (key `P` behavior: $\pm 10$ N impulses) to trunk on each stair flight.
   - Verify that contact forces and torques saturate properly without blowing up.
   - Test standing/falling state transitions and ensure telemetry classification works reliably.
3. Record findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.

## 2026-10-11T19:49:16Z
You are M3 Challenger 2 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m3_challenger_2

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_challenger_2/DISPATCH.md`.

Your objective is to adversarially challenge BAM M6 actuator dynamics, physics limits, and perturbation response in `scripts/infer_stairs.py`:
1. Actuator voltage extremes:
   - Test low voltage (`--vin 4.0`) and high voltage (`--vin 12.0`). Verify torque limit computation and ensure no NaNs or math domain errors.
2. Perturbations and stability:
   - Programmatically apply impulse forces/perturbations (key `P` behavior: +/- 10 N impulses) to trunk on each stair flight.
   - Verify that contact forces and torques saturate properly without blowing up.
   - Test standing/falling state transitions and ensure telemetry classification works reliably.
3. Record findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.
Notify the orchestrator via `send_message` when done.
