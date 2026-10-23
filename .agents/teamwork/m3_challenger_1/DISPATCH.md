# M3 Challenger 1 Dispatch

## Target Files
- `scripts/infer_stairs.py`
- `tests/test_infer_stairs.py`

## Mission
Adversarially challenge and stress-test `scripts/infer_stairs.py` via Python and CLI:
1. Test CLI boundary and edge parameters:
   - Invalid scene path, invalid preset name, negative telemetry Hz, non-existent policy.
   - All 5 spawn presets (`--spawn bottom`, `--spawn flight1`, `--spawn flight2`, `--spawn flight3`, `--spawn top`).
   - Extreme commanded velocities (`--lin-vel-x 2.0`, `--lin-vel-x -2.0`).
2. Verify 100+ steps across all presets in headless mode without crashing or emitting NaNs.
3. Test keyboard control mappings and state transitions programmatically.
4. Record findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.


## 2026-10-11T19:49:16Z
You are M3 Challenger 1 in the Microduck RL project.
Your working directory is: /home/dhp01/microduck_rl/.agents/teamwork/m3_challenger_1

MANDATORY FIRST STEP:
Read `/home/dhp01/microduck_rl/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/dhp01/microduck_rl/.agents/teamwork/PROJECT.md`.
Read `/home/dhp01/microduck_rl/.agents/teamwork/m3_challenger_1/DISPATCH.md`.

Your objective is to adversarially challenge and stress-test `scripts/infer_stairs.py` via Python and CLI:
1. Test CLI boundary and edge parameters:
   - Invalid scene path, invalid preset name, negative telemetry Hz, non-existent policy.
   - All 5 spawn presets (`--spawn bottom`, `--spawn flight1`, `--spawn flight2`, `--spawn flight3`, `--spawn top`).
   - Extreme commanded velocities (`--lin-vel-x 2.0`, `--lin-vel-x -2.0`).
2. Verify 100+ steps across all presets in headless mode without crashing or emitting NaNs.
3. Test keyboard control mappings and state transitions programmatically.
4. Record findings and test code in `report.md` and explicit verdict (`APPROVE` or `REJECT`) in `handoff.md`.
Notify the orchestrator via `send_message` when done.
